# Tensor Core SpMM via WMMA (Milestone Report)

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.

---

## Overview — full progression

| Approach | Technique | Best speedup (DoD cells) |
|---|---|---|
| Baseline | Baseline CSR / warp-per-row memopt | up to 1.14× |
| Memory optimization | Tiled shmem staging, 128-col register tile (tiled_v3) | 2.04–2.69× (2/4 DoD cells ≥ 2×) |
| Tensor Core SpMM | BSR 16×16 + WMMA Tensor Cores, FP16 A+B / FP32 accumulate | **0.10–0.91×** (all DoD cells slower than baseline) |

---

## What I did

Built the Tensor Core acceleration layer on top of the memory optimization infrastructure:

- [`src/bsr.h`](../src/bsr.h) / [`src/bsr.cu`](../src/bsr.cu) — Block-Sparse Row (BSR) format with fixed 16×16 blocks. CSR→BSR conversion on the host: collects unique block-columns per block-row, zero-pads sparse blocks to dense FP16 tiles. Reports fill-in statistics.
- [`src/kernels/spmm_wmma.cu`](../src/kernels/spmm_wmma.cu) — WMMA kernel. Grid: `(block_rows, N/16)`; block: 32 threads (one warp). Each warp accumulates one 16×16 FP32 output tile via `mma_sync` over all non-empty BSR blocks in its block-row.
- Harness extended: `--kernel wmma` builds BSR and converts B to FP16 once (excluded from timing), then benchmarks the TC kernel. Prints `bsr_stats` (fill-in ratio, block density) per cell. `--kernel all` now runs all eight kernels.

---

## Build / footprint (ptxas, sm_75, -O3)

| kernel | regs/thread | shmem/block | spills |
|---|---|---|---|
| `memopt`    | 63 | 0 B    | 0 |
| `tiled_v3`  | 41 | 4096 B | 0 |
| `wmma`      | 62 | 1024 B | 0 |

The `wmma` kernel reports 1024 B shmem even though the slow-path (boundary tiles) is dead code for all test sizes — ptxas allocates shared memory for any `__shared__` declaration in scope. Occupancy: 32 threads/block → 16 blocks/SM (hardware limit) → 16 warps/SM = 50 % warp occupancy on T4.

---

## BSR fill-in analysis

For **uniform random** sparsity, the expected fraction of non-empty 16×16 blocks is ≈ 1 − (1 − d)²⁵⁶. At d = 0.05 that is essentially 1.0 — every block is occupied. The fill-in ratios measured from the harness:

| m=k | density | num_blocks | stored_elems | fill_in_ratio | block_density |
|---|---|---|---|---|---|
| 1024 | 0.050 | 4 096 | 1 048 576 | 20.1× | 1.000 |
| 1024 | 0.010 | 3 761 | 962 816 | 92.7× | 0.918 |
| 1024 | 0.001 | 917 | 234 752 | 224.9× | 0.224 |
| 4096 | 0.050 | 65 536 | 16 777 216 | 20.0× | 1.000 |
| 4096 | 0.010 | 60 630 | 15 521 280 | 92.6× | 0.925 |
| 4096 | 0.001 | 14 924 | 3 820 544 | 225.6× | 0.228 |
| 8192 | 0.050 | 262 144 | 67 108 864 | 20.0× | 1.000 |
| 8192 | 0.010 | 242 248 | 62 015 488 | 92.4× | 0.924 |
| 8192 | 0.001 | 59 324 | 15 186 944 | 226.2× | 0.226 |
| 16384 | 0.050 | 1 048 574 | 268 434 944 | 20.0× | 1.000 |
| 16384 | 0.010 | 968 648 | 247 973 888 | 92.4× | 0.924 |
| 16384 | 0.001 | 236 629 | 60 577 024 | 225.9× | 0.226 |

Tensor Cores on T4 deliver ~8× raw FP16 throughput vs FP32. But at 20–226× fill-in, the kernel must read and multiply 20–226× more data than the original nnz. TC cannot recover from a 20–226× memory bandwidth penalty with an 8× compute advantage.

---

## Correctness

The WMMA kernel accumulates in FP32 but the individual multiply operands are FP16 (10-bit mantissa, ~3 decimal digits). Summing many FP16 products — especially in the dense blocks that BSR creates — causes rounding error to accumulate.

| DoD threshold | wmma rel_l2_err (all cells) | passes? |
|---|---|---|
| < 1e-4 | **≈ 2.61e-4** (uniform) | ❌ |

The error is consistent and structural — it does not vary with m or density. The FP32 kernels (the CSR kernels) all achieved rel_l2_err ≤ 4×10⁻⁷. Tensor Core SpMM introduces a ~4-order-of-magnitude regression in accuracy alongside the performance regression. The `max_rel_err` on individual elements reaches ~16% (1.6×10⁻¹), confirming that some output elements are seriously wrong due to catastrophic cancellation in the FP16 products.

---

## Results — Tensor Core SpMM sweep on Colab T4

Median ms over 20 timed iterations, 5 warmup, `n = 256`. Baseline and tiled_v3 values from the memory-optimization report (same hardware). Speedup = baseline_ms / wmma_ms (>1 means wmma faster).

| m=k | density | baseline (ms) | tiled_v3 (ms) | wmma (ms) | wmma vs baseline | wmma vs tiled_v3 |
|---|---|---|---|---|---|---|
| 1024 | 0.050 | 0.0804 | 0.0610 | **0.2683** | 0.30× | 0.23× |
| 1024 | 0.010 | 0.0225 | 0.0196 | **0.1406** | 0.16× | 0.14× |
| 1024 | 0.001 | 0.0129 | 0.0080 | **0.0388** | 0.33× | 0.21× |
| 4096 | 0.050 | 1.0651 | 0.8537 | **2.1197** | 0.50× | 0.40× |
| 4096 | 0.010 | 0.1903 | 0.1599 | **1.9872** | 0.096× | 0.080× |
| 4096 | 0.001 | 0.0735 | 0.0589 | **0.6101** | 0.12× | 0.097× |
| 8192 | 0.050 | 10.21 | 3.96 | **14.1382** | 0.72× | 0.28× |
| 8192 | 0.010 | 1.7320 | 0.8497 | **8.1406** | 0.21× | 0.10× |
| 8192 | 0.001 | 0.2502 | 0.2089 | **2.0966** | 0.12× | 0.10× |
| 16384 | 0.050 | 46.67 | 28.46 | **37.1856** | **1.26×** | 0.77× |
| 16384 | 0.010 | 10.03 | 6.75 | **34.2521** | 0.29× | 0.20× |
| 16384 | 0.001 | 1.131 | 0.929 | **9.0851** | 0.12× | 0.10× |

The single partial win at m=16384 d=0.05 (1.26× vs baseline) is explained by the L2 cache pressure at that size: with B = 16384×256×4 B ≈ 16 MB of dense matrix data well beyond the 4 MB L2, the baseline's memory-bound regime is almost as bad as BSR's inflated working set. Even so, tiled_v3 at 28.46 ms is still 1.31× faster than wmma's 37.19 ms.

---

## DoD verdict

Head-to-head on the 4 DoD cells (all kernels, same session):

| DoD cell | baseline | tiled_v3 | wmma | wmma vs baseline | passes ≥ 2×? |
|---|---|---|---|---|---|
| m=4096 d=0.05 | 0.8364 ms | 0.8570 ms | 2.0799 ms | **0.40×** (2.5× slower) | ❌ |
| m=4096 d=0.01 | 0.1910 ms | 0.1537 ms | 1.9693 ms | **0.097×** (10× slower) | ❌ |
| m=8192 d=0.05 | 8.0697 ms | 3.6471 ms | 8.8918 ms | **0.91×** (≈ same speed) | ❌ |
| m=8192 d=0.01 | 1.7323 ms | 0.8335 ms | 8.1674 ms | **0.21×** (4.7× slower) | ❌ |

**0 of 4 DoD cells pass** (and the correctness bar also fails at rel_l2_err ≈ 2.61e-4 > 1e-4).

The result confirms the fill-in analysis exactly: for uniform random sparsity, BSR+WMMA is substantially worse than the baseline CSR kernel in every DoD cell. The m=4096 gap that memory optimization could not close to ≥ 2× remains unclosed; TC makes it significantly worse.

---

## Performance analysis

### Why WMMA loses at every scale

**GFLOPS computed against original nnz:**

| density | wmma GFLOPS (measured) | fill-in ratio | actual ops / original ops | effective GFLOPS |
|---|---|---|---|---|
| 0.050 | ~100–203 | 20× | 20× | **~9 GFLOPS** |
| 0.010 | ~38–43 | 92× | 92× | **~0.5 GFLOPS** |
| 0.001 | ~14–16 | 225× | 225× | **~0.07 GFLOPS** |

The "measured GFLOPS" is normalised against original nnz and gives a flattering but misleading picture. The kernel is actually doing 20–225× more FMA operations than the nnz count implies. Even at 200 GFLOPS (near the T4's FP16 TC roof), the 20× fill-in means only ~10 useful GFLOPS reach the output. The tiled_v3 CSR kernel achieves 395–556 GFLOPS (FP32, memory optimization data) with zero fill-in waste.

### Why m=16384 d=0.05 is the only partial win

At m=16384 the dense matrix B is 16384 × 256 × 4 B ≈ 16 MB — far beyond the 4 MB L2 cache. The baseline is fully DRAM-bandwidth-bound. BSR at d=0.05 has a 20× fill-in but also benefits from coalesced TC loads; the ratio of wasted bandwidth is smaller relative to the already-saturated DRAM bus. This is the only regime where fill-in overhead is partly hidden by already-saturated memory bandwidth. But tiled_v3 still wins because it carries no fill-in penalty at all.

### ncu profile

The ncu cells failed in this Colab session due to a `getcwd` path error — the shell's working directory was corrupted after the Python `subprocess.run` calls in Step 13. The ncu cells in the notebook have been updated to use absolute paths (`/content/spmm/build/spmm_bench`) so they will run correctly in the next session.

---

## Full-project progression summary

| kernel | m=8192 d=0.01 (ms) | speedup | m=4096 d=0.05 (ms) | speedup |
|---|---|---|---|---|
| baseline  | 1.7320 | 1.00× | 1.0651 | 1.00× |
| memopt    | 1.9132 | 0.91× | 1.2421 | 0.86× |
| tiled     | 0.9300 | 1.86× | 0.9963 | 1.07× |
| tiled_v3  | **0.8497** | **2.04×** | **0.8537** | **1.25×** |
| tiled_v4  | 0.9202 | 1.88× | 0.9162 | 1.16× |
| **wmma**  | 8.1406 | **0.21×** | 2.1197 | **0.50×** |

tiled_v3 remains the best overall kernel. The TC path is not competitive for random sparsity.

---

## Key lessons

**the baseline:** Warp-per-row (memopt) wins only for latency-bound regimes (very short rows); loses in bandwidth-bound regimes due to lower occupancy and register pressure.

**memory optimization:** Shared-memory staging of nnz tiles with 128-col register tiles (tiled_v3) cuts DRAM re-fetches and wins the bandwidth-bound regime (2.04–2.69×). The m=4096 cells hit an L2-cache ceiling (~1.25× max) — there is no DRAM bandwidth to recover.

**Tensor Core SpMM:** For *random* sparse matrices, BSR 16×16 blocks carry a 20–226× fill-in penalty. T4 Tensor Cores provide only 8× raw throughput over FP32. The memory overhead completely erases the compute advantage, making WMMA uniformly slower than the FP32 baseline and 2.4–13× slower than tiled_v3 across the DoD cells. Additionally, FP16 arithmetic introduces rel_l2_err ≈ 2.6×10⁻⁴, exceeding the 1×10⁻⁴ correctness threshold.

**Conclusion:** TC acceleration for SpMM is beneficial only for *structured block-sparse* matrices — attention patterns, GNN adjacency blocks, convolution im2col — where each block is densely populated and fill-in overhead approaches 1×. For uniform random sparsity the optimal strategy on T4 remains FP32 CSR with shared-memory tiling (tiled_v3).

**Potential future directions:**
- Row-packing: merge multiple short CSR rows into one 16×16 WMMA tile to amortize overhead.
- Adaptive format selection: use CSR for low-density regimes, BSR for high-density structured patterns.
- Structured matrices: re-run the sweep with block-diagonal patterns (`gen_matrices.py --pattern block_diagonal`) where BSR fill-in = 1×.

---

## AI - USE

Utilized AI tools such as ***ChatGPT*** and ***Perplexity AI*** to assist with research, gathering relevant information, and developing an appropriate design structure and implementation plan for the project. ***Claude AI*** was used for code validation and testing, formatting, adding comments, and refining the language and presentation of the report.
