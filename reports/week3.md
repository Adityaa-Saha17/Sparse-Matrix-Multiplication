# Week 3 — Tensor Core SpMM via WMMA (Phase 3 — Milestone Report)

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.
**Week dates:** 2026-05-27 to 2026-06-02.

---

## Overview — full progression

This is the milestone formal report covering all three phases of the project.

| Phase | Technique | Best speedup (DoD cells) |
|---|---|---|
| Phase 1 (week 1) | Baseline CSR / warp-per-row | up to 1.14× |
| Phase 2 (week 2) | Tiled shmem staging, scalar lane-strided loads | 2.04–2.69× (2/4 DoD cells ≥2×) |
| Phase 3 (week 3) | BSR 16×16 + WMMA Tensor Cores, FP16→FP32 | [TBD from Colab run] |

---

## What I did this week

Built the Tensor Core acceleration layer on top of the Phase 2 infrastructure:

- [`src/bsr.h`](../src/bsr.h) / [`src/bsr.cu`](../src/bsr.cu) — Block-Sparse Row (BSR) format with fixed 16×16 blocks. CSR→BSR conversion on the host: collects unique block-columns per block-row, zero-pads sparse blocks to dense 16×16 FP16 tiles.
- [`src/kernels/spmm_wmma.cu`](../src/kernels/spmm_wmma.cu) — WMMA kernel. Grid: `(block_rows, N/16)`; block: 32 threads (one warp). Each warp accumulates one 16×16 FP32 output tile via `mma_sync` over all non-empty BSR blocks in its block-row.
- Harness extended: `--kernel wmma` builds BSR and converts B to FP16 once (excluded from timing), then benchmarks the TC kernel. `--kernel all` now runs all eight kernels.

**Key design choices:**
- FP16 A+B, FP32 accumulate — matches the WMMA `m=16 n=16 k=16` fragment type supported on Turing (sm_75).
- One warp per output tile — simplest possible WMMA mapping; occupancy is 50% (16 warps/SM) but TC throughput compensates in compute-bound regimes.
- FLOPS denominator = original nnz (not padded block elements) — keeps speedup numbers comparable across phases.

---

## BSR format analysis — fill-in overhead

For **uniform random** sparsity, almost every 16×16 block contains at least one nonzero once density ≥ 0.001. This means BSR stores many near-empty blocks as dense FP16 tiles, which increases memory traffic relative to CSR.

Expected fill-in ratios (probability a block is non-empty × 256 elements / avg nnz per block):

| density | P(block non-empty) | avg nnz/block | fill-in ratio |
|---|---|---|---|
| 0.050 | ≈ 1.000 | 12.80 | **20×** |
| 0.010 | ≈ 0.923 | 2.78  | **92×** |
| 0.001 | ≈ 0.224 | 1.14  | **225×** |

This means the WMMA kernel must deliver a very large compute speedup (from TC throughput) to overcome the memory overhead. Tensor Cores on T4 provide ~65 TFLOPS FP16 vs ~8.1 TFLOPS FP32 — an 8× raw throughput advantage — but for random sparse inputs the fill-in erases most of this gain. BSR+TC is architecturally optimal for **structured block-sparse** matrices (e.g., block-diagonal, attention pattern, GNN). The sweep below tests the random case and confirms or refutes this analysis.

---

## Build / footprint (ptxas, sm_75, -O3)

| kernel | regs/thread | shmem/block | spills |
|---|---|---|---|
| `tiled_v3` (Phase 2 winner) | 41 | 4096 B | 0 |
| `wmma` | [TBD] | 0 B (fast path) | [TBD] |

*Fill in after running Step 11 of colabRunner.ipynb.*

---

## Results — Phase 3 sweep on Colab T4

Median ms over 20 timed iterations, 5 warmup, `n = 256`. Bold = fastest in row.
Speedup = wmma vs. baseline (same denominator as Phases 1 & 2).

| m=k | density | baseline | tiled_v3 | **wmma** | wmma vs baseline | wmma vs tiled_v3 |
|---|---|---|---|---|---|---|
| 1024 | 0.050 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 1024 | 0.010 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 1024 | 0.001 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 4096 | 0.050 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 4096 | 0.010 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 4096 | 0.001 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 8192 | 0.050 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 8192 | 0.010 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 8192 | 0.001 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 16384 | 0.050 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 16384 | 0.010 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 16384 | 0.001 | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |

*Fill in from Step 12 + Step 13 of colabRunner.ipynb.*

---

## BSR statistics (from harness output)

The harness prints a `bsr_stats` line before each wmma timing row:

| m=k | density | num_blocks | stored_elems | fill_in_ratio | block_density |
|---|---|---|---|---|---|
| 1024 | 0.050 | [TBD] | [TBD] | [TBD] | [TBD] |
| 1024 | 0.010 | [TBD] | [TBD] | [TBD] | [TBD] |
| 1024 | 0.001 | [TBD] | [TBD] | [TBD] | [TBD] |
| 4096 | 0.050 | [TBD] | [TBD] | [TBD] | [TBD] |
| 4096 | 0.010 | [TBD] | [TBD] | [TBD] | [TBD] |
| 4096 | 0.001 | [TBD] | [TBD] | [TBD] | [TBD] |
| 8192 | 0.050 | [TBD] | [TBD] | [TBD] | [TBD] |
| 8192 | 0.010 | [TBD] | [TBD] | [TBD] | [TBD] |
| 8192 | 0.001 | [TBD] | [TBD] | [TBD] | [TBD] |
| 16384 | 0.050 | [TBD] | [TBD] | [TBD] | [TBD] |
| 16384 | 0.010 | [TBD] | [TBD] | [TBD] | [TBD] |
| 16384 | 0.001 | [TBD] | [TBD] | [TBD] | [TBD] |

---

## Correctness

All WMMA outputs must agree with `cusparseSpMM` to `rel_l2_err < 1e-4`.
Expected: FP16 computation introduces rounding; typical error floor is ~1e-3 to 1e-4,
which is higher than Phase 2's FP32 floor of ~4e-7 but still within DoD.

| sweep cell | rel_l2_err |
|---|---|
| m=1024 d=0.05 | [TBD] |
| m=1024 d=0.01 | [TBD] |
| m=1024 d=0.001 | [TBD] |
| m=4096 d=0.05 | [TBD] |
| m=4096 d=0.01 | [TBD] |
| m=4096 d=0.001 | [TBD] |
| m=8192 d=0.05 | [TBD] |
| m=8192 d=0.01 | [TBD] |
| m=8192 d=0.001 | [TBD] |
| m=16384 d=0.05 | [TBD] |
| m=16384 d=0.01 | [TBD] |
| m=16384 d=0.001 | [TBD] |

---

## DoD verdict

Three independent Colab T4 runs, wmma vs baseline on the 4 DoD cells:

| DoD cell | run 1 | run 2 | run 3 | **median** | passes ≥2×? |
|---|---|---|---|---|---|
| m=4096 d=0.05 | [TBD] | [TBD] | [TBD] | **[TBD]** | [?] |
| m=4096 d=0.01 | [TBD] | [TBD] | [TBD] | **[TBD]** | [?] |
| m=8192 d=0.05 | [TBD] | [TBD] | [TBD] | **[TBD]** | [?] |
| m=8192 d=0.01 | [TBD] | [TBD] | [TBD] | **[TBD]** | [?] |

**Expected outcome (pre-run analysis):** The m=4096 cells are unlikely to improve over
Phase 2 for *random* sparsity because the fill-in ratio (~20–92×) forces far more memory
traffic than the original CSR. The m=8192 cells are bandwidth-bound; the extra memory
traffic from BSR zero-padding will make things worse, not better. TC throughput helps
only when the kernel is genuinely compute-bound AND the block structure is dense — neither
condition holds for uniform random sparse matrices.

If the above is confirmed by the run, the conclusion is: **for random sparsity, BSR+WMMA
is dominated by tiled CSR; TC gains only materialise for structured (block-sparse) matrices.**

---

## Performance analysis — ncu (Nsight Compute)

### C2 (m=4096 d=0.05) — expected L2/compute-bound

| metric | tiled_v3 (Phase 2) | wmma |
|---|---|---|
| Compute SM Throughput | [TBD] % | [TBD] % |
| Memory Throughput | [TBD] % | [TBD] % |
| DRAM Throughput | [TBD] % | [TBD] % |
| Achieved Occupancy | [TBD] % | [TBD] % |
| SM Active Cycles | [TBD] | [TBD] |

### A2 (m=8192 d=0.01) — expected bandwidth-bound

| metric | tiled_v3 (Phase 2) | wmma |
|---|---|---|
| Compute SM Throughput | [TBD] % | [TBD] % |
| Memory Throughput | [TBD] % | [TBD] % |
| DRAM Throughput | [TBD] % | [TBD] % |
| Achieved Occupancy | [TBD] % | [TBD] % |
| SM Active Cycles | [TBD] | [TBD] |

*Fill in from Steps 14–15 of colabRunner.ipynb.*

---

## Full-project progression summary (baseline → TC)

| kernel | m=8192 d=0.01 (ms) | speedup vs baseline | m=4096 d=0.05 (ms) | speedup vs baseline |
|---|---|---|---|---|
| baseline | 1.7320 | 1.00× | 1.0651 | 1.00× |
| memopt | 1.9132 | 0.91× | 1.2421 | 0.86× |
| tiled | 0.9300 | 1.86× | 0.9963 | 1.07× |
| tiled_v3 | 0.8497 | **2.04×** | 0.8537 | **1.25×** |
| tiled_v4 | 0.9202 | 1.88× | 0.9162 | 1.16× |
| **wmma** | [TBD] | **[TBD]** | [TBD] | **[TBD]** |

---

## Key lessons

**Phase 1:** Warp-per-row (memopt) wins only when rows are very short (latency-bound); loses to baseline in bandwidth-bound regimes due to lower occupancy.

**Phase 2:** Shared-memory staging of nnz tiles with 128-col register tiles (tiled_v3) cuts DRAM re-fetches by 4× in bandwidth-bound regimes (→ 2.04–2.69× speedup). The m=4096 cells hit an L2-cache roofline ceiling (~1.3×): B fits in the 4 MB L2, so there is no DRAM traffic to optimise away.

**Phase 3 (to be confirmed):** For *random* sparsity, BSR 16×16 blocks have 20–225× fill-in ratios that erase TC's 8× throughput advantage. TC acceleration is most valuable for *structured* block-sparse patterns (attention heads, GNN, convolution im2col) where blocks are nearly full. An avenue for future work is adaptive BSR with variable block sizes, or a CSR-to-TC path that packs multiple short rows into a single 16×16 WMMA tile.

---

## AI - USE

Utilized AI tools such as ***ChatGPT*** and ***Perplexity AI*** to assist with research, gathering relevant information, and developing an appropriate design structure and implementation plan for the project. ***Claude AI*** was used for code validation and testing, formatting, adding comments, and refining the language and presentation of the report.
