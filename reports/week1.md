# Week 1 — Baseline CSR SpMM + Memory Optimization

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.
**Phase 1 goal:** Working baseline CSR SpMM kernel and a memory-optimized variant,
both validated against cuSPARSE, with timing recorded across sizes and sparsity levels.

---

## What I did this week

- Scaffolded the project (Makefile targeting `sm_75`, source layout under
  `src/`, scripts under `scripts/`).
- Implemented CSR data structures and host↔device transfer
  ([src/csr.h](../src/csr.h), [src/csr.cu](../src/csr.cu)).
- Wrote utilities: CUDA / cuSPARSE error-check macros and a `GpuTimer` based on
  CUDA events ([src/utils.h](../src/utils.h)).

**Baseline kernel** ([src/kernels/spmm_baseline.cu](../src/kernels/spmm_baseline.cu)):
- One thread per output element; threads in a warp have consecutive C-columns,
  so reads of the dense matrix B are coalesced.

**Memory-optimized kernel** ([src/kernels/spmm_memopt.cu](../src/kernels/spmm_memopt.cu)):
- Warp-per-row: each 32-thread warp handles one row of A.
- Each lane computes columns `lane, lane+32, lane+64, …` (column-strided), so
  all 32 threads in a warp access consecutive B columns — coalesced.
- Reduces redundant `row_ptr` lookups that plague the baseline.

**Benchmark harness** ([src/bench/harness.cu](../src/bench/harness.cu)):
- Generates a synthetic uniform-random CSR matrix and a dense B.
- Runs `cusparseSpMM` as the correctness reference.
- Runs each kernel: 3 warmup + 20 timed iterations, reports median time.
- Correctness metrics: `max_abs_err`, `max_rel_err` (filtered: only cells
  where `|ref| ≥ 0.1% × max|ref|`), and `rel_l2_err = ‖ours−ref‖₂/‖ref‖₂`.

**Synthetic matrix generator** ([scripts/gen_matrices.py](../scripts/gen_matrices.py)):
- Supports uniform, banded, block-diagonal, and power-law sparsity patterns.
- Outputs binary CSR files for use with `--bin`.

---

## Results

Hardware: **NVIDIA T4 (sm_75)** on Kaggle. All runs: `n=256`, `iters=20`.
Median over 20 timed iterations; speedup = baseline_ms / memopt_ms.

| m=k   | density | nnz (approx) | base (ms) | base (GFLOPS) | memopt (ms) | memopt (GFLOPS) | speedup | verdict        |
|-------|---------|-------------|-----------|--------------|-------------|-----------------|---------|----------------|
| 1024  | 0.050   | 52 429      | 0.121     | 221.9        | 0.149       | 180.2           | 0.81×   | baseline       |
| 1024  | 0.010   | 10 486      | 0.031     | 173.2        | 0.037       | 145.1           | 0.84×   | baseline       |
| 1024  | 0.001   | 1 049       | 0.018     | 29.8         | 0.016       | 33.6            | 1.07×   | ~tie           |
| 4096  | 0.050   | 838 861     | 1.378     | 311.7        | 1.397       | 307.4           | 0.99×   | tie            |
| 4096  | 0.010   | 167 772     | 0.339     | 253.4        | 0.341       | 251.9           | 0.99×   | tie            |
| 4096  | 0.001   | 16 777      | 0.108     | 79.5         | 0.082       | 104.7           | **1.32×** | **memopt** |
| 8192  | 0.050   | 3 355 443   | 8.140     | 211.1        | 7.520       | 228.4           | **1.08×** | **memopt** |
| 8192  | 0.010   | 671 089     | 1.733     | 198.3        | 1.894       | 181.4           | 0.92×   | baseline       |
| 8192  | 0.001   | 67 109      | 0.251     | 136.9        | 0.267       | 128.7           | 0.94×   | baseline       |
| 16384 | 0.050   | 13 421 773  | ~43.8     | ~156.9       | ~43.5       | ~157.9          | ~1.01×  | tie (noisy)    |
| 16384 | 0.010   | 2 684 355   | 10.010    | 137.3        | 10.910      | 125.9           | 0.92×   | baseline       |
| 16384 | 0.001   | 268 436     | 1.096     | 125.4        | 1.227       | 112.0           | 0.89×   | baseline       |

*Numbers stable across 3 independent sweeps; 1024-size cells show ~20–30% run-to-run variance
(low GPU utilisation at small problem sizes makes timing noisy).*

---

## Correctness

Both kernels agree with `cusparseSpMM` across all 12 cells:

| metric     | range observed        | DoD threshold |
|------------|-----------------------|---------------|
| rel_l2_err | 7.9e-9 – 5.3e-7       | < 1e-4  ✓     |
| max_rel_err (filtered) | < 1e-5   | < 1e-4  ✓     |

`rel_l2_err` is the primary pass criterion (global L2 relative error, robust to
cells near zero). `max_rel_err` uses a floor of `0.1% × max|ref|` to suppress
noise from near-zero reference values.

---

## Performance analysis

**Baseline GFLOPS vs. matrix size (density 0.05)**

| m    | baseline GFLOPS |
|------|-----------------|
| 1024 | 221.9           |
| 4096 | 311.7  ← peak   |
| 8192 | 211.1           |
| 16384| 156.9           |

GFLOPS peaks at m=4096 and *decreases* at larger sizes — a sign of load imbalance
(highly variable row lengths → warp divergence) and/or L2 cache thrashing once
the working set of B exceeds ~40 MB (T4 has 4 MB L2).

**When does memopt win vs. lose?**

memopt is a warp-per-row kernel. It outperforms the baseline in the **sparse,
large-row** regime (m=4096, density=0.001; m=8192, density=0.05) where each warp
processes a long row and the warp-stride access pattern amortises the overhead of
the outer column loop.

It loses in the **dense, small-problem** regime (m=1024, density≥0.01) where the
overhead of the warp-centric dispatch (extra arithmetic in the column-stride loop)
outweighs its memory access advantage, and overall occupancy is low enough that
the scheduler cannot hide the latency.

**Root cause (to be confirmed by ncu):**
Hypothesis — at m=8192, density=0.01 (memopt loses), memory throughput is not the
bottleneck; warp stalls due to control-flow divergence (rows with very different
lengths) dominate. At m=4096, density=0.001 (memopt wins), the working set fits
better in L1/L2 and the warp-stride access pattern achieves higher memory efficiency.

---

## Profiling (pending — Phase 1 close-out)

Target cells for `ncu --set basic`:

| cell | kernel | purpose |
|------|--------|---------|
| m=8192, density=0.01 | both | why memopt loses (−8%) |
| m=4096, density=0.001 | both | why memopt wins (+32%) |

Commands (run on Kaggle T4):
```bash
ncu --set basic --target-processes all --kernel-name regex:spmm \
    ./build/spmm_bench --m 8192 --k 8192 --n 256 --density 0.01 \
    --iters 1 --warmup 0 --kernel baseline 2>&1 | tail -60
```
Repeat with `--kernel memopt` and the m=4096 d=0.001 pair.
Key counters to compare: `sm__throughput`, `l1tex__t_bytes`, `smsp__warp_issue_stalled_*`.

---

## Next week (Phase 2 — deeper memory optimizations)

1. Run ncu profiles for the two target cells above; add findings here.
2. Shared-memory tiling of dense B (tile size 16×64).
3. Register accumulation + manual loop unrolling for the inner dot-product.
4. Vectorized loads (`float4`, `__ldg`).

Target: cumulative ≥2× speedup over baseline in the 4096–8192 range (density 0.01–0.05);
nsight-compute profiling for validation.
