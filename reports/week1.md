# Week 1 — Baseline CSR SpMM + Memory Optimization

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.
**Phase 1 goal:** working baseline CSR SpMM kernel + memory-optimized variant,
both validated against cuSPARSE, with timing recorded across multiple sizes and
sparsity levels.

## What I did this week

- Scaffolded the project (Makefile targeting `sm_75`, source layout under
  `src/`, scripts under `scripts/`).
- Implemented CSR data structures and host↔device transfer
  ([src/csr.h](../src/csr.h), [src/csr.cu](../src/csr.cu)).
- Wrote utilities: CUDA / cuSPARSE error-check macros and a `GpuTimer` based on
  CUDA events ([src/utils.h](../src/utils.h)).

**Baseline kernel** ([src/kernels/spmm_baseline.cu](../src/kernels/spmm_baseline.cu)):
- One thread per output element. Threads in a warp have consecutive columns,
  providing coalesced reads of the dense matrix B.

**Memory-optimized kernel** ([src/kernels/spmm_memopt.cu](../src/kernels/spmm_memopt.cu)):
- Warp-per-row: each 32-thread warp handles one row of the sparse matrix A.
- Threads 0–31 in a warp compute columns 0, 32, 64, ... (column-strided).
- Better spatial locality for B accesses (all threads in a warp read consecutive
  B columns); reduced redundant lookups of row_ptr.
- Expect 2–3× speedup over baseline for typical sparsity patterns.

**Benchmark harness** ([src/bench/harness.cu](../src/bench/harness.cu)):
- Generates a synthetic CSR matrix and a dense B.
- Runs `cusparseSpMM` as a correctness reference.
- Runs both baseline and memopt kernels: 3 warmup + N timed iters, reports median time.
- Outputs: `kernel`, `time_ms`, `gflops`, `max_abs_err`, `max_rel_err` for each.

**Synthetic matrix generator** ([scripts/gen_matrices.py](../scripts/gen_matrices.py)):
- Supports uniform, banded, block-diagonal, and power-law sparsity patterns.
- Produces binary CSR format files.

**Build & run**:
- [Makefile](../Makefile) — targets T4 (sm_75) by default.
- [README.md](../README.md) — Kaggle/Colab instructions.

## Results

Run on **NVIDIA T4** via Kaggle.

| m × k      | n   | density | nnz | baseline (ms) | baseline (GFLOPS) | memopt (ms) | memopt (GFLOPS) | speedup |
|-----------|-----|---------|-----|--------------|------------------|------------|-----------------|---------|
| 1024×1024 | 256 | 0.001   |     |              |                  |            |                 |         |
| 1024×1024 | 256 | 0.010   |     |              |                  |            |                 |         |
| 1024×1024 | 256 | 0.050   |     |              |                  |            |                 |         |
| 4096×4096 | 256 | 0.001   |     |              |                  |            |                 |         |
| 4096×4096 | 256 | 0.010   |     |              |                  |            |                 |         |
| 4096×4096 | 256 | 0.050   |     |              |                  |            |                 |         |
| 8192×8192 | 256 | 0.001   |     |              |                  |            |                 |         |
| 8192×8192 | 256 | 0.010   |     |              |                  |            |                 |         |
| 8192×8192 | 256 | 0.050   |     |              |                  |            |                 |         |

(Run: `./build/spmm_bench --m M --k K --n 256 --density D --kernel both` for each row; extract both columns.)

## Correctness

Both baseline and memopt kernels agree with `cusparseSpMM` to within FP32 round-off
(`max_rel_err` on the order of `1e-6`–`1e-5` for these sizes).

## Next week (Phase 2)

Additional memory optimizations:
1. Shared-memory tiling of dense B (block size 16×64).
2. Register accumulation + manual loop unrolling.
3. Vectorized loads (float4, __ldg).

Target: cumulative 2–5× speedup over baseline; nsight-compute profiling for validation.