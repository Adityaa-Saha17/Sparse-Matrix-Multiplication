# Baseline CSR SpMM + Memory Optimization

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.

---

## What I did

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

Hardware: **NVIDIA T4 (sm_75)** on Google Colab. All runs: `n=256`,
`iters=20`. Median over 20 timed iterations; speedup = baseline_ms / memopt_ms.

| m=k   | density | nnz        | base (ms) | base (GFLOPS) | memopt (ms) | memopt (GFLOPS) | speedup   | verdict     |
|-------|---------|------------|-----------|---------------|-------------|-----------------|-----------|-------------|
| 1024  | 0.050   | 52 260     | 0.0746    | 358.6         | 0.0928      | 288.3           | 0.80×     | baseline    |
| 1024  | 0.010   | 10 388     | 0.0227    | 234.1         | 0.0268      | 198.3           | 0.85×     | baseline    |
| 1024  | 0.001   | 1 044      | 0.0127    | 42.2          | 0.0119      | 44.8            | 1.07×     | memopt (noisy) |
| 4096  | 0.050   | 839 236    | 0.9358    | 459.2         | 1.2059      | 356.3           | 0.78×     | baseline    |
| 4096  | 0.010   | 167 686    | 0.2124    | 404.3         | 0.2112      | 406.6           | 1.01×     | tie         |
| 4096  | 0.001   | 16 935     | 0.0818    | 106.0         | 0.0715      | 121.2           | **1.14×** | **memopt**  |
| 8192  | 0.050   | 3 354 625  | 7.8790    | 218.0         | 7.8228      | 219.6           | 1.01×     | tie         |
| 8192  | 0.010   | 671 114    | 1.7319    | 198.4         | 1.8964      | 181.2           | 0.91×     | baseline    |
| 8192  | 0.001   | 67 137     | 0.2519    | 136.5         | 0.2661      | 129.2           | 0.95×     | baseline    |
| 16384 | 0.050   | 13 421 456 | 41.2235   | 166.7         | 48.0457     | 143.0           | 0.86×     | baseline    |
| 16384 | 0.010   | 2 684 748  | 9.9961    | 137.5         | 10.9173     | 125.9           | 0.92×     | baseline    |
| 16384 | 0.001   | 268 160    | 1.0978    | 125.1         | 1.2255      | 112.0           | 0.90×     | baseline    |

*Single Colab T4 session, `iters=20`, median timing. Run-to-run variance
observed across sessions on shared-tenant T4s: ±5–10% for m≥4096, up to
±30% for m=1024 cells (low GPU utilisation makes small-problem timing
noisy). The m=1024 d=0.001 cell's memopt win is within the noise floor.*

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

| m     | baseline GFLOPS |
|-------|-----------------|
| 1024  | 358.6           |
| 4096  | 459.2  ← peak   |
| 8192  | 218.0           |
| 16384 | 166.7           |

GFLOPS peaks at m=4096 and drops sharply at m=8192 and beyond — consistent
with the dense matrix B's working set exceeding the T4's 4 MB L2 around
m=4096 (B is m × n × 4 B = 4 MB at m=4096, n=256). Past that point the
kernel re-reads B from DRAM on every row, and aggregate GFLOPS falls
to roughly the DRAM-bandwidth-limited ceiling (~165–220 GFLOPS).

**When does memopt win vs. lose?**

memopt's only clean win on this sweep is **m=4096, density=0.001 (+14%)**.
The m=1024 d=0.001 cell also shows memopt ahead but is inside the
±20–30% small-problem noise floor. Everywhere else is a tie (m=4096
d=0.010, m=8192 d=0.050) or a baseline win — particularly **at m≥4096,
density=0.050**, where baseline beats memopt by 14–22%.

The pattern: memopt wins only where rows are very short (few nonzeros)
and the kernel's bottleneck is launch/scheduling overhead rather than
memory bandwidth. As soon as rows get long enough to saturate DRAM,
baseline's higher occupancy (8× more thread blocks) wins out.

**Root cause confirmed by ncu** (see the Profiling section below): at m=8192 d=0.01 both kernels are bandwidth-bound (90% / 85% DRAM throughput) and memopt's 63-register-per-thread footprint costs it ~3 pp of occupancy. At m=4096 d=0.001 the baseline is latency-bound (39% DRAM) and memopt's lower thread count + higher per-thread work hides the scheduling cost.

---

## Profiling — Nsight Compute findings

ncu was blocked on Kaggle (`ERR_NVGPUCTRPERM`, shared-tenant counter lockdown). Re-ran on Google Colab — same T4 hardware (`Tesla T4, CC 7.5, 40 SMs, 15 GB`), counters available. Profiles: `colabRunner.ipynb`, saved outputs `ncu_*.txt`.

### Configuration A — m=8192, density=0.01 (memopt loses ~10 %)

| metric                         | baseline   | memopt     |
|--------------------------------|-----------:|-----------:|
| Duration                       | 1.85 ms    | 2.04 ms    |
| **DRAM throughput**            | **90.3 %** | 85.2 %     |
| L1/TEX cache throughput        | 76.4 %     | 69.5 %     |
| L2 cache throughput            | 45.0 %     | 40.1 %     |
| Compute (SM) throughput        | 76.1 %     | 68.1 %     |
| Achieved occupancy             | 86.2 %     | 83.7 %     |
| Registers per thread           | 54         | **63**     |
| Grid / threads                 | 8192 / 2.1 M | 1024 / 262 K |
| Waves per SM                   | 51.2       | 6.4        |

**Both kernels are DRAM-bound** (>85 % memory throughput; ncu flags both
as "utilizing greater than 80 % of available memory performance"). In a
bandwidth-bound regime, occupancy and wave count govern who wins. Memopt
spills more state (`63` vs `54` regs/thread, hitting the SM register-block
limit of 4 blocks/SM in both cases) and dispatches 8× fewer thread blocks
(warp-per-row collapses parallelism). The result: 5 percentage points lost
on DRAM throughput and 8 pp on compute throughput → 10 % slower.

### Configuration B — m=4096, density=0.001 (memopt wins +14 %)

| metric                         | baseline    | memopt           |
|--------------------------------|------------:|-----------------:|
| Duration                       | 122.6 µs    | 71.5 µs (from sweep) |
| **DRAM throughput**            | **39.0 %**  | (n/a — output trimmed) |
| L1/TEX cache throughput        | 42.3 %      | n/a              |
| Compute (SM) throughput        | 41.1 %      | n/a              |
| Achieved occupancy             | 75.6 %      | 75.1 %           |
| Grid / threads                 | 4096 / 1.05 M | 512 / 131 K   |
| Waves per SM                   | 25.6        | 3.2              |
| Avg nnz / row                  | ≈ 4         | ≈ 4              |

ncu's own diagnostic on the baseline:
> *Achieved compute throughput and/or memory bandwidth below 60.0 % of peak
> typically indicate latency issues. Look at Scheduler Statistics and Warp
> State Statistics for potential reasons.*

The baseline is **latency-bound**, not bandwidth-bound. With ~4 nonzeros
per row, each baseline thread does almost no arithmetic — most of its
time is the row_ptr lookup, the loop prologue, and the final store. It
launches 1 M threads with 8 rows per block, so every block re-reads the
same row_ptr values 32 times.

memopt's warp-per-row pattern launches 8× fewer threads but gives each
thread 256× more arithmetic work (the `for col = lane; col < N; col += 32`
loop). Per-row state (`row_start`, `row_end`) is read once per warp instead
of 32 times. The reduction in launch/scheduling overhead and the higher
per-thread arithmetic intensity win the regime — even though peak DRAM
throughput is well below the bandwidth ceiling.

### Why the pattern flips across the sweep

| regime                          | bottleneck   | who wins   | why                                        |
|---------------------------------|-------------|-----------|---------------------------------------------|
| dense, large-row (e.g. 8192 d=0.01) | DRAM bandwidth | baseline | higher occupancy + more waves/SM            |
| sparse, short-row (e.g. 4096 d=0.001) | scheduling latency | memopt | fewer threads, more work each, less overhead |

This explains the bimodal verdicts in the sweep table: memopt only beats
baseline when the baseline is *not* saturating DRAM. The "memopt sweet
spot" is therefore narrow on T4 — bounded above by register pressure
(63 regs/thread limits occupancy in dense regimes) and below by the small
problem-size noise floor.

### AI - USE

Utilized AI tools such as ***ChatGPT*** and ***Perplexity AI*** to assist with research, gathering relevant information, and developing an appropriate design structure and implementation plan for the project. ***Claude AI*** was used for code validation and testing, formatting, adding comments, and refining the language and presentation of the report.