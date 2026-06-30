# CUDA Sparse Matrix Multiplication

Architecture-aware Sparse × Dense matrix multiplication (SpMM) on NVIDIA GPUs.
Target hardware: **NVIDIA T4** (sm_75). Development and dual-GPU experiments
run on Kaggle (free dual-T4); profiling runs on Google Colab (single T4,
where Nsight Compute counters are accessible).

---

## Project status

| # | Task                                                              | Status         |
|---|-------------------------------------------------------------------|----------------|
| 1 | Baseline CSR SpMM on CUDA cores                                   | ✅ done         |
| 2 | Memory optimization (coalescing, shared-mem tiling, register reuse) | ✅ done (6 kernels shipped; DoD 2/4 cells pass — empirical cap on m=4096 documented in the memory-optimization report) |
| 3 | Tensor Core acceleration (block-sparse on T4)                     | ✅ done (BSR 16×16 + WMMA; loses on random sparsity from fill-in, but 851 GFLOP/s / up to 1.81× on structured block-diagonal — see the Tensor-Core and roofline reports) |
| 4 | Hybrid execution (CUDA cores vs. Tensor Cores)                    | ✅ done (structure-aware auto-dispatch in `spmm_hybrid`; routes by BSR fill-in) |
| 5 | Memory-aware execution (unified memory / multi-GPU)               | ✅ done (nnz-balanced dual-GPU row split + unified-memory path in `spmm_multigpu`; measured **2.40× on 2×T4** at m=16384 d=0.01 — 199→478 GFLOP/s) |
| 6 | Performance evaluation across patterns and sizes                  | 🟡 ongoing      |
| 7 | Final integrated system                                           | ✅ done (`spmm_run` end-to-end driver: auto-dispatch, multi-GPU/unified, cuSPARSE-verified) |

**Latest report:** [reports/roofline_analysis.md](reports/roofline_analysis.md) —
roofline consolidation on Colab T4: every kernel placed against the T4 hardware
roofline, restored ncu Speed-of-Light profiles, and a structured-block-sparse
WMMA re-measurement (851 GFLOP/s, the fastest kernel in the project). See also
[reports/tensor_core_wmma.md](reports/tensor_core_wmma.md) (WMMA) and
[reports/memory_optimization.md](reports/memory_optimization.md).

**Memory optimization:** the per-cell winner depends on the
bottleneck regime — `tiled_v3` dominates bandwidth-bound m=8192 cells
(2.04× and 2.56× over baseline), `tiled_v2` wins overhead-dominated
small-m / very-sparse cells, `tiled_v4` wins large-m / dense cells where
the wider per-warp work amortises. The DoD bar of ≥2× is met on the two
m=8192 cells but not on the two m=4096 cells (median 1.27× / 1.32× across
six kernels and three runs). The ncu evidence shows m=4096 is
compute-bound on T4 (B fits in L2, AI ≈ 29 FLOP/byte, above the 25
FLOP/byte roofline balance) with the baseline already at ~5% of T4 peak,
which is why pure CUDA-core SpMM caps out at ~1.5× there.

**Tensor Cores:** BSR 16×16 + WMMA on
Tensor Cores *loses* on uniform-random sparsity (0.10–0.91×) because random
nonzeros inflate 16×16 blocks 20–226× (fill-in), starving the Tensor Cores —
measured as L1/TEX saturation at 98.6–99.2% with SM at ~28%. But on
**structured block-diagonal** input (fill-in 1.00) the *same binary* hits
**851 GFLOP/s — 1.61× the best FP32 kernel and the fastest kernel in the
project** at m=8192 (1.81× at m=4096). The roofline analysis also proved the unmet m=4096 d≥0.01
DoD bars are a hardware cache-bandwidth ceiling (L1/TEX 82%, DRAM 10%,
≤1.22× headroom), not optimization debt. This split — TC for structured /
low-fill-in, CSR for random — is exactly the decision the hybrid dispatcher
(`spmm_hybrid`) now automates.

**Baseline vs. memopt:** memopt's win/loss vs.
baseline is not random — it tracks the kernel's bottleneck. memopt wins
in the **latency-bound** regime (short rows, sparse) by amortising
scheduling overhead; baseline wins in the **bandwidth-bound** regime
(long rows, dense) thanks to higher occupancy. DRAM throughput is 39 %
in memopt's win cell and 90 % in its loss cell.

---

## Repository layout

```
src/
  csr.{h,cu}                  CSR data structure + host↔device transfer + row slicing
  bsr.{h,cu}                  Block-Sparse Row (16×16) + fill-in stats
  utils.h                     CUDA / cuSPARSE error checks, GpuTimer
  spmm_hybrid.{h,cu}          structure-aware CUDA-core/Tensor-Core auto-dispatch
  spmm_multigpu.{h,cu}        nnz-balanced multi-GPU row split + unified-memory path
  kernels/
    spmm_baseline.{h,cu}      one-thread-per-output baseline
    spmm_memopt.{h,cu}        warp-per-row, coalesced reads
    spmm_memopt_v2.{h,cu}     register-footprint reduction (no-op)
    spmm_tiled.{h,cu}         col-tile streaming, shmem-staged CSR
    spmm_tiled_v2.{h,cu}      wider tile + float4 (regression)
    spmm_tiled_v3.{h,cu}      fix for the v2 regression (best FP32 kernel)
    spmm_tiled_v4.{h,cu}      single-pass 8-cols/lane (m=4096 negative result)
    spmm_wmma.{h,cu}          BSR + WMMA Tensor Core kernel (FP16 in, FP32 accumulate)
  bench/
    harness.cu                benchmark + cuSPARSE check (per-kernel + hybrid/multigpu/unified)
    spmm_run.cu               integrated driver: auto-dispatch, multi-GPU/unified, verified
scripts/
  gen_matrices.py             synthetic CSR generator (uniform/banded/block/power-law)
  roofline.py                 roofline analysis; emits sweep/roofline CSVs + plot
  ncu_extract.py              parses notebook ncu output → ncu_sol.csv (SOL, real AI)
reports/
  baseline_csr.md             baseline CSR + first memory-optimization pass
  memory_optimization.md      shared-mem tiling / register-tile kernels
  tensor_core_wmma.md         WMMA / Tensor Cores (milestone report)
  roofline_analysis.md        roofline consolidation + ncu Speed-of-Light
  verify_*.md                 per-kernel verification recipes
  data/                       sweep.csv, roofline.csv, ncu_sol.csv
  figures/roofline_t4.png     T4 roofline with each cell's best CSR kernel
colabRunner.ipynb             Colab/Kaggle notebook: profiling + integrated-system demos
Makefile                      builds spmm_bench + spmm_run; targets sm_75 (T4)
```

---

This README focuses on **running the project on Kaggle Notebooks** (the
free dual-T4 environment used for development). Two ingestion methods are
supported: pulling from GitHub, or attaching this repo as a Kaggle Dataset.

---

## 0. Notebook setup (one-time, both methods)

In any new Kaggle notebook:

1. **Settings** → **Accelerator** → select **GPU T4 x2**.
2. **Settings** → **Internet** → **On** (required only for the GitHub method).
3. Confirm the GPU is visible:

   ```python
   !nvidia-smi
   ```

   You should see one or two `Tesla T4` entries.

---

## Method A — Run from GitHub

Use this when the project is pushed to a GitHub repository (recommended for
iterative development — `git pull` to refresh).

```bash
# Cell 1 — clone into /kaggle/working (writable)
!cd /kaggle/working && git clone https://github.com/Adityaa-Saha17/Sparse-Matrix-Multiplication.git spmm
%cd /kaggle/working/spmm
```

```bash
# Cell 2 — build (targets sm_75 = T4 by default)
!make -j
```

```bash
# Cell 3 — run the benchmark (both kernels, default size)
!./build/spmm_bench
```

```bash
# Cell 4 — bigger sweep example
!./build/spmm_bench --m 4096 --k 4096 --n 256 --density 0.01 --iters 20
```

To pick up new commits in a later session:

```bash
%cd /kaggle/working/spmm
!git pull
!make -j
```

---

## Method B — Run from a Kaggle Dataset

Use this when you don't want a public GitHub repo, or you don't have internet
in the notebook.

### One-time: upload the project as a Kaggle Dataset

1. Zip the project locally (excluding `build/`, `data/`, etc.):

   ```bash
   cd "/Users/adityaasaha/Documents/CUDA/Sparse Matrix Multiplication"
   zip -r ../spmm-src.zip . -x 'build/*' 'data/*' '.git/*' 'dump/*'
   ```

2. On Kaggle, **Datasets → New Dataset**, upload `spmm-src.zip`, and publish
   it (private is fine). Note the dataset slug — e.g. `your-username/spmm-src`.

### In the notebook

1. **Add Data** (right sidebar) → search for your dataset → attach it.
   It will appear at `/kaggle/input/<dataset-slug>/`.

2. Copy it into the writable working directory and build:

   ```bash
   # Cell 1 — copy from read-only /kaggle/input to writable /kaggle/working
   !cp -r /kaggle/input/<dataset-slug> /kaggle/working/spmm
   %cd /kaggle/working/spmm
   ```

   ```bash
   # Cell 2 — build
   !make -j
   ```

   ```bash
   # Cell 3 — run
   !./build/spmm_bench
   ```

> `/kaggle/input` is read-only, so `make` cannot write object files there.
> Always copy to `/kaggle/working` first.

To refresh the code after editing locally: re-zip, **upload a new version** of
the dataset (Kaggle versions datasets automatically), then re-attach.

---

## Running the benchmark

Whichever ingestion method you used, the binary is `./build/spmm_bench`.

```bash
# Defaults: m=1024, k=1024, n=256, density=0.01, both kernels
!./build/spmm_bench

# Pick one kernel
!./build/spmm_bench --kernel baseline
!./build/spmm_bench --kernel memopt

# Custom size & sparsity
!./build/spmm_bench --m 8192 --k 8192 --n 256 --density 0.005 --iters 20

# Load a generated matrix from disk
!python3 scripts/gen_matrices.py --pattern banded --rows 8192 --cols 8192 \
    --bandwidth 32 --out data/banded_8k.bin
!./build/spmm_bench --bin data/banded_8k.bin --n 256
```

Output is one line per kernel (example below from a Colab T4 run):

```
kernel=baseline m=4096 k=4096 n=256 nnz=167686 density=0.009995 time_ms=0.3118 gflops=275.32 max_abs_err=4.768e-06 max_rel_err=9.153e-05 rel_l2_err=1.241e-07
kernel=memopt   m=4096 k=4096 n=256 nnz=167686 density=0.009995 time_ms=0.3127 gflops=274.53 max_abs_err=4.768e-06 max_rel_err=9.153e-05 rel_l2_err=1.241e-07
```

A correct run has `rel_l2_err < 1e-4` (the FP32 correctness bar). Observed values are
`7.9e-9` to `5.3e-7` across all tested sizes — FP32 round-off only.
`max_rel_err` is filtered: only cells where `|ref| ≥ 0.1% × max|ref|` are
counted, to suppress noise from near-zero reference values.

---

## Integrated driver and system features

Beyond the per-kernel benchmark, the build produces `./build/spmm_run` — the
end-to-end front end. It ingests a matrix, inspects its block structure, and
**auto-selects** the CUDA-core (`tiled_v3`) or Tensor-Core (`wmma`) back end by
BSR fill-in, then verifies the result against cuSPARSE:

```bash
# Auto-dispatch. Uniform-random -> high fill-in -> CUDA-core CSR.
!./build/spmm_run --m 4096 --k 4096 --n 256 --density 0.01

# Block-diagonal (block=16) -> fill-in ~1 -> Tensor Cores chosen automatically.
!python3 scripts/gen_matrices.py --pattern block-diagonal --rows 4096 --cols 4096 \
    --block 16 --out data/blockdiag_4096.bin
!./build/spmm_run --bin data/blockdiag_4096.bin --n 256

# Memory-aware paths (CUDA-core kernel under the hood):
!./build/spmm_run --m 16384 --k 16384 --n 256 --density 0.01 --multi-gpu   # dual-T4 row split
!./build/spmm_run --m 8192  --k 8192  --n 256 --density 0.01 --unified     # unified memory
```

Each run prints the chosen back end, the fill-in evidence behind the decision,
the timing/GFLOPS, and a `[PASS]/[FAIL]` against cuSPARSE. The same features are
exposed in the benchmark harness for side-by-side comparison:

```bash
!./build/spmm_bench --kernel hybrid   --m 4096 --k 4096 --n 256 --density 0.01
!./build/spmm_bench --kernel multigpu --gpus 2 --m 16384 --k 16384 --n 256 --density 0.01
!./build/spmm_bench --kernel unified  --m 8192 --k 8192 --n 256 --density 0.01
```

## Saving results

Anything written under `/kaggle/working` (logs, CSVs, plots) persists in the
notebook's output and can be downloaded after the session, or committed to the
notebook to make it part of a Kaggle "version".
