# CUDA Sparse Matrix Multiplication

Architecture-aware Sparse × Dense matrix multiplication (SpMM) on NVIDIA GPUs.
Target hardware: **NVIDIA T4** (sm_75). Development and dual-GPU experiments
run on Kaggle (free dual-T4); profiling runs on Google Colab (single T4,
where Nsight Compute counters are accessible).

---

## Project status

| # | Task                                                              | Phase | Status         |
|---|-------------------------------------------------------------------|-------|----------------|
| 1 | Baseline CSR SpMM on CUDA cores                                   | 1     | ✅ done         |
| 2 | Memory optimization (coalescing, shared-mem tiling, register reuse) | 1–2 | 🟡 in progress (coalescing/warp-per-row done; tiling + register reuse next) |
| 3 | Tensor Core acceleration (block-sparse on T4)                     | 3     | ⏳ not started  |
| 4 | Hybrid execution (CUDA cores vs. Tensor Cores)                    | 4     | ⏳ not started  |
| 5 | Memory-aware execution (unified memory / multi-GPU)               | 5     | ⏳ not started  |
| 6 | Performance evaluation across patterns and sizes                  | all   | 🟡 ongoing      |
| 7 | Final integrated system                                           | 6     | ⏳ not started  |

**Latest report:** [reports/week1.md](reports/week1.md) — Phase 1 close-out
with `ncu` profiling findings on Colab T4.

**Headline result from Phase 1:** memopt's win/loss vs. baseline is not
random — it tracks the kernel's bottleneck. memopt wins in the
**latency-bound** regime (short rows, sparse) by amortising scheduling
overhead; baseline wins in the **bandwidth-bound** regime (long rows,
dense) thanks to higher occupancy. DRAM throughput is 39 % in memopt's
win cell and 90 % in its loss cell — the cleanest signal we could ask for.

---

## Repository layout

```
src/
  csr.{h,cu}                  CSR data structure + host↔device transfer
  utils.h                     CUDA / cuSPARSE error checks, GpuTimer
  kernels/
    spmm_baseline.{h,cu}      one-thread-per-output baseline
    spmm_memopt.{h,cu}        warp-per-row, coalesced reads
  bench/
    harness.cu                benchmark + cuSPARSE correctness check
scripts/
  gen_matrices.py             synthetic CSR generator (uniform/banded/block/power-law)
reports/
  week1.md                    Phase 1 weekly report
colabRunner.ipynb             Colab notebook for ncu profiling
Makefile                      targets sm_75 (T4)
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

A correct run has `rel_l2_err < 1e-4` (Phase 1 DoD). Observed values are
`7.9e-9` to `5.3e-7` across all tested sizes — FP32 round-off only.
`max_rel_err` is filtered: only cells where `|ref| ≥ 0.1% × max|ref|` are
counted, to suppress noise from near-zero reference values.

---

## Saving results

Anything written under `/kaggle/working` (logs, CSVs, plots) persists in the
notebook's output and can be downloaded after the session, or committed to the
notebook to make it part of a Kaggle "version".
