# CUDA Sparse Matrix Multiplication

Architecture-aware Sparse × Dense matrix multiplication (SpMM) on NVIDIA GPUs.

This README is focused on **running the project on Kaggle Notebooks** (the
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

Output is one line per kernel:

```
kernel=baseline m=4096 k=4096 n=256 nnz=167680 density=0.010000 time_ms=2.34 gflops=36.7 max_abs_err=4.77e-07 max_rel_err=2.38e-06
kernel=memopt   m=4096 k=4096 n=256 nnz=167680 density=0.010000 time_ms=0.91 gflops=94.4 max_abs_err=4.77e-07 max_rel_err=2.38e-06
```

A correct run has `max_rel_err` on the order of `1e-6` (FP32 round-off only).

---

## Saving results

Anything written under `/kaggle/working` (logs, CSVs, plots) persists in the
notebook's output and can be downloaded after the session, or committed to the
notebook to make it part of a Kaggle "version".
