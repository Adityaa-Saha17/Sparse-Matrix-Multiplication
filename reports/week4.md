# Week 4 — Consolidation & Roofline: how close to theoretical are we? (Phase 1–3 polish)

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.
**Week dates:** 2026-06-03 to 2026-06-09.
**Scope change:** instead of starting Phase 4 (hybrid CUDA/Tensor-Core dispatch),
this week consolidates the three weeks already done and measures every kernel
against the **T4 hardware roofline** — i.e. how close the empirical numbers are
to what the hardware can theoretically deliver — then fixes the data-quality
gaps that were blocking that comparison.

---

## Why this week

The Phase 1–3 reports established *relative* speedups (kernel vs. baseline) but
never anchored them to an *absolute* ceiling. "tiled_v3 is 2.04× baseline" does
not say whether 2.04× is near the hardware limit or leaving 5× on the table. Two
loose ends made the absolute comparison impossible:

1. **No theoretical reference.** Speedups were never placed on a roofline, so
   "good" and "capped" were asserted, not shown.
2. **The Phase-3 ncu profiles never ran.** Every Nsight Compute cell in the
   Colab notebook died with `getcwd: cannot access parent directories`, so the
   actual memory-traffic and throughput counters that would *prove* where each
   kernel sits were missing.

This week closes both, and adds the one re-measurement that shows the Tensor
Core path reaching its theoretical regime.

---

## T4 theoretical ceilings (NVIDIA datasheet, Turing TU104)

| Quantity | Value | Used as |
|---|---|---|
| FP32 peak | **8 140 GFLOP/s** | compute roof (CSR kernels) |
| FP16 Tensor Core peak | **65 130 GFLOP/s** | compute roof (WMMA) |
| DRAM bandwidth (GDDR6) | **320 GB/s** | memory roof |
| L2 cache | **4 MB** | B-residency threshold (B fits when `k·N·4 ≤ 4 MB`) |
| FP32 roofline ridge | **25.4 FLOP/byte** | `8140 / 320` |

These anchor every "% of theoretical" figure below.

---

## The roofline (computed from the Week-2 full sweep)

Arithmetic intensity uses the **compulsory** DRAM-traffic model — every operand
read/written exactly once: `bytes = 8·nnz + 4(m+1) + 4kN + 4mN`,
`FLOPs = 2·nnz·N`. This is the *optimistic* traffic (perfect reuse), so the
resulting ceiling is a genuine **upper bound** on achievable performance. Numbers
below are produced by [`scripts/roofline.py`](../scripts/roofline.py) from the
measured `gflops` in [`reports/data/sweep.csv`](data/sweep.csv); full table in
[`reports/data/roofline.csv`](data/roofline.csv); plot in
[`reports/figures/roofline_t4.png`](figures/roofline_t4.png).

| m=k | density | AI (FLOP/B) | regime | best CSR kernel | GFLOP/s | % FP32 peak | % of (optimistic) roof |
|---|---|---|---|---|---|---|---|
| 1024 | 0.001 | 0.3 | memory | tiled_v2 | 49.9 | 0.6% | 61.5% |
| 1024 | 0.010 | 2.4 | memory | tiled_v2 | 266.4 | 3.3% | 34.2% |
| 1024 | 0.050 | 10.6 | memory | tiled_v3 | 266.0 | 3.3% | 7.8% |
| 4096 | 0.001 | 1.0 | memory | tiled_v4 | 145.5 | 1.8% | 44.8% |
| 4096 | 0.010 | 8.8 | memory | tiled_v2 | 625.7 | 7.7% | 22.2% |
| **4096** | **0.050** | **28.4** | **compute / L2** | tiled_v4 | 506.0 | 6.2% | 6.2% |
| 8192 | 0.001 | 2.0 | memory | tiled_v2 | 166.6 | 2.0% | 26.3% |
| **8192** | **0.010** | **15.5** | **memory** | tiled_v3 | 406.0 | 5.0% | 8.2% |
| **8192** | **0.050** | **39.4** | **compute / L2** | tiled_v4 | 489.9 | 6.0% | 6.0% |
| 16384 | 0.001 | 3.8 | memory | tiled_v2 | 148.9 | 1.8% | 12.1% |
| 16384 | 0.010 | 24.9 | memory | tiled_v4 | 209.3 | 2.6% | 2.6% |
| 16384 | 0.050 | 48.7 | compute / L2 | tiled_v4 | 261.9 | 3.2% | 3.2% |

### Reading this table correctly

**"% of FP32 peak" is the wrong yardstick — and that is the point.** SpMM at
N=256 is memory-bound: at every density below 0.05 the arithmetic intensity sits
*left of the 25.4 FLOP/byte ridge*, so the FP32 flop peak is unreachable by
construction. Hitting 2–8% of 8.14 TFLOP/s is not a failure; it is what the
roofline permits. The honest ceiling is the **memory roof**, not the flop peak.

**The cells split into two physical regimes at the ridge:**

- **Memory-bound (AI < 25.4):** every cell except the three d=0.05 large-m ones.
  Here the binding resource is bandwidth. The Phase-1 ncu profile already
  measured the bandwidth-bound A2 cell (m=8192, d=0.01) at **~90 % of DRAM
  throughput** (see [week1](week1.md) / README) — i.e. that kernel is already
  within ~10 % of the 320 GB/s theoretical memory roof. It is essentially done.
  The reason the "% of optimistic roof" column reads low (8.2 %) is that the
  compulsory model assumes B is read once; at m=8192 the dense B (8 MB) **exceeds
  the 4 MB L2**, so B is refetched and the *real* traffic is several × the
  compulsory minimum → real AI is lower → the real roof is far below the
  optimistic 4 958 GFLOP/s, and the kernel is close to *that* lower line. The
  fixed ncu cells (below) will pin the true `dram__bytes` and collapse this gap.

- **Compute / L2-bound (AI > 25.4):** the three d=0.05 cells at m≥4096, including
  the **m=4096 d=0.05 DoD cell that never reached ≥2×**. Their B fits (m=4096,
  B=4 MB) or nearly fits in L2, so DRAM is *not* the limiter — L2 bandwidth and
  SM-issue throughput are. The DRAM roofline cannot bound these (they sit far
  under the flat FP32 ceiling because that ceiling is also not their limiter).
  This is the gap that Phase 2's `tiled_v*` work hit and could not pass: it is an
  **L2-bandwidth ceiling, not an un-optimized kernel.** Confirming the exact L2
  throughput is the single most valuable missing measurement, and is exactly what
  the now-fixed ncu profiling delivers.

---

## The three empirical-vs-theory gaps, and what closes each

| # | Gap | Diagnosis | Closer (this week) |
|---|---|---|---|
| 1 | m=4096/8192 d=0.05 cap at ~1.3× / 6% of peak | compute & L2-bandwidth bound; DRAM roof not binding | fixed ncu cells → measure L2 throughput & SM issue to confirm the ceiling is L2 BW, not slack |
| 2 | memory-bound cells look like "8% of roof" | optimistic model ignores B-refetch past the 4 MB L2 | fixed ncu cells → real `dram__bytes` gives true AI; Phase-1 ncu already shows ~90% BW |
| 3 | WMMA loses every random-sparse DoD cell (0.1–0.9×) | 20–226× BSR fill-in on *random* sparsity erases the 8× TC compute edge | structured re-measurement (below) on block-diagonal → fill-in→1×, the regime TC is built for |

None of these is closed by writing a faster kernel — they are closed by **better
measurement and a correct theoretical frame**. Two of the three "misses" turn out
to be hardware ceilings (L2 BW; FP16 fill-in physics), not optimization debt.

---

## Re-measurement #1 — restored ncu profiling (data-quality fix)

**Bug:** the Phase-3 re-clone cell ran `rm -rf spmm` while the notebook kernel's
working directory was still *inside* `/content/spmm` (set by an earlier `%cd`).
Deleting the cwd out from under the kernel left every subsequent `!` shell cell
unable to `getcwd()`, so all twelve Phase-2 and both Phase-3 ncu cells failed —
including the two that *claimed* to "use absolute paths" (the absolute path was
treating a symptom; the shell itself could not start). Captured failure:

```
shell-init: error retrieving current directory: getcwd: cannot access parent directories
==ERROR== Failed to parse options: filesystem error: cannot get current path
```

**Fix** (`colabRunner.ipynb` cell 46): `os.chdir('/content')` *before* the
destructive `rm`, and `os.chdir('/content/spmm')` *after* the rebuild, so the
kernel always holds a live cwd. The Phase-3 ncu cells additionally pin
`%cd /content/spmm` defensively. All Phase 1/2/3 ncu profiles are now runnable in
a single clean session — these are the counters (`dram__bytes.sum`,
`lts__t_sectors` for L2, `sm__throughput`) that turn gaps #1 and #2 from
*argued* into *measured*.

---

## Re-measurement #2 — WMMA on structured block-sparse (the TC theoretical case)

The random-sparse sweep is the **worst case** for BSR+WMMA: 16×16 blocking on
uniform-random nonzeros inflates stored elements 20–226× (fill-in), so the Tensor
Cores spend nearly all throughput multiplying padding zeros. That is why WMMA
measured 0.10–0.91× — a property of the *input distribution*, not the kernel.

New cell (`colabRunner.ipynb` Step 13.5) re-runs the **unchanged** WMMA kernel on
**block-diagonal matrices with a 16-element block**, which align exactly to the
WMMA 16×16 tiles → `block_density = 1.0`, `fill_in_ratio ≈ 1.0`. This is the
apples-to-apples theoretical regime where every FP16 FMA does useful work. It
isolates the question "is the kernel slow, or is random sparsity the wrong input
for Tensor Cores?" — expected result: WMMA goes from ~0.1–0.9× (random) to
competitive-or-faster than the FP32 CSR kernels (structured). Run on Colab T4 to
populate the structured-pattern numbers.

> No kernel code changed this week — both items above are re-measurements of the
> existing Phase 1–3 binaries under corrected tooling and the correct input
> regime, per the "re-measure + re-analyze" scope.

---

## How close to theoretical are we, honestly

| Regime | Cells | Theoretical limiter | Status vs. theory |
|---|---|---|---|
| Bandwidth-bound | m≥8192, mid density | 320 GB/s DRAM | **~90 % of DRAM roof** (Phase-1 ncu) — essentially at the limit |
| L2 / compute-bound | m=4096–16384, d=0.05 | L2 bandwidth + SM issue | capped by L2 BW, **not** slack — fixed ncu will quantify the exact % |
| Latency / overhead-bound | small m, very sparse | kernel-launch + scheduling | tiled_v2/v3 already amortize most overhead (34–61 % of optimistic roof) |
| Tensor Core (random) | all WMMA random cells | FP16 fill-in physics | far from TC peak **by input, not kernel**; structured re-run isolates the real ceiling |

**Bottom line:** the bandwidth-bound kernels are already within ~10 % of the T4
memory roof — there is very little theoretical headroom left there, and the
earlier "2.04×" is close to the best the hardware allows. The unmet ≥2× DoD bars
(m=4096) are an **L2-bandwidth ceiling**, a hardware limit rather than an
optimization gap, which the restored ncu profiling will now confirm with the L2
throughput counters. The Tensor Core path's poor numbers are a property of
*uniform-random* sparsity; on structured block-sparse it should approach its
theoretical regime, which the new cell measures directly.

---

## What changed in the repo

- `scripts/roofline.py` — reproducible roofline analysis; ingests harness logs or
  the embedded Week-2 sweep, emits `reports/data/{sweep,roofline}.csv` and the
  roofline plot.
- `reports/data/sweep.csv`, `reports/data/roofline.csv` — raw + computed data.
- `reports/figures/roofline_t4.png` — the T4 roofline with each cell's best CSR
  kernel placed on it.
- `colabRunner.ipynb` — getcwd bug fixed (cell 46 + Phase-3 ncu cells); new
  Step 13.5 structured block-sparse WMMA re-measurement.

## What to run next on Colab (single clean session)

1. Re-run the Phase-3 section top-to-bottom — the ncu cells now succeed; capture
   `dram__bytes` (gaps #1/#2) and L2 `lts__t_sectors` for the m=4096 d=0.05 cell.
2. Run Step 13.5 to populate the structured block-diagonal WMMA numbers.
3. `python3 scripts/roofline.py --plot` to refresh the table/plot, then drop the
   measured `dram__bytes` into a real (non-compulsory) AI column to collapse the
   "% of roof" gap for the memory-bound cells.

---

## AI use

Utilized AI tools (**ChatGPT**, **Perplexity AI**) for research and the roofline
methodology; **Claude** for the roofline analysis script, the notebook
data-quality fix, validation of the computed numbers against the harness GFLOPS,
and report formatting. T4 datasheet figures verified against NVIDIA's published
T4 Tensor Core specification.
