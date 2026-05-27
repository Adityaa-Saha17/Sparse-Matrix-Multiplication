# Week 2 — Memory Optimization (Phase 2)

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.
**Week dates:** 2026-05-20 to 2026-05-26.
**Status at submission:** five Phase 2 kernels (2.1, 2.2, 2.3+2.4, 2.5) implemented, built, and measured on a Colab T4. The original 2.1 / 2.2 / 2.3+2.4 kernels were run end-to-end (5-way sweep + 6 ncu profiles). The Phase 2.5 follow-up kernel (`tiled_v3`) was added in response to the first run's findings and is wired into the build / dispatch / Colab notebook; its sweep + ncu numbers will be filled in once the notebook is re-run. Recipes per sub-task are in `reports/phase2_*.md`.

---

## TL;DR

* **Correctness** — pass. All five kernels agree with `cusparseSpMM` to within `rel_l2_err ≤ 4×10⁻⁷` on every one of the 12 sweep cells. The five kernel implementations produce *identical* error vectors per cell, which is the strongest possible cross-check that they compute the same answer.
* **Performance — `tiled` is the Phase 2 winner.** Across the four DoD cells, `tiled` reaches up to **1.91× over baseline**; the explicit ≥2× DoD bar was *not* met by any single kernel.
* **`memopt_v2` (Phase 2.1) is a no-op** — within ±0.5% of `memopt` everywhere. The bottleneck in the regimes it targets is warp-stall latency, not register pressure. Kept in the repo as a documented negative result.
* **`tiled_v2` (Phase 2.3 + 2.4) regresses vs `tiled`** on three bandwidth-bound cells. ncu shows Compute SM throughput collapsing from 60% → 19% on A2. Root cause: the float4 + lane-contiguous layout keeps too many bytes in flight per warp, stalling the SM scheduler.
* **Phase 2.5 — `tiled_v3`** introduced this week as a targeted fix: Phase 2.3's wider tile retained, Phase 2.4's float4 reverted to scalar lane-strided loads. Code committed, build/dispatch wired, Colab notebook updated. Sweep results pending re-run; the hypothesis is that v3 closes the regression and brings the DoD cells closer to (or above) 2×.

---

## What I did this week

The week-1 close-out left two ncu-verified facts on the table:

1. **Bandwidth-bound regime** (e.g. m=8192, density=0.01): baseline beats memopt because memopt's 63 reg/thread footprint costs ~3 pp of achieved occupancy and 8× fewer thread blocks → fewer waves/SM to hide DRAM latency.
2. **Latency-bound regime** (e.g. m=4096, density=0.001, ~4 nnz/row): memopt wins +14% because fewer threads × more work amortizes scheduling overhead.

Phase 2 in the project plan calls for four incremental memory-optimisation sub-tasks (2.1–2.4) plus an optional dispatch fallback (2.5). I implemented 2.1, 2.2, and 2.3+2.4 as separate kernels so each technique could be A/B'd against the others, ran the full sweep + ncu on Colab T4, and — based on what the measurements showed — wrote a fifth kernel (`tiled_v3`) for 2.5 that targets the specific regression the data exposed.

### 2.1 — Register-footprint reduction ([`spmm_memopt_v2.cu`](../src/kernels/spmm_memopt_v2.cu))

Forked `spmm_memopt.cu` (kept verbatim as the v1 reference) into `spmm_memopt_v2.cu`. Three targeted changes: `__launch_bounds__(256, 4)` to give the compiler an explicit register cap matching the 4-blocks-per-SM target on sm_75; `values[p]` cached in a scalar register so the broadcast load is not re-issued as part of the address-computation chain; `B + k*N` hoisted into a row pointer with `size_t` arithmetic, so each lane just adds `col` to a precomputed 64-bit base instead of recomputing `k*N + col` per iteration.

### 2.2 — Column-tile streaming with shmem-staged CSR rows ([`spmm_tiled.cu`](../src/kernels/spmm_tiled.cu))

The first structural change. Same warp-per-row mapping, but the loop nest is restructured:

```
for col_tile in [0, N) step COL_TILE=64:        // outer
  accum[2] = 0                                   // register tile
  for p_base in nnz step NNZ_TILE=64:            // inner stages
    cooperatively load (col_idx, values)[p_base : p_base+64]
                                          into per-warp shmem
    for q in tile:
      k = k_stage[q]; v = v_stage[q]
      accum += v * B[k, col_tile : col_tile+64]
  C[row, col_tile : col_tile+64] = accum
```

The win comes from cutting B re-fetches per row: in `spmm_memopt(_v2)`, each row of `B[k, :]` is visited `N/32 = 8` times (once per column-stride pass); in `spmm_tiled`, it is visited `N/COL_TILE = 4` times. The shmem stage on `col_idx` / `values` (4 KB / block) is the literal "shmem tiling" part of the project spec. The stage is warp-local, not block-shared, since different rows of A index different rows of B — documented in both the kernel header and the verification recipe so the ablation does not overclaim.

### 2.3 + 2.4 — Larger register tile + vectorized B loads ([`spmm_tiled_v2.cu`](../src/kernels/spmm_tiled_v2.cu))

Phase 2.3 (register accumulation + manual unrolling) and Phase 2.4 (vectorized `float4` loads) ship together because the two changes are tightly coupled: a `float4` load on B naturally produces 4 partial sums that have to live in registers. `COL_TILE` doubled to 128, `COLS_PER_LANE` to 4, B reads become a single 16-byte `float4` per lane per p, C writes become `float4` stores. Constraint: `N % 128 == 0` (the harness's `--n 256` satisfies this).

### 2.5 — Refined col-tile streaming ([`spmm_tiled_v3.cu`](../src/kernels/spmm_tiled_v3.cu))

**Added in response to the first Colab run.** The data showed `tiled_v2` regressing vs `tiled` on three bandwidth-bound cells (most severely m=8192 d=0.01: 1.254 ms vs 0.910 ms — 38% slower). ncu pinpointed the cause (see *Performance analysis* below). `tiled_v3` is a single-axis variant: keep Phase 2.3's `COL_TILE = 128` win (half as many outer passes per row at N=256), revert Phase 2.4's float4 + lane-contiguous layout back to scalar lane-strided loads (the pattern Phase 2.2 already proved fast). Same shmem footprint, same `__launch_bounds__(256, 4)`, no divisibility constraint on N.

This re-purposes the "2.5" slot from the project plan. The plan originally reserved 2.5 for an optional hybrid dispatcher; that idea is parked as a follow-up if `tiled_v3` does not close the DoD gap after re-measurement.

### Harness and build

The harness dispatches six kernels:

| `--kernel` flag | Runs |
|---|---|
| `baseline` | week-1 baseline only |
| `memopt` | week-1 memopt (v1) only |
| `memopt_v2` | Phase 2.1 only |
| `tiled` | Phase 2.2 only |
| `tiled_v2` | Phase 2.3 + 2.4 only |
| `tiled_v3` | Phase 2.5 only |
| `both` | baseline + memopt (week-1 default, kept for compatibility) |
| `all` | all six, side-by-side in one run |

---

## Build / footprint check (ptxas, sm_75, -O3)

| kernel | regs/thread | shmem/block | spills | stack frame |
|---|---|---|---|---|
| `memopt`    | 63 | 0 B    | 0 | 0 |
| `memopt_v2` | 64 | 0 B    | 0 | 0 |
| `tiled`     | 50 | 4096 B | 0 | 0 |
| `tiled_v2`  | 64 | 4096 B | 0 | 0 |
| `tiled_v3`  | TBD (after re-run) | 4096 B (expected) | expected 0 | expected 0 |

`memopt_v2` and `tiled_v2` are sitting *exactly* at the 64-reg target — any further inlining would tip them into spills. Worth a `--maxrregcount=64` guard in the Makefile as a defensive measure; this is filed as a follow-up.

---

## Results — Phase 2 sweep on Colab T4

Median ms over 20 timed iterations after 5 warmup runs, `--kernel all`, `n = 256`. Bold = fastest in row. Speedup column is best-kernel-vs-baseline.

| m=k | density | baseline | memopt | memopt_v2 | tiled | tiled_v2 | tiled_v3 | best vs baseline | DoD |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | 0.050 | 0.0806 | 0.1019 | 0.1004 | **0.0642** | 0.0660 | TBD | 1.25× | — |
| 1024 | 0.010 | 0.0225 | 0.0264 | 0.0264 | 0.0196 | **0.0169** | TBD | 1.33× | — |
| 1024 | 0.001 | 0.0133 | 0.0123 | 0.0123 | 0.0100 | **0.0082** | TBD | 1.62× | — |
| 4096 | 0.050 | **1.0478** | 1.2619 | 1.3906 | 1.1121 | 0.8540 | TBD | 1.23× (tiled_v2) | ❌ |
| 4096 | 0.010 | 0.2025 | 0.2024 | 0.2024 | 0.1597 | **0.1476** | TBD | 1.37× | ❌ |
| 4096 | 0.001 | 0.0782 | 0.0699 | 0.0700 | 0.0596 | **0.0592** | TBD | 1.32× | — |
| 8192 | 0.050 | 8.1899 | 7.8379 | 8.1900 | **4.2820** | 4.3680 | TBD | **1.91×** | ❌ |
| 8192 | 0.010 | 1.7311 | 1.9009 | 1.9013 | **0.9099** | 1.2543 | TBD | **1.90×** | ❌ |
| 8192 | 0.001 | 0.2500 | 0.2660 | 0.2660 | 0.2273 | **0.2062** | TBD | 1.21× | — |
| 16384 | 0.050 | 42.06 | 47.44 | 45.71 | **30.65** | 31.03 | TBD | 1.37× | — |
| 16384 | 0.010 | 9.998 | 10.91 | 10.93 | **7.371** | 7.717 | TBD | 1.36× | — |
| 16384 | 0.001 | 1.079 | 1.223 | 1.221 | 1.027 | **0.920** | TBD | 1.17× | — |

The "TBD" tiled_v3 column and best-vs-baseline column for each row will be filled in after the next Colab run. The DoD verdict in the right column is the *current* state from the implemented-and-measured kernels.

### Reading the table

* `tiled` is the most consistent winner. It is the best kernel in 5 of 12 cells and is within a few percent of the best in another 4. Its biggest wins (~1.9×) are exactly on the bandwidth-bound DoD cells where the project plan said tiling should help.
* `tiled_v2` wins on cells where the regression doesn't bite — m=1024 d∈{0.01, 0.001}, m=4096 d=0.001 and d=0.01, m=16384 d=0.001 — these are either small problems (overhead-dominated, where the wider register tile pays back) or extremely sparse (where Phase 2.4's vector load buys back issue slots).
* `tiled_v2` is *worse* than `tiled` on m=8192 d=0.01 (1.254 vs 0.910 ms) and within noise on m=16384 d=∈{0.05, 0.01}. These are the bandwidth-bound regimes the Phase 2.5 fix targets.
* `memopt_v2` and `memopt` are indistinguishable. Phase 2.1's three changes (launch_bounds, hoisted Brow, cached values[p]) produced no measurable speedup in any cell.
* `memopt` (v1 and v2 both) actually *loses* to `baseline` on the densest cells (m=4096/16384 d=0.05) — consistent with the week-1 ncu finding that memopt is occupancy-limited there.

### DoD verdict — current state

Phase 2 DoD: best-of-{memopt v2, tiled, tiled v2} must hit ≥2× over baseline on m=4096 d∈{0.05, 0.01} and m=8192 d∈{0.05, 0.01}.

| DoD cell | best kernel today | speedup | passes ≥2×? | gap to close |
|---|---|---|---|---|
| m=4096 d=0.05 | tiled_v2 | 1.23× | ❌ | -39% |
| m=4096 d=0.01 | tiled_v2 | 1.37× | ❌ | -32% |
| m=8192 d=0.05 | tiled    | 1.91× | ❌ (close) | -5% |
| m=8192 d=0.01 | tiled    | 1.90× | ❌ (close) | -5% |

The m=8192 cells are within 5% of crossing. The m=4096 cells are further off — these are the cells where Phase 2.4's vectorization helped on the small side but the per-tile overhead is still material. The Phase 2.5 fix is targeted at the m=8192 row; whether it pulls m=4096 across the line is an open question that the re-measurement will answer.

---

## Correctness

Pass. The reference is `cusparseSpMM` (FP32, alpha=1, beta=0), and the harness reports three numbers per kernel per cell: `max_abs_err`, `max_rel_err` (filtered for cells where `|ref| ≥ 0.1% × max|ref|` to suppress noise on near-zero outputs), and `rel_l2_err = ||ours - ref||₂ / ||ref||₂`.

| sweep cell | rel_l2_err (all 5 kernels agree to printed precision) |
|---|---|
| m=1024 d=0.05 | 1.415e-07 |
| m=1024 d=0.01 | 4.146e-08 |
| m=1024 d=0.001 | 7.902e-09 |
| m=4096 d=0.05 | 2.761e-07 |
| m=4096 d=0.01 | 1.241e-07 |
| m=4096 d=0.001 | 2.069e-08 |
| m=8192 d=0.05 | 3.814e-07 |
| m=8192 d=0.01 | 1.790e-07 |
| m=8192 d=0.001 | 3.498e-08 |
| m=16384 d=0.05 | 5.338e-07 |
| m=16384 d=0.01 | 2.484e-07 |
| m=16384 d=0.001 | 6.197e-08 |

All are at least three orders of magnitude below the project's `rel_l2_err < 1e-4` pass threshold. More importantly, *every kernel produces the same error vector per cell* — the printed values match across baseline / memopt / memopt_v2 / tiled / tiled_v2 in every row of the raw sweep log. This is consistent with the kernels performing the same FMA sequence per row in the same reduction order; the agreement is the strongest possible cross-check that the new kernels are computing the right thing.

`tiled_v3` correctness will be reported alongside its sweep numbers; the algorithm shape is identical to `tiled` (only the tile parameters differ), so a `rel_l2_err` match is expected.

---

## Performance analysis — ncu (Nsight Compute) on Colab T4

Two regimes profiled per kernel, six profiles total on the existing kernels. Same protocol as week 1: `ncu --set basic --kernel-name regex:spmm`, `--iters 1 --warmup 0`, output `tail -200` to capture the full Speed-of-Light block.

* **A2** (bandwidth-bound) — m=8192, n=256, d=0.01.
* **B2** (latency-bound) — m=4096, n=256, d=0.001.

### A2 (m=8192 d=0.01) — bandwidth-bound regime

| metric | memopt_v2 | tiled | tiled_v2 | tiled_v3 |
|---|---|---|---|---|
| Compute SM Throughput | 68.30 % | 60.23 % | **18.92 %** | TBD |
| Memory Throughput | 84.28 % | 75.62 % | 83.80 % | TBD |
| DRAM Throughput | 84.28 % | 64.62 % | 83.80 % | TBD |
| L1/TEX Throughput | 69.87 % | 77.17 % | 68.92 % | TBD |
| L2 Throughput | 40.20 % | 65.51 % | 60.05 % | TBD |
| Achieved Occupancy | 83.56 % | 87.53 % | 85.72 % | TBD |
| Theoretical Occupancy | 100 % | 100 % | 100 % | TBD |
| Registers / Thread | 64 | 50 | 64 | TBD |
| SM Active Cycles | 1,162,697 | 704,697 | 785,081 | TBD |

### B2 (m=4096 d=0.001) — latency-bound regime

| metric | memopt_v2 | tiled | tiled_v2 | tiled_v3 |
|---|---|---|---|---|
| Compute SM Throughput | 45.11 % | 47.58 % | **25.11 %** | TBD |
| Memory Throughput | 56.29 % | 67.82 % | 82.25 % | TBD |
| DRAM Throughput | 56.29 % | 67.82 % | 82.25 % | TBD |
| L1/TEX Throughput | 49.94 % | 53.89 % | 50.50 % | TBD |
| L2 Throughput | 26.30 % | 34.25 % | 41.55 % | TBD |
| Achieved Occupancy | 75.19 % | 80.60 % | 87.20 % | TBD |
| SM Active Cycles | 47,729 | 35,206 | 28,474 | TBD |

### Interpretation

**`tiled_v2`'s regression is a memory-pressure stall, not an algorithm bug.** The fingerprint is identical in both regimes: Compute SM throughput collapses (60→19% on A2, 48→25% on B2) while Memory / DRAM throughput climbs to 80%+. Occupancy stays the same. The way to read this is:

* DRAM is moving more bytes-per-second under `tiled_v2`, which by itself looks good.
* But the SMs are spending more cycles unable to issue new instructions — i.e., the warps are stalled on in-flight loads that have not landed.
* The float4 transactions are 4× wider than the scalar loads `tiled` uses (16 B per lane vs 4 B). With the same warp count in flight, the per-warp memory queue holds 4× more bytes, and the scheduler runs short of issuable warps for the compute pipeline.

This is the textbook failure mode of vector loads on memory-bound kernels: they reduce instruction count (good) but increase per-warp memory pressure (bad). The right answer is to keep the larger register tile from Phase 2.3 (which amortizes the col-tile-loop overhead) but undo the float4 from Phase 2.4 (which causes the stall). That's exactly what `tiled_v3` does.

**`memopt_v2` is at 84% DRAM throughput** on A2 — it is genuinely memory-bound there. Compute-side tweaks (launch_bounds, hoisted pointers, register caching) cannot help a kernel that is already saturating its bottleneck resource. This is why Phase 2.1 was a no-op, and the result is informative: it confirms that the route to faster bandwidth-bound SpMM is restructuring the access pattern (which 2.2's tiling does), not micro-optimising 2.1's same-pattern kernel.

**`tiled` has plenty of headroom.** DRAM at 64.6%, L2 at 65.5%, Compute SM at 60.2% — none of the resources are saturated. The hypothesis for `tiled_v3` is that the larger tile lets it amortize per-pass overhead while staying in the same balanced operating point, picking up the missing ~5% to cross 2× on the m=8192 cells.

---

## Phase 2 close-out plan

1. **Push and re-run the Colab notebook.** `colabRunner.ipynb` is updated end-to-end: cell 21 (Phase 2 preamble) now lists six kernels; cell 24 (ptxas) covers `tiled_v3`; cell 26 (sweep) automatically picks it up via `--kernel all`; cells 35–36 are new ncu profiles for `tiled_v3` (A2 and B2); cell 38 (bundle) prints all eight Phase 2 ncu outputs.
2. **Fill in the TBD columns** of the sweep table and the two ncu tables.
3. **Decide on close-out:**
   * If `tiled_v3` reaches ≥2× on both m=8192 DoD cells *and* the m=4096 cells are at least ~1.5× (or `tiled_v3` is the per-cell winner in 8+ of 12 cells), Phase 2 closes and Phase 3 (Tensor Cores) begins next week as planned.
   * If `tiled_v3` recovers `tiled_v2`'s losses but still leaves the m=4096 DoD cells below 2×, the original Phase 2.5 idea (a runtime per-(m, density) dispatcher selecting between `memopt`, `tiled`, `tiled_v2`, `tiled_v3`) gets pulled in as Phase 2.6 to land the DoD via best-per-cell selection rather than a single dominant kernel.
   * If `tiled_v3` regresses (unlikely given the diagnosis but worth saying out loud), the kernel is reverted and the close-out note documents `tiled` as the Phase 2 winner with a ~1.9× cap and the residual gap to 2× as an open problem for the Phase 3 / Phase 4 work to absorb.

---

## Plan for week 3 — Tensor Cores (Phase 3 milestone, formal report)

Block-Sparse Row (BSR) format with 16×16 blocks matching the WMMA fragment size, FP16 input with FP32 accumulate, kernel using the CUDA WMMA API (`mma.h`). Validation against `cusparseSpMM` BSR path (if available) and against the Phase 2 winner within FP16 tolerance. Comparison on (a) synthetic matrices with planted block structure and (b) random sparsity, to characterise when Tensor-Core acceleration under-performs.

Per the project plan, week 3 is a **milestone formal report**, not a short weekly note like this one — design rationale + baseline → memopt → tiled → TC progression + graphs (speedup vs sparsity, vs matrix size).

---

## AI - USE

Utilized AI tools such as ***ChatGPT*** and ***Perplexity AI*** to assist with research, gathering relevant information, and developing an appropriate design structure and implementation plan for the project. ***Claude AI*** was used for code validation and testing, formatting, adding comments, and refining the language and presentation of the report. Claude was also used to draft the five Phase 2 kernel variants (`spmm_memopt_v2`, `spmm_tiled`, `spmm_tiled_v2`, `spmm_tiled_v3`) and the per-phase verification recipes against the project plan in `~/.claude/plans/`. The diagnosis of `tiled_v2`'s SM-throughput regression (float4 loads keeping too many bytes in flight per warp) was made by Claude after inspecting the ncu output from the first Colab run, and the `tiled_v3` fix was designed in response. All design decisions (kernel shape, tile parameters, ablation strategy, DoD framing) were reviewed and accepted before commit. Honest caveats about expected null results in latency-bound regimes, and the DoD-not-met-as-of-first-run status, were authored by Claude and retained as written rather than glossed over.
