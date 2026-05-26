# Week 2 — Memory Optimization (Phase 2)

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.
**Week dates:** 2026-05-20 to 2026-05-26.
**Status at submission:** all four Phase 2 kernels implemented and wired into the build; correctness + sweep + ncu measurements pending the next Colab session. Verification recipes for each sub-task are in `reports/phase2_*.md` and are written to drop-in to `colabRunner.ipynb`.

---

## What I did this week

The week 1 close-out left two ncu-verified facts on the table:

1. **Bandwidth-bound regime** (e.g. m=8192, density=0.01): baseline beats memopt because memopt's 63 reg/thread footprint costs ~3 pp of achieved occupancy and 8× fewer thread blocks → fewer waves/SM to hide DRAM latency.
2. **Latency-bound regime** (e.g. m=4096, density=0.001, ~4 nnz/row): memopt wins +14% because fewer threads × more work amortizes scheduling overhead.

Phase 2 in the project plan calls for four incremental memory-optimisation sub-tasks (2.1–2.4) plus an optional dispatch fallback (2.5). I implemented 2.1, 2.2, and 2.3+2.4 as separate kernels so each technique can be A/B'd against the others.

### 2.1 — Register-footprint reduction ([src/kernels/spmm_memopt_v2.cu](../src/kernels/spmm_memopt_v2.cu))

Forked `spmm_memopt.cu` (kept verbatim as the v1 reference) into `spmm_memopt_v2.cu`. Three targeted changes:

- `__launch_bounds__(256, 4)` to give the compiler an explicit register cap matching the 4-blocks-per-SM target on sm_75.
- `values[p]` cached in a scalar register so the broadcast load is not re-issued as part of the address-computation chain.
- Hoisted `B + k*N` into a row pointer with `size_t` arithmetic, so each lane just adds `col` to a precomputed 64-bit base instead of recomputing `k*N + col` per iteration.

Mathematically identical to v1 within FP rounding. Honest caveat: theoretical occupancy at 4 blocks × 8 warps already pegs sm_75's 32-warp slot at 100% for v1 — the Phase 1 *achieved* occupancy of 83.7% is stall-bound, not register-bound — so 2.1 alone is expected to move ncu numbers only a few percent. The verification recipe ([phase2_1_verification.md](phase2_1_verification.md)) is explicit about this and includes a decision rule: if 2.1 is no-op, we skip further reg tuning and go straight to 2.2.

### 2.2 — Column-tile streaming with shmem-staged CSR rows ([src/kernels/spmm_tiled.cu](../src/kernels/spmm_tiled.cu))

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

The win comes from cutting B re-fetches per row. In `spmm_memopt(_v2)`, each row of `B[k, :]` is visited `N/32 = 8` times (once per column-stride pass of the outer loop). In `spmm_tiled`, it's visited `N/COL_TILE = 4` times — **2× reduction in B traffic per row** at N=256, COL_TILE=64.

The shmem stage on `col_idx`/`values` (4 KB per block: 8 warps × 64 entries × (int + float)) is the literal "shmem tiling" part of the project spec. It does not give cross-row reuse — different rows of A index different rows of B, so the staged data is warp-local rather than block-shared. This is documented in both the kernel header and the verification recipe so the ablation report does not overclaim.

### 2.3 + 2.4 — Larger register tile + vectorized B loads ([src/kernels/spmm_tiled_v2.cu](../src/kernels/spmm_tiled_v2.cu))

Phase 2.3 (register accumulation + manual unrolling) and Phase 2.4 (vectorized `float4`/`__ldg` loads) ship together as one kernel because the two changes are tightly coupled: a `float4` load on B naturally produces 4 partial sums that have to live in registers. Splitting them across two kernels would create an interim state that's neither faster nor easier to reason about.

Changes vs `spmm_tiled`:

- `COL_TILE` doubled to **128**; `COLS_PER_LANE` doubled to **4** (`float4` register accumulator per lane).
- B reads are a single 16-byte `float4` load per lane per p. The 32 lanes of a warp together issue a 512-byte coalesced burst (4 contiguous 128-byte segments).
- C writes use `float4` stores at the end of each col tile (one 16-byte store per lane).

Constraint: `N % 128 == 0`. The harness's default `--n 256` satisfies this; under `--kernel all`, tiled_v2 is silently skipped if N is not divisible by 128.

### Harness and build ([src/bench/harness.cu](../src/bench/harness.cu), [Makefile](../Makefile))

The harness now dispatches five kernels:

| `--kernel` flag | Runs |
|---|---|
| `baseline` | week-1 baseline only |
| `memopt` | week-1 memopt (v1) only |
| `memopt_v2` | Phase 2.1 only |
| `tiled` | Phase 2.2 only |
| `tiled_v2` | Phase 2.3 + 2.4 only |
| `both` | baseline + memopt (week-1 default, kept for compatibility) |
| `all` | all five, side-by-side in one run |

Existing Colab cells from week 1 continue to work unchanged because `both` was not redefined.

---

## Results

Pending the next Colab session. The 5-way 12-cell sweep template:

| m=k | density | baseline | memopt v1 | memopt v2 | tiled | tiled v2 | winner | DoD met? |
|---|---|---|---|---|---|---|---|---|
| 1024 | 0.050 | — | — | — | — | — | — | — |
| 1024 | 0.010 | — | — | — | — | — | — | — |
| 1024 | 0.001 | — | — | — | — | — | — | — |
| 4096 | 0.050 | — | — | — | — | — | — | DoD cell |
| 4096 | 0.010 | — | — | — | — | — | — | DoD cell |
| 4096 | 0.001 | — | — | — | — | — | — | — |
| 8192 | 0.050 | — | — | — | — | — | — | DoD cell |
| 8192 | 0.010 | — | — | — | — | — | — | DoD cell |
| 8192 | 0.001 | — | — | — | — | — | — | — |
| 16384 | 0.050 | — | — | — | — | — | — | — |
| 16384 | 0.010 | — | — | — | — | — | — | — |
| 16384 | 0.001 | — | — | — | — | — | — | — |

Each cell reports median ms over 20 timed iterations after 5 warmup runs (same protocol as week 1).

The four cells marked **DoD cell** are the gate for closing Phase 2: the best of {memopt v2, tiled, tiled v2} in each of those cells must hit **≥ 2× over baseline**. This is the definition of done from the project plan.

---

## Correctness

Pending. Pass criterion is unchanged from week 1: `rel_l2_err < 1e-4` for every kernel × every sweep cell, with `cusparseSpMM` as the reference. The harness already reports `max_abs_err`, `max_rel_err` (filtered for cells where `|ref| ≥ 0.1% × max|ref|`), and `rel_l2_err` per kernel per run.

---

## Performance analysis — what each kernel is expected to do

Until Colab numbers land, the analysis here is mechanistic, not measured.

**spmm_memopt_v2 vs spmm_memopt.** The change is small and conservative. The most plausible outcome is a 1–3% improvement in bandwidth-bound cells from removing the inner-loop register churn, and no movement in latency-bound cells. If 2.1 turns out to be a no-op, that is itself a publishable finding — it tells us that the bottleneck in those regimes is warp-stall latency, not register pressure.

**spmm_tiled vs spmm_memopt.** Bandwidth-bound corner (m=16384, d=0.05) should narrow or close the gap to baseline because the kernel re-fetches B about half as often. Latency-bound corner (m=4096, d=0.001 with ~4 nnz/row) is likely **worse** than memopt — the col-tile loop's per-pass overhead (accumulator init, shmem stage repeat, syncwarp) is amortized over almost no arithmetic when the row is empty. This is the expected ablation finding and is exactly the evidence the Phase 4 hybrid dispatcher will need.

**spmm_tiled_v2 vs spmm_tiled.** Bandwidth-bound corner should gain again, this time from fewer outer col-tile passes (2 instead of 4 at N=256) and from the float4 loads issuing 4× fewer memory instructions per p. Latency-bound corner is likely worst-in-class — for the same reason as tiled, only more so.

**Across all four kernels combined.** The week-2 hypothesis is that no single kernel will dominate every cell of the sweep. Different bottlenecks favour different designs:

| Regime | Bottleneck | Expected winner |
|---|---|---|
| dense, large m (m≥4096, d≥0.01) | DRAM bandwidth | tiled or tiled_v2 |
| sparse, short-row (d=0.001) | warp scheduling latency | memopt v1/v2 |
| small problem (m=1024) | launch overhead, noise floor | indistinguishable |

If that hypothesis holds, the optional Phase 2.5 fallback (a runtime dispatcher choosing per (m, density) bucket) is the natural close-out for week 2 once measurements confirm the boundary.

---

## Profiling — Nsight Compute plan

Same protocol as week 1 (Colab, not Kaggle — Kaggle blocks performance counters with `ERR_NVGPUCTRPERM`). Four profiles per new kernel:

- **A2** (bandwidth-bound): `m=8192 n=256 d=0.01` — the strongest comparator to week 1's memopt A2 profile (90.3% / 85.2% DRAM, 86.2% / 83.7% occupancy).
- **B2** (latency-bound): `m=4096 n=256 d=0.001` — the regime where memopt won +14% in week 1.

For each (kernel × point) we record: registers/thread, shmem/block, achieved occupancy, DRAM throughput, L1/TEX throughput, L2 throughput, waves/SM, compute throughput. This produces a 5-kernel × 2-point ncu matrix that maps directly to the "where does this kernel win and why" narrative for the final report.

Verification recipes:
- [phase2_1_verification.md](phase2_1_verification.md) — 2.1 (memopt v2).
- [phase2_2_verification.md](phase2_2_verification.md) — 2.2 (tiled).
- [phase2_3_4_verification.md](phase2_3_4_verification.md) — 2.3 + 2.4 (tiled v2).

Each recipe specifies the exact ptxas command (for register / spill / shmem footprint), the exact `--kernel all` sweep, and the ncu invocations.

---

## What's left for week 2 close-out

1. Run the three verification recipes on Colab (one session, ~30 minutes wall-clock).
2. Fill in the results, correctness, and profiling tables in this file.
3. Run the final `--kernel all` sweep and pick the best kernel per cell.
4. Decide whether Phase 2.5 (hybrid dispatcher) is needed to close any remaining DoD gaps. If the answer is no (one of the three new kernels dominates the four DoD cells with ≥2× over baseline), Phase 2 closes. If yes, that becomes the first item next week before Phase 3 starts.

---

## Plan for week 3 — Tensor Cores (Phase 3 milestone, formal report)

Block-Sparse Row (BSR) format with 16×16 blocks matching the WMMA fragment size, FP16 input with FP32 accumulate, kernel using the CUDA WMMA API (`mma.h`). Validation against `cusparseSpMM` BSR path (if available) and against `spmm_tiled_v2` within FP16 tolerance. Comparison on (a) synthetic matrices with planted block structure and (b) random sparsity, to characterise when Tensor-Core acceleration under-performs.

Per the project plan, week 3 is a **milestone formal report**, not a short weekly note like this one — design rationale + baseline → memopt → tiled → TC progression + graphs (speedup vs sparsity, vs matrix size).

---

## AI - USE

Utilized AI tools such as ***ChatGPT*** and ***Perplexity AI*** to assist with research, gathering relevant information, and developing an appropriate design structure and implementation plan for the project. ***Claude AI*** was used for code validation and testing, formatting, adding comments, and refining the language and presentation of the report. Claude was also used to draft the four Phase 2 kernel variants (`spmm_memopt_v2`, `spmm_tiled`, `spmm_tiled_v2`) and the per-phase verification recipes against the project plan in `~/.claude/plans/`. All design decisions (kernel shape, tile parameters, ablation strategy, DoD framing) were reviewed and accepted before commit. Honest caveats about expected null results in latency-bound regimes were authored by Claude and retained as written rather than glossed over.
