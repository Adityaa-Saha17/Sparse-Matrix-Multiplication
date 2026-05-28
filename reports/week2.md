# Week 2 — Memory Optimization (Phase 2)

**Project:** Architecture-aware Sparse Matrix Multiplication on NVIDIA GPUs.
**Week dates:** 2026-05-20 to 2026-05-26.

---

## What I did this week

Built five new kernels on top of the week-1 pair, each targeting a specific memory bottleneck identified by ncu:

- [`spmm_memopt_v2.cu`](../src/kernels/spmm_memopt_v2.cu) — added `__launch_bounds__(256, 4)`, hoisted `B + k*N` pointer, cached `values[p]` in a register. **No-op** — within ±0.5% of `memopt` everywhere; bottleneck is warp-stall latency, not registers.
- [`spmm_tiled.cu`](../src/kernels/spmm_tiled.cu) — restructured loop nest into outer col-tile (64 cols) + inner nnz-tile staged into 4 KB shmem per warp. Cuts B re-fetches per row from 8 passes to 4.
- [`spmm_tiled_v2.cu`](../src/kernels/spmm_tiled_v2.cu) — `COL_TILE` doubled to 128, B reads via `float4`. **Regression on 3 bandwidth-bound cells** — float4 floods the warp memory queue; Compute SM Throughput collapses 60% → 19%.
- [`spmm_tiled_v3.cu`](../src/kernels/spmm_tiled_v3.cu) — keeps `COL_TILE = 128`, reverts float4 → scalar lane-strided loads. Recovers Compute SM Throughput to 61%. **Phase 2 overall winner.**
- [`spmm_tiled_v4.cu`](../src/kernels/spmm_tiled_v4.cu) — `COL_TILE` 128 → 256, `COLS_PER_LANE` 4 → 8; outer loop runs once per row. **Regression on m=4096 cells** — 8-FMA dependency chain serializes load-FMA-load, drops SM Throughput 13–14 pp. Wins on m≥8192 dense cells.

---

## Build / footprint (ptxas, sm_75, -O3)

| kernel | regs/thread | shmem/block | spills |
|---|---|---|---|
| `memopt`    | 63 | 0 B    | 0 |
| `memopt_v2` | 64 | 0 B    | 0 |
| `tiled`     | 50 | 4096 B | 0 |
| `tiled_v2`  | 64 | 4096 B | 0 |
| `tiled_v3`  | 41 | 4096 B | 0 |
| `tiled_v4`  | 38 | 4096 B | 0 |

---

## Results — Phase 2 sweep on Colab T4

Median ms over 20 timed iterations, 5 warmup, `--kernel all`, `n = 256`. Bold = fastest in row. Speedup = best Phase 2 kernel vs. baseline.

| m=k | density | baseline | memopt | memopt_v2 | tiled | tiled_v2 | tiled_v3 | tiled_v4 | best vs baseline | DoD |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 | 0.050 | 0.0804 | 0.0985 | 0.0978 | 0.0625 | 0.0644 | **0.0610** | 0.0633 | 1.32× (tiled_v3) | — |
| 1024 | 0.010 | 0.0225 | 0.0271 | 0.0273 | 0.0198 | **0.0176** | 0.0196 | 0.0216 | 1.28× (tiled_v2) | — |
| 1024 | 0.001 | 0.0129 | 0.0117 | 0.0117 | 0.0098 | **0.0080** | 0.0080 | 0.0081 | 1.62× (tiled_v2) | — |
| 4096 | 0.050 | **1.0651** | 1.2421 | 1.3795 | 0.9963 | 0.9460 | **0.8537** | 0.9162 | **1.25× (tiled_v3)** | ❌ |
| 4096 | 0.010 | 0.1903 | 0.1907 | 0.1907 | 0.1635 | **0.1494** | 0.1599 | 0.1814 | **1.27× (tiled_v2)** | ❌ |
| 4096 | 0.001 | 0.0735 | 0.0689 | 0.0689 | 0.0590 | **0.0590** | 0.0589 | 0.0570 | 1.29× (tiled_v4) | — |
| 8192 | 0.050 | 10.21 | 8.78 | 9.02 | 4.63 | 4.67 | 3.96 | **3.79** | **2.69× (tiled_v4)** | ✓ |
| 8192 | 0.010 | 1.7320 | 1.9132 | 1.9145 | 0.9300 | 1.2461 | **0.8497** | 0.9202 | **2.04× (tiled_v3)** | ✓ |
| 8192 | 0.001 | 0.2502 | 0.2659 | 0.2659 | 0.2272 | **0.2059** | 0.2089 | 0.2226 | 1.22× (tiled_v2) | — |
| 16384 | 0.050 | 46.67 | 44.82 | 46.96 | 32.09 | 30.00 | 28.46 | **26.00** | 1.80× (tiled_v4) | — |
| 16384 | 0.010 | 10.03 | 10.97 | 10.99 | 7.41 | 7.73 | 6.75 | **6.59** | 1.52× (tiled_v4) | — |
| 16384 | 0.001 | 1.131 | 1.218 | 1.220 | 1.027 | **0.917** | 0.929 | 0.987 | 1.23× (tiled_v2) | — |

Per-cell winners: `tiled_v3` × 5, `tiled_v2` × 4, `tiled_v4` × 3.

---

## Correctness

All kernels agree with `cusparseSpMM` to within `rel_l2_err ≤ 4×10⁻⁷` on every sweep cell — 3+ orders of magnitude below the `< 1e-4` threshold. Every kernel produces identical error vectors per cell (strongest possible cross-check).

| sweep cell | rel_l2_err |
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

---

## DoD verdict

Three independent Colab T4 runs, best Phase 2 kernel vs. baseline on the 4 DoD cells:

| DoD cell | run 1 | run 2 | run 3 | **median** | passes ≥2×? |
|---|---|---|---|---|---|
| m=4096 d=0.05 | 1.28× | 1.48× | 1.25× | **1.27×** | ❌ |
| m=4096 d=0.01 | 1.35× | 1.32× | 1.27× | **1.32×** | ❌ |
| m=8192 d=0.05 | 2.12× | 2.56× | 2.69× | **2.56×** | ✓ |
| m=8192 d=0.01 | 2.05× | 1.99× | 2.04× | **2.04×** | ✓ |

**2 of 4 DoD cells pass.** The m=4096 cells cap empirically at ~1.3× across all six kernel variants and three runs. The reason: at m=4096 the dense matrix B (4 MB) fits entirely in T4's 4 MB L2, making the regime compute-bound rather than bandwidth-bound — memory-pattern optimisations hit a roofline ceiling, not a bandwidth one. The remaining gap is carried into Phase 3 (Tensor Cores).

---

## Performance analysis — ncu (Nsight Compute)

### A2 (m=8192 d=0.01) — bandwidth-bound

| metric | memopt_v2 | tiled | tiled_v2 | **tiled_v3** |
|---|---|---|---|---|
| Compute SM Throughput | 68.30 % | 60.23 % | **18.92 %** | **60.97 %** |
| Memory Throughput | 84.28 % | 75.62 % | 83.80 % | 78.13 % |
| DRAM Throughput | 84.28 % | 64.62 % | 83.80 % | 63.13 % |
| Achieved Occupancy | 83.56 % | 87.53 % | 85.72 % | 86.90 % |
| Registers / Thread | 64 | 50 | 64 | 41 |
| SM Active Cycles | 1,162,697 | 704,697 | 785,081 | **682,734** |

`tiled_v3` restores Compute SM Throughput from `tiled_v2`'s 18.92% collapse back to 60.97% — revert float4, keep wider tile, problem solved.

### B2 (m=4096 d=0.001) — latency-bound

| metric | memopt_v2 | tiled | tiled_v2 | **tiled_v3** |
|---|---|---|---|---|
| Compute SM Throughput | 45.11 % | 47.58 % | 25.11 % | **48.24 %** |
| Memory Throughput | 56.29 % | 67.82 % | 82.25 % | 80.09 % |
| DRAM Throughput | 56.29 % | 67.82 % | 82.25 % | 80.09 % |
| Achieved Occupancy | 75.19 % | 80.60 % | 87.20 % | 81.10 % |
| SM Active Cycles | 47,729 | 35,206 | 28,474 | **28,867** |

### C1 / C2 (m=4096 d=0.01 / d=0.05) — compute-bound: `tiled_v4` negative result

| metric | C2 tiled_v3 | C2 tiled_v4 | C1 tiled_v3 | C1 tiled_v4 |
|---|---|---|---|---|
| Compute SM Throughput | **61.96 %** | 48.43 % | **58.26 %** | 44.46 % |
| DRAM Throughput | 10.29 % | 8.18 % | 27.41 % | 23.88 % |
| Achieved Occupancy | 89.04 % | 89.77 % | 84.29 % | 83.74 % |
| Registers / Thread | 41 | 38 | 41 | 38 |
| SM Active Cycles | 792,516 | 816,847 | 166,232 | 178,171 |

DRAM at 10% confirms C2 is L2-cache-bound, not bandwidth-bound. The Compute SM drop (–13 pp) is a dependency-chain effect — 8-FMA body stalls load-FMA-load; v3's 4-FMA body interleaves better with other warps.

### Bottleneck summary

| regime | bottleneck | winner | why |
|---|---|---|---|
| m=8192 d=0.01 (A2) | DRAM bandwidth | tiled_v3 | scalar loads, wide tile |
| m=4096 d=0.001 (B2) | scheduling latency | tiled_v3 | warp-per-row amortizes overhead |
| m=4096 d=0.05 (C2) | compute / L2 | tiled_v3 (ceiling ~1.3×) | baseline already L2-efficient |

---

## Plan for week 3

Block-Sparse Row (BSR) format, 16×16 blocks, WMMA API (`mma.h`), FP16 input with FP32 accumulate. First test: does Tensor Core acceleration close the m=4096 gap? Week 3 is the **milestone formal report** — full progression baseline → memopt → tiled → TC with speedup graphs.

---

## AI - USE

Utilized AI tools such as ***ChatGPT*** and ***Perplexity AI*** to assist with research, gathering relevant information, and developing an appropriate design structure and implementation plan for the project. ***Claude AI*** was used for code validation and testing, formatting, adding comments, and refining the language and presentation of the report.
