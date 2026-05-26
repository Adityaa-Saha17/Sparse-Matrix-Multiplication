# Phase 2.3 + 2.4 verification recipe (Colab T4)

Status: code merged locally on `main`; not yet committed. Run after (or together with) the 2.1 / 2.2 recipes — all kernels coexist in the same build, and `--kernel all` now exercises five kernels (baseline, memopt v1, memopt v2, tiled, tiled_v2) in one harness call.

## What was changed

- New kernel `src/kernels/spmm_tiled_v2.{h,cu}` — extends `spmm_tiled` with:
  - **COL_TILE = 128** (was 64), **COLS_PER_LANE = 4** (was 2) — Phase 2.3.
  - **`float4` vectorized B loads and C stores** — Phase 2.4. One 16-byte load per lane per p; 32 lanes together span 512 bytes (4 × 128-byte coalesced segments).
- `src/bench/harness.cu` extended with `--kernel tiled_v2` and skipped silently under `--kernel all` when `N % 128 != 0`. The harness's default `--n 256` satisfies this.
- `Makefile` SRCS extended.

## Design — expected wins and risks

| Quantity | spmm_tiled | spmm_tiled_v2 |
|---|---|---|
| Col-tile passes per row at N=256 | 4 (N/COL_TILE=64) | 2 (N/COL_TILE=128) |
| Per-lane register accumulators | 2 floats | 4 floats (float4) |
| B loads per p per lane | 2 scalar | 1 float4 |
| B traffic per warp per p | 32 × 4 B = 128 B | 32 × 16 B = 512 B in one transaction |
| C stores per col tile per lane | 2 scalar | 1 float4 |

**Expected wins** (bandwidth-bound regime):
- Fewer outer col-tile passes → less per-pass overhead (accumulator init, address recompute, shmem stage repeat) amortized over more arithmetic.
- `float4` loads issue fewer memory instructions, giving the scheduler more headroom to hide latency.
- Vectorized stores remove the 2× scalar C-write loop.

**Risks**:
- Register pressure: `float4 sum` + address regs + loop state may push past 64 regs/thread and drop occupancy. If ptxas reports >64 regs, the launch_bounds is forcing spills — that needs to be backed off (drop `__launch_bounds__(256, 4)` to `(256, 3)` or remove).
- Alignment: every B and C access in this kernel assumes 16-byte alignment. `cudaMalloc` guarantees 256-byte alignment, and `col_tile + lane*4` is always a multiple of 4 floats = 16 B, so this should hold for any harness run. If it ever fires a misaligned-access fault, that's the first thing to suspect.
- Latency-bound regime: tiled_v2's bigger per-pass setup is even more wasteful when nnz/row is tiny. Expect it to be **worse** than tiled (and possibly worse than memopt_v2) in the m=4096 d=0.001 corner. That's the expected ablation finding.

## Cell 1 — Build

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
make clean
make -j2 2>&1 | tail -40
```

## Cell 2 — ptxas: regs, shmem, spills

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
echo "=== tiled_v2 (Phase 2.3 + 2.4) ==="
nvcc --ptxas-options=-v -arch=sm_75 -O3 -std=c++17 -Isrc \
     -c src/kernels/spmm_tiled_v2.cu -o /tmp/tiled_v2.o 2>&1 \
     | grep -E "registers|smem|spill|stack frame"
```

Targets:
- Registers per thread: **≤ 64** (4 blocks/SM). Most-likely landing: 56–64.
- Shared memory per block: ~4 KB (same warp-local stage as tiled).
- **Zero spills.** If spills appear, the most likely cause is the unrolled inner FMA expanding to many independent issue slots. Try removing `#pragma unroll` on the NNZ_TILE staging loop first.

## Cell 3 — 5-way sweep

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
BIN=build/spmm_bench
for m in 1024 4096 8192 16384; do
  for d in 0.05 0.01 0.001; do
    $BIN --m $m --k $m --n 256 --density $d --iters 20 --warmup 5 --kernel all
  done
done
```

12 × 5 = 60 rows. Build the comparison table per cell with columns: baseline / memopt v1 / memopt v2 / tiled / tiled_v2 / winner / DoD-met?

**Phase 2 DoD recap**: the best kernel in each cell must hit **≥ 2× over baseline at m=4096–8192, d=0.01–0.05**. The four cells in that range are the gate for closing Phase 2.

## Cell 4 — ncu profile for tiled_v2

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
BIN=build/spmm_bench
NCU="ncu --set full --target-processes all --kernel-name regex:spmm_csr_tiled_v2_kernel"

echo "=== A2-tiled_v2: m=8192 d=0.01 (bandwidth-bound) ==="
$NCU $BIN --m 8192 --k 8192 --n 256 --density 0.01 \
     --iters 1 --warmup 0 --kernel tiled_v2 2>&1 | tail -200

echo
echo "=== B2-tiled_v2: m=4096 d=0.001 (latency-bound) ==="
$NCU $BIN --m 4096 --k 4096 --n 256 --density 0.001 \
     --iters 1 --warmup 0 --kernel tiled_v2 2>&1 | tail -200
```

Numbers to compare against tiled (2.2) and memopt v1 (Phase 1):
- Memory chart — should show fewer global B sectors / requests due to float4 coalescing.
- L1/TEX throughput — should improve from tiled's already-improved baseline.
- "Long Scoreboard Stalls" — vector loads typically reduce these.

## Decision rule for closing Phase 2

Phase 2 is done when **all four** hold:
1. tiled_v2 builds clean (≤64 regs, zero spills, correct shmem footprint).
2. Correctness: `rel_l2_err < 1e-4` on every sweep cell for every kernel.
3. In each of the four Phase-2 DoD cells (m=4096 d=0.05, m=4096 d=0.01, m=8192 d=0.05, m=8192 d=0.01), the best of {memopt v2, tiled, tiled_v2} achieves **≥ 2× over baseline**.
4. ncu shows reduced global B traffic and improved L1/shmem hit rate vs Phase 1.

If criterion 3 misses, the options on the table are:
- 2.5 (optional from the original plan): runtime hybrid dispatch fallback — pick the best of v2/tiled/tiled_v2 per (density, m) bucket. Easy win if no single kernel dominates.
- Push COLS_PER_LANE higher (8) with two float4 loads per p — bigger register tile, risky.
- Reduce shmem stage to free regs for a bigger output tile.

## After the run

Paste Cell 2's ptxas output and Cell 3's 60-row sweep back here. I'll then either:
1. Finalize `reports/week2.md` with the measured numbers, or
2. Pick exactly the right 2.5 fallback / extension to close the DoD gap, write it, and re-iterate.
