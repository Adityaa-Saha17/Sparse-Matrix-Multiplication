# Tiled-streaming kernel (tiled) — verification recipe (Colab T4)

Status: code merged locally on `main`; not yet committed. Run after (or together with) `verify_register_footprint.md` — both kernels live in the same build now, and `--kernel all` exercises all four (baseline, memopt v1, memopt v2, tiled) in one harness call.

## What was changed

- New kernel `src/kernels/spmm_tiled.{h,cu}` — warp-per-row CSR SpMM, restructured as:
  - **Outer loop over col tiles** of B (`COL_TILE = 64`).
  - **Per-warp shmem stage** of `col_idx` / `values` (`NNZ_TILE = 64`).
  - **Per-lane register accumulators** (`COLS_PER_LANE = 2`) for the col tile.
  - `__launch_bounds__(256, 4)`.
- `src/bench/harness.cu` extended with `--kernel tiled`. `--kernel all` is now 4-way (baseline + memopt + memopt_v2 + tiled).
- `Makefile` SRCS extended.

## Design — what we expect to move and what we don't

The handshake spec phrased the goal as "stage B tiles into shmem; reuse across rows in a block." Strict cross-row B reuse only pays off when multiple rows of A index the same `k`, which random CSR doesn't give us. The actual mechanical change in this kernel is:

| Quantity | spmm_memopt(_v2) | spmm_tiled |
|---|---|---|
| Times B[k, :] is visited per row | N/32 = 8 (at N=256) | N/COL_TILE = 4 (at N=256, COL_TILE=64) |
| col_idx / values residence during inner loop | broadcast load via L1 | warp-local shmem |
| Per-lane accumulator | 1 (one col at a time) | 2 (COLS_PER_LANE=2) |

**Expected wins** (bandwidth-bound regime, e.g. m=16384 d=0.05):
- ~2× reduction in B re-fetches → L2 hit rate up, DRAM throughput possibly down (less re-traffic) **or** up (better latency hiding); ncu will tell us which.
- col_idx/values shmem stage isolates their access pattern from L1 thrashing.

**Likely null result** (latency-bound regime, e.g. m=4096 d=0.001 with ~4 nnz/row):
- Only 1 col_tile pass uses real work; the others run the staged inner loop with tiny `nnz_in_tile`. Overhead per col_tile (setup, accumulator init, shmem sync) is amortized over very little work. Probably **slower** than memopt_v2 in this regime. That's the expected ablation finding: tiled wins in bandwidth-bound, memopt_v2 wins in latency-bound — strong evidence for the hybrid dispatcher.

## Cell 1 — Build

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
make clean
make -j2 2>&1 | tail -40
```

Same as 2.1 cell. If the new `spmm_tiled.cu` is missing from the build, check `git pull` and `ls src/kernels/`.

## Cell 2 — ptxas: regs, shmem, spills

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
echo "=== tiled (the tiled-streaming kernel) ==="
nvcc --ptxas-options=-v -arch=sm_75 -O3 -std=c++17 -Isrc \
     -c src/kernels/spmm_tiled.cu -o /tmp/tiled.o 2>&1 \
     | grep -E "registers|smem|spill|stack frame"
```

What we want to see:
- Registers per thread: **≤ 64** (the 4-blocks/SM launch_bounds target).
- Shared memory per block: ~4096 bytes (8 warps × 64 entries × (int+float) = 4 KB) plus a small ABI overhead.
- **Zero stack frame, zero spills.** If spills appear, the register tile in the inner loop is too aggressive — first thing to try is dropping `#pragma unroll` on the inner-most `for (i = 0; i < COLS_PER_LANE; ++i)`.

## Cell 3 — 4-way 12-cell sweep

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

12 × 4 = 48 rows. The table to build for `reports/memory_optimization.md` should be one row per (m, d) cell with columns: baseline / memopt v1 / memopt v2 / tiled / winner.

Decision cuts to look for:
- **Bandwidth-bound corner** (m=16384 d=0.05): tiled should approach or pass baseline; v1 and v2 will both be ~0.85–0.90× baseline as in the baseline.
- **Latency-bound corner** (m=4096 d=0.001): tiled likely worst; memopt_v2 should hold the win.
- **Mid regime** (m=4096–8192, d=0.01): the most interesting cells. The DoD for memory optimization close-out is **≥ 2× over baseline** here. If tiled is well short of 2×, that flags 2.3 / 2.4 (register accumulation in a more aggressive form, vectorized loads) as required, not optional.

## Cell 4 — ncu profiles for tiled

Two profile points, matching the baseline's A2 (bandwidth-bound) and B2 (latency-bound):

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
BIN=build/spmm_bench
NCU="ncu --set full --target-processes all --kernel-name regex:spmm_csr_tiled_kernel"

echo "=== A2-tiled: m=8192 d=0.01 tiled (bandwidth-bound) ==="
$NCU $BIN --m 8192 --k 8192 --n 256 --density 0.01 \
     --iters 1 --warmup 0 --kernel tiled 2>&1 | tail -200

echo
echo "=== B2-tiled: m=4096 d=0.001 tiled (latency-bound) ==="
$NCU $BIN --m 4096 --k 4096 --n 256 --density 0.001 \
     --iters 1 --warmup 0 --kernel tiled 2>&1 | tail -200
```

Numbers to record:
- Registers per thread + shmem per block (should match Cell 2's ptxas).
- Achieved occupancy.
- DRAM throughput % — compare to memopt v1's 85.2% at A2.
- L2 hit rate — this is where 2.2 should show movement.
- Memory chart: hopefully fewer global B loads than v1's profile.

## Decision rule for closing memory optimization

the tiled-streaming kernel done when:
1. ptxas: ≤ 64 regs, shmem ≈ 4 KB, zero spills.
2. Correctness: `rel_l2_err < 1e-4` across all sweep cells.
3. tiled beats v1/v2 in the bandwidth-bound corner (m ≥ 8192, d ≥ 0.01).

memory optimization (the whole stage) done when:
1. The memory optimization DoD holds: **≥ 2× over baseline at m=4096–8192, d=0.01–0.05** for the best kernel in each cell (which may be different kernels in different cells — that's fine, it's the input to the hybrid dispatcher).
2. ncu shows improved L1 / shared-mem hit rate vs the baseline's A2/B2 profiles.
3. `reports/memory_optimization.md` written and pushed.

If 2.2 closes but the 2× DoD doesn't, the remaining headroom is in 2.3 (more aggressive register tiling — e.g., bigger COLS_PER_LANE, ≥4) and 2.4 (`float4` loads on B, explicit `__ldg` on the row pointer). I'll write those if needed once we have the 2.2 measurements.

## After the run

Paste Cell 2's ptxas line and Cell 3's sweep table back here. Based on which cells move and which don't, we either:
- Close memory optimization with `reports/memory_optimization.md`, or
- Pick exactly the next sub-task (2.3 or 2.4) that targets the remaining gap.
