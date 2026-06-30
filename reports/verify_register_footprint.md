# Phase 2.1 verification recipe (Colab T4)

Status: code merged locally on `main`; not yet committed. After you push, run the cells below on Colab to collect the v2 numbers. The notebook is `colabRunner.ipynb` — paste these as new cells at the bottom.

## What was changed

- New kernel `src/kernels/spmm_memopt_v2.{h,cu}` — warp-per-row CSR SpMM with:
  - `__launch_bounds__(256, 4)` to cap per-thread register usage at the 4-blocks/SM target,
  - cached `values[p]` in a scalar register,
  - hoisted `B + k*N` row pointer with `size_t` arithmetic.
- `src/bench/harness.cu` extended with `--kernel memopt_v2` and `--kernel all` (baseline + memopt + memopt_v2). `--kernel both` is unchanged (still baseline + memopt) so existing Phase 1 sweep cells still work.
- `Makefile` SRCS extended.

## Cell 1 — Build

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
make clean
make -j2 2>&1 | tail -40
```

Expect a clean build. If you see "missing spmm_memopt_v2.h" the git pull didn't include the new files — check `git status` and `ls src/kernels/`.

## Cell 2 — Register-count check (ptxas)

The single number that says whether 2.1 actually moved the needle.

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
echo "=== v1 (spmm_memopt) ==="
nvcc --ptxas-options=-v -arch=sm_75 -O3 -std=c++17 -Isrc \
     -c src/kernels/spmm_memopt.cu -o /tmp/v1.o 2>&1 \
     | grep -E "registers|spill|stack frame"
echo
echo "=== v2 (spmm_memopt_v2) ==="
nvcc --ptxas-options=-v -arch=sm_75 -O3 -std=c++17 -Isrc \
     -c src/kernels/spmm_memopt_v2.cu -o /tmp/v2.o 2>&1 \
     | grep -E "registers|spill|stack frame"
```

What we want to see:
- v1 reports ~63 registers (matches the Phase 1 ncu measurement).
- v2 reports ≤60 registers, and **zero stack frame / zero spills**. If v2 introduces spills, the launch_bounds is too tight — back off to `__launch_bounds__(256, 3)` or drop it entirely.

## Cell 3 — Re-run the 12-cell sweep with all three kernels

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

This produces a 12 × 3 = 36-row table. The diff to track:

|  | v2 vs baseline | v2 vs memopt |
|--|---|---|
| bandwidth-bound (e.g. m=16384, d=0.05) | the regime baseline currently wins | v2 should narrow or close the gap |
| latency-bound (e.g. m=4096, d=0.001) | memopt already wins 1.14× | v2 should match or slightly beat memopt |

**Honest expectation**: 2.1 alone is a register/occupancy tweak; theoretical occupancy at 4 blocks × 8 warps already maxes the SM's 32-warp slot at v1. Achieved occupancy of 83.7% is stall-bound, not reg-bound, so v2 likely buys a few percent at best in the bandwidth-bound cells. The real wins are 2.2 (shmem tiling) and 2.4 (vectorized loads). If 2.1 moves nothing, that is a finding worth reporting — it tells us the bottleneck isn't where the spec assumed.

## Cell 4 — ncu profile for v2

Match the four Phase 1 profile points so the comparison is apples-to-apples. Two are sufficient for 2.1 sign-off (A2 bandwidth-bound + B2 latency-bound).

```bash
%%bash
cd /content/Sparse-Matrix-Multiplication
BIN=build/spmm_bench
NCU="ncu --set full --target-processes all --kernel-name regex:spmm_csr_memopt_v2_kernel"

echo "=== A2-v2: m=8192 d=0.01 memopt_v2 (bandwidth-bound) ==="
$NCU $BIN --m 8192 --k 8192 --n 256 --density 0.01 \
     --iters 1 --warmup 0 --kernel memopt_v2 2>&1 | tail -200

echo
echo "=== B2-v2: m=4096 d=0.001 memopt_v2 (latency-bound) ==="
$NCU $BIN --m 4096 --k 4096 --n 256 --density 0.001 \
     --iters 1 --warmup 0 --kernel memopt_v2 2>&1 | tail -200
```

Note the `tail -200`, not `tail -80` — the Phase 1 B2 record lost its Speed-of-Light section to the shorter tail. We want the full SoL block this time.

Numbers to record from each profile:
- Registers per thread (should match ptxas number from Cell 2).
- Achieved occupancy (the v1 baseline is 83.7% for A2, 75.1% for B2).
- DRAM throughput % (A2-v1: 85.2%).
- Waves per SM (A2-v1: 6.4, B2-v1: small).
- Any new warnings or the disappearance of v1's warnings.

## Decision rule for closing Phase 2.1

Phase 2.1 is done when **all three** hold:
1. ptxas reports v2 ≤ 60 registers with zero spills.
2. v2 correctness: `rel_l2_err < 1e-4` on every sweep cell (same DoD as Phase 1).
3. v2 is **not worse** than memopt in any cell, and is at least tied with baseline in the bandwidth-bound cells (m=16384, d=0.05).

If criterion 3 fails, that's evidence that register pressure isn't the binding constraint and we move directly to 2.2 (shmem tiling of B) without further reg tuning.

## After the run

Paste the Cell 2 ptxas output and the Cell 3 sweep table back here. I'll fold them into `reports/week2.md` and decide whether to proceed to 2.2 or revisit 2.1.
