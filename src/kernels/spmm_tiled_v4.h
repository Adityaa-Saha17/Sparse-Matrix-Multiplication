#pragma once

#include "../csr.h"

namespace spmm {

// High-ILP, single-outer-pass variant of spmm_tiled_v3.
//
// Motivation (m=4096 DoD gap). Earlier work left the m=4096 DoD cells
// short of the 2x bar:
//
//   cell                   baseline    best prior   (kernel)    speedup
//   m=4096 d=0.05          1.04 ms     0.82 ms (tiled_v3)       1.27x
//   m=4096 d=0.01          0.19 ms     0.14 ms (tiled_v2)       1.36x
//
// At m=4096, B (K*N*4B = 4 MB) fits entirely in T4's 4 MB L2, so after the
// first pass through the kernel the arithmetic intensity (AI) is high
// enough that the cells are **compute-bound** on T4, not bandwidth-bound:
//
//   FLOPs    = 2 * nnz * N             = 430 MF  (d=0.05) / 86 MF  (d=0.01)
//   resident = A + B + C bytes         = ~14.7 MB (d=0.05) / ~9 MB (d=0.01)
//   AI       = FLOPs / resident bytes  ~  29 FLOP/B (d=0.05)
//   roofline: 320 GB/s * 29 FLOP/B     =  9.3 TF (above T4's 8.1 TF peak)
//
// tiled_v3 at m=4096 d=0.05 reaches 526 GF = ~6.5% of T4 peak. The 2x DoD
// bar is 826 GF = ~10% of peak. The gap is not a bandwidth problem; it is
// a warp-issue / arithmetic-throughput problem.
//
// Single-axis change vs tiled_v3:
//   - COL_TILE      128 -> 256
//   - COLS_PER_LANE   4 -> 8
//   - everything else identical (warp-per-row mapping, NNZ_TILE=64 shmem
//     staging of (col_idx, values), scalar lane-strided B loads, the
//     __launch_bounds__(256, 4) register cap)
//
// What this is supposed to buy:
//   - At N=256 the outer col_tile loop now runs ONCE per row instead of
//     twice. Per-row __syncwarp() count and per-row CSR-row re-stage cost
//     are both halved.
//   - The inner consume loop emits 8 independent FMAs per staged nnz
//     instead of 4. Higher per-warp ILP gives the scheduler more issuable
//     work in flight, which is the lever for a compute-bound kernel.
//   - Same shmem footprint (4 KB/block) and same launch shape, so
//     occupancy should not move much.
//
// What this is NOT:
//   - Not a memory-pattern change (B layout, vector loads, prefetching).
//     v2 already showed float4 loads at this scale stall the SM scheduler;
//     v4 keeps v3's scalar lane-strided loads on purpose.
//   - Not a B-row-sharing scheme (multi-row-per-warp). At d=0.05 with
//     random sparsity, expected col overlap between two adjacent rows is
//     ~205^2 / 4096 ~ 10 of 205 entries -- too small to pay back the
//     coordination cost.
//
// Honest expected outcome: v4 likely improves m=4096 by single-digit to
// low-double-digit percent over v3; whether that crosses 2x over baseline
// is data-dependent and not guaranteed by the design. The header is being
// explicit about this because the project's optimization narrative has been
// honest throughout (memopt_v2 was a no-op, tiled_v2 was a regression).
//
// Constraints: same as tiled_v3 -- N is not required to be a multiple of
// COL_TILE; the inner bounds check guards partial tiles.
void spmm_tiled_v4(const CSR& d_A, const float* d_B, float* d_C, int N);

}  // namespace spmm
