#pragma once

#include "../csr.h"

namespace spmm {

// Phase 2.5 — refined col-tile streaming. Combines Phase 2.3's larger
// register tile (COL_TILE = 128) with Phase 2.2's scalar lane-strided
// load pattern.
//
// Built to address the regression observed in Phase 2.3 + 2.4 (spmm_tiled_v2)
// vs Phase 2.2 (spmm_tiled). On the Colab T4 sweep (A2: m=8192 d=0.01):
//
//                          A2_tiled    A2_tiled_v2
//   Compute SM throughput   60.2 %      18.9 %       (3.2x drop)
//   DRAM throughput         64.6 %      83.8 %       (bytes/sec up)
//   SM Active Cycles         705k        785k        (longer wall time)
//
// Reading: tiled_v2 saturates DRAM bandwidth but the SMs stall waiting on
// in-flight float4 loads. The wider per-lane transactions (16 B vs 4 B)
// keep more bytes in flight per warp than the compute pipeline can absorb,
// so the scheduler runs short of issuable warps.
//
// Single-axis change vs tiled_v2:
//   - keep COL_TILE = 128, COLS_PER_LANE = 4 (Phase 2.3's win: half as
//     many outer passes per row at N=256, 4 -> 2)
//   - revert the float4 loads/stores back to scalar (Phase 2.4 undone)
//   - revert the lane-contiguous column layout to lane-strided
//     (lane l owns cols {l, l+32, l+64, l+96} within the tile), so the
//     compiler can pipeline 4 independent scalar loads per p instead of
//     blocking on one 16-byte transaction
//
// Expected: SM-side throughput recovers to ~tiled's level while keeping
// the half-as-many-passes saving from the larger tile. Same shmem
// footprint (4 KB/block) and same __launch_bounds__(256, 4).
//
// No alignment / divisibility constraints: N is not required to be a
// multiple of COL_TILE (the inner bounds check handles partial tiles
// the same way Phase 2.2 does).
void spmm_tiled_v3(const CSR& d_A, const float* d_B, float* d_C, int N);

}  // namespace spmm
