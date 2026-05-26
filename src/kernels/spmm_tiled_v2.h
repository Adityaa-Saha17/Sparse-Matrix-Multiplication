#pragma once

#include "../csr.h"

namespace spmm {

// Phase 2.3 + 2.4 — col-tile streaming, larger register tile, vectorized B loads.
//
// Builds on spmm_tiled (Phase 2.2). Two changes:
//
//   2.3  Register accumulation widened: COL_TILE = 128, COLS_PER_LANE = 4
//        (each lane holds 4 partial sums per col-tile instead of 2). This
//        cuts the number of outer col-tile passes per row in half (at N=256,
//        from 4 → 2) and gives the compiler more inner-loop independent
//        FMAs to schedule.
//
//   2.4  Vectorized B reads: each lane issues one `float4` load per p,
//        covering its 4 consecutive output columns inside the current col
//        tile. The 32 lanes of a warp together issue a single 512-byte
//        coalesced transaction per p (4 * 128 B segments). Writes to C use
//        `float4` stores at the end of each col tile.
//
// Constraints (asserted in the launcher):
//   - N must be a multiple of COL_TILE (128). The harness uses N=256 which
//     satisfies this; for arbitrary N, fall back to spmm_tiled.
//   - Device pointers must be 16-byte aligned (cudaMalloc guarantees 256-byte
//     alignment, so this holds for B and C allocated by the harness).
//
// Same shmem stage of (col_idx, values) as Phase 2.2; same launch shape
// (256 threads/block, __launch_bounds__(256, 4)).
void spmm_tiled_v2(const CSR& d_A, const float* d_B, float* d_C, int N);

}  // namespace spmm
