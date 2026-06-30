#pragma once

#include "../csr.h"

namespace spmm {

// Column-tile streaming with shmem-staged CSR rows.
//
// Still warp-per-row, but the loop nest is restructured:
//
//   for col_tile in [0, N) step COL_TILE:        // outer
//     accum[COLS_PER_LANE] = 0                   // register tile
//     for p_base in nnz step NNZ_TILE:           // inner stages
//       cooperatively load (col_idx, values)[p_base : p_base+NNZ_TILE]
//                                            into per-warp shmem
//       for q in tile:
//         k = k_stage[q]; v = v_stage[q]
//         accum += v * B[k, col_tile : col_tile+COL_TILE]
//     C[row, col_tile : col_tile+COL_TILE] = accum
//
// vs spmm_memopt(_v2):
//   - B traffic: each row of B[k, :] is visited N/COL_TILE times instead of
//     N/32 times. At COL_TILE=64, N=256 → 2x reduction in B re-fetches.
//   - col_idx / values: hot in shmem during one col_tile pass. (In v1 these
//     were broadcast loads with L1 hits; in tiled we make the reuse explicit
//     in shmem so the access pattern is independent of L1 behavior.)
//   - C writes: COLS_PER_LANE coalesced stores per lane per col_tile.
//
// Caveat called out for the ablation report: this restructure does NOT give
// cross-row reuse of B within a block (different rows of A index different
// rows of B). The B-traffic reduction is per-row, from collapsing the N/32
// col passes into N/COL_TILE col passes.
//
// Defaults: COL_TILE=64, NNZ_TILE=64, 256 threads/block (8 warps).
// __launch_bounds__(256, 4) matches the v2 occupancy target on sm_75.
void spmm_tiled(const CSR& d_A, const float* d_B, float* d_C, int N);

}  // namespace spmm
