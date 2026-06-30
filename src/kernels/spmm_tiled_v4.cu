#include "spmm_tiled_v4.h"

#include <cstddef>
#include <cuda_runtime.h>

#include "../utils.h"

namespace spmm {

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / 32;  // 8
constexpr int COL_TILE          = 256;
constexpr int NNZ_TILE          = 64;
constexpr int COLS_PER_LANE     = COL_TILE / 32;            // 8

// Single-outer-pass kernel. See spmm_tiled_v4.h for design rationale.
//
// Algorithm shape is identical to spmm_tiled_v3; only COL_TILE and
// COLS_PER_LANE differ. The hypothesis under test: at N=256, a single
// outer pass per row with 8 FMAs/nnz inside the warp gives the SM
// scheduler enough independent instructions to push past tiled_v3's
// ~6.5%-of-peak ceiling in the compute-bound m=4096 regime.
__global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void spmm_csr_tiled_v4_kernel(int M, int N,
                              const int*   __restrict__ row_ptr,
                              const int*   __restrict__ col_idx,
                              const float* __restrict__ values,
                              const float* __restrict__ B,
                              float*       __restrict__ C) {
    const int warp_in_block = threadIdx.x / 32;
    const int lane          = threadIdx.x & 31;
    const int warp_id       = blockIdx.x * WARPS_PER_BLOCK + warp_in_block;

    if (warp_id >= M) return;

    const int row       = warp_id;
    const int row_start = row_ptr[row];
    const int row_end   = row_ptr[row + 1];

    // Per-warp shmem stage for the row's nnz. Same layout/size as
    // spmm_tiled / spmm_tiled_v2 / spmm_tiled_v3:
    // 8 warps * 64 entries * (int + float) = 4 KB/block.
    __shared__ int   k_smem[WARPS_PER_BLOCK][NNZ_TILE];
    __shared__ float v_smem[WARPS_PER_BLOCK][NNZ_TILE];
    int*   k_stage = k_smem[warp_in_block];
    float* v_stage = v_smem[warp_in_block];

    const std::size_t Nz = static_cast<std::size_t>(N);
    float* const Crow = C + static_cast<std::size_t>(row) * Nz;

    for (int col_tile = 0; col_tile < N; col_tile += COL_TILE) {
        // Register tile: 8 partial sums per lane, covering cols
        // {col_tile + lane + i*32} for i in [0, 8). At N=256 the outer
        // loop runs exactly once -- one warp covers a whole output row.
        float sum[COLS_PER_LANE];
        #pragma unroll
        for (int i = 0; i < COLS_PER_LANE; ++i) sum[i] = 0.f;

        for (int p_base = row_start; p_base < row_end; p_base += NNZ_TILE) {
            const int nnz_in_tile =
                (NNZ_TILE < row_end - p_base) ? NNZ_TILE : (row_end - p_base);

            // Cooperative warp-local stage. 32 lanes load NNZ_TILE entries
            // in NNZ_TILE/32 = 2 strides. Bounds check guards partial tail.
            #pragma unroll
            for (int off = 0; off < NNZ_TILE; off += 32) {
                const int idx = off + lane;
                if (idx < nnz_in_tile) {
                    k_stage[idx] = col_idx[p_base + idx];
                    v_stage[idx] = values [p_base + idx];
                }
            }
            __syncwarp();

            // Consume the staged tile. Eight independent scalar loads per
            // nnz at col offsets {lane + i*32} for i in [0, 8). Each stride
            // is a single coalesced 128-byte segment (32 lanes * 4 B). With
            // 8 loads + 8 FMAs unrolled per p, the compiler has wide ILP to
            // schedule against the in-flight memory ops -- the per-warp
            // arithmetic density doubles vs tiled_v3.
            for (int q = 0; q < nnz_in_tile; ++q) {
                const int    k    = k_stage[q];
                const float  v    = v_stage[q];
                const float* Brow = B + static_cast<std::size_t>(k) * Nz + col_tile;
                #pragma unroll
                for (int i = 0; i < COLS_PER_LANE; ++i) {
                    const int col_off = lane + i * 32;
                    if (col_tile + col_off < N) {
                        sum[i] += v * Brow[col_off];
                    }
                }
            }
            __syncwarp();
        }

        // Write back this row's col-tile slice. Eight coalesced scalar
        // stores per warp (one per stride). At N=256 this is the only
        // write per row and covers the entire output row.
        #pragma unroll
        for (int i = 0; i < COLS_PER_LANE; ++i) {
            const int col_off = lane + i * 32;
            const int col_g   = col_tile + col_off;
            if (col_g < N) {
                Crow[col_g] = sum[i];
            }
        }
    }
}

}  // namespace

void spmm_tiled_v4(const CSR& d_A, const float* d_B, float* d_C, int N) {
    dim3 block(THREADS_PER_BLOCK, 1);
    dim3 grid(ceil_div(d_A.num_rows, WARPS_PER_BLOCK), 1);
    spmm_csr_tiled_v4_kernel<<<grid, block>>>(d_A.num_rows, N,
                                              d_A.row_ptr, d_A.col_idx, d_A.values,
                                              d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace spmm
