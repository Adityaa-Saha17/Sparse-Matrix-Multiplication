#include "spmm_tiled_v2.h"

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "../utils.h"

namespace spmm {

namespace {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / 32;  // 8
constexpr int COL_TILE          = 128;
constexpr int NNZ_TILE          = 64;
constexpr int COLS_PER_LANE     = 4;                       // float4 load per p

// Wider-tile + vectorized-load kernel. See spmm_tiled_v2.h for the rationale and the
// constraints (N % COL_TILE == 0; 16-byte aligned device pointers).
__global__ __launch_bounds__(THREADS_PER_BLOCK, 4)
void spmm_csr_tiled_v2_kernel(int M, int N,
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

    // Per-warp shmem stage for the row's nnz. Same layout as spmm_tiled.
    __shared__ int   k_smem[WARPS_PER_BLOCK][NNZ_TILE];
    __shared__ float v_smem[WARPS_PER_BLOCK][NNZ_TILE];
    int*   k_stage = k_smem[warp_in_block];
    float* v_stage = v_smem[warp_in_block];

    const std::size_t Nz = static_cast<std::size_t>(N);
    float* const Crow = C + static_cast<std::size_t>(row) * Nz;

    // Each lane owns COLS_PER_LANE = 4 consecutive columns inside the
    // current col tile: cols [col_tile + lane*4 + 0 .. col_tile + lane*4 + 3].
    const int lane_col_off = lane * COLS_PER_LANE;

    for (int col_tile = 0; col_tile < N; col_tile += COL_TILE) {
        // Register tile of 4 partial sums per lane.
        float4 sum = make_float4(0.f, 0.f, 0.f, 0.f);

        for (int p_base = row_start; p_base < row_end; p_base += NNZ_TILE) {
            const int nnz_in_tile =
                (NNZ_TILE < row_end - p_base) ? NNZ_TILE : (row_end - p_base);

            // Cooperative warp-local stage of (col_idx, values).
            #pragma unroll
            for (int off = 0; off < NNZ_TILE; off += 32) {
                const int idx = off + lane;
                if (idx < nnz_in_tile) {
                    k_stage[idx] = col_idx[p_base + idx];
                    v_stage[idx] = values [p_base + idx];
                }
            }
            __syncwarp();

            // Consume the staged tile. One float4 B-load per p, 4 FMAs.
            for (int q = 0; q < nnz_in_tile; ++q) {
                const int   k = k_stage[q];
                const float v = v_stage[q];
                const float* Brow_base =
                    B + static_cast<std::size_t>(k) * Nz + col_tile + lane_col_off;
                // Single 16-byte aligned load: the 32 lanes' loads together
                // span 512 B (4 contiguous 128 B coalesced segments).
                const float4 b = *reinterpret_cast<const float4*>(Brow_base);
                sum.x += v * b.x;
                sum.y += v * b.y;
                sum.z += v * b.z;
                sum.w += v * b.w;
            }
            __syncwarp();
        }

        // Vectorized store back into C. col_tile + lane*4 is 16-byte aligned
        // by construction.
        float4* Crow_v4 =
            reinterpret_cast<float4*>(Crow + col_tile + lane_col_off);
        *Crow_v4 = sum;
    }
}

}  // namespace

void spmm_tiled_v2(const CSR& d_A, const float* d_B, float* d_C, int N) {
    if (N % COL_TILE != 0) {
        std::fprintf(stderr,
            "spmm_tiled_v2: N=%d is not a multiple of COL_TILE=%d. "
            "Use spmm_tiled for general N, or pad N up.\n",
            N, COL_TILE);
        std::abort();
    }
    dim3 block(THREADS_PER_BLOCK, 1);
    dim3 grid(ceil_div(d_A.num_rows, WARPS_PER_BLOCK), 1);
    spmm_csr_tiled_v2_kernel<<<grid, block>>>(d_A.num_rows, N,
                                              d_A.row_ptr, d_A.col_idx, d_A.values,
                                              d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace spmm
