#include "spmm_memopt.h"

#include <cuda_runtime.h>

#include "../utils.h"

namespace spmm {

namespace {

__global__ void spmm_csr_memopt_kernel(int M, int N,
                                       const int* __restrict__ row_ptr,
                                       const int* __restrict__ col_idx,
                                       const float* __restrict__ values,
                                       const float* __restrict__ B,
                                       float* __restrict__ C) {
    // Warp-per-row: each 32-thread warp handles one row of A.
    // Multiple warps per block for better occupancy.
    int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    int lane    = threadIdx.x & 31;

    if (warp_id >= M) return;

    int row = warp_id;
    int row_start = row_ptr[row];
    int row_end   = row_ptr[row + 1];

    // Each thread in the warp computes columns: lane, lane+32, lane+64, ...
    // This ensures coalesced reads of B and reduces redundant row_ptr lookups.
    for (int col = lane; col < N; col += 32) {
        float sum = 0.f;
        for (int p = row_start; p < row_end; ++p) {
            int k = col_idx[p];
            sum += values[p] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

}  // namespace

void spmm_memopt(const CSR& d_A, const float* d_B, float* d_C, int N) {
    constexpr int THREADS_PER_BLOCK = 256;  // 8 warps for occupancy
    constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / 32;
    dim3 block(THREADS_PER_BLOCK, 1);
    dim3 grid(ceil_div(d_A.num_rows, WARPS_PER_BLOCK), 1);
    spmm_csr_memopt_kernel<<<grid, block>>>(d_A.num_rows, N,
                                            d_A.row_ptr, d_A.col_idx, d_A.values,
                                            d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace spmm
