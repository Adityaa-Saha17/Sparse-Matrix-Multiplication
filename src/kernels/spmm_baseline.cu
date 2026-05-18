#include "spmm_baseline.h"

#include <cuda_runtime.h>

#include "../utils.h"

namespace spmm {

namespace {

__global__ void spmm_csr_naive_kernel(int M, int N, const int* __restrict__ row_ptr, const int* __restrict__ col_idx, const float* __restrict__ values, const float* __restrict__ B, float* __restrict__ C) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;  // column of C
    const int i = blockIdx.y * blockDim.y + threadIdx.y;  // row of C
    if (i >= M || j >= N) return;

    const int start = row_ptr[i];
    const int end   = row_ptr[i + 1];

    float sum = 0.f;
    for (int p = start; p < end; ++p) {
        const int k = col_idx[p];
        sum += values[p] * B[k * N + j];
    }
    C[i * N + j] = sum;
}

}  // namespace

void spmm_baseline(const CSR& d_A, const float* d_B, float* d_C, int N) {
    constexpr int TX = 32;
    constexpr int TY = 8;
    dim3 block(TX, TY);
    dim3 grid(ceil_div(N, TX), ceil_div(d_A.num_rows, TY));
    spmm_csr_naive_kernel<<<grid, block>>>(d_A.num_rows, N,
                                           d_A.row_ptr, d_A.col_idx, d_A.values,
                                           d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace spmm
