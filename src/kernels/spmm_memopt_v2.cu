#include "spmm_memopt_v2.h"

#include <cstddef>
#include <cuda_runtime.h>

#include "../utils.h"

namespace spmm {

namespace {

// Warp-per-row CSR SpMM, v2 (Phase 2.1). See spmm_memopt_v2.h for the rationale
// and the diff vs v1 (spmm_memopt.cu). Intentionally a near-clone of v1 so the
// only deltas under measurement are register pressure and address arithmetic.
__global__ __launch_bounds__(256, 4)
void spmm_csr_memopt_v2_kernel(int M, int N,
                               const int*   __restrict__ row_ptr,
                               const int*   __restrict__ col_idx,
                               const float* __restrict__ values,
                               const float* __restrict__ B,
                               float*       __restrict__ C) {
    // Warp-per-row: each 32-thread warp handles one row of A.
    const int warp_id = blockIdx.x * (blockDim.x / 32) + (threadIdx.x / 32);
    const int lane    = threadIdx.x & 31;

    if (warp_id >= M) return;

    const int row       = warp_id;
    const int row_start = row_ptr[row];
    const int row_end   = row_ptr[row + 1];

    // Use size_t for B / C indexing so the compiler doesn't have to widen
    // 32-bit indices to 64 bits inside the hot loop.
    const std::size_t Nz = static_cast<std::size_t>(N);
    float* const Crow = C + static_cast<std::size_t>(row) * Nz;

    for (int col = lane; col < N; col += 32) {
        float sum = 0.f;
        for (int p = row_start; p < row_end; ++p) {
            const int    k    = col_idx[p];                 // broadcast load
            const float  v    = values[p];                  // broadcast load, kept in reg
            const float* Brow = B + static_cast<std::size_t>(k) * Nz;  // hoisted row base
            sum += v * Brow[col];
        }
        Crow[col] = sum;
    }
}

}  // namespace

void spmm_memopt_v2(const CSR& d_A, const float* d_B, float* d_C, int N) {
    constexpr int THREADS_PER_BLOCK = 256;  // 8 warps; matches v1 for A/B parity
    constexpr int WARPS_PER_BLOCK   = THREADS_PER_BLOCK / 32;
    dim3 block(THREADS_PER_BLOCK, 1);
    dim3 grid(ceil_div(d_A.num_rows, WARPS_PER_BLOCK), 1);
    spmm_csr_memopt_v2_kernel<<<grid, block>>>(d_A.num_rows, N,
                                               d_A.row_ptr, d_A.col_idx, d_A.values,
                                               d_B, d_C);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace spmm
