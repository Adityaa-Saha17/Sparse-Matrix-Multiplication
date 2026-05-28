// Phase 3 — Tensor Core SpMM (BSR + WMMA).
// See spmm_wmma.h for the full algorithm description.

#include "spmm_wmma.h"
#include "../utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>        // nvcuda::wmma — requires sm_75+ and -arch=sm_75+

using namespace nvcuda;

namespace spmm {

namespace {

// ── Float → Half conversion kernel ────────────────────────────────────────
__global__ void float_to_half_kernel(const float* __restrict__ src,
                                      half*        __restrict__ dst,
                                      int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2half(src[i]);
}

// ── BSR SpMM WMMA kernel ──────────────────────────────────────────────────
//
// Grid  : (block_rows, ceil(N / 16))
// Block : 32 threads  (one warp — WMMA intrinsics require full-warp participation)
//
// Memory layout of BSR values (block b, element at local-row lr, local-col lc):
//   bsr_values[ b * 16 * 16  +  lr * 16  +  lc ]    (row-major within block)
//
// Memory layout of d_B_fp16 (row r, column c):
//   d_B_fp16[ r * N + c ]    (row-major, stride = N)
//
// Each warp accumulates its 16×16 output tile entirely in registers (the WMMA
// accumulator fragment), then writes the FP32 result to global C in a single
// store_matrix_sync call.  No shared memory is used on the fast path (M and N
// are both multiples of 16 for all test-case sizes); the boundary slow-path
// uses 1 KB of shared memory per block.
//
// Occupancy note (sm_75 / T4):
//   Registers : limited by WMMA fragment state; ~64 regs/thread expected.
//   Shared mem: 0 B on fast path (1 KB on slow path, never triggered in sweep).
//   With 32 threads/block the hardware can schedule up to 16 blocks/SM,
//   giving 16 warps/SM (50 % warp occupancy).  This is lower than the tiled
//   CSR kernels but is intentional: TC throughput compensates for the lower
//   warp count, especially in the compute-bound m=4096 regime.
__global__
void spmm_wmma_kernel(int M, int N,
                       int block_rows,
                       const int*  __restrict__ bsr_row_ptr,
                       const int*  __restrict__ bsr_col_idx,
                       const half* __restrict__ bsr_values,
                       const half* __restrict__ d_B_fp16,
                       float*      __restrict__ d_C) {
    const int br = blockIdx.x;   // block-row of A
    const int nt = blockIdx.y;   // N-tile (column tile of B/C)

    if (br >= block_rows) return;

    // Initialise accumulator to 0.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    const int blk_start = bsr_row_ptr[br];
    const int blk_end   = bsr_row_ptr[br + 1];

    for (int blk = blk_start; blk < blk_end; ++blk) {
        const int bc = bsr_col_idx[blk];

        // ── A fragment ────────────────────────────────────────────────────
        // 16×16 FP16 tile from BSR; stride within block = 16 (BSR_BLOCK).
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        const half* a_ptr = bsr_values + static_cast<size_t>(blk) * 16 * 16;
        wmma::load_matrix_sync(a_frag, a_ptr, 16);

        // ── B fragment ────────────────────────────────────────────────────
        // Rows bc*16 .. bc*16+15, cols nt*16 .. nt*16+15 of d_B_fp16.
        // Stride = N (the full column width of B).
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        const half* b_ptr = d_B_fp16
                            + static_cast<size_t>(bc) * 16 * N
                            + nt * 16;
        wmma::load_matrix_sync(b_frag, b_ptr, N);

        // ── TC multiply-accumulate ─────────────────────────────────────────
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // ── Store 16×16 FP32 output tile to C ────────────────────────────────
    const int row_base = br * 16;
    const int col_base = nt * 16;

    if ((row_base + 16) <= M && (col_base + 16) <= N) {
        // Fast path: tile is fully inside the output matrix — write directly.
        float* c_ptr = d_C + static_cast<size_t>(row_base) * N + col_base;
        wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
    } else {
        // Slow path: partial tile at the boundary — stage through shared memory.
        // Only triggered when M or N is not a multiple of 16; all benchmark
        // cases have M ∈ {1024,4096,8192,16384} and N = 256, so this path is
        // dead code in practice.  Kept for correctness with arbitrary inputs.
        __shared__ float shmem[16 * 16];
        wmma::store_matrix_sync(shmem, c_frag, 16, wmma::mem_row_major);
        __syncwarp();

        const int rows_rem = M - row_base;
        const int cols_rem = N - col_base;
        for (int t = threadIdx.x; t < 256; t += 32) {
            const int r = t / 16, c = t % 16;
            if (r < rows_rem && c < cols_rem) {
                d_C[static_cast<size_t>(row_base + r) * N + col_base + c] = shmem[t];
            }
        }
    }
}

}  // namespace (anonymous)

// ─────────────────────────────────────────────────────────────────────────────
// Public API implementations
// ─────────────────────────────────────────────────────────────────────────────

void float_to_half(const float* d_src, half* d_dst, int n_elements) {
    constexpr int BLOCK = 256;
    const int grid = (n_elements + BLOCK - 1) / BLOCK;
    float_to_half_kernel<<<grid, BLOCK>>>(d_src, d_dst, n_elements);
    CUDA_CHECK(cudaGetLastError());
}

void spmm_wmma(const BSR& d_bsr, const half* d_B_fp16, float* d_C, int N) {
    const int n_tiles = (N + 15) / 16;
    dim3 block(32, 1);                          // one warp per block
    dim3 grid(d_bsr.block_rows, n_tiles);       // one block per output tile
    spmm_wmma_kernel<<<grid, block>>>(
        d_bsr.num_rows, N,
        d_bsr.block_rows,
        d_bsr.row_ptr, d_bsr.col_idx, d_bsr.values,
        d_B_fp16, d_C);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace spmm
