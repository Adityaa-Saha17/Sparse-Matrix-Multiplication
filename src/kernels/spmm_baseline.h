#pragma once

#include "../csr.h"

namespace spmm {

// Naive CSR SpMM: C(MxN) = A(MxK, CSR) * B(KxN, row-major).
//
// Launch policy: one CUDA thread per output element (i, j).
//   - threadIdx.x indexes columns of C (j) so warps coalesce reads from B.
//   - Each thread independently re-reads row i of A from global memory.
//   - This kernel exists purely as the baseline; optimizations come
//     in later kernels (warp-per-row, shared-mem tiling, etc.).
//
// d_A is on the device. d_B and d_C are device pointers with leading dim N.
void spmm_baseline(const CSR& d_A, const float* d_B, float* d_C, int N);

}  // namespace spmm
