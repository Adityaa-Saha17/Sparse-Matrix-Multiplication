#pragma once

#include "../csr.h"

namespace spmm {

// Memory-optimized CSR SpMM: C(MxN) = A(MxK, CSR) * B(KxN, row-major).
//
// Optimizations over baseline:
//   1. Warp-per-row: 16 threads per row of A. Each thread computes a block
//      of N_BLOCK consecutive columns of C in that row.
//   2. Shared-memory tiling of B: tile B into shared memory (16x16 blocks)
//      to improve L1 hit rate and reduce global memory reads.
//   3. Register accumulation: unroll inner loop to accumulate partial sums
//      in registers before writing to global memory.
//
// d_A is on the device. d_B and d_C are device pointers with leading dim N.
void spmm_memopt(const CSR& d_A, const float* d_B, float* d_C, int N);

}  // namespace spmm
