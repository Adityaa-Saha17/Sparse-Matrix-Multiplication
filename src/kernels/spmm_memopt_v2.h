#pragma once

#include "../csr.h"

namespace spmm {

// Phase 2.1 — register-footprint-reduced version of spmm_memopt.
//
// Same warp-per-row CSR SpMM as spmm_memopt.cu (v1), with three targeted
// changes aimed at lifting bandwidth-bound achieved occupancy on T4 (sm_75):
//
//   1. __launch_bounds__(256, 4)  — tells nvcc that this kernel must be
//      schedulable at 4 blocks of 256 threads per SM, capping per-thread
//      register usage so the register file is the binding constraint, not
//      software-pipelining heuristics.
//   2. Cached `values[p]` in a scalar register, so the broadcast load is
//      not re-issued as part of an address-computation chain.
//   3. Hoisted `B + k*N` into a row pointer (size_t arithmetic), so each
//      lane only adds `col` to a precomputed 64-bit base instead of
//      recomputing `k * N + col` per iteration.
//
// Math is identical to spmm_memopt within FP rounding. Used as an A/B
// against the v1 kernel for the Phase 2 ablation; both remain in the build.
void spmm_memopt_v2(const CSR& d_A, const float* d_B, float* d_C, int N);

}  // namespace spmm
