#pragma once

// Phase 3 — Tensor Core SpMM via the WMMA API.
//
// Requires: CUDA sm_75+ (Turing), compiled with -arch=sm_75 or higher.
//
// Algorithm overview
// ------------------
// The sparse matrix A is stored in BSR format with 16×16 blocks (see bsr.h).
// The dense input B and output C remain FP32 in the harness; the caller
// converts B to FP16 once with float_to_half() before the timed loop, and
// the kernel accumulates in FP32 so C is written back as FP32.
//
// Kernel shape:
//   Grid  : (block_rows,  ceil(N / 16))
//   Block : 32 threads  (exactly one warp — required by the WMMA API)
//
// Each warp computes one 16×16 tile of C:
//   C[ br*16 : (br+1)*16 ][ nt*16 : (nt+1)*16 ]
//     = Σ_{BSR blocks in block-row br}  A_block  ×  B_slice
//
//   A_block : 16×16 FP16 from BSR values (stride 16 within block).
//   B_slice : 16 rows × 16 cols FP16 slice of B (stride N).

#include <cuda_fp16.h>
#include "../bsr.h"

namespace spmm {

// Convert d_src (float, n_elements contiguous) → d_dst (half, pre-allocated).
// Launches a simple element-wise conversion kernel.
void float_to_half(const float* d_src, half* d_dst, int n_elements);

// BSR SpMM using Tensor Cores.
//   d_bsr    : device BSR of A (FP16 values).
//   d_B_fp16 : K×N FP16 row-major dense B (call float_to_half once upfront).
//   d_C      : M×N FP32 row-major dense output.
//   N        : number of columns in B and C.
void spmm_wmma(const BSR& d_bsr, const half* d_B_fp16, float* d_C, int N);

}  // namespace spmm
