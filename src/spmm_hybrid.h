#pragma once

// Hybrid execution: automatic CUDA-core vs. Tensor-Core dispatch.
//
// The project ships two competitive SpMM back ends with opposite sweet spots:
//
//   * tiled_v3  — CSR on the CUDA cores. Robust across sparsity patterns;
//                 the fastest FP32 kernel on bandwidth-bound cells.
//   * wmma      — BSR (16x16) on the Tensor Cores. Unbeatable when the
//                 nonzeros already cluster into dense 16x16 blocks, but it
//                 collapses on uniform-random sparsity because scattered
//                 nonzeros inflate every touched block (fill-in), forcing the
//                 Tensor Cores to multiply mostly zeros.
//
// The single number that separates these regimes is the BSR fill-in ratio
// (stored block elements / original nonzeros). Near 1.0 the blocks are dense
// and the Tensor Cores win; large values mean the BSR is mostly padding and
// the CUDA-core CSR kernel wins. hybrid_choose() builds the BSR structure on
// the host (cheap relative to the SpMM itself), measures the fill-in, and
// picks the back end before any device work happens.

#include "csr.h"

namespace spmm {

enum class Backend {
    CSR_TILED,    // tiled_v3 on CUDA cores
    TENSOR_WMMA   // BSR + WMMA on Tensor Cores
};

// A back-end decision plus the evidence behind it.
struct HybridPlan {
    Backend backend       = Backend::CSR_TILED;
    double  fill_in_ratio = 0.0;   // BSR stored elements / original nnz
    double  block_density = 0.0;   // fraction of 16x16 blocks that are non-empty
    float   threshold     = 0.0f;  // fill-in cut-off used for the decision
    bool    n_aligned     = false; // N % 16 == 0  (required by WMMA)
    const char* reason    = "";    // human-readable justification
};

// Decide the back end from A's block structure.
//   h_A                : host CSR (full matrix).
//   N                  : number of columns of B/C (WMMA needs N % 16 == 0).
//   fill_in_threshold  : pick Tensor Cores only when fill-in <= this value.
// The default threshold of 2.0 cleanly separates structured inputs
// (fill-in ~1) from uniform-random inputs (fill-in >= ~20 at the densities
// used in this project).
HybridPlan hybrid_choose(const CSR& h_A, int N, float fill_in_threshold = 2.0f);

// String form of a back-end choice ("csr" / "wmma").
const char* backend_name(Backend b);

// End-to-end hybrid SpMM (the integrated single-GPU path).
// Builds whatever device-side representation the chosen back end needs,
// runs `warmup` + `iters` timed launches, fills h_C (host, M*N row-major),
// and returns the median per-launch kernel time in milliseconds.
//   plan : the decision from hybrid_choose (so the caller can log it).
//   h_A  : host CSR.
//   h_B  : host row-major K*N dense B.
//   h_C  : host row-major M*N output (written on return).
//   N    : columns of B/C.
float spmm_hybrid_run(const HybridPlan& plan,
                      const CSR& h_A, const float* h_B, float* h_C,
                      int N, int warmup, int iters);

}  // namespace spmm
