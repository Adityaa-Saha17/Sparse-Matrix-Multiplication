#pragma once

// BSR (Block-Sparse Row) format with a fixed 16×16 block size.
//
// 16×16 is the WMMA tile size supported by all Turing/Ampere/Hopper GPUs
// for FP16 inputs with FP32 accumulation.  Every non-empty 16×16 sub-block
// of the original sparse matrix is stored as a dense FP16 tile in row-major
// order; missing elements within a block are zero-padded.
//
// Two structs are provided:
//   BSR_host  — host memory, uses std::vector for RAII lifetime.
//   BSR       — device memory, raw pointers (mirrors CSR in csr.h).

#include <cuda_fp16.h>
#include <vector>

#include "csr.h"

namespace spmm {

// Fixed block size for WMMA m=16 n=16 k=16 on sm_75+.
constexpr int BSR_BLOCK = 16;

// ── Host BSR ──────────────────────────────────────────────────────────────
struct BSR_host {
    int num_rows   = 0;  // original M (rows of A)
    int num_cols   = 0;  // original K (cols of A)
    int block_rows = 0;  // ceil(M / BSR_BLOCK)
    int block_cols = 0;  // ceil(K / BSR_BLOCK)
    int num_blocks = 0;  // number of non-empty BSR blocks
    int nnz        = 0;  // original nonzero count (before block fill-in)

    // row_ptr[br+1] - row_ptr[br] = number of non-empty blocks in block-row br.
    std::vector<int>  row_ptr;  // length: block_rows + 1

    // Block-column index of each non-empty block (sorted within each block-row).
    std::vector<int>  col_idx;  // length: num_blocks

    // Dense FP16 data for each block in row-major order (stride = BSR_BLOCK).
    // Block b occupies values[b * BSR_BLOCK * BSR_BLOCK .. (b+1) * BSR_BLOCK^2 - 1].
    std::vector<half> values;   // length: num_blocks * BSR_BLOCK * BSR_BLOCK
};

// ── Device BSR ────────────────────────────────────────────────────────────
struct BSR {
    int num_rows   = 0;
    int num_cols   = 0;
    int block_rows = 0;
    int block_cols = 0;
    int num_blocks = 0;
    int nnz        = 0;

    int*  row_ptr = nullptr;  // device pointer, length: block_rows + 1
    int*  col_idx = nullptr;  // device pointer, length: num_blocks
    half* values  = nullptr;  // device pointer, length: num_blocks * BSR_BLOCK^2, FP16
};

// ── BSR statistics ─────────────────────────────────────────────────────────
struct BSR_stats {
    int    num_blocks;         // non-empty blocks
    long   stored_elements;    // num_blocks * BSR_BLOCK^2 (includes zero-padding)
    int    original_nnz;       // original nonzeros (without padding)
    double fill_in_ratio;      // stored_elements / original_nnz  (>1 means overhead)
    double block_density;      // num_blocks / (block_rows * block_cols)
};

// ── API ─────────────────────────────────────────────────────────────────────

// Build a BSR_host from a host-side CSR.  O(nnz log nnz).
// h_csr must be a valid host CSR (row_ptr, col_idx, values allocated on host).
BSR_host bsr_from_csr_host(const CSR& h_csr);

// Upload BSR_host → device BSR (allocates device memory).
BSR  bsr_to_device(const BSR_host& h);

// Free all device memory in a device BSR.
void bsr_free_device(BSR& d);

// Compute and return fill-in statistics for a host BSR.
BSR_stats bsr_compute_stats(const BSR_host& h);

}  // namespace spmm
