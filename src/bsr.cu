// BSR (Block-Sparse Row) implementation.
// See bsr.h for the public API and rationale.

#include "bsr.h"
#include "utils.h"

#include <algorithm>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace spmm {

// ─────────────────────────────────────────────────────────────────────────────
// bsr_from_csr_host
// ─────────────────────────────────────────────────────────────────────────────
//
// Algorithm (3 passes over the CSR):
//
//  Pass 1 — collect unique block-column indices per block-row.
//  Pass 2 — build row_ptr (exclusive-scan over block counts) and col_idx.
//  Pass 3 — fill values: for each CSR nonzero, compute its (block, local-row,
//            local-col) coordinates and write to the FP32 staging buffer.
//
// Finally, convert the FP32 staging buffer to FP16 in-place.
//
// Complexity: O(nnz · log(avg_blocks_per_row)) due to the binary search in
// Pass 3.  For practical inputs this is effectively O(nnz).

BSR_host bsr_from_csr_host(const CSR& h_csr) {
    const int M          = h_csr.num_rows;
    const int K          = h_csr.num_cols;
    const int block_rows = (M + BSR_BLOCK - 1) / BSR_BLOCK;
    const int block_cols = (K + BSR_BLOCK - 1) / BSR_BLOCK;
    const int bs2        = BSR_BLOCK * BSR_BLOCK;

    // ── Pass 1: find non-empty block-columns per block-row ──────────────────
    // nonempty[br] = sorted, unique list of block-column indices that have at
    // least one nonzero in block-row br.
    std::vector<std::vector<int>> nonempty(block_rows);

    for (int r = 0; r < M; ++r) {
        const int br = r / BSR_BLOCK;
        for (int p = h_csr.row_ptr[r]; p < h_csr.row_ptr[r + 1]; ++p) {
            nonempty[br].push_back(h_csr.col_idx[p] / BSR_BLOCK);
        }
    }
    for (auto& v : nonempty) {
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
    }

    // ── Pass 2: build row_ptr and col_idx ───────────────────────────────────
    std::vector<int> row_ptr(block_rows + 1, 0);
    for (int br = 0; br < block_rows; ++br)
        row_ptr[br + 1] = row_ptr[br] + static_cast<int>(nonempty[br].size());

    const int num_blocks = row_ptr[block_rows];

    std::vector<int> col_idx(num_blocks);
    for (int br = 0; br < block_rows; ++br) {
        const int base = row_ptr[br];
        for (int i = 0; i < static_cast<int>(nonempty[br].size()); ++i)
            col_idx[base + i] = nonempty[br][i];
    }

    // ── Pass 3: fill dense block values (FP32 staging) ──────────────────────
    // Zero-initialise the entire value buffer (zero-padding is the default for
    // BSR blocks that are not fully occupied by nonzeros).
    std::vector<float> vals_f32(static_cast<size_t>(num_blocks) * bs2, 0.0f);

    for (int r = 0; r < M; ++r) {
        const int br = r / BSR_BLOCK;
        const int lr = r % BSR_BLOCK;  // local row within block

        for (int p = h_csr.row_ptr[r]; p < h_csr.row_ptr[r + 1]; ++p) {
            const int c  = h_csr.col_idx[p];
            const int bc = c / BSR_BLOCK;
            const int lc = c % BSR_BLOCK;  // local col within block

            // Binary search: find bc in nonempty[br] → local index → global block index.
            const auto& row_bcs = nonempty[br];
            const int local_blk = static_cast<int>(
                std::lower_bound(row_bcs.begin(), row_bcs.end(), bc) - row_bcs.begin());
            const int blk = row_ptr[br] + local_blk;

            // Row-major layout within the block: element (lr, lc) at offset lr*BSR_BLOCK + lc.
            vals_f32[static_cast<size_t>(blk) * bs2 + lr * BSR_BLOCK + lc] = h_csr.values[p];
        }
    }

    // ── Convert FP32 → FP16 ─────────────────────────────────────────────────
    // __float2half is available as a host function in cuda_fp16.h (CUDA ≥ 7.5).
    std::vector<half> vals_f16(static_cast<size_t>(num_blocks) * bs2);
    for (size_t i = 0; i < vals_f16.size(); ++i)
        vals_f16[i] = __float2half(vals_f32[i]);

    BSR_host h;
    h.num_rows   = M;
    h.num_cols   = K;
    h.block_rows = block_rows;
    h.block_cols = block_cols;
    h.num_blocks = num_blocks;
    h.nnz        = h_csr.nnz;
    h.row_ptr    = std::move(row_ptr);
    h.col_idx    = std::move(col_idx);
    h.values     = std::move(vals_f16);
    return h;
}

// ─────────────────────────────────────────────────────────────────────────────
// bsr_to_device / bsr_free_device
// ─────────────────────────────────────────────────────────────────────────────

BSR bsr_to_device(const BSR_host& h) {
    BSR d;
    d.num_rows   = h.num_rows;
    d.num_cols   = h.num_cols;
    d.block_rows = h.block_rows;
    d.block_cols = h.block_cols;
    d.num_blocks = h.num_blocks;
    d.nnz        = h.nnz;

    const size_t rp_bytes  = (h.block_rows + 1) * sizeof(int);
    const size_t ci_bytes  = h.num_blocks * sizeof(int);
    const size_t val_bytes = static_cast<size_t>(h.num_blocks) * BSR_BLOCK * BSR_BLOCK * sizeof(half);

    CUDA_CHECK(cudaMalloc(&d.row_ptr, rp_bytes));
    CUDA_CHECK(cudaMalloc(&d.col_idx, ci_bytes));
    CUDA_CHECK(cudaMalloc(&d.values,  val_bytes));

    CUDA_CHECK(cudaMemcpy(d.row_ptr, h.row_ptr.data(), rp_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.col_idx, h.col_idx.data(), ci_bytes,  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.values,  h.values.data(),  val_bytes, cudaMemcpyHostToDevice));

    return d;
}

void bsr_free_device(BSR& d) {
    if (d.row_ptr) { CUDA_CHECK(cudaFree(d.row_ptr)); d.row_ptr = nullptr; }
    if (d.col_idx) { CUDA_CHECK(cudaFree(d.col_idx)); d.col_idx = nullptr; }
    if (d.values)  { CUDA_CHECK(cudaFree(d.values));  d.values  = nullptr; }
}

// ─────────────────────────────────────────────────────────────────────────────
// bsr_compute_stats
// ─────────────────────────────────────────────────────────────────────────────

BSR_stats bsr_compute_stats(const BSR_host& h) {
    BSR_stats s;
    s.num_blocks      = h.num_blocks;
    s.stored_elements = static_cast<long>(h.num_blocks) * BSR_BLOCK * BSR_BLOCK;
    s.original_nnz    = h.nnz;
    s.fill_in_ratio   = (h.nnz > 0)
                        ? static_cast<double>(s.stored_elements) / h.nnz
                        : 0.0;
    const long total  = static_cast<long>(h.block_rows) * h.block_cols;
    s.block_density   = (total > 0)
                        ? static_cast<double>(h.num_blocks) / total
                        : 0.0;
    return s;
}

}  // namespace spmm
