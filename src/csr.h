#pragma once

#include <string>
#include <vector>

namespace spmm {
    struct CSR {
        int  num_rows = 0;
        int  num_cols = 0;
        int  nnz      = 0;
        int*   row_ptr = nullptr;
        int*   col_idx = nullptr;
        float* values  = nullptr;
    };

    // Host-side allocation / deallocation.
    CSR  csr_alloc_host(int rows, int cols, int nnz);
    void csr_free_host(CSR& m);

    // Build CSR from COO triplets on the host. Sorts and de-duplicates entries.
    CSR csr_from_coo_host(int rows, int cols, const std::vector<int>&   row_indices, const std::vector<int>&   col_indices, const std::vector<float>& values);

    // Generate a uniform-random sparse matrix in CSR (host).
    // `density` is the fraction of nonzero entries.
    CSR generate_uniform_random_csr(int rows, int cols, float density, int seed);

    // Extract rows [row_begin, row_end) of a host CSR as a new host CSR.
    // Column count is unchanged; row_ptr is rebased to start at 0. Used by the
    // multi-GPU path to hand each device an independent row slice of A.
    CSR csr_row_slice_host(const CSR& m, int row_begin, int row_end);

    // Persistence — simple binary format: [int32 rows][int32 cols][int32 nnz]
    //                                      [row_ptr][col_idx][values]
    CSR  csr_read_binary_host(const std::string& path);
    void csr_write_binary_host(const CSR& m, const std::string& path);

    // Move CSR between host and device.
    CSR  csr_to_device(const CSR& host);
    void csr_free_device(CSR& device);

}
