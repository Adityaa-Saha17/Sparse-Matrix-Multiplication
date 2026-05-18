#include "csr.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "utils.h"

namespace spmm {

CSR csr_alloc_host(int rows, int cols, int nnz) {
    CSR m;
    m.num_rows = rows;
    m.num_cols = cols;
    m.nnz      = nnz;
    m.row_ptr  = static_cast<int*>(std::malloc(sizeof(int) * (rows + 1)));
    m.col_idx  = static_cast<int*>(std::malloc(sizeof(int) * nnz));
    m.values   = static_cast<float*>(std::malloc(sizeof(float) * nnz));
    if (!m.row_ptr || !m.col_idx || !m.values) {
        std::fprintf(stderr, "csr_alloc_host: out of memory\n");
        std::exit(EXIT_FAILURE);
    }
    return m;
}

void csr_free_host(CSR& m) {
    std::free(m.row_ptr);
    std::free(m.col_idx);
    std::free(m.values);
    m.row_ptr = nullptr;
    m.col_idx = nullptr;
    m.values  = nullptr;
    m.nnz = m.num_rows = m.num_cols = 0;
}

CSR csr_from_coo_host(int rows, int cols, const std::vector<int>&   row_indices, const std::vector<int>&   col_indices, const std::vector<float>& values) {
    const int n = static_cast<int>(values.size());
    if (n != static_cast<int>(row_indices.size()) ||
        n != static_cast<int>(col_indices.size())) {
        throw std::runtime_error("csr_from_coo_host: triplet size mismatch");
    }

    // Sort triplets by (row, col).
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (row_indices[a] != row_indices[b]) return row_indices[a] < row_indices[b];
        return col_indices[a] < col_indices[b];
    });

    CSR m = csr_alloc_host(rows, cols, n);
    std::memset(m.row_ptr, 0, sizeof(int) * (rows + 1));

    for (int i = 0; i < n; ++i) {
        int p = order[i];
        m.col_idx[i] = col_indices[p];
        m.values[i]  = values[p];
        ++m.row_ptr[row_indices[p] + 1];
    }
    for (int r = 0; r < rows; ++r) {
        m.row_ptr[r + 1] += m.row_ptr[r];
    }
    return m;
}

CSR generate_uniform_random_csr(int rows, int cols, float density, int seed) {
    if (density < 0.f || density > 1.f) {
        throw std::runtime_error("generate_uniform_random_csr: invalid density");
    }

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> uni(0.f, 1.f);
    std::uniform_real_distribution<float> vals(-1.f, 1.f);

    // Reservoir-style: per-row, decide which columns are nonzero by Bernoulli.
    // For large matrices with low density this is O(rows*cols) but acceptable for
    // benchmarking sizes used in Phase 1. Replace with sparser sampling if needed.
    std::vector<int>   rs;
    std::vector<int>   cs;
    std::vector<float> vs;
    rs.reserve(static_cast<size_t>(rows) * cols * density * 1.1);
    cs.reserve(rs.capacity());
    vs.reserve(rs.capacity());

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (uni(rng) < density) {
                rs.push_back(r);
                cs.push_back(c);
                vs.push_back(vals(rng));
            }
        }
    }
    return csr_from_coo_host(rows, cols, rs, cs, vs);
}

CSR csr_read_binary_host(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("csr_read_binary_host: cannot open " + path);

    int32_t header[3];
    f.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!f) throw std::runtime_error("csr_read_binary_host: short read on header");

    CSR m = csr_alloc_host(header[0], header[1], header[2]);
    f.read(reinterpret_cast<char*>(m.row_ptr), sizeof(int) * (m.num_rows + 1));
    f.read(reinterpret_cast<char*>(m.col_idx), sizeof(int) * m.nnz);
    f.read(reinterpret_cast<char*>(m.values),  sizeof(float) * m.nnz);
    if (!f) throw std::runtime_error("csr_read_binary_host: short read on body");
    return m;
}

void csr_write_binary_host(const CSR& m, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("csr_write_binary_host: cannot open " + path);

    int32_t header[3] = {m.num_rows, m.num_cols, m.nnz};
    f.write(reinterpret_cast<const char*>(header), sizeof(header));
    f.write(reinterpret_cast<const char*>(m.row_ptr), sizeof(int) * (m.num_rows + 1));
    f.write(reinterpret_cast<const char*>(m.col_idx), sizeof(int) * m.nnz);
    f.write(reinterpret_cast<const char*>(m.values),  sizeof(float) * m.nnz);
}

CSR csr_to_device(const CSR& host) {
    CSR d;
    d.num_rows = host.num_rows;
    d.num_cols = host.num_cols;
    d.nnz      = host.nnz;
    CUDA_CHECK(cudaMalloc(&d.row_ptr, sizeof(int) * (host.num_rows + 1)));
    CUDA_CHECK(cudaMalloc(&d.col_idx, sizeof(int) * host.nnz));
    CUDA_CHECK(cudaMalloc(&d.values,  sizeof(float) * host.nnz));
    CUDA_CHECK(cudaMemcpy(d.row_ptr, host.row_ptr,
                          sizeof(int) * (host.num_rows + 1),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.col_idx, host.col_idx,
                          sizeof(int) * host.nnz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.values, host.values,
                          sizeof(float) * host.nnz, cudaMemcpyHostToDevice));
    return d;
}

void csr_free_device(CSR& d) {
    if (d.row_ptr) CUDA_CHECK(cudaFree(d.row_ptr));
    if (d.col_idx) CUDA_CHECK(cudaFree(d.col_idx));
    if (d.values)  CUDA_CHECK(cudaFree(d.values));
    d.row_ptr = nullptr;
    d.col_idx = nullptr;
    d.values  = nullptr;
    d.nnz = d.num_rows = d.num_cols = 0;
}

}  // namespace spmm
