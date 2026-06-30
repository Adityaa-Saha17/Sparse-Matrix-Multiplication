// Memory-aware execution: multi-GPU row splitting and unified memory.
// See spmm_multigpu.h for the rationale.

#include "spmm_multigpu.h"

#include <algorithm>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "utils.h"
#include "kernels/spmm_tiled_v3.h"

namespace spmm {

namespace {

float median_ms(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.f : v[v.size() / 2];
}

// nnz-balanced row boundaries: bounds[g]..bounds[g+1] is device g's row range,
// chosen so each device owns roughly total_nnz / num_gpus nonzeros.
std::vector<int> nnz_balanced_bounds(const CSR& h_A, int num_gpus) {
    std::vector<int> bounds(num_gpus + 1, 0);
    bounds[num_gpus] = h_A.num_rows;
    const long total = h_A.nnz;
    int r = 0;
    for (int g = 1; g < num_gpus; ++g) {
        const long target = (total * g) / num_gpus;
        while (r < h_A.num_rows && h_A.row_ptr[r] < target) ++r;
        bounds[g] = r;
    }
    // Guarantee monotonic, non-overlapping ranges even on degenerate inputs.
    for (int g = 1; g <= num_gpus; ++g)
        bounds[g] = std::max(bounds[g], bounds[g - 1]);
    return bounds;
}

}  // namespace

int gpu_count() {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) return 0;
    return n;
}

MultiGpuResult spmm_multigpu(const CSR& h_A, const float* h_B, float* h_C,
                             int N, int num_gpus, int warmup, int iters) {
    const int visible = gpu_count();
    if (num_gpus > visible) num_gpus = visible;
    if (num_gpus < 1)       num_gpus = 1;

    const int K = h_A.num_cols;
    const std::vector<int> bounds = nnz_balanced_bounds(h_A, num_gpus);

    MultiGpuResult result;
    result.num_gpus = num_gpus;
    result.rows.assign(num_gpus, 0);
    result.nnz.assign(num_gpus, 0);
    std::vector<float> per_gpu_ms(num_gpus, 0.f);

    auto worker = [&](int g) {
        const int row_begin = bounds[g];
        const int row_end   = bounds[g + 1];
        const int rows_g    = row_end - row_begin;
        result.rows[g] = rows_g;

        CUDA_CHECK(cudaSetDevice(g));

        // Build and upload this device's row slice of A plus a full copy of B.
        CSR slice = csr_row_slice_host(h_A, row_begin, row_end);
        result.nnz[g] = slice.nnz;
        CSR d_A = csr_to_device(slice);

        float* d_B = nullptr;
        float* d_C = nullptr;
        const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
        const size_t c_bytes = sizeof(float) * static_cast<size_t>(rows_g) * N;
        CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
        if (c_bytes > 0) CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
        CUDA_CHECK(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice));

        if (rows_g > 0) {
            for (int i = 0; i < warmup; ++i) spmm_tiled_v3(d_A, d_B, d_C, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            std::vector<float> times(iters);
            GpuTimer t;
            for (int i = 0; i < iters; ++i) {
                t.start();
                spmm_tiled_v3(d_A, d_B, d_C, N);
                t.stop();
                times[i] = t.elapsed_ms();
            }
            per_gpu_ms[g] = median_ms(times);

            CUDA_CHECK(cudaMemcpy(h_C + static_cast<size_t>(row_begin) * N, d_C,
                                  c_bytes, cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaFree(d_B));
        if (d_C) CUDA_CHECK(cudaFree(d_C));
        csr_free_device(d_A);
        csr_free_host(slice);
    };

    // One host thread per device so the kernels overlap across GPUs.
    std::vector<std::thread> pool;
    pool.reserve(num_gpus);
    for (int g = 0; g < num_gpus; ++g) pool.emplace_back(worker, g);
    for (auto& th : pool) th.join();

    // The slowest device sets the achievable wall time.
    result.time_ms = *std::max_element(per_gpu_ms.begin(), per_gpu_ms.end());
    return result;
}

float spmm_unified(const CSR& h_A, const float* h_B, float* h_C,
                   int N, int warmup, int iters) {
    const int M = h_A.num_rows;
    const int K = h_A.num_cols;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));

    // Managed (unified) allocations for A, B and C.
    CSR u_A;
    u_A.num_rows = M;
    u_A.num_cols = K;
    u_A.nnz      = h_A.nnz;
    CUDA_CHECK(cudaMallocManaged(&u_A.row_ptr, sizeof(int) * (M + 1)));
    CUDA_CHECK(cudaMallocManaged(&u_A.col_idx, sizeof(int) * h_A.nnz));
    CUDA_CHECK(cudaMallocManaged(&u_A.values,  sizeof(float) * h_A.nnz));

    float* u_B = nullptr;
    float* u_C = nullptr;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;
    CUDA_CHECK(cudaMallocManaged(&u_B, b_bytes));
    CUDA_CHECK(cudaMallocManaged(&u_C, c_bytes));

    std::copy(h_A.row_ptr, h_A.row_ptr + (M + 1), u_A.row_ptr);
    std::copy(h_A.col_idx, h_A.col_idx + h_A.nnz, u_A.col_idx);
    std::copy(h_A.values,  h_A.values  + h_A.nnz, u_A.values);
    std::copy(h_B, h_B + (static_cast<size_t>(K) * N), u_B);

    // Prefetch the read-only working set to the device. Not all platforms
    // support prefetch; ignore the status rather than aborting if unsupported.
    cudaMemPrefetchAsync(u_A.row_ptr, sizeof(int) * (M + 1), dev);
    cudaMemPrefetchAsync(u_A.col_idx, sizeof(int) * h_A.nnz, dev);
    cudaMemPrefetchAsync(u_A.values,  sizeof(float) * h_A.nnz, dev);
    cudaMemPrefetchAsync(u_B, b_bytes, dev);
    cudaMemPrefetchAsync(u_C, c_bytes, dev);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < warmup; ++i) spmm_tiled_v3(u_A, u_B, u_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> times(iters);
    GpuTimer t;
    for (int i = 0; i < iters; ++i) {
        t.start();
        spmm_tiled_v3(u_A, u_B, u_C, N);
        t.stop();
        times[i] = t.elapsed_ms();
    }
    const float ms = median_ms(times);

    CUDA_CHECK(cudaDeviceSynchronize());
    std::copy(u_C, u_C + (static_cast<size_t>(M) * N), h_C);

    CUDA_CHECK(cudaFree(u_A.row_ptr));
    CUDA_CHECK(cudaFree(u_A.col_idx));
    CUDA_CHECK(cudaFree(u_A.values));
    CUDA_CHECK(cudaFree(u_B));
    CUDA_CHECK(cudaFree(u_C));
    return ms;
}

}  // namespace spmm
