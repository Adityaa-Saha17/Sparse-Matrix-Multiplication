// Hybrid execution: automatic CUDA-core vs. Tensor-Core dispatch.
// See spmm_hybrid.h for the rationale.

#include "spmm_hybrid.h"

#include <algorithm>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "bsr.h"
#include "utils.h"
#include "kernels/spmm_tiled_v3.h"
#include "kernels/spmm_wmma.h"

namespace spmm {

namespace {

float median_ms(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.f : v[v.size() / 2];
}

}  // namespace

const char* backend_name(Backend b) {
    return (b == Backend::TENSOR_WMMA) ? "wmma" : "csr";
}

HybridPlan hybrid_choose(const CSR& h_A, int N, float fill_in_threshold) {
    HybridPlan plan;
    plan.threshold = fill_in_threshold;
    plan.n_aligned = (N % BSR_BLOCK == 0) && (N > 0);

    // Build the BSR structure once on the host and read its fill-in.
    BSR_host  h_bsr = bsr_from_csr_host(h_A);
    BSR_stats st    = bsr_compute_stats(h_bsr);
    plan.fill_in_ratio = st.fill_in_ratio;
    plan.block_density = st.block_density;

    const bool dense_blocks = st.fill_in_ratio <= fill_in_threshold;

    if (plan.n_aligned && dense_blocks) {
        plan.backend = Backend::TENSOR_WMMA;
        plan.reason  = "low BSR fill-in and N aligned to 16 -> Tensor Cores";
    } else if (!plan.n_aligned) {
        plan.backend = Backend::CSR_TILED;
        plan.reason  = "N not a multiple of 16 -> CUDA-core CSR";
    } else {
        plan.backend = Backend::CSR_TILED;
        plan.reason  = "high BSR fill-in -> CUDA-core CSR";
    }
    return plan;
}

float spmm_hybrid_run(const HybridPlan& plan,
                      const CSR& h_A, const float* h_B, float* h_C,
                      int N, int warmup, int iters) {
    const int M = h_A.num_rows;
    const int K = h_A.num_cols;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;

    // Dense B and output C are common to both back ends.
    float* d_B = nullptr;
    float* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, b_bytes, cudaMemcpyHostToDevice));

    float ms = 0.f;

    if (plan.backend == Backend::TENSOR_WMMA) {
        // BSR on the device + a one-off FP16 copy of B (conversion excluded
        // from the timed loop, mirroring the benchmark harness).
        BSR_host h_bsr = bsr_from_csr_host(h_A);
        BSR      d_bsr = bsr_to_device(h_bsr);

        half* d_B_fp16 = nullptr;
        CUDA_CHECK(cudaMalloc(&d_B_fp16,
                              static_cast<size_t>(K) * N * sizeof(half)));
        float_to_half(d_B, d_B_fp16, K * N);
        CUDA_CHECK(cudaDeviceSynchronize());

        for (int i = 0; i < warmup; ++i) spmm_wmma(d_bsr, d_B_fp16, d_C, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> times(iters);
        GpuTimer t;
        for (int i = 0; i < iters; ++i) {
            t.start();
            spmm_wmma(d_bsr, d_B_fp16, d_C, N);
            t.stop();
            times[i] = t.elapsed_ms();
        }
        ms = median_ms(times);

        CUDA_CHECK(cudaFree(d_B_fp16));
        bsr_free_device(d_bsr);
    } else {
        CSR d_A = csr_to_device(h_A);

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
        ms = median_ms(times);

        csr_free_device(d_A);
    }

    CUDA_CHECK(cudaMemcpy(h_C, d_C, c_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    return ms;
}

}  // namespace spmm
