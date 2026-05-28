// Benchmark harness — Phases 1, 2, and 3.
//   - Generates (or loads) a sparse matrix A and a dense matrix B.
//   - Runs the project's SpMM kernels and cuSPARSE's cusparseSpMM as reference.
//   - Reports correctness (max |abs| / |rel| error) and median time + GFLOPS.
//
// Phase 3 additions (--kernel wmma):
//   - Builds BSR from CSR on the host.
//   - Converts B to FP16 once (excluded from timed loop).
//   - Benchmarks spmm_wmma (Tensor Core kernel) and reports BSR fill-in stats.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cusparse.h>

#include "../csr.h"
#include "../bsr.h"
#include "../kernels/spmm_baseline.h"
#include "../kernels/spmm_memopt.h"
#include "../kernels/spmm_memopt_v2.h"
#include "../kernels/spmm_tiled.h"
#include "../kernels/spmm_tiled_v2.h"
#include "../kernels/spmm_tiled_v3.h"
#include "../kernels/spmm_tiled_v4.h"
#include "../kernels/spmm_wmma.h"
#include "../utils.h"

namespace {

struct Args {
    int   m       = 1024;
    int   k       = 1024;
    int   n       = 256;
    float density = 0.01f;
    int   seed    = 42;
    int   warmup  = 3;
    int   iters   = 10;
    std::string mtx_path;  // optional binary CSR file; overrides synthetic
    std::string kernel = "both";  // "baseline" | "memopt" | "memopt_v2" | "tiled" | "tiled_v2" | "tiled_v3" | "tiled_v4" | "wmma" | "both" | "all"
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --m INT          rows of A and C        (default 1024)\n"
        "  --k INT          cols of A / rows of B  (default 1024)\n"
        "  --n INT          cols of B and C        (default 256)\n"
        "  --density FLOAT  fraction of nonzeros   (default 0.01)\n"
        "  --seed INT       RNG seed               (default 42)\n"
        "  --warmup INT     warmup iterations      (default 3)\n"
        "  --iters INT      timed iterations       (default 10)\n"
        "  --bin PATH       load CSR from binary file (overrides --m/--k/--density)\n"
        "  --kernel STR     'baseline' | 'memopt' | 'memopt_v2' | 'tiled' | 'tiled_v2'\n"
        "                   | 'tiled_v3' | 'tiled_v4' | 'wmma' | 'both' | 'all'\n"
        "                   'both' = baseline + memopt (Phase 1 default)\n"
        "                   'all'  = baseline + memopt + memopt_v2 + tiled + tiled_v2\n"
        "                            + tiled_v3 + tiled_v4 + wmma\n"
        "                            (full ablation: all eight kernels side by side)\n"
        "                   note: tiled_v2 requires N %% 128 == 0\n"
        "                   note: wmma requires sm_75+ and N %% 16 == 0\n"
        "                   default: 'both'\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string opt = argv[i];
        auto needs = [&](const char* name) {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", name);
                print_usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            return std::string(argv[++i]);
        };
        if      (opt == "--m")       a.m       = std::stoi(needs("--m"));
        else if (opt == "--k")       a.k       = std::stoi(needs("--k"));
        else if (opt == "--n")       a.n       = std::stoi(needs("--n"));
        else if (opt == "--density") a.density = std::stof(needs("--density"));
        else if (opt == "--seed")    a.seed    = std::stoi(needs("--seed"));
        else if (opt == "--warmup")  a.warmup  = std::stoi(needs("--warmup"));
        else if (opt == "--iters")   a.iters   = std::stoi(needs("--iters"));
        else if (opt == "--bin")     a.mtx_path = needs("--bin");
        else if (opt == "--kernel")  a.kernel = needs("--kernel");
        else if (opt == "-h" || opt == "--help") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown option: %s\n", opt.c_str());
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return a;
}

std::vector<float> generate_dense_row_major(int rows, int cols, int seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> v(static_cast<size_t>(rows) * cols);
    for (auto& x : v) x = uni(rng);
    return v;
}

// cuSPARSE reference: row-major B and C, FP32, alpha=1, beta=0.
void run_cusparse_spmm(int M, int N, int K, const spmm::CSR& d_A, const float* d_B, float* d_C) {
    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));

    cusparseSpMatDescr_t mat_a;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_a, M, K, d_A.nnz,
        d_A.row_ptr, d_A.col_idx, d_A.values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

    cusparseDnMatDescr_t mat_b;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_b, K, N, /*ld=*/N,
        const_cast<float*>(d_B), CUDA_R_32F, CUSPARSE_ORDER_ROW));

    cusparseDnMatDescr_t mat_c;
    CUSPARSE_CHECK(cusparseCreateDnMat(
        &mat_c, M, N, /*ld=*/N,
        d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    const float alpha = 1.f, beta = 0.f;
    size_t buffer_size = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_a, mat_b, &beta, mat_c,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &buffer_size));

    void* d_buffer = nullptr;
    if (buffer_size > 0) {
        CUDA_CHECK(cudaMalloc(&d_buffer, buffer_size));
    }

    CUSPARSE_CHECK(cusparseSpMM(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_a, mat_b, &beta, mat_c,
        CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, d_buffer));
    CUDA_CHECK(cudaDeviceSynchronize());

    if (d_buffer) CUDA_CHECK(cudaFree(d_buffer));
    CUSPARSE_CHECK(cusparseDestroyDnMat(mat_c));
    CUSPARSE_CHECK(cusparseDestroyDnMat(mat_b));
    CUSPARSE_CHECK(cusparseDestroySpMat(mat_a));
    CUSPARSE_CHECK(cusparseDestroy(handle));
}

struct ErrorStats {
    float max_abs = 0.f;
    float max_rel = 0.f;  // filtered: only cells whose |ref| is >=0.1% of peak |ref|
    float rel_l2  = 0.f;  // ||ours - ref||_2 / ||ref||_2
};

ErrorStats compare(const std::vector<float>& ours, const std::vector<float>& ref) {
    ErrorStats s;
    float ref_max = 0.f;
    for (float v : ref) ref_max = std::max(ref_max, std::fabs(v));
    const float rel_floor = 1e-3f * ref_max;

    double err_sq = 0.0, ref_sq = 0.0;
    for (size_t i = 0; i < ours.size(); ++i) {
        float d = std::fabs(ours[i] - ref[i]);
        s.max_abs = std::max(s.max_abs, d);
        float a = std::fabs(ref[i]);
        if (a > rel_floor) s.max_rel = std::max(s.max_rel, d / a);
        err_sq += static_cast<double>(d) * d;
        ref_sq += static_cast<double>(a) * a;
    }
    s.rel_l2 = (ref_sq > 0.0) ? static_cast<float>(std::sqrt(err_sq / ref_sq)) : 0.f;
    return s;
}

float median(std::vector<float> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct KernelResult {
    std::string name;
    float time_ms;
    float gflops;
    float max_abs_err;
    float max_rel_err;
    float rel_l2_err;
};

KernelResult run_and_benchmark(const std::string& name,
                                void (*kernel_fn)(const spmm::CSR&, const float*, float*, int),
                                int M, int N, int warmup, int iters,
                                const spmm::CSR& d_A, const float* d_B, float* d_C,
                                const std::vector<float>& h_C_ref) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel_fn(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times(iters);
    GpuTimer t;
    for (int i = 0; i < iters; ++i) {
        t.start();
        kernel_fn(d_A, d_B, d_C, N);
        t.stop();
        times[i] = t.elapsed_ms();
    }
    float ms = median(times);

    // Copy result and compare
    std::vector<float> h_C(M * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));
    ErrorStats err = compare(h_C, h_C_ref);

    double flops = 2.0 * static_cast<double>(d_A.nnz) * N;
    double gflops = flops / (static_cast<double>(ms) * 1.0e6);

    return {name, ms, static_cast<float>(gflops), err.max_abs, err.max_rel, err.rel_l2};
}

// Phase 3: benchmark the WMMA kernel which takes BSR + FP16 B instead of CSR + FP32 B.
// BSR and d_B_fp16 must already be prepared (construction time is excluded from timing).
// FLOPS are computed against the original nnz (not the padded block count) so that
// speedup numbers are apples-to-apples with the CSR kernels.
KernelResult run_and_benchmark_bsr(const std::string& name,
                                    void (*kernel_fn)(const spmm::BSR&, const half*, float*, int),
                                    int M, int N, int original_nnz,
                                    int warmup, int iters,
                                    const spmm::BSR& d_bsr, const half* d_B_fp16,
                                    float* d_C, const std::vector<float>& h_C_ref) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        kernel_fn(d_bsr, d_B_fp16, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs
    std::vector<float> times(iters);
    GpuTimer t;
    for (int i = 0; i < iters; ++i) {
        t.start();
        kernel_fn(d_bsr, d_B_fp16, d_C, N);
        t.stop();
        times[i] = t.elapsed_ms();
    }
    float ms = median(times);

    // Copy result and compare
    std::vector<float> h_C(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, static_cast<size_t>(M) * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    ErrorStats err = compare(h_C, h_C_ref);

    // FLOPS: 2 * original_nnz * N (same denominator as CSR kernels for fair comparison).
    double flops  = 2.0 * static_cast<double>(original_nnz) * N;
    double gflops = flops / (static_cast<double>(ms) * 1.0e6);

    return {name, ms, static_cast<float>(gflops), err.max_abs, err.max_rel, err.rel_l2};
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // Build A.
    spmm::CSR h_A;
    if (!args.mtx_path.empty()) {
        h_A = spmm::csr_read_binary_host(args.mtx_path);
        args.m = h_A.num_rows;
        args.k = h_A.num_cols;
    } else {
        h_A = spmm::generate_uniform_random_csr(args.m, args.k, args.density, args.seed);
    }
    const float actual_density =
        static_cast<float>(h_A.nnz) /
        (static_cast<float>(h_A.num_rows) * h_A.num_cols);

    // Build dense B.
    std::vector<float> h_B = generate_dense_row_major(args.k, args.n, args.seed + 1);

    // Move to device.
    spmm::CSR d_A = spmm::csr_to_device(h_A);
    float *d_B = nullptr, *d_C_ours = nullptr, *d_C_ref = nullptr;
    const size_t b_bytes = sizeof(float) * args.k * args.n;
    const size_t c_bytes = sizeof(float) * args.m * args.n;
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_ours, c_bytes));
    CUDA_CHECK(cudaMalloc(&d_C_ref,  c_bytes));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));

    // Reference
    run_cusparse_spmm(args.m, args.n, args.k, d_A, d_B, d_C_ref);

    // Copy reference result to host for comparison
    std::vector<float> h_C_ref(args.m * args.n);
    CUDA_CHECK(cudaMemcpy(h_C_ref.data(), d_C_ref, c_bytes, cudaMemcpyDeviceToHost));

    // Benchmark kernels
    std::vector<KernelResult> results;
    if (args.kernel == "baseline" || args.kernel == "both" || args.kernel == "all") {
        results.push_back(run_and_benchmark("baseline", spmm::spmm_baseline,
                                            args.m, args.n, args.warmup, args.iters,
                                            d_A, d_B, d_C_ours, h_C_ref));
    }
    if (args.kernel == "memopt" || args.kernel == "both" || args.kernel == "all") {
        results.push_back(run_and_benchmark("memopt", spmm::spmm_memopt,
                                            args.m, args.n, args.warmup, args.iters,
                                            d_A, d_B, d_C_ours, h_C_ref));
    }
    if (args.kernel == "memopt_v2" || args.kernel == "all") {
        results.push_back(run_and_benchmark("memopt_v2", spmm::spmm_memopt_v2,
                                            args.m, args.n, args.warmup, args.iters,
                                            d_A, d_B, d_C_ours, h_C_ref));
    }
    if (args.kernel == "tiled" || args.kernel == "all") {
        results.push_back(run_and_benchmark("tiled", spmm::spmm_tiled,
                                            args.m, args.n, args.warmup, args.iters,
                                            d_A, d_B, d_C_ours, h_C_ref));
    }
    if (args.kernel == "tiled_v2" || args.kernel == "all") {
        // tiled_v2 requires N % 128 == 0; skip silently under 'all' if not.
        if (args.n % 128 == 0) {
            results.push_back(run_and_benchmark("tiled_v2", spmm::spmm_tiled_v2,
                                                args.m, args.n, args.warmup, args.iters,
                                                d_A, d_B, d_C_ours, h_C_ref));
        } else if (args.kernel == "tiled_v2") {
            std::fprintf(stderr,
                "tiled_v2 requires --n divisible by 128 (got n=%d). Skipping.\n",
                args.n);
        }
    }
    if (args.kernel == "tiled_v3" || args.kernel == "all") {
        results.push_back(run_and_benchmark("tiled_v3", spmm::spmm_tiled_v3,
                                            args.m, args.n, args.warmup, args.iters,
                                            d_A, d_B, d_C_ours, h_C_ref));
    }
    if (args.kernel == "tiled_v4" || args.kernel == "all") {
        results.push_back(run_and_benchmark("tiled_v4", spmm::spmm_tiled_v4,
                                            args.m, args.n, args.warmup, args.iters,
                                            d_A, d_B, d_C_ours, h_C_ref));
    }

    // ── Phase 3: Tensor Core / WMMA kernel (BSR format) ────────────────────
    if (args.kernel == "wmma" || args.kernel == "all") {
        if (args.n % 16 != 0) {
            std::fprintf(stderr,
                "wmma requires --n divisible by 16 (got n=%d). Skipping.\n",
                args.n);
        } else {
            // Build BSR on host from the same h_A used above.
            spmm::BSR_host h_bsr = spmm::bsr_from_csr_host(h_A);
            spmm::BSR_stats bsr_stats = spmm::bsr_compute_stats(h_bsr);

            // Print fill-in statistics so the user can see the cost of block padding.
            std::printf(
                "bsr_stats blocks=%d stored_elems=%ld original_nnz=%d "
                "fill_in_ratio=%.2f block_density=%.4f\n",
                bsr_stats.num_blocks, bsr_stats.stored_elements,
                bsr_stats.original_nnz, bsr_stats.fill_in_ratio,
                bsr_stats.block_density);

            // Upload BSR to device.
            spmm::BSR d_bsr = spmm::bsr_to_device(h_bsr);

            // Convert d_B (FP32) to FP16 once — excluded from the timed loop.
            half* d_B_fp16 = nullptr;
            const size_t b_fp16_bytes = static_cast<size_t>(args.k) * args.n * sizeof(half);
            CUDA_CHECK(cudaMalloc(&d_B_fp16, b_fp16_bytes));
            spmm::float_to_half(d_B, d_B_fp16, args.k * args.n);
            CUDA_CHECK(cudaDeviceSynchronize());

            results.push_back(run_and_benchmark_bsr(
                "wmma", spmm::spmm_wmma,
                args.m, args.n, h_A.nnz,
                args.warmup, args.iters,
                d_bsr, d_B_fp16, d_C_ours, h_C_ref));

            CUDA_CHECK(cudaFree(d_B_fp16));
            spmm::bsr_free_device(d_bsr);
        }
    }

    // Report results
    for (const auto& r : results) {
        std::printf(
            "kernel=%s m=%d k=%d n=%d nnz=%d density=%.6f "
            "time_ms=%.4f gflops=%.2f max_abs_err=%.3e max_rel_err=%.3e rel_l2_err=%.3e\n",
            r.name.c_str(), args.m, args.k, args.n, h_A.nnz, actual_density,
            r.time_ms, r.gflops, r.max_abs_err, r.max_rel_err, r.rel_l2_err);
    }

    // Cleanup.
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C_ours));
    CUDA_CHECK(cudaFree(d_C_ref));
    spmm::csr_free_device(d_A);
    spmm::csr_free_host(h_A);
    return 0;
}
