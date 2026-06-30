// Integrated SpMM driver.
//
// One entry point that ties the project together: it ingests a sparse matrix
// (synthetic or from disk), inspects its block structure, automatically selects
// the CUDA-core or Tensor-Core back end, optionally spreads the work across
// multiple GPUs or runs it through unified memory, and verifies the result
// against cuSPARSE. This is the "use the whole system" front end, as opposed to
// the benchmark harness which exercises one kernel at a time.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "../csr.h"
#include "../spmm_hybrid.h"
#include "../spmm_multigpu.h"
#include "../utils.h"

namespace {

struct Args {
    int   m       = 4096;
    int   k       = 4096;
    int   n       = 256;
    float density = 0.01f;
    int   seed    = 42;
    int   warmup  = 3;
    int   iters   = 10;
    float threshold = 2.0f;   // BSR fill-in cut-off for Tensor-Core selection
    int   gpus    = 1;        // devices to use with --multi-gpu
    bool  multi_gpu = false;
    bool  unified   = false;
    bool  verify    = true;
    std::string mtx_path;     // optional binary CSR file
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
        "Usage: %s [options]\n"
        "  --m INT          rows of A and C            (default 4096)\n"
        "  --k INT          cols of A / rows of B      (default 4096)\n"
        "  --n INT          cols of B and C            (default 256)\n"
        "  --density FLOAT  fraction of nonzeros       (default 0.01)\n"
        "  --seed INT       RNG seed                   (default 42)\n"
        "  --warmup INT     warmup iterations          (default 3)\n"
        "  --iters INT      timed iterations           (default 10)\n"
        "  --threshold FLT  BSR fill-in cut-off for Tensor Cores (default 2.0)\n"
        "  --bin PATH       load CSR from a binary file (overrides --m/--k/--density)\n"
        "  --multi-gpu      split A by rows across GPUs (CUDA-core path)\n"
        "  --gpus INT       device count for --multi-gpu (default: all visible)\n"
        "  --unified        single-GPU unified-memory path (CUDA-core)\n"
        "  --no-verify      skip the cuSPARSE correctness check\n",
        prog);
}

Args parse_args(int argc, char** argv) {
    Args a;
    bool gpus_set = false;
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
        if      (opt == "--m")         a.m         = std::stoi(needs("--m"));
        else if (opt == "--k")         a.k         = std::stoi(needs("--k"));
        else if (opt == "--n")         a.n         = std::stoi(needs("--n"));
        else if (opt == "--density")   a.density   = std::stof(needs("--density"));
        else if (opt == "--seed")      a.seed      = std::stoi(needs("--seed"));
        else if (opt == "--warmup")    a.warmup    = std::stoi(needs("--warmup"));
        else if (opt == "--iters")     a.iters     = std::stoi(needs("--iters"));
        else if (opt == "--threshold") a.threshold = std::stof(needs("--threshold"));
        else if (opt == "--gpus")    { a.gpus = std::stoi(needs("--gpus")); gpus_set = true; }
        else if (opt == "--bin")       a.mtx_path  = needs("--bin");
        else if (opt == "--multi-gpu") a.multi_gpu = true;
        else if (opt == "--unified")   a.unified   = true;
        else if (opt == "--no-verify") a.verify    = false;
        else if (opt == "-h" || opt == "--help") { print_usage(argv[0]); std::exit(0); }
        else {
            std::fprintf(stderr, "unknown option: %s\n", opt.c_str());
            print_usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    if (a.multi_gpu && !gpus_set) a.gpus = spmm::gpu_count();
    return a;
}

std::vector<float> generate_dense_row_major(int rows, int cols, int seed) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> v(static_cast<size_t>(rows) * cols);
    for (auto& x : v) x = uni(rng);
    return v;
}

// cuSPARSE reference: row-major B and C, FP32, alpha=1, beta=0. Result on host.
std::vector<float> cusparse_reference(int M, int N, int K, const spmm::CSR& h_A,
                                      const std::vector<float>& h_B) {
    spmm::CSR d_A = spmm::csr_to_device(h_A);
    float *d_B = nullptr, *d_C = nullptr;
    const size_t b_bytes = sizeof(float) * static_cast<size_t>(K) * N;
    const size_t c_bytes = sizeof(float) * static_cast<size_t>(M) * N;
    CUDA_CHECK(cudaMalloc(&d_B, b_bytes));
    CUDA_CHECK(cudaMalloc(&d_C, c_bytes));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), b_bytes, cudaMemcpyHostToDevice));

    cusparseHandle_t handle;
    CUSPARSE_CHECK(cusparseCreate(&handle));
    cusparseSpMatDescr_t mat_a;
    CUSPARSE_CHECK(cusparseCreateCsr(
        &mat_a, M, K, d_A.nnz, d_A.row_ptr, d_A.col_idx, d_A.values,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
    cusparseDnMatDescr_t mat_b, mat_c;
    CUSPARSE_CHECK(cusparseCreateDnMat(&mat_b, K, N, N, d_B, CUDA_R_32F, CUSPARSE_ORDER_ROW));
    CUSPARSE_CHECK(cusparseCreateDnMat(&mat_c, M, N, N, d_C, CUDA_R_32F, CUSPARSE_ORDER_ROW));

    const float alpha = 1.f, beta = 0.f;
    size_t buf = 0;
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, &buf));
    void* d_buf = nullptr;
    if (buf > 0) CUDA_CHECK(cudaMalloc(&d_buf, buf));
    CUSPARSE_CHECK(cusparseSpMM(
        handle, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F,
        CUSPARSE_SPMM_ALG_DEFAULT, d_buf));
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> h_C(static_cast<size_t>(M) * N);
    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, c_bytes, cudaMemcpyDeviceToHost));

    if (d_buf) CUDA_CHECK(cudaFree(d_buf));
    CUSPARSE_CHECK(cusparseDestroyDnMat(mat_c));
    CUSPARSE_CHECK(cusparseDestroyDnMat(mat_b));
    CUSPARSE_CHECK(cusparseDestroySpMat(mat_a));
    CUSPARSE_CHECK(cusparseDestroy(handle));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    spmm::csr_free_device(d_A);
    return h_C;
}

float rel_l2(const std::vector<float>& ours, const std::vector<float>& ref) {
    double err = 0.0, den = 0.0;
    for (size_t i = 0; i < ours.size(); ++i) {
        double d = static_cast<double>(ours[i]) - ref[i];
        err += d * d;
        den += static_cast<double>(ref[i]) * ref[i];
    }
    return den > 0.0 ? static_cast<float>(std::sqrt(err / den)) : 0.f;
}

}  // namespace

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    // Ingest A.
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

    std::vector<float> h_B = generate_dense_row_major(args.k, args.n, args.seed + 1);
    std::vector<float> h_C(static_cast<size_t>(args.m) * args.n, 0.f);

    // Inspect structure and choose a back end.
    spmm::HybridPlan plan = spmm::hybrid_choose(h_A, args.n, args.threshold);

    std::string mode;
    float ms = 0.f;

    if (args.multi_gpu) {
        spmm::MultiGpuResult r =
            spmm::spmm_multigpu(h_A, h_B.data(), h_C.data(), args.n,
                                args.gpus, args.warmup, args.iters);
        ms   = r.time_ms;
        mode = "multigpu[csr]";
        std::printf("split gpus=%d", r.num_gpus);
        for (int g = 0; g < r.num_gpus; ++g)
            std::printf(" gpu%d:rows=%d,nnz=%d", g, r.rows[g], r.nnz[g]);
        std::printf("\n");
    } else if (args.unified) {
        ms   = spmm::spmm_unified(h_A, h_B.data(), h_C.data(), args.n,
                                  args.warmup, args.iters);
        mode = "unified[csr]";
    } else {
        ms   = spmm::spmm_hybrid_run(plan, h_A, h_B.data(), h_C.data(), args.n,
                                     args.warmup, args.iters);
        mode = std::string("hybrid[") + spmm::backend_name(plan.backend) + "]";
    }

    const double flops  = 2.0 * static_cast<double>(h_A.nnz) * args.n;
    const double gflops = flops / (static_cast<double>(ms) * 1.0e6);

    std::printf(
        "plan backend=%s fill_in=%.2f block_density=%.4f threshold=%.2f "
        "n_aligned=%d reason=\"%s\"\n",
        spmm::backend_name(plan.backend), plan.fill_in_ratio,
        plan.block_density, plan.threshold, plan.n_aligned ? 1 : 0, plan.reason);

    float l2 = -1.f;
    if (args.verify) {
        std::vector<float> ref = cusparse_reference(args.m, args.n, args.k, h_A, h_B);
        l2 = rel_l2(h_C, ref);
    }

    std::printf(
        "run mode=%s m=%d k=%d n=%d nnz=%d density=%.6f "
        "time_ms=%.4f gflops=%.2f rel_l2_err=%.3e %s\n",
        mode.c_str(), args.m, args.k, args.n, h_A.nnz, actual_density,
        ms, gflops, l2,
        args.verify ? (l2 < 1e-2f ? "[PASS]" : "[FAIL]") : "[unverified]");

    spmm::csr_free_host(h_A);
    return 0;
}
