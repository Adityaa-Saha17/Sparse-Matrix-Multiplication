#pragma once

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err__ = (call);                                            \
        if (err__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",               \
                         #call, __FILE__, __LINE__,                            \
                         cudaGetErrorString(err__));                           \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

#define CUSPARSE_CHECK(call)                                                   \
    do {                                                                       \
        cusparseStatus_t status__ = (call);                                    \
        if (status__ != CUSPARSE_STATUS_SUCCESS) {                             \
            std::fprintf(stderr, "cuSPARSE error %s at %s:%d: %s\n",           \
                         #call, __FILE__, __LINE__,                            \
                         cusparseGetErrorString(status__));                    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

constexpr int ceil_div(int a, int b) { return (a + b - 1) / b; }

class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }
    ~GpuTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }
    void start() { cudaEventRecord(start_); }
    void stop() {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
    }
    float elapsed_ms() {
        float t = 0.f;
        cudaEventElapsedTime(&t, start_, stop_);
        return t;
    }

private:
    cudaEvent_t start_;
    cudaEvent_t stop_;
};
