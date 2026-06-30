#pragma once

// Memory-aware execution: multi-GPU row splitting and unified memory.
//
// SpMM with a row-partitioned A is embarrassingly parallel across the output
// rows: C = A * B computes row r of C purely from row r of A and all of B.
// So A can be split by rows across several GPUs, each holding its own row slice
// plus a full copy of the (small, dense) B, and the row slices of C concatenate
// into the final result with no cross-GPU communication.
//
// Two memory strategies are provided:
//   * spmm_multigpu — explicit per-device copies, one host thread per GPU,
//                     nnz-balanced row split (so each device gets roughly equal
//                     work rather than equal row count). Targets the free
//                     dual-T4 environment.
//   * spmm_unified  — single GPU, but A/B/C live in CUDA managed (unified)
//                     memory and are prefetched to the device. Lets the working
//                     set be driven by the page migrator instead of explicit
//                     cudaMemcpy, which matters once a matrix no longer fits in
//                     device memory alongside its dense operands.
//
// Both paths run the CSR tiled_v3 kernel (the project's most robust CUDA-core
// SpMM) so the only variable under study is the memory strategy.

#include <vector>

#include "csr.h"

namespace spmm {

// Number of CUDA-capable devices visible to the process.
int gpu_count();

// Outcome of a multi-GPU run, including the realised work split.
struct MultiGpuResult {
    float            time_ms  = 0.f;  // estimated per-iteration wall time
    int              num_gpus = 0;    // devices actually used
    std::vector<int> rows;            // rows assigned to each device
    std::vector<int> nnz;             // nonzeros assigned to each device
};

// Multi-GPU SpMM. A is split by rows across `num_gpus` (clamped to the number
// of visible devices) with an nnz-balanced partition; each device runs tiled_v3
// on its slice with a full copy of B and writes its row slice of C. The H2D
// upload of A and B happens once up front and is excluded from the timed loop;
// the returned time is the max over devices of each device's median per-launch
// kernel time (i.e. the slowest device sets the pace).
//   h_A : host CSR (full matrix).
//   h_B : host row-major K*N dense B.
//   h_C : host row-major M*N output (written on return).
MultiGpuResult spmm_multigpu(const CSR& h_A, const float* h_B, float* h_C,
                             int N, int num_gpus, int warmup, int iters);

// Single-GPU unified-memory SpMM. A/B/C are allocated with cudaMallocManaged,
// prefetched to the current device, and processed with tiled_v3. Returns the
// median per-launch kernel time in milliseconds; h_C is filled on return.
float spmm_unified(const CSR& h_A, const float* h_B, float* h_C,
                   int N, int warmup, int iters);

}  // namespace spmm
