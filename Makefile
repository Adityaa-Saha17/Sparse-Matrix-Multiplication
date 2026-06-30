# SpMM project Makefile — target NVIDIA T4 (Turing, sm_75) by default.
# Override CUDA_ARCH to retarget, e.g.: make CUDA_ARCH=sm_80

NVCC      ?= nvcc
CUDA_ARCH ?= sm_75

NVCCFLAGS  = -O3 -std=c++17 -arch=$(CUDA_ARCH) \
             --expt-relaxed-constexpr \
             -Xcompiler -Wall,-Wextra,-Wno-unused-parameter,-pthread
LDFLAGS    = -lcusparse -lcudart -lpthread
INC        = -Isrc

BUILD_DIR  = build

# Shared library sources (everything except the per-executable main).
LIB_SRCS = src/csr.cu \
           src/bsr.cu \
           src/kernels/spmm_baseline.cu \
           src/kernels/spmm_memopt.cu \
           src/kernels/spmm_memopt_v2.cu \
           src/kernels/spmm_tiled.cu \
           src/kernels/spmm_tiled_v2.cu \
           src/kernels/spmm_tiled_v3.cu \
           src/kernels/spmm_tiled_v4.cu \
           src/kernels/spmm_wmma.cu \
           src/spmm_hybrid.cu \
           src/spmm_multigpu.cu

LIB_OBJS  = $(LIB_SRCS:%.cu=$(BUILD_DIR)/%.o)

BENCH_OBJ = $(BUILD_DIR)/src/bench/harness.o
RUN_OBJ   = $(BUILD_DIR)/src/bench/spmm_run.o

BENCH = $(BUILD_DIR)/spmm_bench   # one kernel at a time (benchmark + correctness)
RUN   = $(BUILD_DIR)/spmm_run     # integrated driver (auto-dispatch / multi-GPU)

all: $(BENCH) $(RUN)

$(BENCH): $(LIB_OBJS) $(BENCH_OBJ)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(RUN): $(LIB_OBJS) $(RUN_OBJ)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(INC) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
