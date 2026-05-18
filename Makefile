# SpMM project Makefile — target NVIDIA T4 (Turing, sm_75) by default.
# Override CUDA_ARCH to retarget, e.g.: make CUDA_ARCH=sm_80

NVCC      ?= nvcc
CUDA_ARCH ?= sm_75

NVCCFLAGS  = -O3 -std=c++17 -arch=$(CUDA_ARCH) \
             --expt-relaxed-constexpr \
             -Xcompiler -Wall,-Wextra,-Wno-unused-parameter
LDFLAGS    = -lcusparse -lcudart
INC        = -Isrc

BUILD_DIR  = build

SRCS = src/csr.cu \
       src/kernels/spmm_baseline.cu \
       src/kernels/spmm_memopt.cu \
       src/bench/harness.cu

OBJS = $(SRCS:%.cu=$(BUILD_DIR)/%.o)
TARGET = $(BUILD_DIR)/spmm_bench

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(OBJS) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(NVCCFLAGS) $(INC) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all clean
