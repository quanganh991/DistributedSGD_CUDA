# HPC_20261B_Group9 - Parallel & Distributed Deep Learning (CUDA)
#
#   make            -> bin/cnn_seq (C tuần tự) + bin/cnn_cuda (CUDA)
#   make seq        -> chỉ bản tuần tự (không cần CUDA)
#   make cuda       -> chỉ bản CUDA
#   make check      -> xem binary build cho kiến trúc GPU nào + thư viện động phụ thuộc
#
# Binary là "fat binary": chứa mã máy cho GTX 1080/1080 Ti (sm_61) và RTX 2060 (sm_75)
# + PTX compute_75 (JIT cho GPU mới hơn). cudart được link TĨNH -> server chỉ cần driver NVIDIA.

CC      ?= gcc
NVCC    ?= nvcc
# KHÔNG dùng -march=native: binary build trên máy local phải chạy được trên CPU của server
CFLAGS  ?= -O3 -std=gnu11 -Wall -Wextra -Wno-unused-parameter
GENCODE ?= -gencode arch=compute_61,code=sm_61 \
           -gencode arch=compute_75,code=sm_75 \
           -gencode arch=compute_75,code=compute_75
NVFLAGS ?= -O3 -std=c++14 $(GENCODE) -Xcompiler -Wall -lineinfo
LDSTATIC = -Xcompiler -static-libgcc -Xcompiler -static-libstdc++

BIN  = bin
OBJ  = build
COMMON_C = src/net_desc.c src/data.c src/cpu_nn.c
COMMON_O = $(patsubst src/%.c,$(OBJ)/%.o,$(COMMON_C))
CU_SRC   = src/gpu_kernels.cu src/gpu_train.cu src/main_cuda.cu
CU_O     = $(patsubst src/%.cu,$(OBJ)/%.cu.o,$(CU_SRC))
HDRS     = $(wildcard src/*.h src/*.cuh)

.PHONY: all seq cuda clean check

all: seq cuda
seq:  $(BIN)/cnn_seq
cuda: $(BIN)/cnn_cuda

$(OBJ) $(BIN):
	mkdir -p $@

$(OBJ)/%.o: src/%.c $(HDRS) | $(OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJ)/%.cu.o: src/%.cu $(HDRS) | $(OBJ)
	$(NVCC) $(NVFLAGS) -c $< -o $@

$(BIN)/cnn_seq: $(OBJ)/main_seq.o $(COMMON_O) | $(BIN)
	$(CC) $(CFLAGS) -static-libgcc -o $@ $^ -lm

$(BIN)/cnn_cuda: $(CU_O) $(COMMON_O) | $(BIN)
	$(NVCC) $(GENCODE) -cudart static $(LDSTATIC) -o $@ $^ -lm

check: all
	@echo "== kiến trúc GPU có trong cnn_cuda =="; cuobjdump --list-elf $(BIN)/cnn_cuda 2>/dev/null || true
	@echo "== thư viện động =="; ldd $(BIN)/cnn_cuda; ldd $(BIN)/cnn_seq
	@echo "== GLIBC tối thiểu cần =="; objdump -T $(BIN)/cnn_cuda $(BIN)/cnn_seq | grep -o 'GLIBC_[0-9.]*' | sort -V | uniq | tail -1

clean:
	rm -rf $(OBJ) $(BIN)
