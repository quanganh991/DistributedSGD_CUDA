# HPC_20261B_Group9_DistributedSGD_CUDA

Hiện thực bài báo **"Parallel and Distributed Deep Learning"** (V. Hegde, S. Usmani, Stanford) bằng C (tuần tự) và CUDA (song song), **không dùng thư viện** (không cuBLAS/cuDNN — GEMM, im2col, pooling, softmax, ... đều tự viết).

| Phần trong paper | Hiện thực trong code |
|---|---|
| §3.1 Convolution = im2col + GEMM (Caffe) | `im2col_cpu` / `k_im2col` + GEMM, unroll **cả batch** một lần |
| §3.2.1 CPU vs GPU, conv 11×11 stride 4, 96 filter, ảnh 224×224, theo batch size | `bench-conv` |
| §3.2.2 FC layer = mat-vec, GEMM theo kích thước ma trận | `bench-fc` |
| §4 SGD, softmax/logistic loss (2)–(4) | `train` (CNN cho MNIST) |
| §5.1.1 ParallelSGD + ShuffleSGD | `--mode localsgd` (`--avg-period 0` = trung bình 1 lần/epoch) |
| §5.1.2 Synchronous update (ADMM.SGD) | `--mode sync` (all-reduce gradient mỗi bước) |
| §5.2.1 Downpour SGD (bất đồng bộ) | `--mode downpour` (`--nfetch`, `--npush`) |

## Cấu trúc

```
src/common.h        timer, RNG, đọc tham số
src/net_desc.[ch]   mô tả kiến trúc CNN, khởi tạo He (dùng chung CPU/GPU)
src/data.[ch]       nạp MNIST (IDX) / dữ liệu tổng hợp
src/cpu_nn.[ch]     CNN tuần tự: GEMM, im2col/col2im, conv, pool, FC, softmax-CE, backprop
src/main_seq.c      chương trình tuần tự  -> bin/cnn_seq
src/gpu_nn.cuh      khai báo phần CUDA
src/gpu_kernels.cu  kernel: GEMM naive + GEMM tiled (shared mem 64x64x16, 4x4 register blocking),
                    im2col/col2im, ReLU, max-pool, softmax-CE, SGD, average, atomic push
src/gpu_train.cu    data parallelism: K worker (stream/GPU), sync / localsgd / downpour
src/main_cuda.cu    chương trình song song -> bin/cnn_cuda (info, verify, train, bench-*)
scripts/            tải data, build cho server, deploy, chạy thí nghiệm, vẽ hình
```

Kiến trúc mặc định (`--arch c16k5,p2,c32k5,p2,f128`): Conv5×5(16)–ReLU–Pool2–Conv5×5(32)–ReLU–Pool2–FC128–ReLU–FC10, 80 202 tham số.

## Build

Yêu cầu: `gcc`, `make`; bản CUDA cần CUDA Toolkit ≤ 12.x (CUDA 13 bỏ hỗ trợ Pascal/sm_61).

```bash
make seq      # chỉ bản tuần tự
make cuda     # bản CUDA
make          # cả hai
make check    # xem kiến trúc GPU trong binary, thư viện động, GLIBC cần
```

Hoặc dùng CMake (CLion):

```bash
cmake -S . -B cmake-build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cmake-build-release -j      # binary vẫn ra bin/
```

### Dùng với CLion
1. **File → Open** thư mục project (chọn `CMakeLists.txt` → *Open as Project*).
2. **Settings → Build, Execution, Deployment → CMake**: thêm profile *Release*. Nếu CLion không thấy nvcc, điền vào *CMake options*:
   `-DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc`
3. Chọn target `cnn_cuda` hoặc `cnn_seq` → **Edit Configurations**:
   - *Program arguments*: ví dụ `train --data data/mnist --mode sync --workers 4 --epochs 5`, hoặc `verify`
   - *Working directory*: `$ProjectFileDir$` (để đường dẫn `data/mnist` đúng)
4. Chạy `scripts/get_mnist.sh` một lần (Terminal của CLion) để có dữ liệu.
5. Debug: bản `cnn_seq` debug bình thường bằng GDB của CLion. Với bản CUDA, CLion chỉ debug được phần host; kernel thì dùng `cuda-gdb` hoặc `compute-sanitizer ./bin/cnn_cuda verify`.

Build cho server vẫn dùng `scripts/build_for_server.sh` (Makefile trong Docker 22.04), không phụ thuộc CLion.

Binary là **fat binary** `sm_61` (GTX 1080 / 1080 Ti) + `sm_75` (RTX 2060) + PTX; `cudart` link tĩnh → máy chạy chỉ cần driver NVIDIA.

## Chạy

```bash
scripts/get_mnist.sh                       # tải MNIST vào data/mnist

# tuần tự (C)
./bin/cnn_seq gradcheck                    # kiểm tra backprop bằng sai phân hữu hạn
./bin/cnn_seq train --data data/mnist --epochs 1 --batch 64 --lr 0.05
./bin/cnn_seq bench-conv --batches 1,2,4,8,16,32,64,128
./bin/cnn_seq bench-fc   --sizes 256,512,1024,2048,4096

# song song (CUDA)
./bin/cnn_cuda info
./bin/cnn_cuda verify                      # so GPU với CPU: GEMM, forward, loss, gradient từng lớp
./bin/cnn_cuda train --data data/mnist --epochs 10                     # 1 worker
./bin/cnn_cuda train --data data/mnist --mode sync     --workers 4
./bin/cnn_cuda train --data data/mnist --mode localsgd --workers 4 --avg-period 0
./bin/cnn_cuda train --data data/mnist --mode downpour --workers 4 --nfetch 5 --npush 5
./bin/cnn_cuda train --data data/mnist --mode sync --workers 2 --gpus 0,1   # 2 GPU
./bin/cnn_cuda bench-conv --gemm both
./bin/cnn_cuda bench-fc
```

Tham số chung: `--batch` (mỗi worker), `--lr`, `--epochs`, `--seed`, `--arch`, `--gemm tiled|naive`, `--log file.csv`, `--synthetic` (không cần MNIST), `--train-limit N`.

## Quy trình: build ở local (RTX 2060), chạy trên server (GTX 1080)

```bash
# 1) LOCAL: test
make && ./bin/cnn_cuda verify && ./bin/cnn_cuda train --synthetic --epochs 2

# 2) LOCAL: build bằng container Ubuntu 22.04 (glibc khớp server) + đóng gói binary + data (không có source)
scripts/get_mnist.sh
scripts/build_for_server.sh
#    nếu `nvidia-smi` trên server báo "CUDA Version" < 12.0:
#    CUDA_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04 scripts/build_for_server.sh

# 3) LOCAL: đẩy lên server (mặc định /home/hpc/nhom9/dsgd)
REMOTE=hpc@<server> PORT=<port> scripts/deploy.sh

# 4) SERVER: chạy thí nghiệm (chọn GPU rảnh theo nvidia-smi)
cd /home/hpc/nhom9/dsgd
GPU=0 EPOCHS=10 nohup ./scripts/run_experiments.sh > run.log 2>&1 &
tail -f run.log

# 5) LOCAL: lấy kết quả về và vẽ hình
REMOTE=hpc@<server> PORT=<port> scripts/deploy.sh fetch
python3 scripts/plot_results.py results     # -> results/figs/*.png, summary.csv
```

### Lỗi thường gặp trên server
| Lỗi | Nguyên nhân / cách sửa |
|---|---|
| `GLIBC_2.38 not found` | build trên Ubuntu 24.04 → dùng `scripts/build_for_server.sh` (Docker 22.04) |
| `CUDA driver version is insufficient` | driver server cũ hơn toolkit → build lại với `CUDA_IMAGE=...11.8...` |
| `no kernel image is available` | thiếu `sm_61` trong `GENCODE` của Makefile |
| `out of memory` | GPU đang bị người khác dùng → đổi `GPU=1` hoặc giảm `--workers` |

## Thiết kế song song (tóm tắt)

- **Conv**: im2col cho cả batch thành ma trận `[Cin·k·k] × [N·OH·OW]`, sau đó 1 GEMM `W[OC×K] · col`. Backward: `dW = dY·colᵀ`, `dcol = Wᵀ·dY`, col2im kiểu *gather* (mỗi thread 1 pixel, không cần atomic).
- **GEMM tiled**: block 256 thread tính tile 64×64 của C, tile K = 16 trong shared memory (+1 padding chống bank conflict), mỗi thread 4×4 phần tử trong thanh ghi. Cách nạp tile thay đổi theo chuyển vị để đọc global memory luôn coalesced. So sánh với bản naive bằng `--gemm naive`.
- **Data parallelism**: mỗi worker là 1 bản sao mô hình + 1 CUDA stream; các worker đặt vòng tròn lên `--gpus`. GPU đầu tiên làm driver/parameter server. Mỗi epoch xáo trộn toàn cục rồi chia shard (ShuffleSGD). Giao tiếp: `cudaMemcpyPeerAsync` (P2P nếu có, nếu không thì đi qua host) và đồng bộ giữa các stream bằng `cudaEvent` — host không bao giờ bị chặn trong vòng lặp.
  - `sync`: gradient → server → trung bình → phát lại → mọi worker cập nhật giống hệt nhau (tương đương batch K·B).
  - `localsgd`: worker cập nhật độc lập, cứ H bước trung bình tham số (H = cả epoch là ParallelSGD gốc).
  - `downpour`: không có rào đồng bộ; worker fetch tham số mỗi `n_fetch` bước, push gradient tích luỹ mỗi `n_push` bước; server cộng bằng `atomicAdd` (lock-free).

## Kết quả

Các file CSV nằm trong `results/`, hình nằm trong `results/figs/` (xem `scripts/plot_results.py`).