/* gpu_nn.cuh - khai báo phần GPU (CUDA C). Không dùng cuBLAS/cuDNN:
 * mọi kernel (GEMM, im2col, pooling, softmax, ...) đều tự viết.
 */
#ifndef GPU_NN_CUH
#define GPU_NN_CUH
#include <cuda_runtime.h>
#include "common.h"
#include "net_desc.h"

#define CK(call)                                                                  \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), \
                    __FILE__, __LINE__, cudaGetErrorString(e_));                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)
#define CK_LAUNCH() CK(cudaGetLastError())

/* 0 = GEMM naive (1 thread / phần tử), 1 = GEMM tiled shared-memory + register blocking */
extern int g_gemm_tiled;

/* ---------------- primitive ---------------- */
void gemm_gpu(int tA, int tB, int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C, cudaStream_t s);
void matvec_gpu(int M, int K, const float *A, const float *x, float *y, cudaStream_t s);
void conv_forward_gpu(const Layer *l, int N, const float *x, const float *W, const float *b,
                      float *y, float *col, float *tmp, cudaStream_t s);
void conv_backward_gpu(const Layer *l, int N, const float *x, const float *W, const float *dy,
                       float *dW, float *db, float *dx, float *col, float *tmp, cudaStream_t s);

/* ---------------- các phép trên vector tham số ---------------- */
void vec_sgd(float *p, const float *g, int n, float lr, cudaStream_t s);           /* p -= lr*g      */
void vec_axpy(float *y, const float *x, int n, float a, cudaStream_t s);           /* y += a*x       */
void vec_atomic_axpy(float *y, const float *x, int n, float a, cudaStream_t s);    /* y += a*x (atomic) */
void vec_average(const float *buf, int K, int n, float *out, cudaStream_t s);      /* out = mean_k buf[k] */
void gather_batch(const float *X, const int *Y, const int *idx, int N, int sz,
                  float *xb, int *yb, cudaStream_t s);

/* ---------------- mạng trên 1 GPU ---------------- */
struct GpuNet {
    int dev;
    NetDesc d;
    int batch;
    cudaStream_t st;
    float *params, *grads;
    float *act[MAXL + 1], *dact[MAXL + 1];
    int *pidx[MAXL];
    float *col, *tmp;
    int *labels;
    float *loss_sum;   /* tích luỹ trên device (atomic) */
    int *correct;
};

void gnet_create(GpuNet *n, const NetDesc *d, int batch, int dev);
void gnet_free(GpuNet *n);
void gnet_forward(GpuNet *n, int N);                 /* input ở act[0] */
void gnet_loss(GpuNet *n, int N, int need_grad);     /* nhãn ở labels; cộng dồn loss_sum, correct */
void gnet_backward(GpuNet *n, int N);
void gnet_reset_stats(GpuNet *n);
void gnet_read_stats(GpuNet *n, double *loss_sum, long *correct);

/* copy giữa 2 GPU (hoặc cùng GPU) bất đồng bộ trên stream s */
void copy_dev(float *dst, int dst_dev, const float *src, int src_dev, size_t count, cudaStream_t s);

/* chạy huấn luyện (gpu_train.cu) */
int cmd_train_gpu(int argc, char **argv);

#endif
