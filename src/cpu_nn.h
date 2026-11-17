/* cpu_nn.h - CNN tuần tự trên CPU (1 lõi), không dùng thư viện ngoài.
 * Đây vừa là baseline tuần tự, vừa là "reference" để kiểm tra kết quả GPU.
 *
 * Bố cục dữ liệu (row-major):
 *   activation: [N][C][H][W]
 *   Conv W:     [out_c][in_c*k*k],   FC W: [out][in]
 *   im2col:     col[(c*k+ki)*k+kj][n*P + oy*OW + ox]   (P = OH*OW)
 */
#ifndef CPU_NN_H
#define CPU_NN_H
#include "net_desc.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    NetDesc d;
    int batch;                       /* dung lượng batch tối đa */
    float *params, *grads;           /* mảng phẳng nparams */
    float *act[MAXL + 1];            /* act[0] = input, act[i+1] = output lớp i */
    float *dact[MAXL + 1];
    int *pidx[MAXL];                 /* argmax của max-pool */
    float *col, *tmp;                /* workspace */
} CpuNet;

void  cpu_net_create(CpuNet *n, const NetDesc *d, int batch);
void  cpu_net_free(CpuNet *n);
void  cpu_net_forward(CpuNet *n, int N);                       /* input ở act[0] */
/* softmax + cross-entropy; ghi dlogits vào dact[nl]; trả về TỔNG loss của batch */
float cpu_net_loss(CpuNet *n, const int *y, int N, int *correct);
void  cpu_net_backward(CpuNet *n, int N);                      /* ghi grads */
void  cpu_sgd(float *p, const float *g, int n, float lr);

/* ---- primitive (được dùng cho benchmark & kiểm tra) ---- */
/* C[MxN] = alpha * op(A)[MxK] * op(B)[KxN] + beta*C
 * A lưu MxK (tA=0) hoặc KxM (tA=1); B lưu KxN (tB=0) hoặc NxK (tB=1) */
void gemm_cpu(int tA, int tB, int M, int N, int K, float alpha,
              const float *A, const float *B, float beta, float *C);
void matvec_cpu(int M, int K, const float *A, const float *x, float *y);  /* y = A x */
void im2col_cpu(const float *x, int N, int C, int H, int W, int k, int s, int pad,
                int OH, int OW, float *col);
void col2im_cpu(const float *col, int N, int C, int H, int W, int k, int s, int pad,
                int OH, int OW, float *dx);
void conv_forward_cpu(const Layer *l, int N, const float *x, const float *W, const float *b,
                      float *y, float *col, float *tmp);
void conv_backward_cpu(const Layer *l, int N, const float *x, const float *W, const float *dy,
                       float *dW, float *db, float *dx, float *col, float *tmp);
void fc_forward_cpu(const Layer *l, int N, const float *x, const float *W, const float *b, float *y);
void fc_backward_cpu(const Layer *l, int N, const float *x, const float *W, const float *dy,
                     float *dW, float *db, float *dx);

#ifdef __cplusplus
}
#endif
#endif
