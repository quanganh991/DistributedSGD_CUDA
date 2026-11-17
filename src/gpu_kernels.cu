/* gpu_kernels.cu - toàn bộ kernel CUDA tự viết + forward/backward CNN trên GPU */
#include "gpu_nn.cuh"
#include <float.h>

int g_gemm_tiled = 1;

static inline unsigned nblocks(size_t n, int bs = 256) {
    size_t b = (n + bs - 1) / bs;
    if (b > 0x7fffffff) b = 0x7fffffff;
    return (unsigned)(b ? b : 1);
}
#define GRID_STRIDE(i, n) \
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += (size_t)gridDim.x * blockDim.x)

/* =====================================================================
 *  GEMM:  C[MxN] = alpha * op(A)[MxK] * op(B)[KxN] + beta * C   (row-major)
 * ===================================================================== */

/* (1) naive: mỗi thread tính 1 phần tử C, đọc thẳng từ global memory */
template <bool TA, bool TB>
__global__ void k_gemm_naive(int M, int N, int K, float alpha, const float *__restrict__ A,
                             const float *__restrict__ B, float beta, float *__restrict__ C) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) return;
    float s = 0.0f;
    for (int k = 0; k < K; k++) {
        float a = TA ? A[(size_t)k * M + i] : A[(size_t)i * K + k];
        float b = TB ? B[(size_t)j * K + k] : B[(size_t)k * N + j];
        s += a * b;
    }
    size_t o = (size_t)i * N + j;
    C[o] = alpha * s + (beta != 0.0f ? beta * C[o] : 0.0f);
}

/* (2) tiled: block 256 thread tính tile 64x64 của C, mỗi thread 4x4 phần tử
 *     (register blocking). Mỗi vòng nạp tile A[64x16], B[16x64] vào shared memory.
 *     Cách nạp phụ thuộc chuyển vị để đọc global memory luôn coalesced. */
#define TBM 64
#define TBN 64
#define TBK 16
template <bool TA, bool TB>
__global__ void __launch_bounds__(256)
k_gemm_tiled(int M, int N, int K, float alpha, const float *__restrict__ A,
             const float *__restrict__ B, float beta, float *__restrict__ C) {
    __shared__ float As[TBK][TBM + 1];
    __shared__ float Bs[TBK][TBN + 1];
    const int tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int bm = blockIdx.y * TBM, bn = blockIdx.x * TBN;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += TBK) {
#pragma unroll
        for (int r = 0; r < 4; r++) {
            int t = tid + r * 256;               /* 0..1023 = 64*16 phần tử */
            int mm, kk;
            if (!TA) { kk = t % TBK; mm = t / TBK; }   /* A[m][k]: liên tục theo k */
            else     { mm = t % TBM; kk = t / TBM; }   /* A[k][m]: liên tục theo m */
            int gm = bm + mm, gk = k0 + kk;
            float v = 0.0f;
            if (gm < M && gk < K) v = TA ? A[(size_t)gk * M + gm] : A[(size_t)gm * K + gk];
            As[kk][mm] = v;

            int nn;
            if (!TB) { nn = t % TBN; kk = t / TBN; }   /* B[k][n]: liên tục theo n */
            else     { kk = t % TBK; nn = t / TBK; }   /* B[n][k]: liên tục theo k */
            int gn = bn + nn;
            gk = k0 + kk;
            v = 0.0f;
            if (gn < N && gk < K) v = TB ? B[(size_t)gn * K + gk] : B[(size_t)gk * N + gn];
            Bs[kk][nn] = v;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < TBK; kk++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = As[kk][ty + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Bs[kk][tx + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int gm = bm + ty + 16 * i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            int gn = bn + tx + 16 * j;
            if (gn >= N) continue;
            size_t o = (size_t)gm * N + gn;
            C[o] = alpha * acc[i][j] + (beta != 0.0f ? beta * C[o] : 0.0f);
        }
    }
}

template <bool TA, bool TB>
static void gemm_launch(int M, int N, int K, float alpha, const float *A, const float *B, float beta,
                        float *C, cudaStream_t s) {
    if (g_gemm_tiled) {
        dim3 grid((N + TBN - 1) / TBN, (M + TBM - 1) / TBM);
        k_gemm_tiled<TA, TB><<<grid, 256, 0, s>>>(M, N, K, alpha, A, B, beta, C);
    } else {
        dim3 blk(16, 16), grid((N + 15) / 16, (M + 15) / 16);
        k_gemm_naive<TA, TB><<<grid, blk, 0, s>>>(M, N, K, alpha, A, B, beta, C);
    }
    CK_LAUNCH();
}

void gemm_gpu(int tA, int tB, int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C, cudaStream_t s) {
    if (!tA && !tB) gemm_launch<false, false>(M, N, K, alpha, A, B, beta, C, s);
    else if (!tA && tB) gemm_launch<false, true>(M, N, K, alpha, A, B, beta, C, s);
    else if (tA && !tB) gemm_launch<true, false>(M, N, K, alpha, A, B, beta, C, s);
    else gemm_launch<true, true>(M, N, K, alpha, A, B, beta, C, s);
}

/* mat-vec: mỗi warp tính 1 hàng, rút gọn bằng warp shuffle */
__global__ void k_matvec(int M, int K, const float *__restrict__ A, const float *__restrict__ x,
                         float *__restrict__ y) {
    int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= M) return;
    const float *a = A + (size_t)warp * K;
    float s = 0.0f;
    for (int k = lane; k < K; k += 32) s += a[k] * x[k];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) y[warp] = s;
}
void matvec_gpu(int M, int K, const float *A, const float *x, float *y, cudaStream_t s) {
    k_matvec<<<nblocks((size_t)M * 32), 256, 0, s>>>(M, K, A, x, y);
    CK_LAUNCH();
}

/* =====================================================================
 *  Convolution = im2col + GEMM (giống Caffe, mục 3.1 của paper),
 *  nhưng unroll CẢ BATCH một lần thay vì lần lượt từng ảnh.
 * ===================================================================== */
__global__ void k_im2col(const float *__restrict__ x, int N, int C, int H, int W, int k, int s,
                         int pad, int OH, int OW, float *__restrict__ col) {
    size_t P = (size_t)OH * OW, NP = (size_t)N * P, total = (size_t)C * k * k * NP;
    GRID_STRIDE(i, total) {
        size_t row = i / NP, cj = i - row * NP;
        int n = (int)(cj / P), p = (int)(cj - (size_t)n * P);
        int oy = p / OW, ox = p - oy * OW;
        int kj = (int)(row % k), ki = (int)((row / k) % k), c = (int)(row / ((size_t)k * k));
        int iy = oy * s - pad + ki, ix = ox * s - pad + kj;
        col[i] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? x[(((size_t)n * C + c) * H + iy) * W + ix] : 0.0f;
    }
}

/* col2im kiểu gather: mỗi thread = 1 pixel input, cộng dồn các cột liên quan (không atomic) */
__global__ void k_col2im(const float *__restrict__ col, int N, int C, int H, int W, int k, int s,
                         int pad, int OH, int OW, float *__restrict__ dx) {
    size_t P = (size_t)OH * OW, NP = (size_t)N * P, total = (size_t)N * C * H * W;
    GRID_STRIDE(i, total) {
        int ix = (int)(i % W), iy = (int)((i / W) % H);
        int c = (int)((i / ((size_t)W * H)) % C), n = (int)(i / ((size_t)W * H * C));
        float sum = 0.0f;
        for (int ki = 0; ki < k; ki++) {
            int ty = iy + pad - ki;
            if (ty < 0 || ty % s) continue;
            int oy = ty / s;
            if (oy >= OH) continue;
            for (int kj = 0; kj < k; kj++) {
                int tx = ix + pad - kj;
                if (tx < 0 || tx % s) continue;
                int ox = tx / s;
                if (ox >= OW) continue;
                size_t row = (size_t)(c * k + ki) * k + kj;
                sum += col[row * NP + (size_t)n * P + oy * OW + ox];
            }
        }
        dx[i] = sum;
    }
}

/* tmp[OC][N*P] + b -> y[N][OC][P] */
__global__ void k_conv_out(const float *__restrict__ tmp, const float *__restrict__ b, int N, int OC,
                           size_t P, float *__restrict__ y) {
    size_t NP = (size_t)N * P, total = (size_t)N * OC * P;
    GRID_STRIDE(i, total) {
        size_t p = i % P;
        int oc = (int)((i / P) % OC), n = (int)(i / (P * OC));
        y[i] = tmp[(size_t)oc * NP + (size_t)n * P + p] + b[oc];
    }
}
/* dy[N][OC][P] -> tmp[OC][N*P] */
__global__ void k_conv_dy(const float *__restrict__ dy, int N, int OC, size_t P, float *__restrict__ tmp) {
    size_t NP = (size_t)N * P, total = (size_t)N * OC * P;
    GRID_STRIDE(i, total) {
        size_t p = i % P;
        int oc = (int)((i / P) % OC), n = (int)(i / (P * OC));
        tmp[(size_t)oc * NP + (size_t)n * P + p] = dy[i];
    }
}

/* tổng theo hàng: 1 block / hàng, rút gọn trong shared memory */
__global__ void k_row_sum(const float *__restrict__ A, size_t cols, float *__restrict__ out) {
    __shared__ float sh[256];
    const float *a = A + (size_t)blockIdx.x * cols;
    float s = 0.0f;
    for (size_t j = threadIdx.x; j < cols; j += blockDim.x) s += a[j];
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int o = blockDim.x / 2; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) sh[threadIdx.x] += sh[threadIdx.x + o];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sh[0];
}
/* tổng theo cột (cho bias FC): mỗi thread 1 cột */
__global__ void k_col_sum(const float *__restrict__ A, int rows, int cols, float *__restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= cols) return;
    float s = 0.0f;
    for (int r = 0; r < rows; r++) s += A[(size_t)r * cols + j];
    out[j] = s;
}
__global__ void k_add_bias_rows(float *__restrict__ y, const float *__restrict__ b, int N, int out) {
    size_t total = (size_t)N * out;
    GRID_STRIDE(i, total) y[i] += b[i % out];
}

void conv_forward_gpu(const Layer *l, int N, const float *x, const float *W, const float *b,
                      float *y, float *col, float *tmp, cudaStream_t s) {
    int K = l->in_c * l->k * l->k;
    size_t P = (size_t)l->out_h * l->out_w, NP = (size_t)N * P;
    k_im2col<<<nblocks((size_t)K * NP), 256, 0, s>>>(x, N, l->in_c, l->in_h, l->in_w, l->k, l->stride,
                                                    l->pad, l->out_h, l->out_w, col);
    CK_LAUNCH();
    gemm_gpu(0, 0, l->out_c, (int)NP, K, 1.0f, W, col, 0.0f, tmp, s);
    k_conv_out<<<nblocks((size_t)l->out_c * NP), 256, 0, s>>>(tmp, b, N, l->out_c, P, y);
    CK_LAUNCH();
}

void conv_backward_gpu(const Layer *l, int N, const float *x, const float *W, const float *dy,
                       float *dW, float *db, float *dx, float *col, float *tmp, cudaStream_t s) {
    int K = l->in_c * l->k * l->k, OC = l->out_c;
    size_t P = (size_t)l->out_h * l->out_w, NP = (size_t)N * P;
    k_conv_dy<<<nblocks((size_t)OC * NP), 256, 0, s>>>(dy, N, OC, P, tmp);
    CK_LAUNCH();
    k_row_sum<<<OC, 256, 0, s>>>(tmp, NP, db);
    CK_LAUNCH();
    k_im2col<<<nblocks((size_t)K * NP), 256, 0, s>>>(x, N, l->in_c, l->in_h, l->in_w, l->k, l->stride,
                                                    l->pad, l->out_h, l->out_w, col);
    CK_LAUNCH();
    gemm_gpu(0, 1, OC, K, (int)NP, 1.0f, tmp, col, 0.0f, dW, s);      /* dW = dY col^T  */
    if (dx) {
        gemm_gpu(1, 0, K, (int)NP, OC, 1.0f, W, tmp, 0.0f, col, s);   /* dcol = W^T dY */
        k_col2im<<<nblocks((size_t)N * layer_in_size(l)), 256, 0, s>>>(col, N, l->in_c, l->in_h, l->in_w,
                                                                       l->k, l->stride, l->pad, l->out_h,
                                                                       l->out_w, dx);
        CK_LAUNCH();
    }
}

/* =====================================================================
 *  ReLU, MaxPool, Softmax-CE
 * ===================================================================== */
__global__ void k_relu_f(const float *__restrict__ x, float *__restrict__ y, size_t n) {
    GRID_STRIDE(i, n) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
__global__ void k_relu_b(const float *__restrict__ y, const float *__restrict__ dy, float *__restrict__ dx, size_t n) {
    GRID_STRIDE(i, n) dx[i] = y[i] > 0.0f ? dy[i] : 0.0f;
}
__global__ void k_pool_f(const float *__restrict__ x, int N, int C, int H, int W, int k, int OH, int OW,
                         float *__restrict__ y, int *__restrict__ idx) {
    size_t total = (size_t)N * C * OH * OW;
    GRID_STRIDE(i, total) {
        int ox = (int)(i % OW), oy = (int)((i / OW) % OH);
        size_t nc = i / ((size_t)OW * OH);
        size_t base = nc * H * W;
        float best = -FLT_MAX;
        int bi = 0;
        for (int ki = 0; ki < k; ki++)
            for (int kj = 0; kj < k; kj++) {
                int id = (int)(base + (size_t)(oy * k + ki) * W + ox * k + kj);
                float v = x[id];
                if (v > best) { best = v; bi = id; }
            }
        y[i] = best;
        idx[i] = bi;
    }
}
/* pool không chồng lấn -> mỗi input nhận tối đa 1 gradient, không cần atomic */
__global__ void k_pool_b(const float *__restrict__ dy, const int *__restrict__ idx, size_t n, float *__restrict__ dx) {
    GRID_STRIDE(i, n) dx[idx[i]] = dy[i];
}
__global__ void k_softmax_ce(const float *__restrict__ z, const int *__restrict__ y, int N, int C,
                             float *__restrict__ dz, float *loss_sum, int *correct) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float *zi = z + (size_t)i * C;
    float m = zi[0];
    int am = 0;
    for (int j = 1; j < C; j++) if (zi[j] > m) { m = zi[j]; am = j; }
    float s = 0.0f;
    for (int j = 0; j < C; j++) s += __expf(zi[j] - m);
    float ls = logf(s);
    atomicAdd(loss_sum, -(zi[y[i]] - m - ls));
    if (am == y[i]) atomicAdd(correct, 1);
    if (dz)
        for (int j = 0; j < C; j++)
            dz[(size_t)i * C + j] = (__expf(zi[j] - m - ls) - (j == y[i] ? 1.0f : 0.0f)) / (float)N;
}

/* =====================================================================
 *  Vector ops cho tham số (SGD, trung bình giữa worker, Downpour push)
 * ===================================================================== */
__global__ void k_sgd(float *__restrict__ p, const float *__restrict__ g, int n, float lr) {
    GRID_STRIDE(i, (size_t)n) p[i] -= lr * g[i];
}
__global__ void k_axpy(float *__restrict__ y, const float *__restrict__ x, int n, float a) {
    GRID_STRIDE(i, (size_t)n) y[i] += a * x[i];
}
__global__ void k_atomic_axpy(float *y, const float *__restrict__ x, int n, float a) {
    GRID_STRIDE(i, (size_t)n) atomicAdd(&y[i], a * x[i]);
}
__global__ void k_average(const float *__restrict__ buf, int K, int n, float *__restrict__ out) {
    GRID_STRIDE(i, (size_t)n) {
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += buf[(size_t)k * n + i];
        out[i] = s / (float)K;
    }
}
__global__ void k_gather(const float *__restrict__ X, const int *__restrict__ Y, const int *__restrict__ idx,
                         int N, int sz, float *__restrict__ xb, int *__restrict__ yb) {
    size_t total = (size_t)N * sz;
    GRID_STRIDE(i, total) {
        int n = (int)(i / sz), j = (int)(i - (size_t)n * sz);
        int id = idx[n];
        xb[i] = X[(size_t)id * sz + j];
        if (j == 0) yb[n] = Y[id];
    }
}

void vec_sgd(float *p, const float *g, int n, float lr, cudaStream_t s) {
    k_sgd<<<nblocks(n), 256, 0, s>>>(p, g, n, lr); CK_LAUNCH();
}
void vec_axpy(float *y, const float *x, int n, float a, cudaStream_t s) {
    k_axpy<<<nblocks(n), 256, 0, s>>>(y, x, n, a); CK_LAUNCH();
}
void vec_atomic_axpy(float *y, const float *x, int n, float a, cudaStream_t s) {
    k_atomic_axpy<<<nblocks(n), 256, 0, s>>>(y, x, n, a); CK_LAUNCH();
}
void vec_average(const float *buf, int K, int n, float *out, cudaStream_t s) {
    k_average<<<nblocks(n), 256, 0, s>>>(buf, K, n, out); CK_LAUNCH();
}
void gather_batch(const float *X, const int *Y, const int *idx, int N, int sz, float *xb, int *yb, cudaStream_t s) {
    k_gather<<<nblocks((size_t)N * sz), 256, 0, s>>>(X, Y, idx, N, sz, xb, yb); CK_LAUNCH();
}

void copy_dev(float *dst, int dst_dev, const float *src, int src_dev, size_t count, cudaStream_t s) {
    if (dst_dev == src_dev) CK(cudaMemcpyAsync(dst, src, count * sizeof(float), cudaMemcpyDeviceToDevice, s));
    else CK(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, count * sizeof(float), s));
}

/* =====================================================================
 *  GpuNet
 * ===================================================================== */
template <typename T>
static T *dalloc(size_t n) {
    T *p = NULL;
    CK(cudaMalloc((void **)&p, (n ? n : 1) * sizeof(T)));
    return p;
}

void gnet_create(GpuNet *n, const NetDesc *d, int batch, int dev) {
    memset(n, 0, sizeof(*n));
    n->dev = dev; n->d = *d; n->batch = batch;
    CK(cudaSetDevice(dev));
    CK(cudaStreamCreateWithFlags(&n->st, cudaStreamNonBlocking));
    n->params = dalloc<float>(d->nparams);
    n->grads = dalloc<float>(d->nparams);
    CK(cudaMemset(n->grads, 0, sizeof(float) * d->nparams));
    n->act[0] = dalloc<float>((size_t)batch * d->in_c * d->in_h * d->in_w);
    for (int i = 0; i < d->nl; i++) {
        size_t s = (size_t)batch * layer_out_size(&d->L[i]);
        n->act[i + 1] = dalloc<float>(s);
        n->dact[i + 1] = dalloc<float>(s);
        n->pidx[i] = d->L[i].type == L_POOL ? dalloc<int>(s) : NULL;
    }
    n->col = dalloc<float>(netdesc_col_size(d, batch));
    n->tmp = dalloc<float>(netdesc_tmp_size(d, batch));
    n->labels = dalloc<int>(batch);
    n->loss_sum = dalloc<float>(1);
    n->correct = dalloc<int>(1);
    gnet_reset_stats(n);
    CK(cudaStreamSynchronize(n->st));
}

void gnet_free(GpuNet *n) {
    CK(cudaSetDevice(n->dev));
    cudaFree(n->params); cudaFree(n->grads); cudaFree(n->act[0]);
    for (int i = 0; i < n->d.nl; i++) { cudaFree(n->act[i + 1]); cudaFree(n->dact[i + 1]); if (n->pidx[i]) cudaFree(n->pidx[i]); }
    cudaFree(n->col); cudaFree(n->tmp); cudaFree(n->labels); cudaFree(n->loss_sum); cudaFree(n->correct);
    cudaStreamDestroy(n->st);
}

void gnet_forward(GpuNet *n, int N) {
    CK(cudaSetDevice(n->dev));
    for (int i = 0; i < n->d.nl; i++) {
        const Layer *l = &n->d.L[i];
        const float *x = n->act[i];
        float *y = n->act[i + 1];
        size_t no = (size_t)N * layer_out_size(l);
        switch (l->type) {
        case L_CONV:
            conv_forward_gpu(l, N, x, n->params + l->w_off, n->params + l->b_off, y, n->col, n->tmp, n->st);
            break;
        case L_FC:
            gemm_gpu(0, 1, N, l->out_c, layer_in_size(l), 1.0f, x, n->params + l->w_off, 0.0f, y, n->st);
            k_add_bias_rows<<<nblocks(no), 256, 0, n->st>>>(y, n->params + l->b_off, N, l->out_c);
            CK_LAUNCH();
            break;
        case L_RELU:
            k_relu_f<<<nblocks(no), 256, 0, n->st>>>(x, y, no);
            CK_LAUNCH();
            break;
        case L_POOL:
            k_pool_f<<<nblocks(no), 256, 0, n->st>>>(x, N, l->in_c, l->in_h, l->in_w, l->k, l->out_h, l->out_w,
                                                     y, n->pidx[i]);
            CK_LAUNCH();
            break;
        }
    }
}

void gnet_loss(GpuNet *n, int N, int need_grad) {
    CK(cudaSetDevice(n->dev));
    k_softmax_ce<<<nblocks(N, 128), 128, 0, n->st>>>(n->act[n->d.nl], n->labels, N, n->d.classes,
                                                     need_grad ? n->dact[n->d.nl] : NULL, n->loss_sum, n->correct);
    CK_LAUNCH();
}

void gnet_backward(GpuNet *n, int N) {
    CK(cudaSetDevice(n->dev));
    for (int i = n->d.nl - 1; i >= 0; i--) {
        const Layer *l = &n->d.L[i];
        const float *x = n->act[i], *y = n->act[i + 1], *dy = n->dact[i + 1];
        float *dx = (i > 0) ? n->dact[i] : NULL;
        size_t no = (size_t)N * layer_out_size(l);
        switch (l->type) {
        case L_CONV:
            conv_backward_gpu(l, N, x, n->params + l->w_off, dy, n->grads + l->w_off, n->grads + l->b_off,
                              dx, n->col, n->tmp, n->st);
            break;
        case L_FC: {
            int in = layer_in_size(l), out = l->out_c;
            gemm_gpu(1, 0, out, in, N, 1.0f, dy, x, 0.0f, n->grads + l->w_off, n->st);   /* dW = dy^T x */
            k_col_sum<<<nblocks(out), 256, 0, n->st>>>(dy, N, out, n->grads + l->b_off);
            CK_LAUNCH();
            if (dx) gemm_gpu(0, 0, N, in, out, 1.0f, dy, n->params + l->w_off, 0.0f, dx, n->st); /* dx = dy W */
            break;
        }
        case L_RELU:
            if (dx) { k_relu_b<<<nblocks(no), 256, 0, n->st>>>(y, dy, dx, no); CK_LAUNCH(); }
            break;
        case L_POOL:
            if (dx) {
                CK(cudaMemsetAsync(dx, 0, sizeof(float) * (size_t)N * layer_in_size(l), n->st));
                k_pool_b<<<nblocks(no), 256, 0, n->st>>>(dy, n->pidx[i], no, dx);
                CK_LAUNCH();
            }
            break;
        }
    }
}

void gnet_reset_stats(GpuNet *n) {
    CK(cudaSetDevice(n->dev));
    CK(cudaMemsetAsync(n->loss_sum, 0, sizeof(float), n->st));
    CK(cudaMemsetAsync(n->correct, 0, sizeof(int), n->st));
}

void gnet_read_stats(GpuNet *n, double *loss_sum, long *correct) {
    CK(cudaSetDevice(n->dev));
    float l; int c;
    CK(cudaMemcpyAsync(&l, n->loss_sum, sizeof(float), cudaMemcpyDeviceToHost, n->st));
    CK(cudaMemcpyAsync(&c, n->correct, sizeof(int), cudaMemcpyDeviceToHost, n->st));
    CK(cudaStreamSynchronize(n->st));
    *loss_sum = l; *correct = c;
}
