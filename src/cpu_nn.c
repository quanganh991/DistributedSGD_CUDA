#include "cpu_nn.h"
#include "common.h"
#include <float.h>

/* ======================= GEMM / MATVEC ======================= */
void gemm_cpu(int tA, int tB, int M, int N, int K, float alpha,
              const float *A, const float *B, float beta, float *C) {
    size_t MN = (size_t)M * N;
    if (beta == 0.0f) memset(C, 0, sizeof(float) * MN);
    else if (beta != 1.0f) for (size_t i = 0; i < MN; i++) C[i] *= beta;

    if (!tA && !tB) {            /* thứ tự i-k-j: truy cập B, C liên tục */
        for (int i = 0; i < M; i++) {
            float *c = C + (size_t)i * N;
            for (int k = 0; k < K; k++) {
                float a = alpha * A[(size_t)i * K + k];
                if (a == 0.0f) continue;
                const float *b = B + (size_t)k * N;
                for (int j = 0; j < N; j++) c[j] += a * b[j];
            }
        }
    } else if (!tA && tB) {      /* tích vô hướng hàng A với hàng B */
        for (int i = 0; i < M; i++) {
            const float *a = A + (size_t)i * K;
            for (int j = 0; j < N; j++) {
                const float *b = B + (size_t)j * K;
                float s = 0.0f;
                for (int k = 0; k < K; k++) s += a[k] * b[k];
                C[(size_t)i * N + j] += alpha * s;
            }
        }
    } else if (tA && !tB) {      /* k-i-j */
        for (int k = 0; k < K; k++) {
            const float *b = B + (size_t)k * N;
            for (int i = 0; i < M; i++) {
                float a = alpha * A[(size_t)k * M + i];
                if (a == 0.0f) continue;
                float *c = C + (size_t)i * N;
                for (int j = 0; j < N; j++) c[j] += a * b[j];
            }
        }
    } else {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float s = 0.0f;
                for (int k = 0; k < K; k++) s += A[(size_t)k * M + i] * B[(size_t)j * K + k];
                C[(size_t)i * N + j] += alpha * s;
            }
    }
}

void matvec_cpu(int M, int K, const float *A, const float *x, float *y) {
    for (int i = 0; i < M; i++) {
        const float *a = A + (size_t)i * K;
        float s = 0.0f;
        for (int k = 0; k < K; k++) s += a[k] * x[k];
        y[i] = s;
    }
}

/* ======================= im2col / col2im ======================= */
void im2col_cpu(const float *x, int N, int C, int H, int W, int k, int s, int pad,
                int OH, int OW, float *col) {
    size_t P = (size_t)OH * OW, NP = (size_t)N * P;
    for (int c = 0; c < C; c++)
        for (int ki = 0; ki < k; ki++)
            for (int kj = 0; kj < k; kj++) {
                float *dst = col + ((size_t)(c * k + ki) * k + kj) * NP;
                for (int n = 0; n < N; n++) {
                    const float *xn = x + ((size_t)n * C + c) * H * W;
                    float *d = dst + n * P;
                    for (int oy = 0; oy < OH; oy++) {
                        int iy = oy * s - pad + ki;
                        for (int ox = 0; ox < OW; ox++) {
                            int ix = ox * s - pad + kj;
                            d[oy * OW + ox] = (iy >= 0 && iy < H && ix >= 0 && ix < W) ? xn[iy * W + ix] : 0.0f;
                        }
                    }
                }
            }
}

/* col2im kiểu "gather": mỗi pixel đầu vào cộng dồn mọi vị trí cột tham chiếu tới nó
 * (giống hệt kernel GPU -> không cần atomic). */
void col2im_cpu(const float *col, int N, int C, int H, int W, int k, int s, int pad,
                int OH, int OW, float *dx) {
    size_t P = (size_t)OH * OW, NP = (size_t)N * P;
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int iy = 0; iy < H; iy++)
                for (int ix = 0; ix < W; ix++) {
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
                            sum += col[row * NP + n * P + oy * OW + ox];
                        }
                    }
                    dx[(((size_t)n * C + c) * H + iy) * W + ix] = sum;
                }
}

/* ======================= Conv ======================= */
void conv_forward_cpu(const Layer *l, int N, const float *x, const float *W, const float *b,
                      float *y, float *col, float *tmp) {
    int K = l->in_c * l->k * l->k;
    size_t P = (size_t)l->out_h * l->out_w, NP = (size_t)N * P;
    im2col_cpu(x, N, l->in_c, l->in_h, l->in_w, l->k, l->stride, l->pad, l->out_h, l->out_w, col);
    gemm_cpu(0, 0, l->out_c, (int)NP, K, 1.0f, W, col, 0.0f, tmp);     /* [OC x NP] */
    for (int n = 0; n < N; n++)
        for (int oc = 0; oc < l->out_c; oc++) {
            float *yo = y + ((size_t)n * l->out_c + oc) * P;
            const float *t = tmp + oc * NP + n * P;
            for (size_t p = 0; p < P; p++) yo[p] = t[p] + b[oc];
        }
}

void conv_backward_cpu(const Layer *l, int N, const float *x, const float *W, const float *dy,
                       float *dW, float *db, float *dx, float *col, float *tmp) {
    int K = l->in_c * l->k * l->k, OC = l->out_c;
    size_t P = (size_t)l->out_h * l->out_w, NP = (size_t)N * P;
    /* dY [N][OC][P] -> tmp [OC][N*P] */
    for (int n = 0; n < N; n++)
        for (int oc = 0; oc < OC; oc++)
            memcpy(tmp + oc * NP + n * P, dy + ((size_t)n * OC + oc) * P, sizeof(float) * P);
    for (int oc = 0; oc < OC; oc++) {
        float s = 0.0f;
        for (size_t j = 0; j < NP; j++) s += tmp[oc * NP + j];
        db[oc] = s;
    }
    /* tính lại im2col (workspace col dùng chung giữa các lớp) */
    im2col_cpu(x, N, l->in_c, l->in_h, l->in_w, l->k, l->stride, l->pad, l->out_h, l->out_w, col);
    gemm_cpu(0, 1, OC, K, (int)NP, 1.0f, tmp, col, 0.0f, dW);          /* dW = dY * col^T */
    if (dx) {
        gemm_cpu(1, 0, K, (int)NP, OC, 1.0f, W, tmp, 0.0f, col);       /* dcol = W^T * dY */
        col2im_cpu(col, N, l->in_c, l->in_h, l->in_w, l->k, l->stride, l->pad, l->out_h, l->out_w, dx);
    }
}

/* ======================= FC ======================= */
void fc_forward_cpu(const Layer *l, int N, const float *x, const float *W, const float *b, float *y) {
    int in = layer_in_size(l), out = l->out_c;
    gemm_cpu(0, 1, N, out, in, 1.0f, x, W, 0.0f, y);                   /* y = x W^T */
    for (int n = 0; n < N; n++)
        for (int j = 0; j < out; j++) y[(size_t)n * out + j] += b[j];
}

void fc_backward_cpu(const Layer *l, int N, const float *x, const float *W, const float *dy,
                     float *dW, float *db, float *dx) {
    int in = layer_in_size(l), out = l->out_c;
    gemm_cpu(1, 0, out, in, N, 1.0f, dy, x, 0.0f, dW);                 /* dW = dy^T x */
    for (int j = 0; j < out; j++) {
        float s = 0.0f;
        for (int n = 0; n < N; n++) s += dy[(size_t)n * out + j];
        db[j] = s;
    }
    if (dx) gemm_cpu(0, 0, N, in, out, 1.0f, dy, W, 0.0f, dx);         /* dx = dy W */
}

/* ======================= ReLU / Pool ======================= */
static void relu_forward(const float *x, float *y, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] = x[i] > 0.0f ? x[i] : 0.0f;
}
static void relu_backward(const float *y, const float *dy, float *dx, size_t n) {
    for (size_t i = 0; i < n; i++) dx[i] = y[i] > 0.0f ? dy[i] : 0.0f;
}
static void pool_forward(const Layer *l, int N, const float *x, float *y, int *idx) {
    int C = l->in_c, H = l->in_h, W = l->in_w, k = l->k, OH = l->out_h, OW = l->out_w;
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++) {
            size_t base = ((size_t)n * C + c) * H * W;
            for (int oy = 0; oy < OH; oy++)
                for (int ox = 0; ox < OW; ox++) {
                    float best = -FLT_MAX; int bi = 0;
                    for (int ki = 0; ki < k; ki++)
                        for (int kj = 0; kj < k; kj++) {
                            int id = (int)(base + (size_t)(oy * k + ki) * W + ox * k + kj);
                            if (x[id] > best) { best = x[id]; bi = id; }
                        }
                    size_t o = (((size_t)n * C + c) * OH + oy) * OW + ox;
                    y[o] = best; idx[o] = bi;
                }
        }
}
static void pool_backward(const Layer *l, int N, const float *dy, const int *idx, float *dx) {
    memset(dx, 0, sizeof(float) * (size_t)N * layer_in_size(l));
    size_t no = (size_t)N * layer_out_size(l);
    for (size_t i = 0; i < no; i++) dx[idx[i]] += dy[i];
}

/* ======================= Net ======================= */
void cpu_net_create(CpuNet *n, const NetDesc *d, int batch) {
    memset(n, 0, sizeof(*n));
    n->d = *d; n->batch = batch;
    n->params = (float *)xmalloc(sizeof(float) * d->nparams);
    n->grads = (float *)xmalloc(sizeof(float) * d->nparams);
    memset(n->grads, 0, sizeof(float) * d->nparams);
    size_t in0 = (size_t)batch * d->in_c * d->in_h * d->in_w;
    n->act[0] = (float *)xmalloc(sizeof(float) * in0);
    n->dact[0] = NULL;  /* không cần gradient theo input */
    for (int i = 0; i < d->nl; i++) {
        size_t s = (size_t)batch * layer_out_size(&d->L[i]);
        n->act[i + 1] = (float *)xmalloc(sizeof(float) * s);
        n->dact[i + 1] = (float *)xmalloc(sizeof(float) * s);
        n->pidx[i] = d->L[i].type == L_POOL ? (int *)xmalloc(sizeof(int) * s) : NULL;
    }
    n->col = (float *)xmalloc(sizeof(float) * netdesc_col_size(d, batch));
    n->tmp = (float *)xmalloc(sizeof(float) * netdesc_tmp_size(d, batch));
}

void cpu_net_free(CpuNet *n) {
    free(n->params); free(n->grads); free(n->act[0]);
    for (int i = 0; i < n->d.nl; i++) { free(n->act[i + 1]); free(n->dact[i + 1]); free(n->pidx[i]); }
    free(n->col); free(n->tmp);
}

void cpu_net_forward(CpuNet *n, int N) {
    for (int i = 0; i < n->d.nl; i++) {
        const Layer *l = &n->d.L[i];
        const float *x = n->act[i];
        float *y = n->act[i + 1];
        switch (l->type) {
        case L_CONV: conv_forward_cpu(l, N, x, n->params + l->w_off, n->params + l->b_off, y, n->col, n->tmp); break;
        case L_FC:   fc_forward_cpu(l, N, x, n->params + l->w_off, n->params + l->b_off, y); break;
        case L_RELU: relu_forward(x, y, (size_t)N * layer_out_size(l)); break;
        case L_POOL: pool_forward(l, N, x, y, n->pidx[i]); break;
        }
    }
}

float cpu_net_loss(CpuNet *n, const int *y, int N, int *correct) {
    int C = n->d.classes;
    const float *logits = n->act[n->d.nl];
    float *dl = n->dact[n->d.nl];
    double loss = 0.0; int corr = 0;
    for (int i = 0; i < N; i++) {
        const float *z = logits + (size_t)i * C;
        float m = z[0]; int am = 0;
        for (int j = 1; j < C; j++) if (z[j] > m) { m = z[j]; am = j; }
        float s = 0.0f;
        for (int j = 0; j < C; j++) s += expf(z[j] - m);
        float ls = logf(s);
        loss += -(z[y[i]] - m - ls);
        if (am == y[i]) corr++;
        for (int j = 0; j < C; j++) {
            float p = expf(z[j] - m - ls);
            dl[(size_t)i * C + j] = (p - (j == y[i] ? 1.0f : 0.0f)) / (float)N;
        }
    }
    if (correct) *correct = corr;
    return (float)loss;
}

void cpu_net_backward(CpuNet *n, int N) {
    for (int i = n->d.nl - 1; i >= 0; i--) {
        const Layer *l = &n->d.L[i];
        const float *x = n->act[i], *y = n->act[i + 1], *dy = n->dact[i + 1];
        float *dx = (i > 0) ? n->dact[i] : NULL;
        switch (l->type) {
        case L_CONV:
            conv_backward_cpu(l, N, x, n->params + l->w_off, dy, n->grads + l->w_off,
                              n->grads + l->b_off, dx, n->col, n->tmp);
            break;
        case L_FC:
            fc_backward_cpu(l, N, x, n->params + l->w_off, dy, n->grads + l->w_off, n->grads + l->b_off, dx);
            break;
        case L_RELU: if (dx) relu_backward(y, dy, dx, (size_t)N * layer_out_size(l)); break;
        case L_POOL: if (dx) pool_backward(l, N, dy, n->pidx[i], dx); break;
        }
    }
}

void cpu_sgd(float *p, const float *g, int n, float lr) {
    for (int i = 0; i < n; i++) p[i] -= lr * g[i];
}
