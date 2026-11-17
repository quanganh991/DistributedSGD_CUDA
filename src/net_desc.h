/* net_desc.h - mô tả kiến trúc CNN (dùng chung CPU & GPU)
 *
 * Chuỗi kiến trúc, ví dụ: "c16k5,p2,c32k5,p2,f128"
 *   c<OUT>k<K>[s<S>][p<PAD>] : Conv OUT filter KxK, stride S, padding PAD (+ ReLU)
 *   p<K>                     : MaxPool KxK, stride K
 *   f<OUT>                   : Fully-connected OUT neuron (+ ReLU)
 * Lớp FC cuối cùng (số lớp = classes, không ReLU) được thêm tự động.
 *
 * Tất cả tham số (W, b của mọi lớp) nằm liên tiếp trong MỘT mảng phẳng
 * kích thước nparams -> thuận tiện cho việc gửi/nhận/trung bình tham số
 * giữa các worker (data parallelism).
 */
#ifndef NET_DESC_H
#define NET_DESC_H
#include <stdio.h>
#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAXL 32

typedef enum { L_CONV = 0, L_RELU = 1, L_POOL = 2, L_FC = 3 } LType;

typedef struct {
    LType type;
    int in_c, in_h, in_w;      /* kích thước 1 mẫu đầu vào  */
    int out_c, out_h, out_w;   /* kích thước 1 mẫu đầu ra   */
    int k, stride, pad;        /* conv / pool               */
    int w_off, b_off, nw, nb;  /* vị trí W, b trong mảng tham số phẳng */
} Layer;

typedef struct {
    int nl;
    Layer L[MAXL];
    int in_c, in_h, in_w, classes;
    int nparams;
} NetDesc;

static inline int layer_in_size(const Layer *l)  { return l->in_c * l->in_h * l->in_w; }
static inline int layer_out_size(const Layer *l) { return l->out_c * l->out_h * l->out_w; }

int    netdesc_build(NetDesc *d, const char *spec, int c, int h, int w, int classes);
void   netdesc_print(const NetDesc *d, FILE *f);
void   netdesc_init_params(const NetDesc *d, float *p, uint64_t seed); /* He init */
size_t netdesc_col_size(const NetDesc *d, int batch);  /* workspace im2col lớn nhất */
size_t netdesc_tmp_size(const NetDesc *d, int batch);  /* workspace out_c*N*P lớn nhất */
double netdesc_flops_per_sample(const NetDesc *d);     /* FLOP forward / 1 mẫu */

#ifdef __cplusplus
}
#endif
#endif
