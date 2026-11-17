#include "net_desc.h"
#include "common.h"
#include <ctype.h>

static int add_layer(NetDesc *d, Layer l) {
    if (d->nl >= MAXL) { fprintf(stderr, "Too many layers\n"); return -1; }
    d->L[d->nl++] = l;
    return 0;
}

/* đọc số nguyên sau ký tự key trong token, vd tok="c16k5s1p2", key='k' -> 5 */
static int tok_get(const char *tok, char key, int def) {
    for (const char *p = tok + 1; *p; p++)
        if (*p == key && isdigit((unsigned char)p[1])) return atoi(p + 1);
    return def;
}

int netdesc_build(NetDesc *d, const char *spec, int c, int h, int w, int classes) {
    memset(d, 0, sizeof(*d));
    d->in_c = c; d->in_h = h; d->in_w = w; d->classes = classes;
    int off = 0;
    char buf[512];
    strncpy(buf, spec ? spec : "", sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;

    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        Layer l; memset(&l, 0, sizeof(l));
        l.in_c = c; l.in_h = h; l.in_w = w;
        if (tok[0] == 'c') {
            l.type = L_CONV;
            l.out_c = atoi(tok + 1);
            l.k = tok_get(tok, 'k', 3);
            l.stride = tok_get(tok, 's', 1);
            l.pad = tok_get(tok, 'p', 0);
            l.out_h = (h + 2 * l.pad - l.k) / l.stride + 1;
            l.out_w = (w + 2 * l.pad - l.k) / l.stride + 1;
            if (l.out_c <= 0 || l.out_h <= 0 || l.out_w <= 0) { fprintf(stderr, "Bad conv '%s'\n", tok); return -1; }
            l.nw = l.out_c * c * l.k * l.k; l.nb = l.out_c;
            l.w_off = off; off += l.nw; l.b_off = off; off += l.nb;
            if (add_layer(d, l)) return -1;
            c = l.out_c; h = l.out_h; w = l.out_w;
            Layer r; memset(&r, 0, sizeof(r));
            r.type = L_RELU; r.in_c = r.out_c = c; r.in_h = r.out_h = h; r.in_w = r.out_w = w;
            if (add_layer(d, r)) return -1;
        } else if (tok[0] == 'p') {
            l.type = L_POOL;
            l.k = l.stride = atoi(tok + 1);
            l.out_c = c; l.out_h = h / l.k; l.out_w = w / l.k;
            if (l.k <= 0 || l.out_h <= 0) { fprintf(stderr, "Bad pool '%s'\n", tok); return -1; }
            if (add_layer(d, l)) return -1;
            h = l.out_h; w = l.out_w;
        } else if (tok[0] == 'f') {
            int in = c * h * w;
            l.type = L_FC;
            l.out_c = atoi(tok + 1); l.out_h = l.out_w = 1;
            l.nw = l.out_c * in; l.nb = l.out_c;
            l.w_off = off; off += l.nw; l.b_off = off; off += l.nb;
            if (add_layer(d, l)) return -1;
            c = l.out_c; h = w = 1;
            Layer r; memset(&r, 0, sizeof(r));
            r.type = L_RELU; r.in_c = r.out_c = c; r.in_h = r.out_h = 1; r.in_w = r.out_w = 1;
            if (add_layer(d, r)) return -1;
        } else {
            fprintf(stderr, "Unknown layer token '%s'\n", tok);
            return -1;
        }
    }
    /* lớp phân loại cuối */
    Layer l; memset(&l, 0, sizeof(l));
    l.type = L_FC; l.in_c = c; l.in_h = h; l.in_w = w;
    l.out_c = classes; l.out_h = l.out_w = 1;
    l.nw = classes * c * h * w; l.nb = classes;
    l.w_off = off; off += l.nw; l.b_off = off; off += l.nb;
    if (add_layer(d, l)) return -1;
    d->nparams = off;
    return 0;
}

void netdesc_print(const NetDesc *d, FILE *f) {
    static const char *nm[] = {"CONV", "RELU", "POOL", "FC"};
    fprintf(f, "Network: input %dx%dx%d, %d classes, %d params (%.1f KB)\n",
            d->in_c, d->in_h, d->in_w, d->classes, d->nparams, d->nparams * 4.0 / 1024);
    for (int i = 0; i < d->nl; i++) {
        const Layer *l = &d->L[i];
        fprintf(f, "  [%2d] %-4s %3dx%3dx%3d -> %3dx%3dx%3d", i, nm[l->type],
                l->in_c, l->in_h, l->in_w, l->out_c, l->out_h, l->out_w);
        if (l->type == L_CONV) fprintf(f, "  k=%d s=%d p=%d", l->k, l->stride, l->pad);
        if (l->type == L_POOL) fprintf(f, "  k=%d", l->k);
        if (l->nw) fprintf(f, "  params=%d", l->nw + l->nb);
        fprintf(f, "\n");
    }
}

void netdesc_init_params(const NetDesc *d, float *p, uint64_t seed) {
    Rng r = {seed};
    memset(p, 0, sizeof(float) * d->nparams);
    for (int i = 0; i < d->nl; i++) {
        const Layer *l = &d->L[i];
        if (l->type != L_CONV && l->type != L_FC) continue;
        int fan_in = (l->type == L_CONV) ? l->in_c * l->k * l->k : layer_in_size(l);
        float sd = sqrtf(2.0f / (float)fan_in);
        for (int j = 0; j < l->nw; j++) p[l->w_off + j] = sd * rng_normal(&r);
        /* bias = 0 */
    }
}

size_t netdesc_col_size(const NetDesc *d, int batch) {
    size_t m = 1;
    for (int i = 0; i < d->nl; i++) {
        const Layer *l = &d->L[i];
        if (l->type != L_CONV) continue;
        size_t s = (size_t)l->in_c * l->k * l->k * (size_t)batch * l->out_h * l->out_w;
        if (s > m) m = s;
    }
    return m;
}

size_t netdesc_tmp_size(const NetDesc *d, int batch) {
    size_t m = 1;
    for (int i = 0; i < d->nl; i++) {
        const Layer *l = &d->L[i];
        if (l->type != L_CONV) continue;
        size_t s = (size_t)l->out_c * (size_t)batch * l->out_h * l->out_w;
        if (s > m) m = s;
    }
    return m;
}

double netdesc_flops_per_sample(const NetDesc *d) {
    double f = 0;
    for (int i = 0; i < d->nl; i++) {
        const Layer *l = &d->L[i];
        if (l->type == L_CONV)
            f += 2.0 * l->out_c * l->in_c * l->k * l->k * l->out_h * l->out_w;
        else if (l->type == L_FC)
            f += 2.0 * l->out_c * layer_in_size(l);
    }
    return f;
}
