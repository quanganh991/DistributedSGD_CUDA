#include "data.h"
#include "common.h"

static uint32_t read_be32(FILE *f) {
    unsigned char b[4];
    if (fread(b, 1, 4, f) != 4) return 0;
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | b[3];
}

int dataset_load_mnist(Dataset *ds, const char *dir, int train, int limit) {
    char pi[1024], pl[1024];
    snprintf(pi, sizeof(pi), "%s/%s", dir, train ? "train-images-idx3-ubyte" : "t10k-images-idx3-ubyte");
    snprintf(pl, sizeof(pl), "%s/%s", dir, train ? "train-labels-idx1-ubyte" : "t10k-labels-idx1-ubyte");
    FILE *fi = fopen(pi, "rb"), *fl = fopen(pl, "rb");
    if (!fi || !fl) {
        fprintf(stderr, "Cannot open %s or %s\n  -> chạy scripts/get_mnist.sh hoặc dùng --synthetic\n", pi, pl);
        if (fi) fclose(fi);
        if (fl) fclose(fl);
        return -1;
    }
    uint32_t mi = read_be32(fi), n = read_be32(fi), h = read_be32(fi), w = read_be32(fi);
    uint32_t ml = read_be32(fl), nl = read_be32(fl);
    if (mi != 2051 || ml != 2049 || n != nl) {
        fprintf(stderr, "Bad IDX header in %s / %s\n", pi, pl);
        fclose(fi); fclose(fl);
        return -1;
    }
    if (limit > 0 && (uint32_t)limit < n) n = (uint32_t)limit;
    ds->n = (int)n; ds->c = 1; ds->h = (int)h; ds->w = (int)w; ds->classes = 10;
    size_t sz = (size_t)h * w;
    ds->x = (float *)xmalloc(sizeof(float) * n * sz);
    ds->y = (int *)xmalloc(sizeof(int) * n);
    unsigned char *buf = (unsigned char *)xmalloc(n * sz);
    unsigned char *lb = (unsigned char *)xmalloc(n);
    if (fread(buf, 1, n * sz, fi) != n * sz || fread(lb, 1, n, fl) != n) {
        fprintf(stderr, "Short read in MNIST files\n");
        exit(1);
    }
    /* chuẩn hoá theo mean/std của MNIST */
    for (size_t i = 0; i < n * sz; i++) ds->x[i] = (buf[i] / 255.0f - 0.1307f) / 0.3081f;
    for (uint32_t i = 0; i < n; i++) ds->y[i] = lb[i];
    free(buf); free(lb);
    fclose(fi); fclose(fl);
    return 0;
}

void dataset_synthetic(Dataset *ds, int n, int c, int h, int w, int classes, unsigned long long seed) {
    Rng r = {0xC0FFEEULL}; /* mẫu gốc cố định cho cả train và test */
    size_t sz = (size_t)c * h * w;
    float *proto = (float *)xmalloc(sizeof(float) * classes * sz);
    for (int k = 0; k < classes; k++) {
        /* mẫu gốc = vài "đốm" Gauss ở vị trí ngẫu nhiên */
        float *p = proto + k * sz;
        memset(p, 0, sizeof(float) * sz);
        for (int b = 0; b < 4; b++) {
            float cy = rng_uniform(&r) * h, cx = rng_uniform(&r) * w, s = 2.0f + 3.0f * rng_uniform(&r);
            for (int ch = 0; ch < c; ch++)
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++) {
                        float d2 = (y - cy) * (y - cy) + (x - cx) * (x - cx);
                        p[(ch * h + y) * w + x] += 2.0f * expf(-d2 / (2 * s * s));
                    }
        }
    }
    Rng rs = {seed};
    ds->n = n; ds->c = c; ds->h = h; ds->w = w; ds->classes = classes;
    ds->x = (float *)xmalloc(sizeof(float) * n * sz);
    ds->y = (int *)xmalloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        int k = (int)(rng_next(&rs) % (uint64_t)classes);
        ds->y[i] = k;
        for (size_t j = 0; j < sz; j++) ds->x[i * sz + j] = proto[k * sz + j] + 1.6f * rng_normal(&rs);
    }
    free(proto);
}

int dataset_from_args(int argc, char **argv, Dataset *tr, Dataset *te) {
    int trl = arg_int(argc, argv, "--train-limit", 0);
    int tel = arg_int(argc, argv, "--test-limit", 0);
    if (arg_flag(argc, argv, "--synthetic")) {
        dataset_synthetic(tr, trl > 0 ? trl : 12000, 1, 28, 28, 10, 1);
        dataset_synthetic(te, tel > 0 ? tel : 2000, 1, 28, 28, 10, 2);
        printf("Data: synthetic (train %d, test %d)\n", tr->n, te->n);
        return 0;
    }
    const char *dir = arg_str(argc, argv, "--data", "data/mnist");
    if (dataset_load_mnist(tr, dir, 1, trl)) return -1;
    if (dataset_load_mnist(te, dir, 0, tel)) return -1;
    printf("Data: %s (train %d, test %d, %dx%dx%d)\n", dir, tr->n, te->n, tr->c, tr->h, tr->w);
    return 0;
}

void dataset_free(Dataset *ds) {
    free(ds->x); free(ds->y);
    ds->x = NULL; ds->y = NULL;
}
