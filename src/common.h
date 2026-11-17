/* common.h - tiện ích dùng chung cho bản tuần tự (C) và bản song song (CUDA)
 *  - timer, RNG (splitmix64), xáo trộn, đọc tham số dòng lệnh
 */
#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#ifdef __cplusplus
extern "C" {
#endif

static inline double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static inline void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { fprintf(stderr, "Out of memory (%zu bytes)\n", n); exit(1); }
    return p;
}

/* ---------------- RNG: splitmix64 (tất định theo seed) ---------------- */
typedef struct { uint64_t s; } Rng;

static inline uint64_t rng_next(Rng *r) {
    uint64_t z = (r->s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
static inline float rng_uniform(Rng *r) {            /* [0,1) */
    return (float)(rng_next(r) >> 40) * (1.0f / 16777216.0f);
}
static inline float rng_normal(Rng *r) {             /* N(0,1), Box-Muller */
    float u1 = rng_uniform(r), u2 = rng_uniform(r);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
}
static inline void shuffle_int(int *a, int n, Rng *r) {  /* Fisher-Yates */
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(rng_next(r) % (uint64_t)(i + 1));
        int t = a[i]; a[i] = a[j]; a[j] = t;
    }
}

/* ---------------- tham số dòng lệnh: --name value ---------------- */
static inline const char *arg_str(int argc, char **argv, const char *name, const char *def) {
    for (int i = 1; i < argc - 1; i++)
        if (strcmp(argv[i], name) == 0) return argv[i + 1];
    return def;
}
static inline int arg_int(int argc, char **argv, const char *name, int def) {
    const char *s = arg_str(argc, argv, name, NULL);
    return s ? atoi(s) : def;
}
static inline double arg_double(int argc, char **argv, const char *name, double def) {
    const char *s = arg_str(argc, argv, name, NULL);
    return s ? atof(s) : def;
}
static inline int arg_flag(int argc, char **argv, const char *name) {
    for (int i = 1; i < argc; i++)
        if (strcmp(argv[i], name) == 0) return 1;
    return 0;
}
/* parse "1,2,4,8" -> mảng int, trả về số phần tử */
static inline int parse_int_list(const char *s, int *out, int maxn) {
    int n = 0;
    while (s && *s && n < maxn) {
        out[n++] = atoi(s);
        const char *c = strchr(s, ',');
        if (!c) break;
        s = c + 1;
    }
    return n;
}

#ifdef __cplusplus
}
#endif
#endif /* COMMON_H */
