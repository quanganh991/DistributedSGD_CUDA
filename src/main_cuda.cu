/* main_cuda.cu - chương trình SONG SONG (CUDA)
 *
 *   cnn_cuda info
 *   cnn_cuda verify                     so sánh kết quả GPU với bản CPU tuần tự
 *   cnn_cuda train [...]                xem gpu_train.cu
 *   cnn_cuda bench-conv [--batches ..] [--reps R] [--gemm tiled|naive|both] [--gpu ID] [--out F]
 *   cnn_cuda bench-fc   [--sizes ..]   [--reps R] [--gpu ID] [--out F]
 */
#include "gpu_nn.cuh"
#include "cpu_nn.h"
#include "data.h"

static void usage(void) {
    printf("Usage: cnn_cuda <info|verify|train|bench-conv|bench-fc> [options]\n"
           "  train      --data DIR | --synthetic, --mode sync|localsgd|downpour, --workers K,\n"
           "             --gpus 0[,1], --epochs 5, --batch 64 (per worker), --lr 0.05,\n"
           "             --avg-period H (localsgd, 0=1 lần/epoch), --nfetch 1 --npush 1 (downpour),\n"
           "             --gemm tiled|naive, --arch SPEC, --seed 42, --log FILE\n"
           "  verify     [--gpu ID]\n"
           "  bench-conv --batches 1,2,4,8,16,32,64,128 --reps 10 --gemm both --gpu 0 --out FILE\n"
           "  bench-fc   --sizes 256,512,1024,2048,4096 --reps 10 --gpu 0 --out FILE\n");
}

/* ------------------------------------------------------------------ */
static int cmd_info(void) {
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    int drv = 0, rt = 0;
    cudaDriverGetVersion(&drv);
    cudaRuntimeGetVersion(&rt);
    printf("CUDA driver %d.%d, runtime %d.%d, %d device(s)\n", drv / 1000, (drv % 1000) / 10, rt / 1000, (rt % 1000) / 10, n);
    for (int i = 0; i < n; i++) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, i));
        printf("  GPU%d: %s  sm_%d%d  %d SMs  %.1f GB  clock %.0f MHz  memBus %d-bit\n", i, p.name, p.major, p.minor,
               p.multiProcessorCount, p.totalGlobalMem / 1073741824.0, p.clockRate / 1000.0, p.memoryBusWidth);
    }
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (i != j) {
                int c = 0;
                cudaDeviceCanAccessPeer(&c, i, j);
                printf("  peer %d->%d: %s\n", i, j, c ? "yes" : "no");
            }
    return 0;
}

/* ------------------------------------------------------------------ */
static float *to_dev(const float *h, size_t n) {
    float *d;
    CK(cudaMalloc((void **)&d, sizeof(float) * (n ? n : 1)));
    CK(cudaMemcpy(d, h, sizeof(float) * n, cudaMemcpyHostToDevice));
    return d;
}
static float *to_host(const float *d, size_t n) {
    float *h = (float *)xmalloc(sizeof(float) * n);
    CK(cudaMemcpy(h, d, sizeof(float) * n, cudaMemcpyDeviceToHost));
    return h;
}
/* sai số tương đối: max|a-b| / max|b| */
static double rel_err(const float *a, const float *b, size_t n) {
    double md = 0, mr = 1e-12;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - b[i]);
        if (d > md) md = d;
        if (fabs(b[i]) > mr) mr = fabs(b[i]);
    }
    return md / mr;
}

static int verify_gemm(void) {
    int sizes[][3] = {{1, 1, 1}, {37, 53, 29}, {64, 64, 64}, {130, 70, 200}, {10, 500, 7}, {256, 3, 1000}, {96, 2916, 121}};
    int ns = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int fails = 0;
    Rng r = {5};
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    for (int impl = 0; impl < 2; impl++) {
        g_gemm_tiled = impl;
        for (int t = 0; t < 4; t++) {
            int tA = t >> 1, tB = t & 1;
            double worst = 0;
            for (int si = 0; si < ns; si++) {
                int M = sizes[si][0], N = sizes[si][1], K = sizes[si][2];
                float beta = (si % 2) ? 0.5f : 0.0f;
                size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;
                float *A = (float *)xmalloc(4 * na), *B = (float *)xmalloc(4 * nb), *C = (float *)xmalloc(4 * nc);
                for (size_t i = 0; i < na; i++) A[i] = rng_uniform(&r) - 0.5f;
                for (size_t i = 0; i < nb; i++) B[i] = rng_uniform(&r) - 0.5f;
                for (size_t i = 0; i < nc; i++) C[i] = rng_uniform(&r) - 0.5f;
                float *dA = to_dev(A, na), *dB = to_dev(B, nb), *dC = to_dev(C, nc);
                gemm_cpu(tA, tB, M, N, K, 1.5f, A, B, beta, C);
                gemm_gpu(tA, tB, M, N, K, 1.5f, dA, dB, beta, dC, st);
                CK(cudaStreamSynchronize(st));
                float *hC = to_host(dC, nc);
                double e = rel_err(hC, C, nc);
                if (e > worst) worst = e;
                free(A); free(B); free(C); free(hC);
                cudaFree(dA); cudaFree(dB); cudaFree(dC);
            }
            int ok = worst < 1e-4;
            fails += !ok;
            printf("  GEMM %-5s tA=%d tB=%d : max rel err %.2e  %s\n", impl ? "tiled" : "naive", tA, tB, worst, ok ? "PASS" : "FAIL");
        }
    }
    g_gemm_tiled = 1;
    cudaStreamDestroy(st);
    return fails;
}

static int verify_net(const char *arch, int c, int h, int w) {
    NetDesc d;
    if (netdesc_build(&d, arch, c, h, w, 10)) return 1;
    const int N = 32;
    printf("  Net '%s' on %dx%dx%d, batch %d, %d params\n", arch, c, h, w, N, d.nparams);
    Dataset ds;
    dataset_synthetic(&ds, N, c, h, w, 10, 9);
    CpuNet cn;
    cpu_net_create(&cn, &d, N);
    netdesc_init_params(&d, cn.params, 123);
    Rng r = {77};
    for (int i = 0; i < d.nparams; i++) cn.params[i] += 0.01f * rng_normal(&r);
    size_t in = (size_t)N * c * h * w;
    memcpy(cn.act[0], ds.x, sizeof(float) * in);

    int dev;
    CK(cudaGetDevice(&dev));
    GpuNet gn;
    gnet_create(&gn, &d, N, dev);
    CK(cudaMemcpy(gn.params, cn.params, sizeof(float) * d.nparams, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gn.act[0], ds.x, sizeof(float) * in, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gn.labels, ds.y, sizeof(int) * N, cudaMemcpyHostToDevice));

    int ccor;
    cpu_net_forward(&cn, N);
    float closs = cpu_net_loss(&cn, ds.y, N, &ccor);
    cpu_net_backward(&cn, N);

    gnet_forward(&gn, N);
    gnet_loss(&gn, N, 1);
    gnet_backward(&gn, N);
    CK(cudaDeviceSynchronize());
    double gl; long gc;
    gnet_read_stats(&gn, &gl, &gc);

    int fails = 0;
    float *glog = to_host(gn.act[d.nl], (size_t)N * d.classes);
    double e = rel_err(glog, cn.act[d.nl], (size_t)N * d.classes);
    printf("    logits     rel err %.2e %s\n", e, e < 1e-3 ? "PASS" : "FAIL");
    fails += e >= 1e-3;
    double le = fabs(gl - closs) / fabs(closs);
    printf("    loss       cpu %.6f gpu %.6f (rel %.1e) %s, correct cpu %d gpu %ld\n", closs / N, gl / N, le,
           le < 1e-3 ? "PASS" : "FAIL", ccor, gc);
    fails += le >= 1e-3;
    float *gg = to_host(gn.grads, d.nparams);
    for (int i = 0; i < d.nl; i++) {
        const Layer *l = &d.L[i];
        if (!l->nw) continue;
        double ew = rel_err(gg + l->w_off, cn.grads + l->w_off, l->nw);
        double eb = rel_err(gg + l->b_off, cn.grads + l->b_off, l->nb);
        int ok = ew < 1e-3 && eb < 1e-3;
        printf("    layer %2d %s dW rel err %.2e  db rel err %.2e  %s\n", i, l->type == L_CONV ? "CONV" : "FC  ", ew, eb,
               ok ? "PASS" : "FAIL");
        fails += !ok;
    }
    free(glog); free(gg);
    gnet_free(&gn);
    cpu_net_free(&cn);
    dataset_free(&ds);
    return fails;
}

static int cmd_verify(int argc, char **argv) {
    CK(cudaSetDevice(arg_int(argc, argv, "--gpu", 0)));
    int fails = 0;
    printf("[1] GEMM kernels vs CPU\n");
    fails += verify_gemm();
    printf("[2] CNN forward/backward GPU vs CPU\n");
    fails += verify_net("c16k5,p2,c32k5,p2,f128", 1, 28, 28);
    fails += verify_net("c8k3p1,p2,c16k3s2p1,f32", 1, 28, 28);
    fails += verify_net("c6k5s2p2,p2,f20", 3, 32, 32);
    printf("%s (%d failure(s))\n", fails ? "VERIFY FAILED" : "ALL TESTS PASSED", fails);
    return fails ? 2 : 0;
}

/* ------------------------------------------------------------------ */
/* Mục 3.2.1: Conv forward 1x224x224 -> 96 x 54x54 (11x11, stride 4) theo batch size */
static int cmd_bench_conv(int argc, char **argv) {
    CK(cudaSetDevice(arg_int(argc, argv, "--gpu", 0)));
    int bs[32];
    int nb = parse_int_list(arg_str(argc, argv, "--batches", "1,2,4,8,16,32,64,128"), bs, 32);
    int reps = arg_int(argc, argv, "--reps", 10);
    const char *g = arg_str(argc, argv, "--gemm", "both");
    const char *out = arg_str(argc, argv, "--out", NULL);
    int maxb = 0;
    for (int i = 0; i < nb; i++) if (bs[i] > maxb) maxb = bs[i];

    Layer l;
    memset(&l, 0, sizeof(l));
    l.type = L_CONV; l.in_c = 1; l.in_h = l.in_w = 224; l.out_c = 96; l.k = 11; l.stride = 4;
    l.out_h = l.out_w = 54;
    int K = 121;
    size_t P = 54 * 54, nx = (size_t)maxb * 224 * 224;
    float *hx;
    CK(cudaMallocHost((void **)&hx, sizeof(float) * nx));   /* pinned để đo H2D chuẩn */
    float *hW = (float *)xmalloc(sizeof(float) * 96 * K), *hb = (float *)xmalloc(sizeof(float) * 96);
    Rng r = {1};
    for (size_t i = 0; i < nx; i++) hx[i] = rng_uniform(&r);
    for (int i = 0; i < 96 * K; i++) hW[i] = rng_normal(&r) * 0.1f;
    for (int i = 0; i < 96; i++) hb[i] = 0.01f;
    float *x, *W = to_dev(hW, 96 * K), *b = to_dev(hb, 96), *y, *col, *tmp;
    CK(cudaMalloc((void **)&x, sizeof(float) * nx));
    CK(cudaMalloc((void **)&y, sizeof(float) * maxb * 96 * P));
    CK(cudaMalloc((void **)&col, sizeof(float) * K * maxb * P));
    CK(cudaMalloc((void **)&tmp, sizeof(float) * 96 * maxb * P));
    cudaStream_t st;
    cudaEvent_t e0, e1;
    CK(cudaStreamCreate(&st));
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));

    FILE *f = out ? fopen(out, "w") : NULL;
    if (f) fprintf(f, "impl,batch,ms,gflops,h2d_ms\n");
    printf("Conv forward (GPU): input 1x224x224, 96 filters 11x11 stride 4 -> 96x54x54\n");
    printf("%-10s %6s %10s %10s %10s\n", "impl", "batch", "ms", "GFLOP/s", "H2D ms");
    for (int impl = 0; impl < 2; impl++) {
        if (impl == 0 && !strcmp(g, "tiled")) continue;
        if (impl == 1 && !strcmp(g, "naive")) continue;
        g_gemm_tiled = impl;
        for (int i = 0; i < nb; i++) {
            int N = bs[i];
            float h2d;
            CK(cudaEventRecord(e0, st));
            CK(cudaMemcpyAsync(x, hx, sizeof(float) * N * 224 * 224, cudaMemcpyHostToDevice, st));
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            CK(cudaEventElapsedTime(&h2d, e0, e1));
            conv_forward_gpu(&l, N, x, W, b, y, col, tmp, st); /* warm-up */
            CK(cudaEventRecord(e0, st));
            for (int rr = 0; rr < reps; rr++) conv_forward_gpu(&l, N, x, W, b, y, col, tmp, st);
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= reps;
            double gf = 2.0 * 96 * K * P * N / (ms * 1e6);
            const char *nm = impl ? "gpu_tiled" : "gpu_naive";
            printf("%-10s %6d %10.4f %10.2f %10.4f\n", nm, N, ms, gf, h2d);
            if (f) fprintf(f, "%s,%d,%.5f,%.3f,%.5f\n", nm, N, ms, gf, h2d);
        }
    }
    if (f) fclose(f);
    cudaFreeHost(hx); free(hW); free(hb);
    cudaFree(x); cudaFree(W); cudaFree(b); cudaFree(y); cudaFree(col); cudaFree(tmp);
    return 0;
}

/* Mục 3.2.2: FC = mat-vec, và GEMM n x n */
static int cmd_bench_fc(int argc, char **argv) {
    CK(cudaSetDevice(arg_int(argc, argv, "--gpu", 0)));
    int ns[32];
    int nn = parse_int_list(arg_str(argc, argv, "--sizes", "256,512,1024,2048,4096"), ns, 32);
    int reps = arg_int(argc, argv, "--reps", 10);
    const char *out = arg_str(argc, argv, "--out", NULL);
    FILE *f = out ? fopen(out, "w") : NULL;
    if (f) fprintf(f, "impl,op,n,ms,gflops\n");
    cudaStream_t st;
    cudaEvent_t e0, e1;
    CK(cudaStreamCreate(&st));
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    printf("%-10s %8s %6s %10s %10s\n", "impl", "op", "n", "ms", "GFLOP/s");
    Rng r = {3};
    for (int i = 0; i < nn; i++) {
        int n = ns[i];
        size_t n2 = (size_t)n * n;
        float *hA = (float *)xmalloc(4 * n2);
        for (size_t j = 0; j < n2; j++) hA[j] = rng_uniform(&r) - 0.5f;
        float *A = to_dev(hA, n2), *B = to_dev(hA, n2), *C;
        CK(cudaMalloc((void **)&C, 4 * n2));
        for (int op = 0; op < 3; op++) {     /* 0 matvec, 1 gemm naive, 2 gemm tiled */
            int rp = op == 0 ? reps * 10 : reps;
            if (op == 1 && n > 4096) continue;
            g_gemm_tiled = (op == 2);
#define RUN() (op == 0 ? matvec_gpu(n, n, A, B, C, st) : gemm_gpu(0, 0, n, n, n, 1.0f, A, B, 0.0f, C, st))
            RUN();
            CK(cudaEventRecord(e0, st));
            for (int rr = 0; rr < rp; rr++) RUN();
            CK(cudaEventRecord(e1, st));
            CK(cudaEventSynchronize(e1));
#undef RUN
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            ms /= rp;
            double fl = op == 0 ? 2.0 * n2 : 2.0 * n2 * n;
            const char *impl = op == 0 ? "gpu" : op == 1 ? "gpu_naive" : "gpu_tiled";
            const char *opn = op == 0 ? "matvec" : "gemm";
            printf("%-10s %8s %6d %10.4f %10.2f\n", impl, opn, n, ms, fl / (ms * 1e6));
            if (f) fprintf(f, "%s,%s,%d,%.5f,%.4f\n", impl, opn, n, ms, fl / (ms * 1e6));
        }
        free(hA); cudaFree(A); cudaFree(B); cudaFree(C);
    }
    g_gemm_tiled = 1;
    if (f) fclose(f);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *cmd = argv[1];
    if (!strcmp(cmd, "info")) return cmd_info();
    if (!strcmp(cmd, "verify")) return cmd_verify(argc, argv);
    if (!strcmp(cmd, "train")) return cmd_train_gpu(argc, argv);
    if (!strcmp(cmd, "bench-conv")) return cmd_bench_conv(argc, argv);
    if (!strcmp(cmd, "bench-fc")) return cmd_bench_fc(argc, argv);
    usage();
    return 1;
}
