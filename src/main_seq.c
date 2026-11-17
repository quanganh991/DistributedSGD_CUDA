/* main_seq.c - chương trình TUẦN TỰ (C, 1 lõi CPU)
 *
 *   cnn_seq train      [--data DIR | --synthetic] [--epochs E] [--batch B] [--lr LR]
 *                      [--arch SPEC] [--seed S] [--train-limit N] [--log FILE]
 *   cnn_seq bench-conv [--batches 1,2,4,...] [--reps R] [--out FILE]
 *   cnn_seq bench-fc   [--sizes 256,512,...] [--gemm-max N] [--reps R] [--out FILE]
 *   cnn_seq gradcheck
 */
#include "common.h"
#include "net_desc.h"
#include "data.h"
#include "cpu_nn.h"

#define DEFAULT_ARCH "c16k5,p2,c32k5,p2,f128"

static void usage(void) {
    printf("Usage: cnn_seq <train|bench-conv|bench-fc|gradcheck> [options]\n"
           "  train      --data DIR | --synthetic, --epochs 3, --batch 64, --lr 0.05,\n"
           "             --arch " DEFAULT_ARCH ", --seed 42, --train-limit N, --log FILE\n"
           "  bench-conv --batches 1,2,4,8,16,32,64,128 --reps 3 --out FILE\n"
           "             (conv của paper: 1x224x224, 96 filter 11x11, stride 4)\n"
           "  bench-fc   --sizes 256,512,1024,2048,4096 --gemm-max 2048 --reps 5 --out FILE\n"
           "  gradcheck  (kiểm tra backprop bằng sai phân hữu hạn)\n");
}

/* ------------------------------------------------------------------ */
static double evaluate(CpuNet *net, const Dataset *te) {
    int B = net->batch, sz = te->c * te->h * te->w, correct = 0;
    for (int s = 0; s < te->n; s += B) {
        int N = (te->n - s < B) ? te->n - s : B, c = 0;
        memcpy(net->act[0], te->x + (size_t)s * sz, sizeof(float) * N * sz);
        cpu_net_forward(net, N);
        cpu_net_loss(net, te->y + s, N, &c);
        correct += c;
    }
    return (double)correct / te->n;
}

static int cmd_train(int argc, char **argv) {
    Dataset tr, te;
    if (dataset_from_args(argc, argv, &tr, &te)) return 1;
    int epochs = arg_int(argc, argv, "--epochs", 3);
    int B = arg_int(argc, argv, "--batch", 64);
    float lr = (float)arg_double(argc, argv, "--lr", 0.05);
    int seed = arg_int(argc, argv, "--seed", 42);
    const char *arch = arg_str(argc, argv, "--arch", DEFAULT_ARCH);
    const char *logf_ = arg_str(argc, argv, "--log", NULL);

    NetDesc d;
    if (netdesc_build(&d, arch, tr.c, tr.h, tr.w, tr.classes)) return 1;
    netdesc_print(&d, stdout);
    CpuNet net;
    cpu_net_create(&net, &d, B);
    netdesc_init_params(&d, net.params, (uint64_t)seed);

    FILE *lf = logf_ ? fopen(logf_, "w") : NULL;
    if (lf) fprintf(lf, "# impl=cpu_seq mode=single workers=1 batch=%d lr=%g arch=%s\n"
                        "epoch,epoch_time_s,total_time_s,train_loss,train_acc,test_acc,img_per_s\n",
                    B, lr, arch);
    printf("Training (sequential CPU): epochs=%d batch=%d lr=%g\n", epochs, B, lr);

    int sz = tr.c * tr.h * tr.w;
    int *perm = (int *)xmalloc(sizeof(int) * tr.n);
    int *yb = (int *)xmalloc(sizeof(int) * B);
    for (int i = 0; i < tr.n; i++) perm[i] = i;
    Rng rng = {(uint64_t)seed * 7919 + 1};
    int steps = tr.n / B;
    double total = 0;

    for (int ep = 1; ep <= epochs; ep++) {
        shuffle_int(perm, tr.n, &rng);
        double t0 = now_sec(), loss = 0;
        long corr = 0;
        for (int s = 0; s < steps; s++) {
            for (int i = 0; i < B; i++) {
                int id = perm[s * B + i];
                memcpy(net.act[0] + (size_t)i * sz, tr.x + (size_t)id * sz, sizeof(float) * sz);
                yb[i] = tr.y[id];
            }
            int c;
            cpu_net_forward(&net, B);
            loss += cpu_net_loss(&net, yb, B, &c);
            corr += c;
            cpu_net_backward(&net, B);
            cpu_sgd(net.params, net.grads, d.nparams, lr);
            if ((s + 1) % 100 == 0) {
                printf("\r  epoch %d step %d/%d loss %.4f", ep, s + 1, steps, loss / ((s + 1.0) * B));
                fflush(stdout);
            }
        }
        double et = now_sec() - t0;
        total += et;
        double acc = evaluate(&net, &te);
        double tl = loss / ((double)steps * B), ta = (double)corr / ((double)steps * B);
        printf("\rEpoch %d: time %.2fs  loss %.4f  train_acc %.4f  test_acc %.4f  (%.0f img/s)\n",
               ep, et, tl, ta, acc, steps * B / et);
        if (lf) { fprintf(lf, "%d,%.4f,%.4f,%.5f,%.5f,%.5f,%.1f\n", ep, et, total, tl, ta, acc, steps * B / et); fflush(lf); }
    }
    printf("TOTAL train time: %.2fs\n", total);
    if (lf) fclose(lf);
    free(perm); free(yb);
    cpu_net_free(&net);
    dataset_free(&tr); dataset_free(&te);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Thí nghiệm mục 3.2.1 của paper: thời gian forward conv theo batch size */
static int cmd_bench_conv(int argc, char **argv) {
    int bs[32];
    int nb = parse_int_list(arg_str(argc, argv, "--batches", "1,2,4,8,16,32,64,128"), bs, 32);
    int reps = arg_int(argc, argv, "--reps", 3);
    const char *out = arg_str(argc, argv, "--out", NULL);
    int maxb = 0;
    for (int i = 0; i < nb; i++) if (bs[i] > maxb) maxb = bs[i];

    Layer l; memset(&l, 0, sizeof(l));
    l.type = L_CONV; l.in_c = 1; l.in_h = l.in_w = 224; l.out_c = 96; l.k = 11; l.stride = 4; l.pad = 0;
    l.out_h = l.out_w = (224 - 11) / 4 + 1;  /* 54 */
    int K = l.in_c * l.k * l.k;
    size_t P = (size_t)l.out_h * l.out_w;
    float *x = (float *)xmalloc(sizeof(float) * maxb * 224 * 224);
    float *W = (float *)xmalloc(sizeof(float) * l.out_c * K);
    float *b = (float *)xmalloc(sizeof(float) * l.out_c);
    float *y = (float *)xmalloc(sizeof(float) * maxb * l.out_c * P);
    float *col = (float *)xmalloc(sizeof(float) * K * maxb * P);
    float *tmp = (float *)xmalloc(sizeof(float) * l.out_c * maxb * P);
    Rng r = {1};
    for (size_t i = 0; i < (size_t)maxb * 224 * 224; i++) x[i] = rng_uniform(&r);
    for (int i = 0; i < l.out_c * K; i++) W[i] = rng_normal(&r) * 0.1f;
    for (int i = 0; i < l.out_c; i++) b[i] = 0.01f;

    FILE *f = out ? fopen(out, "w") : NULL;
    if (f) fprintf(f, "impl,batch,ms,gflops\n");
    printf("Conv forward (CPU 1 core): input 1x224x224, 96 filters 11x11 stride 4 -> 96x54x54\n");
    printf("%8s %12s %10s\n", "batch", "ms", "GFLOP/s");
    for (int i = 0; i < nb; i++) {
        int N = bs[i];
        conv_forward_cpu(&l, N, x, W, b, y, col, tmp); /* warm-up */
        double t0 = now_sec();
        for (int rr = 0; rr < reps; rr++) conv_forward_cpu(&l, N, x, W, b, y, col, tmp);
        double ms = (now_sec() - t0) * 1000.0 / reps;
        double gf = 2.0 * l.out_c * K * P * N / (ms * 1e6);
        printf("%8d %12.3f %10.3f\n", N, ms, gf);
        if (f) fprintf(f, "cpu,%d,%.4f,%.4f\n", N, ms, gf);
    }
    if (f) fclose(f);
    free(x); free(W); free(b); free(y); free(col); free(tmp);
    return 0;
}

/* Thí nghiệm mục 3.2.2: FC layer = mat-vec, và GEMM theo kích thước ma trận */
static int cmd_bench_fc(int argc, char **argv) {
    int ns[32];
    int nn = parse_int_list(arg_str(argc, argv, "--sizes", "256,512,1024,2048,4096"), ns, 32);
    int gmax = arg_int(argc, argv, "--gemm-max", 2048);
    int reps = arg_int(argc, argv, "--reps", 5);
    const char *out = arg_str(argc, argv, "--out", NULL);
    FILE *f = out ? fopen(out, "w") : NULL;
    if (f) fprintf(f, "impl,op,n,ms,gflops\n");
    printf("%6s %8s %12s %10s\n", "n", "op", "ms", "GFLOP/s");
    Rng r = {3};
    for (int i = 0; i < nn; i++) {
        int n = ns[i];
        size_t nn2 = (size_t)n * n;
        float *A = (float *)xmalloc(sizeof(float) * nn2), *B = (float *)xmalloc(sizeof(float) * nn2);
        float *C = (float *)xmalloc(sizeof(float) * nn2);
        for (size_t j = 0; j < nn2; j++) { A[j] = rng_uniform(&r) - 0.5f; B[j] = rng_uniform(&r) - 0.5f; }
        /* mat-vec */
        int mvreps = reps * 10;
        matvec_cpu(n, n, A, B, C);
        double t0 = now_sec();
        for (int rr = 0; rr < mvreps; rr++) matvec_cpu(n, n, A, B, C);
        double ms = (now_sec() - t0) * 1000.0 / mvreps;
        printf("%6d %8s %12.4f %10.3f\n", n, "matvec", ms, 2.0 * nn2 / (ms * 1e6));
        if (f) fprintf(f, "cpu,matvec,%d,%.5f,%.4f\n", n, ms, 2.0 * nn2 / (ms * 1e6));
        /* gemm */
        if (n <= gmax) {
            int greps = n >= 1024 ? 1 : reps;
            t0 = now_sec();
            for (int rr = 0; rr < greps; rr++) gemm_cpu(0, 0, n, n, n, 1.0f, A, B, 0.0f, C);
            ms = (now_sec() - t0) * 1000.0 / greps;
            double gf = 2.0 * nn2 * n / (ms * 1e6);
            printf("%6d %8s %12.4f %10.3f\n", n, "gemm", ms, gf);
            if (f) fprintf(f, "cpu,gemm,%d,%.5f,%.4f\n", n, ms, gf);
        }
        free(A); free(B); free(C);
    }
    if (f) fclose(f);
    return 0;
}

/* ------------------------------------------------------------------ */
/* Kiểm tra gradient: so sánh backprop với sai phân trung tâm */
static int cmd_gradcheck(int argc, char **argv) {
    (void)argc; (void)argv;
    NetDesc d;
    if (netdesc_build(&d, "c4k3p1,p2,c6k3s2,f12", 2, 10, 10, 5)) return 1;
    netdesc_print(&d, stdout);
    int N = 4;
    CpuNet net;
    cpu_net_create(&net, &d, N);
    netdesc_init_params(&d, net.params, 7);
    Rng r = {11};
    for (int i = 0; i < d.nparams; i++) net.params[i] += 0.05f * rng_normal(&r); /* bias != 0 */
    size_t in = (size_t)N * d.in_c * d.in_h * d.in_w;
    for (size_t i = 0; i < in; i++) net.act[0][i] = rng_normal(&r);
    int y[4] = {0, 3, 1, 4};

    cpu_net_forward(&net, N);
    cpu_net_loss(&net, y, N, NULL);
    cpu_net_backward(&net, N);

    double max_rel = 0;
    const float eps = 1e-3f;
    int nchk = 0;
    for (int li = 0; li < d.nl; li++) {
        const Layer *l = &d.L[li];
        if (!l->nw) continue;
        for (int t = 0; t < 12; t++) {
            int id = (t < 9) ? l->w_off + (int)(rng_next(&r) % (uint64_t)l->nw)
                             : l->b_off + (int)(rng_next(&r) % (uint64_t)l->nb);
            float old = net.params[id];
            net.params[id] = old + eps;
            cpu_net_forward(&net, N);
            double lp = cpu_net_loss(&net, y, N, NULL) / N;
            net.params[id] = old - eps;
            cpu_net_forward(&net, N);
            double lm = cpu_net_loss(&net, y, N, NULL) / N;
            net.params[id] = old;
            double num = (lp - lm) / (2 * eps), ana = net.grads[id];
            double rel = fabs(num - ana) / fmax(1e-3, fabs(num) + fabs(ana));
            if (rel > 1e-2) printf("  layer %d id %d num %.6f ana %.6f rel %.3e\n", li, id, num, ana, rel);
            if (rel > max_rel) max_rel = rel;
            nchk++;
        }
    }
    printf("gradcheck: %d params checked, max relative error = %.3e -> %s\n",
           nchk, max_rel, max_rel < 2e-2 ? "PASS" : "FAIL");
    cpu_net_free(&net);
    return max_rel < 2e-2 ? 0 : 2;
}

int main(int argc, char **argv) {
    if (argc < 2) { usage(); return 1; }
    const char *cmd = argv[1];
    if (!strcmp(cmd, "train")) return cmd_train(argc, argv);
    if (!strcmp(cmd, "bench-conv")) return cmd_bench_conv(argc, argv);
    if (!strcmp(cmd, "bench-fc")) return cmd_bench_fc(argc, argv);
    if (!strcmp(cmd, "gradcheck")) return cmd_gradcheck(argc, argv);
    usage();
    return 1;
}
