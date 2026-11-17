/* gpu_train.cu - Data parallelism trên GPU (mục 5 của paper)
 *
 *  K worker, mỗi worker = 1 bản sao mô hình + 1 CUDA stream riêng, đặt lên các GPU
 *  theo vòng tròn (--gpus 0,1 -> worker 0 lên GPU0, worker 1 lên GPU1, ...).
 *  GPU đầu tiên trong danh sách đóng vai "driver / parameter server".
 *  Mỗi epoch dữ liệu được xáo trộn rồi chia đều thành K shard (ShuffleSGD).
 *
 *  --mode sync      : Synchronous SGD (ADMM.SGD trong paper): mỗi bước mọi worker tính
 *                     gradient trên shard của mình -> server lấy trung bình -> phát lại
 *                     -> mọi worker cập nhật giống hệt nhau.
 *  --mode localsgd  : ParallelSGD (Zinkevich): mỗi worker chạy SGD độc lập, cứ H bước
 *                     (--avg-period H, 0 = 1 lần/epoch) thì lấy trung bình THAM SỐ.
 *  --mode downpour  : Downpour SGD (bất đồng bộ): server giữ tham số; worker lấy tham số
 *                     mỗi n_fetch bước, đẩy gradient tích luỹ mỗi n_push bước. Không có
 *                     rào đồng bộ giữa các worker: các stream/GPU chạy song song, server
 *                     cộng gradient bằng atomicAdd (lock-free, kiểu Hogwild).
 */
#include "gpu_nn.cuh"
#include "data.h"
#include "cpu_nn.h"

enum Mode { M_SYNC = 0, M_LOCAL = 1, M_DOWNPOUR = 2 };
static const char *mode_name[] = {"sync", "localsgd", "downpour"};

#define MAXDEV 16
#define MAXW 64

struct DevData { float *x; int *y; };

struct Worker {
    GpuNet net;
    int dev;
    int *idx;            /* chỉ số mẫu của shard (trên device của worker) */
    float *accrued;      /* gradient tích luỹ (downpour) */
    cudaEvent_t ev_done; /* trên device worker */
    cudaEvent_t ev_apply;/* trên device server */
    cudaStream_t ps_st;  /* stream trên server để áp dụng gradient của worker này */
};

static void worker_compute(Worker *w, int s, int B, const DevData *dd, int sz) {
    CK(cudaSetDevice(w->dev));
    gather_batch(dd->x, dd->y, w->idx + (size_t)s * B, B, sz, w->net.act[0], w->net.labels, w->net.st);
    gnet_forward(&w->net, B);
    gnet_loss(&w->net, B, 1);
    gnet_backward(&w->net, B);
}

/* AllReduce-trung-bình qua server: buf_k (trên worker) -> srv_buf[k] -> mean -> buf_k
 * Hoàn toàn bất đồng bộ phía host: thứ tự được đảm bảo bằng cudaEvent giữa các stream. */
static void allreduce_mean(Worker *W, int K, int which /*0=grads,1=params*/, float *srv_buf, float *srv_avg,
                           int sdev, cudaStream_t srv_st, cudaEvent_t ev_srv, int P) {
    for (int k = 0; k < K; k++) {
        Worker *w = &W[k];
        CK(cudaSetDevice(w->dev));
        float *src = which ? w->net.params : w->net.grads;
        copy_dev(srv_buf + (size_t)k * P, sdev, src, w->dev, P, w->net.st);
        CK(cudaEventRecord(w->ev_done, w->net.st));
    }
    CK(cudaSetDevice(sdev));
    for (int k = 0; k < K; k++) CK(cudaStreamWaitEvent(srv_st, W[k].ev_done, 0));
    vec_average(srv_buf, K, P, srv_avg, srv_st);
    CK(cudaEventRecord(ev_srv, srv_st));
    for (int k = 0; k < K; k++) {
        Worker *w = &W[k];
        CK(cudaSetDevice(w->dev));
        CK(cudaStreamWaitEvent(w->net.st, ev_srv, 0));
        float *dst = which ? w->net.params : w->net.grads;
        copy_dev(dst, w->dev, srv_avg, sdev, P, w->net.st);
    }
}

static double evaluate(GpuNet *ev, const float *src_params, int src_dev, const float *tx, const int *ty,
                       int ntest, int sz) {
    CK(cudaSetDevice(ev->dev));
    copy_dev(ev->params, ev->dev, src_params, src_dev, ev->d.nparams, ev->st);
    gnet_reset_stats(ev);
    for (int s = 0; s < ntest; s += ev->batch) {
        int N = (ntest - s < ev->batch) ? ntest - s : ev->batch;
        CK(cudaMemcpyAsync(ev->act[0], tx + (size_t)s * sz, sizeof(float) * N * sz, cudaMemcpyDeviceToDevice, ev->st));
        CK(cudaMemcpyAsync(ev->labels, ty + s, sizeof(int) * N, cudaMemcpyDeviceToDevice, ev->st));
        gnet_forward(ev, N);
        gnet_loss(ev, N, 0);
    }
    double l; long c;
    gnet_read_stats(ev, &l, &c);
    return (double)c / ntest;
}

static void sync_all(const int *devs, int nd) {
    for (int i = 0; i < nd; i++) { CK(cudaSetDevice(devs[i])); CK(cudaDeviceSynchronize()); }
}

int cmd_train_gpu(int argc, char **argv) {
    Dataset tr, te;
    if (dataset_from_args(argc, argv, &tr, &te)) return 1;

    const char *ms = arg_str(argc, argv, "--mode", "sync");
    int mode = !strcmp(ms, "sync") ? M_SYNC : !strcmp(ms, "localsgd") ? M_LOCAL : !strcmp(ms, "downpour") ? M_DOWNPOUR : -1;
    if (mode < 0) { fprintf(stderr, "Unknown --mode %s (sync|localsgd|downpour)\n", ms); return 1; }
    int K = arg_int(argc, argv, "--workers", 1);
    int B = arg_int(argc, argv, "--batch", 64);
    float lr = (float)arg_double(argc, argv, "--lr", 0.05);
    int epochs = arg_int(argc, argv, "--epochs", 5);
    int seed = arg_int(argc, argv, "--seed", 42);
    int H = arg_int(argc, argv, "--avg-period", 0);
    int nfetch = arg_int(argc, argv, "--nfetch", 1);
    int npush = arg_int(argc, argv, "--npush", 1);
    int EB = arg_int(argc, argv, "--eval-batch", 1000);
    const char *arch = arg_str(argc, argv, "--arch", "c16k5,p2,c32k5,p2,f128");
    const char *logp = arg_str(argc, argv, "--log", NULL);
    g_gemm_tiled = strcmp(arg_str(argc, argv, "--gemm", "tiled"), "naive") != 0;
    if (K < 1 || K > MAXW) { fprintf(stderr, "--workers must be 1..%d\n", MAXW); return 1; }
    if (nfetch < 1) nfetch = 1;
    if (npush < 1) npush = 1;

    int gpus[MAXDEV];
    int ng = parse_int_list(arg_str(argc, argv, "--gpus", "0"), gpus, MAXDEV);
    int ndev_avail = 0;
    CK(cudaGetDeviceCount(&ndev_avail));
    for (int i = 0; i < ng; i++)
        if (gpus[i] < 0 || gpus[i] >= ndev_avail) { fprintf(stderr, "GPU %d not available (have %d)\n", gpus[i], ndev_avail); return 1; }
    int sdev = gpus[0];
    /* danh sách device không trùng */
    int udev[MAXDEV], nu = 0;
    for (int i = 0; i < ng; i++) {
        int f = 0;
        for (int j = 0; j < nu; j++) f |= (udev[j] == gpus[i]);
        if (!f) udev[nu++] = gpus[i];
    }
    /* bật peer access nếu phần cứng hỗ trợ (nếu không, cudaMemcpyPeer tự đi qua host) */
    for (int i = 0; i < nu; i++)
        for (int j = 0; j < nu; j++) {
            if (i == j) continue;
            int can = 0;
            CK(cudaDeviceCanAccessPeer(&can, udev[i], udev[j]));
            if (can) {
                CK(cudaSetDevice(udev[i]));
                cudaError_t e = cudaDeviceEnablePeerAccess(udev[j], 0);
                if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) CK(e);
                cudaGetLastError();
            }
            printf("Peer access GPU%d -> GPU%d: %s\n", udev[i], udev[j], can ? "yes" : "no (staged via host)");
        }
    for (int i = 0; i < nu; i++) {
        cudaDeviceProp p;
        CK(cudaGetDeviceProperties(&p, udev[i]));
        printf("GPU%d: %s (sm_%d%d, %d SMs, %.1f GB)\n", udev[i], p.name, p.major, p.minor, p.multiProcessorCount,
               p.totalGlobalMem / 1073741824.0);
    }

    NetDesc d;
    if (netdesc_build(&d, arch, tr.c, tr.h, tr.w, tr.classes)) return 1;
    netdesc_print(&d, stdout);
    const int P = d.nparams, sz = tr.c * tr.h * tr.w;
    float *h_params = (float *)xmalloc(sizeof(float) * P);
    netdesc_init_params(&d, h_params, (uint64_t)seed);

    /* dữ liệu train lên mọi GPU dùng tới */
    DevData dd[MAXDEV];
    memset(dd, 0, sizeof(dd));
    for (int i = 0; i < nu; i++) {
        int dv = udev[i];
        CK(cudaSetDevice(dv));
        CK(cudaMalloc((void **)&dd[dv].x, sizeof(float) * (size_t)tr.n * sz));
        CK(cudaMalloc((void **)&dd[dv].y, sizeof(int) * tr.n));
        CK(cudaMemcpy(dd[dv].x, tr.x, sizeof(float) * (size_t)tr.n * sz, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dd[dv].y, tr.y, sizeof(int) * tr.n, cudaMemcpyHostToDevice));
    }

    const int shard = tr.n / K;
    const int steps = shard / B;
    if (steps < 1) { fprintf(stderr, "Too few samples: shard %d < batch %d\n", shard, B); return 1; }
    if (H <= 0 || H > steps) H = steps;

    Worker *W = (Worker *)calloc(K, sizeof(Worker));
    for (int k = 0; k < K; k++) {
        Worker *w = &W[k];
        w->dev = gpus[k % ng];
        gnet_create(&w->net, &d, B, w->dev);
        CK(cudaMemcpy(w->net.params, h_params, sizeof(float) * P, cudaMemcpyHostToDevice));
        CK(cudaMalloc((void **)&w->idx, sizeof(int) * shard));
        CK(cudaEventCreateWithFlags(&w->ev_done, cudaEventDisableTiming));
        if (mode == M_DOWNPOUR) {
            CK(cudaMalloc((void **)&w->accrued, sizeof(float) * P));
            CK(cudaMemset(w->accrued, 0, sizeof(float) * P));
        }
        CK(cudaSetDevice(sdev));
        CK(cudaEventCreateWithFlags(&w->ev_apply, cudaEventDisableTiming));
        CK(cudaStreamCreateWithFlags(&w->ps_st, cudaStreamNonBlocking));
    }

    /* server */
    CK(cudaSetDevice(sdev));
    cudaStream_t srv_st;
    cudaEvent_t ev_srv;
    float *srv_buf, *srv_avg, *ps;
    CK(cudaStreamCreateWithFlags(&srv_st, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&ev_srv, cudaEventDisableTiming));
    CK(cudaMalloc((void **)&srv_buf, sizeof(float) * (size_t)K * P));
    CK(cudaMalloc((void **)&srv_avg, sizeof(float) * P));
    CK(cudaMalloc((void **)&ps, sizeof(float) * P));
    CK(cudaMemcpy(ps, h_params, sizeof(float) * P, cudaMemcpyHostToDevice));
    GpuNet ev;
    gnet_create(&ev, &d, EB, sdev);
    float *tx; int *ty;
    CK(cudaSetDevice(sdev));
    CK(cudaMalloc((void **)&tx, sizeof(float) * (size_t)te.n * sz));
    CK(cudaMalloc((void **)&ty, sizeof(int) * te.n));
    CK(cudaMemcpy(tx, te.x, sizeof(float) * (size_t)te.n * sz, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ty, te.y, sizeof(int) * te.n, cudaMemcpyHostToDevice));
    sync_all(udev, nu);

    printf("Training (CUDA): mode=%s workers=%d gpus=%s batch/worker=%d lr=%g epochs=%d gemm=%s\n",
           mode_name[mode], K, arg_str(argc, argv, "--gpus", "0"), B, lr, epochs, g_gemm_tiled ? "tiled" : "naive");
    printf("  shard=%d samples/worker, %d steps/worker/epoch", shard, steps);
    if (mode == M_LOCAL) printf(", average every %d steps", H);
    if (mode == M_DOWNPOUR) printf(", n_fetch=%d n_push=%d", nfetch, npush);
    printf("\n");

    FILE *lf = logp ? fopen(logp, "w") : NULL;
    if (lf) fprintf(lf, "# impl=cuda mode=%s workers=%d gpus=%s batch=%d lr=%g H=%d nfetch=%d npush=%d gemm=%s arch=%s\n"
                        "epoch,epoch_time_s,total_time_s,train_loss,train_acc,test_acc,img_per_s\n",
                    mode_name[mode], K, arg_str(argc, argv, "--gpus", "0"), B, lr, H, nfetch, npush,
                    g_gemm_tiled ? "tiled" : "naive", arch);

    int *perm = (int *)xmalloc(sizeof(int) * tr.n);
    for (int i = 0; i < tr.n; i++) perm[i] = i;
    Rng rng = {(uint64_t)seed * 7919 + 1};
    double total = 0;
    long comm_rounds = 0;

    for (int ep = 1; ep <= epochs; ep++) {
        /* ShuffleSGD: xáo trộn toàn cục rồi chia shard cho từng worker */
        shuffle_int(perm, tr.n, &rng);
        for (int k = 0; k < K; k++) {
            CK(cudaSetDevice(W[k].dev));
            CK(cudaMemcpy(W[k].idx, perm + (size_t)k * shard, sizeof(int) * shard, cudaMemcpyHostToDevice));
            gnet_reset_stats(&W[k].net);
        }
        sync_all(udev, nu);
        double t0 = now_sec();

        for (int s = 0; s < steps; s++) {
            if (mode == M_SYNC) {
                for (int k = 0; k < K; k++) worker_compute(&W[k], s, B, &dd[W[k].dev], sz);
                if (K > 1) { allreduce_mean(W, K, 0, srv_buf, srv_avg, sdev, srv_st, ev_srv, P); comm_rounds++; }
                for (int k = 0; k < K; k++) {
                    CK(cudaSetDevice(W[k].dev));
                    vec_sgd(W[k].net.params, W[k].net.grads, P, lr, W[k].net.st);
                }
            } else if (mode == M_LOCAL) {
                for (int k = 0; k < K; k++) {
                    worker_compute(&W[k], s, B, &dd[W[k].dev], sz);
                    vec_sgd(W[k].net.params, W[k].net.grads, P, lr, W[k].net.st);
                }
                if (K > 1 && ((s + 1) % H == 0 || s == steps - 1)) {
                    allreduce_mean(W, K, 1, srv_buf, srv_avg, sdev, srv_st, ev_srv, P);
                    comm_rounds++;
                }
            } else { /* DOWNPOUR */
                for (int k = 0; k < K; k++) {
                    Worker *w = &W[k];
                    CK(cudaSetDevice(w->dev));
                    if (s % nfetch == 0)   /* StartAsynchronouslyFetchingParameters */
                        copy_dev(w->net.params, w->dev, ps, sdev, P, w->net.st);
                    worker_compute(w, s, B, &dd[w->dev], sz);
                    vec_sgd(w->net.params, w->net.grads, P, lr, w->net.st);   /* cập nhật cục bộ */
                    vec_axpy(w->accrued, w->net.grads, P, 1.0f, w->net.st);   /* tích luỹ gradient */
                    if ((s + 1) % npush == 0 || s == steps - 1) {             /* StartAsynchronouslyPushingGradients */
                        CK(cudaStreamWaitEvent(w->net.st, w->ev_apply, 0));   /* slot srv_buf[k] đã rảnh */
                        copy_dev(srv_buf + (size_t)k * P, sdev, w->accrued, w->dev, P, w->net.st);
                        CK(cudaMemsetAsync(w->accrued, 0, sizeof(float) * P, w->net.st));
                        CK(cudaEventRecord(w->ev_done, w->net.st));
                        CK(cudaSetDevice(sdev));
                        CK(cudaStreamWaitEvent(w->ps_st, w->ev_done, 0));
                        vec_atomic_axpy(ps, srv_buf + (size_t)k * P, P, -lr, w->ps_st);
                        CK(cudaEventRecord(w->ev_apply, w->ps_st));
                        comm_rounds++;
                    }
                }
            }
        }
        sync_all(udev, nu);
        double et = now_sec() - t0;
        total += et;

        double lsum = 0; long csum = 0;
        for (int k = 0; k < K; k++) {
            double l; long c;
            gnet_read_stats(&W[k].net, &l, &c);
            lsum += l; csum += c;
        }
        double nseen = (double)K * steps * B;
        const float *src = (mode == M_DOWNPOUR) ? ps : W[0].net.params;
        int src_dev = (mode == M_DOWNPOUR) ? sdev : W[0].dev;
        double acc = evaluate(&ev, src, src_dev, tx, ty, te.n, sz);
        printf("Epoch %d: time %.3fs  loss %.4f  train_acc %.4f  test_acc %.4f  (%.0f img/s)\n",
               ep, et, lsum / nseen, csum / nseen, acc, nseen / et);
        if (lf) { fprintf(lf, "%d,%.5f,%.5f,%.5f,%.5f,%.5f,%.1f\n", ep, et, total, lsum / nseen, csum / nseen, acc, nseen / et); fflush(lf); }
    }
    printf("TOTAL train time: %.3fs, communication rounds: %ld\n", total, comm_rounds);
    if (lf) fclose(lf);

    /* dọn dẹp */
    for (int k = 0; k < K; k++) {
        Worker *w = &W[k];
        gnet_free(&w->net);
        CK(cudaSetDevice(w->dev));
        cudaFree(w->idx);
        if (w->accrued) cudaFree(w->accrued);
        cudaEventDestroy(w->ev_done);
        CK(cudaSetDevice(sdev));
        cudaEventDestroy(w->ev_apply);
        cudaStreamDestroy(w->ps_st);
    }
    gnet_free(&ev);
    CK(cudaSetDevice(sdev));
    cudaFree(srv_buf); cudaFree(srv_avg); cudaFree(ps); cudaFree(tx); cudaFree(ty);
    cudaStreamDestroy(srv_st); cudaEventDestroy(ev_srv);
    for (int i = 0; i < nu; i++) { CK(cudaSetDevice(udev[i])); cudaFree(dd[udev[i]].x); cudaFree(dd[udev[i]].y); }
    free(W); free(perm); free(h_params);
    dataset_free(&tr); dataset_free(&te);
    return 0;
}
