/* data.h - nạp MNIST / Fashion-MNIST (định dạng IDX) hoặc sinh dữ liệu tổng hợp */
#ifndef DATA_H
#define DATA_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int n, c, h, w, classes;
    float *x;   /* n * c*h*w, đã chuẩn hoá */
    int *y;     /* n nhãn */
} Dataset;

/* dir chứa train-images-idx3-ubyte, train-labels-idx1-ubyte,
 * t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte (đã giải nén).
 * limit > 0: chỉ lấy limit mẫu đầu. Trả về 0 nếu thành công. */
int  dataset_load_mnist(Dataset *ds, const char *dir, int train, int limit);

/* Dữ liệu tổng hợp 1x28x28, 10 lớp: mỗi lớp là 1 "mẫu gốc" ngẫu nhiên + nhiễu.
 * Dùng để test nhanh khi chưa có MNIST. */
void dataset_synthetic(Dataset *ds, int n, int c, int h, int w, int classes, unsigned long long seed);

void dataset_free(Dataset *ds);

/* đọc --synthetic | --data DIR, --train-limit, --test-limit */
int  dataset_from_args(int argc, char **argv, Dataset *train, Dataset *test);

#ifdef __cplusplus
}
#endif
#endif
