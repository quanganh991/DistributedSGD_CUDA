#!/usr/bin/env bash
# Chạy toàn bộ thí nghiệm trên SERVER (hoặc local). Kết quả CSV vào results/.
#   GPU=0 GPUS2=0,1 EPOCHS=10 ./scripts/run_experiments.sh
#   ONLY=bench ./scripts/run_experiments.sh    # chỉ benchmark (bench|train|dist|multigpu)
# Server dùng chung -> xem nvidia-smi, chọn GPU đang rảnh bằng biến GPU.
set -u -o pipefail
cd "$(dirname "$0")/.."
BIN=./bin
DATA=${DATA:-data/mnist}
OUT=${OUT:-results}
GPU=${GPU:-0}
GPUS2=${GPUS2:-0,1}
EPOCHS=${EPOCHS:-10}
LR=${LR:-0.05}
B=${BATCH:-64}
ONLY=${ONLY:-all}
mkdir -p "$OUT"
for f in "$DATA"/*.gz; do [ -f "$f" ] && { [ -f "${f%.gz}" ] || gunzip -kf "$f"; }; done
run() { echo; echo "\$ $*"; "$@"; }
want() { [ "$ONLY" = all ] || [ "$ONLY" = "$1" ]; }

run $BIN/cnn_cuda info | tee "$OUT/info.txt"
run $BIN/cnn_cuda verify --gpu "$GPU" | tee "$OUT/verify.txt" || { echo "VERIFY FAILED - dừng"; exit 1; }

TR="--data $DATA --lr $LR --batch $B"

if want bench; then   # ---- Mục 3 của paper: CPU vs GPU ----
  run $BIN/cnn_seq  bench-conv --reps 2 --out "$OUT/conv_cpu.csv"
  run $BIN/cnn_cuda bench-conv --gpu "$GPU" --out "$OUT/conv_gpu.csv"
  run $BIN/cnn_seq  bench-fc --out "$OUT/fc_cpu.csv"
  run $BIN/cnn_cuda bench-fc --gpu "$GPU" --sizes 256,512,1024,2048,4096,8192 --out "$OUT/fc_gpu.csv"
fi

if want train; then   # ---- baseline huấn luyện ----
  run $BIN/cnn_seq  train $TR --epochs 1 --log "$OUT/train_cpu_seq.csv"
  run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPU" --gemm naive --log "$OUT/train_gpu_naive.csv"
  run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPU" --log "$OUT/train_gpu_single.csv"
fi

if want dist; then    # ---- Mục 5: data parallelism, K worker trên 1 GPU ----
  for K in 2 4 8; do
    run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPU" --mode sync     --workers $K --log "$OUT/sync_k$K.csv"
    run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPU" --mode localsgd --workers $K --log "$OUT/psgd_epoch_k$K.csv"
    run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPU" --mode localsgd --workers $K --avg-period 10 --log "$OUT/psgd_h10_k$K.csv"
    run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPU" --mode downpour --workers $K --log "$OUT/downpour_k$K.csv"
  done
  run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPU" --mode downpour --workers 4 --nfetch 5 --npush 5 --log "$OUT/downpour_f5p5_k4.csv"
fi

if want multigpu; then  # ---- 2 GPU (1080 Ti + 1080) ----
  NG=$($BIN/cnn_cuda info | grep -c "^  GPU[0-9]")
  if [ "$NG" -ge 2 ]; then
    run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPUS2" --mode sync     --workers 2 --log "$OUT/sync_k2_2gpu.csv"
    run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPUS2" --mode sync     --workers 4 --log "$OUT/sync_k4_2gpu.csv"
    run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPUS2" --mode localsgd --workers 2 --log "$OUT/psgd_epoch_k2_2gpu.csv"
    run $BIN/cnn_cuda train $TR --epochs "$EPOCHS" --gpus "$GPUS2" --mode downpour --workers 2 --log "$OUT/downpour_k2_2gpu.csv"
  else
    echo "Chỉ có $NG GPU - bỏ qua thí nghiệm multi-GPU"
  fi
fi
echo; echo "Done. Kết quả trong $OUT/ — kéo về local: REMOTE=... scripts/deploy.sh fetch"
