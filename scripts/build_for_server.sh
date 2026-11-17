#!/usr/bin/env bash
# Build binary trên máy LOCAL (RTX 2060) để mang lên server (GTX 1080 / 1080 Ti, Ubuntu 22.04).
#
# Build trong container nvidia/cuda:*-ubuntu22.04 để binary chỉ đòi glibc <= 2.35
# (build thẳng trên Ubuntu 24.04 sẽ đòi GLIBC_2.38 -> server báo lỗi "GLIBC_2.38 not found").
# Container chỉ dùng để BIÊN DỊCH, không cần GPU.
#
#   scripts/build_for_server.sh                 # CUDA 12.4 (cần driver server >= 525)
#   CUDA_IMAGE=nvidia/cuda:11.8.0-devel-ubuntu22.04 scripts/build_for_server.sh   # driver >= 450.80
#   NATIVE=1 scripts/build_for_server.sh        # build thẳng bằng nvcc trên máy (không Docker)
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.4.1-devel-ubuntu22.04}

if [ "${NATIVE:-0}" = "1" ]; then
  make clean && make -j"$(nproc)" all
else
  echo ">> Building in $IMAGE"
  docker run --rm -u "$(id -u):$(id -g)" -v "$PWD":/src -w /src "$IMAGE" \
    bash -c 'make clean && make -j$(nproc) all && make check'
fi

echo
echo ">> GLIBC tối đa binary cần (server Ubuntu 22.04 có 2.35):"
NEED=$(objdump -T bin/cnn_cuda bin/cnn_seq | grep -o 'GLIBC_[0-9.]*' | sort -V | uniq | tail -1)
echo "$NEED"
if [ "$(printf '%s\nGLIBC_2.35\n' "$NEED" | sort -V | tail -1)" != "GLIBC_2.35" ]; then
  echo "!! CẢNH BÁO: binary cần $NEED > GLIBC_2.35 -> sẽ KHÔNG chạy trên server Ubuntu 22.04. Hãy build bằng Docker (bỏ NATIVE=1)."
fi

# Đóng gói: chỉ binary + script chạy + dữ liệu (KHÔNG có source code)
PKG=dist/dsgd_nhom9.tar.gz
mkdir -p dist
DATA_FILES=""
for d in data/mnist data/fashion; do
  [ -d "$d" ] && DATA_FILES="$DATA_FILES $(ls $d/*.gz 2>/dev/null | tr '\n' ' ')"
done
tar czf "$PKG" bin/cnn_seq bin/cnn_cuda scripts/run_experiments.sh $DATA_FILES
echo ">> Package: $PKG ($(du -h "$PKG" | cut -f1))"
tar tzf "$PKG"
