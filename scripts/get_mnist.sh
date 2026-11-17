#!/usr/bin/env bash
# Tải MNIST (hoặc Fashion-MNIST) về data/<name>/ — chạy trên máy LOCAL (có internet).
#   scripts/get_mnist.sh            -> data/mnist
#   scripts/get_mnist.sh fashion    -> data/fashion
set -euo pipefail
cd "$(dirname "$0")/.."
KIND=${1:-mnist}
FILES="train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte"
if [ "$KIND" = "fashion" ]; then
  DIR=data/fashion
  MIRRORS="http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
else
  DIR=data/mnist
  MIRRORS="https://ossci-datasets.s3.amazonaws.com/mnist https://storage.googleapis.com/cvdf-datasets/mnist"
fi
mkdir -p "$DIR"
for f in $FILES; do
  if [ -f "$DIR/$f" ]; then echo "ok   $DIR/$f"; continue; fi
  for m in $MIRRORS; do
    if curl -fL --retry 2 -o "$DIR/$f.gz" "$m/$f.gz"; then break; fi
  done
  [ -f "$DIR/$f.gz" ] || { echo "Download failed: $f"; exit 1; }
  gunzip -kf "$DIR/$f.gz"
  echo "done $DIR/$f"
done
