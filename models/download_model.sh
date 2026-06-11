#!/bin/bash
set -e
mkdir -p models
echo "Downloading MobileNetV2 ONNX model..."
wget -q --show-progress \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx" \
  -O models/mobilenetv2.onnx
echo "Done. Model saved to models/mobilenetv2.onnx"
