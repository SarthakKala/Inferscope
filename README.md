# onnx-bench

Command-line C++17 tool for benchmarking ONNX model inference on CPU using ONNX Runtime.

## Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

## Run

```bash
./onnx-bench --model ../models/mobilenetv2.onnx --warmup 20 --runs 200 --output results.json
```

## Requirements

- CMake 3.16+
- GCC 9+ or Clang
- ONNX Runtime 1.18.0 (CPU) in `third_party/onnxruntime`
