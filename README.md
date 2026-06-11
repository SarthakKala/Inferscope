# inferscope

inferscope is a command-line tool that benchmarks how fast an ONNX model runs on your CPU. Point it at any `.onnx` file, and it runs inference hundreds of times, throws away the warmup runs, and gives you real latency stats — p50, p90, p99, stddev, throughput — printed to the terminal and saved as JSON. No GPU, no Python, no cloud. Just C++ and ONNX Runtime.

## 🛠️ Technologies

* C++17
* CMake 3.16+
* ONNX Runtime 1.18.0 (CPU only)
* `std::chrono` for per-run timing
* Hand-built JSON output (no external libraries)

## ✨ Features

* Load any `.onnx` model from disk and run CPU inference immediately
* Warmup phase discards the first N slow runs so your numbers are stable
* Measures p50, p90, p99, mean, stddev, min, max, and throughput (inferences/sec)
* Prints input/output names, shapes, and tensor types from the loaded graph
* Writes machine-readable results to a JSON file for scripts and dashboards
* Works with dynamic input shapes — batch size is auto-filled to 1
* Uses random dummy inputs so you're measuring speed, not prediction accuracy

## 📊 Why Percentiles, Not Just an Average

Most tutorials time one inference call and call it a day. That number lies. The first few runs are always slower (memory allocation, kernel warmup), and a single spike can drag the average up while your typical frame time stays fine. inferscope discards warmup runs, times every inference individually, and reports percentiles so you can see typical latency (p50) and worst-case frame budget (p99) — the numbers that actually matter when you're asking "can this chip run this model in real time?"

## 🔧 Process

The tool loads your model once through ONNX Runtime and keeps the session alive. It inspects the graph to figure out input shapes and types, generates random float tensors to match, then runs inference in two loops — warmup (untimed) and measurement (timed with `high_resolution_clock`). Each timed run feeds into a profiler that sorts the latency vector and computes percentiles. Finally, a reporter prints a human-readable summary and writes the full result to JSON.

```
CLI → SessionWrapper → GraphInspector + BenchmarkRunner → Profiler → Reporter → stdout + results.json
```

## 📚 What I Learned

* **ONNX Runtime C++ API** — loading models, creating tensor views, running sessions, and reading graph metadata without Python
* **Why warmup runs exist** — the first N inferences are not representative; benchmarking without discarding them gives inflated numbers
* **Percentile statistics for latency** — p50 vs p99 tells a completely different story than a single average
* **CMake with prebuilt shared libraries** — linking against `libonnxruntime.so` and copying it next to the binary for runtime loading
* **Dynamic ONNX shapes** — models with `-1` batch dimensions need explicit resolution before allocating input tensors

## 🌱 Overall Growth

This project sits at the intersection of systems programming and ML infrastructure — the layer where a model file on disk becomes a number you can trust on a specific piece of hardware. Building it meant going past "run inference once in a notebook" and into the territory of how production teams actually validate whether a model meets a latency budget.

## 🚀 Running the Project

**Prerequisites:** Linux (Ubuntu 20.04+), GCC 9+ or Clang 10+, CMake 3.16+

```bash
sudo apt update && sudo apt install -y build-essential cmake wget tar git
```

```bash
git clone https://github.com/SarthakKala/inferscope
cd inferscope

# Download ONNX Runtime (CPU, no CUDA)
mkdir -p third_party
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz \
  -O third_party/onnxruntime.tgz
tar -xzf third_party/onnxruntime.tgz -C third_party/
mv third_party/onnxruntime-linux-x64-1.18.0 third_party/onnxruntime
rm third_party/onnxruntime.tgz

# Download a test model (MobileNetV2)
chmod +x models/download_model.sh && ./models/download_model.sh

# Build
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run
./inferscope --model ../models/mobilenetv2.onnx --warmup 20 --runs 200 --output results.json
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to `.onnx` model file |
| `--warmup` | `20` | Untimed runs before measurement |
| `--runs` | `200` | Timed runs for statistics |
| `--output` | `results.json` | JSON output path |
| `--help` | — | Print usage |
