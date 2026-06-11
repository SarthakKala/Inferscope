# inferscope

A command-line profiler for ONNX models: load any `.onnx` file, run inference across configurable warmup and measurement rounds, and get statistically rigorous latency numbers — p50, p90, p99, stddev, throughput — written to stdout and a JSON file. No GPU required. CPU execution provider only.

---

## Architecture

```
CLI (--model · --warmup · --runs · --output)
               │
               ▼
       SessionWrapper
   Ort::Env + Ort::Session
    loads model from disk,
    owns session lifecycle
               │
       ┌───────┴────────┐
       ▼                ▼
 GraphInspector    BenchmarkRunner
 input/output      warmup loop (discarded)
 names · shapes    measurement loop
 tensor types            │
                         ▼
                      Profiler
                 p50 · p90 · p99
                 mean · stddev · min · max
                         │
                         ▼
                      Reporter
                 ┌──────┴──────┐
                 ▼             ▼
           stdout           results.json
          summary         machine-readable
```

---

## Tech stack

| Layer | Technology | Role |
|-------|------------|------|
| Language | C++17 | Entire codebase |
| Build system | CMake 3.16+ | Compilation and linking |
| ML runtime | ONNX Runtime 1.18.0 | Model loading, session management, inference |
| Execution provider | CPU (default) | No GPU dependency |
| Timing | `std::chrono::high_resolution_clock` | Sub-millisecond per-run measurement |
| Output | Manual JSON via `std::ostringstream` | No external serialization library |

---

## Prerequisites

- Linux (Ubuntu 20.04+)
- GCC 9+ or Clang 10+
- CMake 3.16+
- `wget`, `tar` (for ONNX Runtime download)

```bash
sudo apt update && sudo apt install -y build-essential cmake wget tar git
```

---

## Setup

1. **Clone**

    ```bash
    git clone https://github.com/SarthakKala/inferscope
    cd inferscope
    ```

2. **Download ONNX Runtime** (CPU build, no CUDA needed)

    ```bash
    mkdir -p third_party
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz \
      -O third_party/onnxruntime.tgz
    tar -xzf third_party/onnxruntime.tgz -C third_party/
    mv third_party/onnxruntime-linux-x64-1.18.0 third_party/onnxruntime
    rm third_party/onnxruntime.tgz
    ```

    After extraction, confirm the C++ header location (layout varies by ORT version):

    ```bash
    find third_party/onnxruntime/include -name onnxruntime_cxx_api.h
    ```

    For v1.18.0 Linux x64 the file is at `include/onnxruntime_cxx_api.h`, which matches the default `CMakeLists.txt` include path (`${ORT_ROOT}/include`). If your `find` result is nested (e.g. `include/onnxruntime/core/session/onnxruntime_cxx_api.h`), update `target_include_directories` accordingly.

3. **Download a model**

    ```bash
    chmod +x models/download_model.sh && ./models/download_model.sh
    ```

    Downloads MobileNetV2-12 from the ONNX Model Zoo into `models/mobilenetv2.onnx`.

4. **Build**

    ```bash
    mkdir -p build && cd build
    cmake ..
    make -j$(nproc)
    ```

---

## Run

```bash
cd build
./inferscope --model ../models/mobilenetv2.onnx --warmup 20 --runs 200 --output results.json
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | required | Path to `.onnx` model file |
| `--warmup` | `20` | Inference runs discarded before measurement starts |
| `--runs` | `200` | Inference runs used for latency statistics |
| `--output` | `results.json` | Path to write JSON results |
| `--help` | — | Print usage |

---

## Output

**stdout** (representative; latency varies by machine)

```
========================================
  inferscope Results
========================================
Model      : ../models/mobilenetv2.onnx
Warmup runs: 20
Measure runs: 200

--- Graph ---
  Input  : input   shape=[-1, 3, 224, 224]   type=float32
  Output : output  shape=[-1, 1000]           type=float32

--- Latency (ms) ---
  p50    : 19.580 ms
  p90    : 22.786 ms
  p99    : 24.732 ms
  mean   : 20.138 ms
  stddev : 1.674 ms
  min    : 17.878 ms
  max    : 25.054 ms

--- Throughput ---
  49.7 inferences/sec
========================================
```

**results.json**

```json
{
  "model": "../models/mobilenetv2.onnx",
  "warmup_runs": 20,
  "measure_runs": 200,
  "graph": {
    "inputs":  [{ "name": "input",  "shape": [-1, 3, 224, 224], "type": "float32" }],
    "outputs": [{ "name": "output", "shape": [-1, 1000],         "type": "float32" }]
  },
  "latency_ms": {
    "p50": 19.5801, "p90": 22.7859, "p99": 24.7315,
    "mean": 20.1384, "stddev": 1.6735, "min": 17.8779, "max": 25.0540
  },
  "throughput_fps": 49.7
}
```

> Latency numbers are machine-dependent. The example above is from a Docker Ubuntu 22.04 build on a typical laptop CPU, single-threaded (`SetIntraOpNumThreads(1)`), with `ORT_ENABLE_EXTENDED` graph optimization. On a modern CPU, MobileNetV2 p50 is often in the 10–80 ms range.

> Graph metadata reports shapes as ONNX defines them. `-1` is a dynamic batch dimension; the benchmark replaces non-positive dimensions with `1` when allocating input tensors, so inference runs at batch size 1 even when the graph section shows `-1`.

---

## Design notes

> Reporting a single average is how tutorials measure inference. It is not how you validate hardware. A single spike can double the mean while the p50 stays flat — that distinction matters when you are deciding whether a chip is fast enough for a real-time workload. Warmup runs are discarded because the first N inferences are slower due to memory allocation and kernel JIT; including them inflates the reported latency. The p99 tells you your worst-case frame budget.

> The tool accepts any `.onnx` model — not just MobileNetV2. Dynamic input dimensions are detected at load time and replaced with batch size 1. Inputs are filled with random floats; the goal is timing accuracy, not meaningful predictions.

---

## Project layout

| Path | Contents |
|------|----------|
| `src/main.cpp` | CLI argument parsing, wires all components |
| `src/session_wrapper` | `Ort::Env` + `Ort::Session` lifecycle, exposes `run_inference()` |
| `src/graph_inspector` | Reads input/output names, shapes, and types from a loaded session |
| `src/benchmark_runner` | Warmup loop, measurement loop, calls profiler |
| `src/profiler` | Sorts latency vector, computes p50/p90/p99/stddev |
| `src/reporter` | Formats results as human text and JSON, writes to file |
| `models/download_model.sh` | Downloads MobileNetV2-12 from ONNX Model Zoo |
| `third_party/onnxruntime/` | Extracted ORT headers and `.so` (not tracked in git) |
| `CMakeLists.txt` | Build config, links against `libonnxruntime.so` |

---

## License

MIT
