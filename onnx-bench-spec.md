# onnx-bench — Agent Build Specification

## What You Are Building

A command-line C++ tool that:
1. Loads a pretrained `.onnx` model file from disk
2. Creates a dummy input tensor (random floats — we are benchmarking speed, not accuracy)
3. Runs inference N times with a warmup phase
4. Computes p50 / p90 / p99 latency statistics
5. Inspects and prints graph metadata (input/output names, shapes)
6. Outputs all results as JSON to stdout and to a file

No GPU. No CUDA. CPU Execution Provider only.
No external libraries except ONNX Runtime.
No Python. Pure C++17.

---

## Tech Stack

| Component       | Choice                          |
|-----------------|---------------------------------|
| Language        | C++17                           |
| Build System    | CMake 3.16+                     |
| ML Runtime      | ONNX Runtime 1.18.0 (CPU only)  |
| Platform        | Linux (Ubuntu 20.04+)           |
| Compiler        | GCC or Clang                    |

---

## Final Directory Structure

Build this exact layout. Do not deviate.

```
onnx-bench/
├── CMakeLists.txt
├── README.md
├── third_party/
│   └── onnxruntime/         ← extracted ORT package goes here (agent does NOT create this)
├── models/
│   └── download_model.sh    ← shell script to download MobileNetV2
├── src/
│   ├── main.cpp
│   ├── session_wrapper.hpp
│   ├── session_wrapper.cpp
│   ├── profiler.hpp
│   ├── profiler.cpp
│   ├── benchmark_runner.hpp
│   ├── benchmark_runner.cpp
│   ├── graph_inspector.hpp
│   ├── graph_inspector.cpp
│   ├── reporter.hpp
│   └── reporter.cpp
└── build/                   ← created by cmake, do NOT create manually
```

---

## Phase 1: System Prerequisites

Run these commands in the terminal before anything else.

```bash
sudo apt update
sudo apt install -y build-essential cmake wget tar git
```

Verify:
```bash
cmake --version    # must be 3.16 or higher
g++ --version      # must be 9.0 or higher
```

---

## Phase 2: ONNX Runtime Setup

Run these commands from inside the `onnx-bench/` root directory.

```bash
mkdir -p third_party

wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz \
  -O third_party/onnxruntime.tgz

tar -xzf third_party/onnxruntime.tgz -C third_party/

mv third_party/onnxruntime-linux-x64-1.18.0 third_party/onnxruntime

rm third_party/onnxruntime.tgz
```

After extraction, verify this structure exists:
```
third_party/onnxruntime/
├── include/
│   ├── onnxruntime_c_api.h
│   ├── onnxruntime_cxx_api.h
│   └── ... (other headers)
└── lib/
    ├── libonnxruntime.so         ← symlink
    └── libonnxruntime.so.1.18.0  ← actual file
```

**IMPORTANT:** The include path may vary. After extraction, run:
```bash
find third_party/onnxruntime/include -name "onnxruntime_cxx_api.h"
```
Note the exact path. Use that path in CMakeLists.txt for `target_include_directories`.
If the file is at `include/onnxruntime_cxx_api.h`, use `${ORT_ROOT}/include`.
If the file is at `include/onnxruntime/core/session/onnxruntime_cxx_api.h`, use `${ORT_ROOT}/include/onnxruntime/core/session`.

---

## Phase 3: Model Download

Create `models/download_model.sh` with this exact content:

```bash
#!/bin/bash
set -e
mkdir -p models
echo "Downloading MobileNetV2 ONNX model..."
wget -q --show-progress \
  "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx" \
  -O models/mobilenetv2.onnx
echo "Done. Model saved to models/mobilenetv2.onnx"
```

Run it:
```bash
chmod +x models/download_model.sh
./models/download_model.sh
```

MobileNetV2 expects input shape: `[batch_size, 3, 224, 224]` (batch, channels, height, width).
Output shape: `[batch_size, 1000]` (1000 ImageNet classes).

---

## Phase 4: CMakeLists.txt

Create this file exactly at the project root.

```cmake
cmake_minimum_required(VERSION 3.16)
project(onnx-bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Path to extracted ONNX Runtime
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root directory")

# Find the shared library
find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${ORT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

message(STATUS "ONNX Runtime library: ${ORT_LIB}")
message(STATUS "ONNX Runtime include: ${ORT_ROOT}/include")

# All source files
set(SOURCES
    src/main.cpp
    src/session_wrapper.cpp
    src/profiler.cpp
    src/benchmark_runner.cpp
    src/graph_inspector.cpp
    src/reporter.cpp
)

add_executable(onnx-bench ${SOURCES})

target_include_directories(onnx-bench PRIVATE
    "${ORT_ROOT}/include"
    "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(onnx-bench PRIVATE
    ${ORT_LIB}
)

# Copy the .so file next to the binary so it can be found at runtime
add_custom_command(TARGET onnx-bench POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_ROOT}/lib/libonnxruntime.so.1.18.0"
        $<TARGET_FILE_DIR:onnx-bench>/libonnxruntime.so.1.18.0
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        libonnxruntime.so.1.18.0
        $<TARGET_FILE_DIR:onnx-bench>/libonnxruntime.so
    COMMENT "Copying ONNX Runtime shared library to build directory"
)
```

---

## Phase 5: Source Files

Build each file in the order listed below.

---

### 5.1 — `src/profiler.hpp`

```cpp
#pragma once
#include <vector>
#include <string>

struct LatencyStats {
    double p50_ms;
    double p90_ms;
    double p99_ms;
    double mean_ms;
    double stddev_ms;
    double min_ms;
    double max_ms;
};

// Takes a vector of latency measurements in milliseconds.
// Sorts them and computes percentiles and stddev.
LatencyStats compute_stats(std::vector<double> latencies);
```

---

### 5.2 — `src/profiler.cpp`

```cpp
#include "profiler.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <stdexcept>

LatencyStats compute_stats(std::vector<double> latencies) {
    if (latencies.empty()) {
        throw std::runtime_error("Cannot compute stats on empty latency vector");
    }

    std::sort(latencies.begin(), latencies.end());

    size_t n = latencies.size();

    LatencyStats stats;
    stats.min_ms    = latencies.front();
    stats.max_ms    = latencies.back();
    stats.p50_ms    = latencies[static_cast<size_t>(n * 0.50)];
    stats.p90_ms    = latencies[static_cast<size_t>(n * 0.90)];
    stats.p99_ms    = latencies[static_cast<size_t>(n * 0.99)];

    double sum      = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    stats.mean_ms   = sum / static_cast<double>(n);

    double sq_sum = 0.0;
    for (double val : latencies) {
        double diff = val - stats.mean_ms;
        sq_sum += diff * diff;
    }
    stats.stddev_ms = std::sqrt(sq_sum / static_cast<double>(n));

    return stats;
}
```

---

### 5.3 — `src/graph_inspector.hpp`

```cpp
#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

struct TensorInfo {
    std::string name;
    std::vector<int64_t> shape;   // -1 means dynamic dimension
    std::string type;             // e.g. "float32"
};

struct GraphInfo {
    std::vector<TensorInfo> inputs;
    std::vector<TensorInfo> outputs;
};

// Reads input and output metadata from a loaded OrtSession.
GraphInfo inspect_graph(Ort::Session& session);

// Converts ONNXTensorElementDataType enum to a human-readable string.
std::string ort_type_to_string(ONNXTensorElementDataType type);
```

---

### 5.4 — `src/graph_inspector.cpp`

```cpp
#include "graph_inspector.hpp"
#include <stdexcept>

std::string ort_type_to_string(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return "string";
        default:                                    return "unknown";
    }
}

GraphInfo inspect_graph(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    GraphInfo info;

    // Inspect inputs
    size_t input_count = session.GetInputCount();
    for (size_t i = 0; i < input_count; ++i) {
        TensorInfo t;

        auto name_ptr = session.GetInputNameAllocated(i, allocator);
        t.name = std::string(name_ptr.get());

        auto type_info   = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        t.shape = tensor_info.GetShape();
        t.type  = ort_type_to_string(tensor_info.GetElementType());

        info.inputs.push_back(t);
    }

    // Inspect outputs
    size_t output_count = session.GetOutputCount();
    for (size_t i = 0; i < output_count; ++i) {
        TensorInfo t;

        auto name_ptr = session.GetOutputNameAllocated(i, allocator);
        t.name = std::string(name_ptr.get());

        auto type_info   = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        t.shape = tensor_info.GetShape();
        t.type  = ort_type_to_string(tensor_info.GetElementType());

        info.outputs.push_back(t);
    }

    return info;
}
```

---

### 5.5 — `src/session_wrapper.hpp`

```cpp
#pragma once
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// Owns the OrtEnv and OrtSession lifecycle.
// Loads the model once. run_inference() can be called repeatedly.
class SessionWrapper {
public:
    // model_path: absolute or relative path to a .onnx file
    explicit SessionWrapper(const std::string& model_path);

    // Runs one forward pass with the given input data.
    // input_data: flat float array matching the model's first input tensor
    // input_shape: dimensions of the input tensor (e.g. {1, 3, 224, 224})
    // Returns: flat float vector of the first output tensor
    std::vector<float> run_inference(
        const std::vector<float>& input_data,
        const std::vector<int64_t>& input_shape
    );

    // Returns the loaded session (for graph inspection)
    Ort::Session& get_session() { return session_; }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    std::string input_name_;
    std::string output_name_;
};
```

---

### 5.6 — `src/session_wrapper.cpp`

This is the most critical file. Copy the ONNX Runtime API calls exactly.

```cpp
#include "session_wrapper.hpp"
#include <stdexcept>
#include <iostream>

SessionWrapper::SessionWrapper(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "onnx-bench"),
      session_(nullptr)
{
    // Configure session
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED
    );

    // Load model from disk
    // On Linux, model path is const char*
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    // Cache input and output names
    Ort::AllocatorWithDefaultOptions allocator;

    auto input_name_ptr  = session_.GetInputNameAllocated(0, allocator);
    input_name_  = std::string(input_name_ptr.get());

    auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator);
    output_name_ = std::string(output_name_ptr.get());

    std::cout << "[SessionWrapper] Model loaded successfully.\n";
    std::cout << "[SessionWrapper] Input  name: " << input_name_  << "\n";
    std::cout << "[SessionWrapper] Output name: " << output_name_ << "\n";
}

std::vector<float> SessionWrapper::run_inference(
    const std::vector<float>& input_data,
    const std::vector<int64_t>& input_shape)
{
    // Describe where the tensor data lives (CPU, arena allocator)
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault
    );

    // Wrap the raw float buffer in an OrtValue tensor
    // NOTE: This does NOT copy data — it creates a view over input_data
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape.data(),
        input_shape.size()
    );

    // Run inference
    const char* input_names[]  = { input_name_.c_str()  };
    const char* output_names[] = { output_name_.c_str() };

    auto output_tensors = session_.Run(
        Ort::RunOptions{nullptr},
        input_names,  &input_tensor, 1,
        output_names, 1
    );

    // Extract output data from the first output tensor
    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    auto   output_shape_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    size_t output_count = output_shape_info.GetElementCount();

    return std::vector<float>(output_data, output_data + output_count);
}
```

---

### 5.7 — `src/benchmark_runner.hpp`

```cpp
#pragma once
#include "session_wrapper.hpp"
#include "profiler.hpp"
#include <vector>
#include <cstddef>

struct BenchmarkConfig {
    int warmup_runs  = 20;   // Runs to discard before measuring
    int measure_runs = 200;  // Runs to actually measure
};

struct BenchmarkResult {
    std::string   model_path;
    BenchmarkConfig config;
    std::vector<int64_t> input_shape;
    LatencyStats  stats;
    double        throughput_fps;  // 1000.0 / mean_ms
};

// Runs warmup_runs inferences (discarded), then measure_runs inferences.
// Times each run with std::chrono::high_resolution_clock.
// Returns statistical summary.
BenchmarkResult run_benchmark(
    SessionWrapper& session,
    const BenchmarkConfig& config,
    const std::string& model_path
);
```

---

### 5.8 — `src/benchmark_runner.cpp`

```cpp
#include "benchmark_runner.cpp"  // DO NOT USE — see below
```

AGENT NOTE: Do not copy the line above. Write the full implementation.

```cpp
#include "benchmark_runner.hpp"
#include "graph_inspector.hpp"
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>

BenchmarkResult run_benchmark(
    SessionWrapper& session,
    const BenchmarkConfig& config,
    const std::string& model_path)
{
    // Step 1: Get input shape from the model
    GraphInfo graph = inspect_graph(session.get_session());

    if (graph.inputs.empty()) {
        throw std::runtime_error("Model has no inputs");
    }

    std::vector<int64_t> input_shape = graph.inputs[0].shape;

    // Step 2: Replace dynamic dimensions (-1) with 1
    for (auto& dim : input_shape) {
        if (dim <= 0) dim = 1;
    }

    // Step 3: Compute total number of floats needed
    int64_t total_elements = 1;
    for (auto dim : input_shape) {
        total_elements *= dim;
    }

    // Step 4: Create dummy input (random floats between 0 and 1)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> input_data(static_cast<size_t>(total_elements));
    for (auto& val : input_data) val = dist(rng);

    // Step 5: Warmup runs — run inference but discard timing
    std::cout << "[Benchmark] Running " << config.warmup_runs << " warmup runs...\n";
    for (int i = 0; i < config.warmup_runs; ++i) {
        session.run_inference(input_data, input_shape);
    }

    // Step 6: Measurement runs — time each run
    std::cout << "[Benchmark] Running " << config.measure_runs << " measurement runs...\n";
    std::vector<double> latencies;
    latencies.reserve(config.measure_runs);

    for (int i = 0; i < config.measure_runs; ++i) {
        auto t_start = std::chrono::high_resolution_clock::now();
        session.run_inference(input_data, input_shape);
        auto t_end = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
        latencies.push_back(ms);
    }

    // Step 7: Compute statistics
    BenchmarkResult result;
    result.model_path    = model_path;
    result.config        = config;
    result.input_shape   = input_shape;
    result.stats         = compute_stats(latencies);
    result.throughput_fps = 1000.0 / result.stats.mean_ms;

    return result;
}
```

---

### 5.9 — `src/reporter.hpp`

```cpp
#pragma once
#include "benchmark_runner.hpp"
#include "graph_inspector.hpp"
#include <string>

// Prints a human-readable summary to stdout
void print_summary(const BenchmarkResult& result, const GraphInfo& graph);

// Returns a JSON string of the full result
std::string to_json(const BenchmarkResult& result, const GraphInfo& graph);

// Writes the JSON string to a file
void write_json(const std::string& json_str, const std::string& output_path);

// Converts a shape vector like {1, 3, 224, 224} to a JSON array string "[1,3,224,224]"
std::string shape_to_json(const std::vector<int64_t>& shape);
```

---

### 5.10 — `src/reporter.cpp`

```cpp
#include "reporter.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

std::string shape_to_json(const std::vector<int64_t>& shape) {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        ss << shape[i];
        if (i + 1 < shape.size()) ss << ", ";
    }
    ss << "]";
    return ss.str();
}

void print_summary(const BenchmarkResult& result, const GraphInfo& graph) {
    std::cout << "\n========================================\n";
    std::cout << "  onnx-bench Results\n";
    std::cout << "========================================\n";
    std::cout << "Model      : " << result.model_path    << "\n";
    std::cout << "Warmup runs: " << result.config.warmup_runs  << "\n";
    std::cout << "Measure runs:" << result.config.measure_runs << "\n";

    std::cout << "\n--- Graph ---\n";
    for (const auto& inp : graph.inputs) {
        std::cout << "  Input  : " << inp.name
                  << "  shape=" << shape_to_json(inp.shape)
                  << "  type=" << inp.type << "\n";
    }
    for (const auto& out : graph.outputs) {
        std::cout << "  Output : " << out.name
                  << "  shape=" << shape_to_json(out.shape)
                  << "  type=" << out.type << "\n";
    }

    std::cout << "\n--- Latency (ms) ---\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  p50    : " << result.stats.p50_ms    << " ms\n";
    std::cout << "  p90    : " << result.stats.p90_ms    << " ms\n";
    std::cout << "  p99    : " << result.stats.p99_ms    << " ms\n";
    std::cout << "  mean   : " << result.stats.mean_ms   << " ms\n";
    std::cout << "  stddev : " << result.stats.stddev_ms << " ms\n";
    std::cout << "  min    : " << result.stats.min_ms    << " ms\n";
    std::cout << "  max    : " << result.stats.max_ms    << " ms\n";

    std::cout << "\n--- Throughput ---\n";
    std::cout << "  " << std::setprecision(1) << result.throughput_fps << " inferences/sec\n";
    std::cout << "========================================\n\n";
}

std::string to_json(const BenchmarkResult& result, const GraphInfo& graph) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(4);

    ss << "{\n";
    ss << "  \"model\": \"" << result.model_path << "\",\n";
    ss << "  \"warmup_runs\": " << result.config.warmup_runs << ",\n";
    ss << "  \"measure_runs\": " << result.config.measure_runs << ",\n";

    // Graph section
    ss << "  \"graph\": {\n";
    ss << "    \"inputs\": [\n";
    for (size_t i = 0; i < graph.inputs.size(); ++i) {
        const auto& inp = graph.inputs[i];
        ss << "      { \"name\": \"" << inp.name << "\","
           << " \"shape\": " << shape_to_json(inp.shape) << ","
           << " \"type\": \"" << inp.type << "\" }";
        if (i + 1 < graph.inputs.size()) ss << ",";
        ss << "\n";
    }
    ss << "    ],\n";
    ss << "    \"outputs\": [\n";
    for (size_t i = 0; i < graph.outputs.size(); ++i) {
        const auto& out = graph.outputs[i];
        ss << "      { \"name\": \"" << out.name << "\","
           << " \"shape\": " << shape_to_json(out.shape) << ","
           << " \"type\": \"" << out.type << "\" }";
        if (i + 1 < graph.outputs.size()) ss << ",";
        ss << "\n";
    }
    ss << "    ]\n";
    ss << "  },\n";

    // Latency section
    ss << "  \"latency_ms\": {\n";
    ss << "    \"p50\":    " << result.stats.p50_ms    << ",\n";
    ss << "    \"p90\":    " << result.stats.p90_ms    << ",\n";
    ss << "    \"p99\":    " << result.stats.p99_ms    << ",\n";
    ss << "    \"mean\":   " << result.stats.mean_ms   << ",\n";
    ss << "    \"stddev\": " << result.stats.stddev_ms << ",\n";
    ss << "    \"min\":    " << result.stats.min_ms    << ",\n";
    ss << "    \"max\":    " << result.stats.max_ms    << "\n";
    ss << "  },\n";

    ss << "  \"throughput_fps\": " << std::setprecision(1) << result.throughput_fps << "\n";
    ss << "}\n";

    return ss.str();
}

void write_json(const std::string& json_str, const std::string& output_path) {
    std::ofstream file(output_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output file: " + output_path);
    }
    file << json_str;
    std::cout << "[Reporter] JSON results written to: " << output_path << "\n";
}
```

---

### 5.11 — `src/main.cpp`

```cpp
#include "session_wrapper.hpp"
#include "benchmark_runner.hpp"
#include "graph_inspector.hpp"
#include "reporter.hpp"

#include <iostream>
#include <string>
#include <stdexcept>

void print_usage(const char* program_name) {
    std::cout << "Usage:\n";
    std::cout << "  " << program_name << " --model <path.onnx> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --model   <path>   Path to .onnx model file (required)\n";
    std::cout << "  --warmup  <n>      Number of warmup runs (default: 20)\n";
    std::cout << "  --runs    <n>      Number of measurement runs (default: 200)\n";
    std::cout << "  --output  <path>   Path to write JSON results (default: results.json)\n";
    std::cout << "  --help             Show this help message\n\n";
    std::cout << "Example:\n";
    std::cout << "  " << program_name << " --model models/mobilenetv2.onnx --warmup 20 --runs 200\n";
}

int main(int argc, char* argv[]) {
    // Defaults
    std::string model_path  = "";
    std::string output_path = "results.json";
    int warmup_runs  = 20;
    int measure_runs = 200;

    // Parse command-line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup_runs = std::stoi(argv[++i]);
        } else if (arg == "--runs" && i + 1 < argc) {
            measure_runs = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model is required.\n\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        std::cout << "[main] Loading model: " << model_path << "\n";
        SessionWrapper session(model_path);

        GraphInfo graph = inspect_graph(session.get_session());

        BenchmarkConfig config;
        config.warmup_runs  = warmup_runs;
        config.measure_runs = measure_runs;

        BenchmarkResult result = run_benchmark(session, config, model_path);

        print_summary(result, graph);

        std::string json = to_json(result, graph);
        write_json(json, output_path);

    } catch (const Ort::Exception& e) {
        std::cerr << "[ONNX Runtime Error] " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "[Error] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
```

---

## Phase 6: Build

Run these commands from the project root.

```bash
# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Compile
make -j$(nproc)
```

If CMake cannot find the ONNX Runtime library, pass the path explicitly:
```bash
cmake .. -DORT_ROOT=../third_party/onnxruntime
```

Successful build output ends with:
```
[100%] Linking CXX executable onnx-bench
[100%] Built target onnx-bench
```

---

## Phase 7: Run and Verify

Run from inside the `build/` directory:

```bash
./onnx-bench --model ../models/mobilenetv2.onnx --warmup 20 --runs 200 --output results.json
```

### Expected stdout output:

```
[SessionWrapper] Model loaded successfully.
[SessionWrapper] Input  name: input
[SessionWrapper] Output name: output
[Benchmark] Running 20 warmup runs...
[Benchmark] Running 200 measurement runs...

========================================
  onnx-bench Results
========================================
Model      : ../models/mobilenetv2.onnx
Warmup runs: 20
Measure runs:200

--- Graph ---
  Input  : input  shape=[1, 3, 224, 224]  type=float32
  Output : output  shape=[1, 1000]  type=float32

--- Latency (ms) ---
  p50    : X.XXX ms
  p90    : X.XXX ms
  p99    : X.XXX ms
  mean   : X.XXX ms
  stddev : X.XXX ms
  min    : X.XXX ms
  max    : X.XXX ms

--- Throughput ---
  XXX.X inferences/sec
========================================
```

Exact latency numbers will vary by machine. On a modern CPU, MobileNetV2 p50 should be in the range of 10–80ms depending on hardware.

### Expected `results.json` output:

```json
{
  "model": "../models/mobilenetv2.onnx",
  "warmup_runs": 20,
  "measure_runs": 200,
  "graph": {
    "inputs": [
      { "name": "input", "shape": [1, 3, 224, 224], "type": "float32" }
    ],
    "outputs": [
      { "name": "output", "shape": [1, 1000], "type": "float32" }
    ]
  },
  "latency_ms": {
    "p50":    XX.XXXX,
    "p90":    XX.XXXX,
    "p99":    XX.XXXX,
    "mean":   XX.XXXX,
    "stddev": X.XXXX,
    "min":    XX.XXXX,
    "max":    XX.XXXX
  },
  "throughput_fps": XX.X
}
```

---

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `libonnxruntime.so not found at runtime` | .so not in library path | Run from `build/` dir, or `export LD_LIBRARY_PATH=./build:$LD_LIBRARY_PATH` |
| `onnxruntime_cxx_api.h: No such file` | Wrong include path in CMake | Run `find third_party -name onnxruntime_cxx_api.h` and update CMakeLists |
| `Model file not found` | Wrong model path | Use absolute path or verify `models/mobilenetv2.onnx` exists |
| `Invalid tensor size` | Dynamic dim not replaced | Check `benchmark_runner.cpp` — every dim <= 0 must be set to 1 |
| `cmake: version too old` | Ubuntu default CMake is old | `pip install cmake` or download from cmake.org |

---

## What Each File Does (Plain English for Reference)

| File | What it does |
|------|-------------|
| `session_wrapper` | Loads the model, owns the ORT session, runs one inference |
| `graph_inspector` | Reads input/output names and shapes from the model |
| `profiler` | Takes a list of timing numbers and computes p50/p90/p99/stddev |
| `benchmark_runner` | Calls session_wrapper in a loop, collects timings, calls profiler |
| `reporter` | Formats results as human text and as JSON, writes JSON to file |
| `main` | Parses CLI args, wires all the above together |

---

## Git Setup (Do This Last)

```bash
cd onnx-bench

# Create .gitignore
cat > .gitignore << 'EOF'
build/
third_party/
models/*.onnx
*.json
EOF

git init
git add .
git commit -m "Initial commit: ONNX inference profiler with p50/p90/p99 latency stats"
```

Push to GitHub with a descriptive repo name: `onnx-bench` or `onnx-inference-profiler`.
