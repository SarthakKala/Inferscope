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
