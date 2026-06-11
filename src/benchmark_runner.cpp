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
