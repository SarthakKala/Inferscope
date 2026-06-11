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
