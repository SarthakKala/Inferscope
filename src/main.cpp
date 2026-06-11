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
