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
