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
