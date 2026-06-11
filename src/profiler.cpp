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
