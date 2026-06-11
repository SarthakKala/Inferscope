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
