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
