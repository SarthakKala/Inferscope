#include "session_wrapper.hpp"
#include <stdexcept>
#include <iostream>

SessionWrapper::SessionWrapper(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "inferscope"),
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
