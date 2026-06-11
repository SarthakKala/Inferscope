#include "graph_inspector.hpp"
#include <stdexcept>

std::string ort_type_to_string(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "float32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:  return "float64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "int64";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "uint8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:    return "bool";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:  return "string";
        default:                                    return "unknown";
    }
}

GraphInfo inspect_graph(Ort::Session& session) {
    Ort::AllocatorWithDefaultOptions allocator;
    GraphInfo info;

    // Inspect inputs
    size_t input_count = session.GetInputCount();
    for (size_t i = 0; i < input_count; ++i) {
        TensorInfo t;

        auto name_ptr = session.GetInputNameAllocated(i, allocator);
        t.name = std::string(name_ptr.get());

        auto type_info   = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        t.shape = tensor_info.GetShape();
        t.type  = ort_type_to_string(tensor_info.GetElementType());

        info.inputs.push_back(t);
    }

    // Inspect outputs
    size_t output_count = session.GetOutputCount();
    for (size_t i = 0; i < output_count; ++i) {
        TensorInfo t;

        auto name_ptr = session.GetOutputNameAllocated(i, allocator);
        t.name = std::string(name_ptr.get());

        auto type_info   = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        t.shape = tensor_info.GetShape();
        t.type  = ort_type_to_string(tensor_info.GetElementType());

        info.outputs.push_back(t);
    }

    return info;
}
