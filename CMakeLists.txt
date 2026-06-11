cmake_minimum_required(VERSION 3.16)
project(inferscope CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Path to extracted ONNX Runtime
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime" CACHE PATH "ONNX Runtime root directory")

# Find the shared library
find_library(ORT_LIB
    NAMES onnxruntime
    PATHS "${ORT_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED
)

message(STATUS "ONNX Runtime library: ${ORT_LIB}")
message(STATUS "ONNX Runtime include: ${ORT_ROOT}/include")

# All source files
set(SOURCES
    src/main.cpp
    src/session_wrapper.cpp
    src/profiler.cpp
    src/benchmark_runner.cpp
    src/graph_inspector.cpp
    src/reporter.cpp
)

add_executable(inferscope ${SOURCES})

target_include_directories(inferscope PRIVATE
    "${ORT_ROOT}/include"
    "${CMAKE_SOURCE_DIR}/src"
)

target_link_libraries(inferscope PRIVATE
    ${ORT_LIB}
)

# Copy the .so file next to the binary so it can be found at runtime
add_custom_command(TARGET inferscope POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${ORT_ROOT}/lib/libonnxruntime.so.1.18.0"
        $<TARGET_FILE_DIR:inferscope>/libonnxruntime.so.1.18.0
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        libonnxruntime.so.1.18.0
        $<TARGET_FILE_DIR:inferscope>/libonnxruntime.so
    COMMENT "Copying ONNX Runtime shared library to build directory"
)
