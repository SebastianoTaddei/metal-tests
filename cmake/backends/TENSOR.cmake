include(xtensor)

add_library(
    tensor_backend
    STATIC
    "${SRC_DIR}/src/backends/tensor/tensor_device.cpp"
)

target_include_directories(
    tensor_backend
    PRIVATE "${SRC_DIR}/include" "${SRC_DIR}/src/backends/tensor"
)

target_link_libraries(tensor_backend xtensor)

target_compile_definitions(tensor_backend PUBLIC GPU_PLAYGROUND_HAS_TENSOR)
