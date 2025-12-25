include(Eigen3)

add_library(cpu_backend
  "${SRC_DIR}/src/backends/cpu/device.cpp"
)

target_include_directories(cpu_backend PRIVATE
  "${SRC_DIR}/include"
)

target_link_libraries(cpu_backend
    Eigen3::Eigen
)
