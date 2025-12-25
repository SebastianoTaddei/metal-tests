set(BACKEND "CPU" CACHE STRING "Chosen backend")
set_property(CACHE BACKEND PROPERTY STRINGS "CPU" "Metal")

get_property(BACKEND_STRINGS CACHE BACKEND PROPERTY STRINGS)
if(NOT BACKEND IN_LIST BACKEND_STRINGS)
  message(FATAL_ERROR
    "GPU Playground: Wrong value of the parameter 'BACKEND' "
    "(${BACKEND}), expected one of [${BACKEND_STRINGS}]"
  )
endif()

if(BACKEND STREQUAL "CPU")
  message(STATUS "GPU Playground: CPU backend selected")
  include(CPU)
  add_library(gpu_playground::backend ALIAS cpu_backend)
elseif(BACKEND STREQUAL "Metal")
  if(NOT APPLE)
    message(FATAL_ERROR
      "GPU Playground: Metal backend is only supported on macOS"
    )
  endif()
  message(STATUS "GPU Playground: Metal backend selected")
  include(Metal)
  add_library(gpu_playground::backend ALIAS metal_backend)
else()
  message(WARNING
    "GPU Playground: you should not be here, no valid backend selected. "
    "Falling back to CPU"
  )
  include(CPU)
  add_library(gpu_playground::backend ALIAS cpu_backend)
endif()

