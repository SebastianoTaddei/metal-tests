include(xtl)
include(xsimd)

set(XTENSOR_REQUIRED_VERSION 0.27.1)

list(APPEND CMAKE_PREFIX_PATH "${THIRD_PARTY_DIR}")
find_package(xtensor ${XTENSOR_REQUIRED_VERSION} NO_MODULE QUIET)

if(NOT TARGET xtensor)
    message(
        STATUS
        "GPU Playground: "
        "Did not find xtensor ${XTENSOR_REQUIRED_VERSION} installed, "
        "downloading to ${THIRD_PARTY_DIR}"
    )
    include(FetchContent)

    set(FETCHCONTENT_BASE_DIR "${THIRD_PARTY_DIR}")
    FetchContent_Declare(
        xtensor
        GIT_REPOSITORY "https://github.com/xtensor-stack/xtensor"
        GIT_TAG ${XTENSOR_REQUIRED_VERSION}
        OPTIONS
        "XTENSOR_USE_XSIMD=ON"
    )

    FetchContent_MakeAvailable(xtensor)
else()
    get_target_property(
        XTENSOR_INCLUDE_DIRS
        xtensor
        INTERFACE_INCLUDE_DIRECTORIES
    )
    message(
        STATUS
        "GPU Playground: Found xtensor installed in ${XTENSOR_INCLUDE_DIRS}"
    )
endif()
