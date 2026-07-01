set(XTL_REQUIRED_VERSION 0.8.0)

list(APPEND CMAKE_PREFIX_PATH "${THIRD_PARTY_DIR}")
find_package(xtl ${XTL_REQUIRED_VERSION} NO_MODULE QUIET)

if(NOT TARGET xtl)
    message(
        STATUS
        "GPU Playground: "
        "Did not find xtl ${XTL_REQUIRED_VERSION} installed, "
        "downloading to ${THIRD_PARTY_DIR}"
    )
    include(FetchContent)

    set(FETCHCONTENT_BASE_DIR "${THIRD_PARTY_DIR}")
    FetchContent_Declare(
        xtl
        GIT_REPOSITORY "https://github.com/xtensor-stack/xtl"
        GIT_TAG ${XTL_REQUIRED_VERSION}
    )

    FetchContent_MakeAvailable(xtl)
else()
    get_target_property(XTL_INCLUDE_DIRS xtl INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "GPU Playground: Found xtl installed in ${XTL_INCLUDE_DIRS}")
endif()
