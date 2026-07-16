cmake_minimum_required(VERSION 3.24)
project(blockpose VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BLOCKPOSE_BUILD_PYTHON "Build pybind11 module" ON)
option(BLOCKPOSE_BUILD_TESTS  "Build unit tests"      ON)

# --- dependencies (all provided by docker/Dockerfile.core) -------------------
find_package(Eigen3 3.4 REQUIRED)
find_package(Ceres 2.1 REQUIRED)
find_package(Open3D REQUIRED)          # tensor API — see docs/plan §2.3
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CLI11 REQUIRED)

# --- environment smoke test ---------------------------------------------------
# Proves the toolchain + every dependency link and run. CI gate for the image.
add_executable(env_check apps/env_check.cpp)
target_link_libraries(env_check PRIVATE
    Eigen3::Eigen
    Ceres::ceres
    Open3D::Open3D
    yaml-cpp::yaml-cpp
    nlohmann_json::nlohmann_json
    CLI11::CLI11
)

# --- core library (agent: populate per plan §2.4 repo layout) ----------------
# add_subdirectory(src)

if(BLOCKPOSE_BUILD_PYTHON)
    find_package(pybind11 CONFIG REQUIRED)
    # add_subdirectory(python/bindings)
endif()

if(BLOCKPOSE_BUILD_TESTS)
    enable_testing()
    find_package(GTest REQUIRED)
    # add_subdirectory(tests)
    add_test(NAME env_check COMMAND env_check)
endif()
