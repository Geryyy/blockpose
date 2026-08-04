# Stands in for the ROS node: consumes the INSTALLED blockpose_core package the
# way a downstream ament/catkin target does, via find_package + blockpose::core.
# Building the core in-tree does not prove that surface works -- the exported
# CMake package, the installed headers and the parameter defaults a downstream
# node actually compiles against are all separate failure points.
cmake_minimum_required(VERSION 3.16)
project(blockpose_core_consumer CXX)
find_package(blockpose_core REQUIRED)
add_executable(blockpose_core_consumer main.cpp)
target_link_libraries(blockpose_core_consumer PRIVATE blockpose::core)
