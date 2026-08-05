# VERSION 0.2.0: bumped from 0.1.0 because the core's behaviour and public
# surface changed materially since that tag was implied -- the cluster gate
# moved 1.5 -> 2.25 (a detection-behaviour change, see
# experiments/detector_cluster_failures.md), the PZS100 gripper self-filter,
# annotation_snap and per-stage diagnostics were added. All additive/behaviour
# changes, no removed or renamed symbols on the blockpose::core interface, so
# this is a minor bump, not a major one -- and it stays below 1.0.0 because
# the shipped detector (46/62 centre recall, 78% precision on the
# Seibersdorf set, see docs/production-readiness-plan.md) is not yet claiming
# production-grade accuracy; 1.0.0 would overstate that.
#
# Do NOT read this bump as a machine-enforced incompatibility signal. Measured,
# not assumed: write_basic_package_version_file() below uses SameMajorVersion,
# and CMake's generated SameMajorVersion file compares only the major field
# with no 0.x special case, so a downstream `find_package(blockpose_core 0.1
# REQUIRED)` still resolves happily against 0.2.0 (verified against a real
# install of this package: it configured and reported
# blockpose_core_VERSION=0.2.0). The version string is therefore documentation
# for a human re-pinning after the parameter defaults moved, not a build-time
# guard. If we ever want the build to refuse an out-of-date pin, that requires
# changing COMPATIBILITY here -- it is not what the bump alone buys.
cmake_minimum_required(VERSION 3.16)
project(blockpose_core VERSION 0.2.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

add_library(blockpose_core INTERFACE)
add_library(blockpose::core ALIAS blockpose_core)
set_target_properties(blockpose_core PROPERTIES EXPORT_NAME core)
target_compile_features(blockpose_core INTERFACE cxx_std_17)
target_link_libraries(blockpose_core INTERFACE Eigen3::Eigen)
target_include_directories(blockpose_core INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Optional classical vision support.  Keep the geometry core dependency-free:
# consumers only need OpenCV when they explicitly link blockpose::vision.
option(BLOCKPOSE_VISION_BUILD "Build the optional OpenCV vision library" ON)
set(BLOCKPOSE_VISION_ENABLED OFF)
if(BLOCKPOSE_VISION_BUILD)
  find_package(OpenCV QUIET COMPONENTS core imgproc calib3d)
  if(OpenCV_FOUND)
    add_library(blockpose_vision src/rgb_edge_prior.cpp)
    add_library(blockpose::vision ALIAS blockpose_vision)
    set_target_properties(blockpose_vision PROPERTIES EXPORT_NAME vision)
    target_compile_features(blockpose_vision PUBLIC cxx_std_17)
    target_link_libraries(blockpose_vision PUBLIC blockpose::core ${OpenCV_LIBS})
    target_include_directories(blockpose_vision PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    set(BLOCKPOSE_VISION_ENABLED ON)
  else()
    message(STATUS "OpenCV not found: blockpose::vision will not be built")
  endif()
endif()

option(BLOCKPOSE_CORE_BUILD_TESTING "Build standalone blockpose_core tests" ON)
if(BLOCKPOSE_CORE_BUILD_TESTING)
  enable_testing()
  find_package(GTest REQUIRED)

  function(add_blockpose_core_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE blockpose::core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(${name})
  endfunction()

  add_executable(blockpose_core_smoke_test tests/smoke_test.cpp)
  target_link_libraries(blockpose_core_smoke_test PRIVATE blockpose::core)
  add_test(NAME blockpose_core_smoke_test COMMAND blockpose_core_smoke_test)

  foreach(name geometry proposals pipeline refine evidence gripper_self_filter annotation_snap)
    add_blockpose_core_test(blockpose_core_${name}_test)
  endforeach()

  if(BLOCKPOSE_VISION_ENABLED)
    add_executable(blockpose_vision_rgb_edge_prior_test tests/blockpose_vision_rgb_edge_prior_test.cpp)
    target_link_libraries(blockpose_vision_rgb_edge_prior_test PRIVATE blockpose::vision GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(blockpose_vision_rgb_edge_prior_test)
  endif()
endif()

# Python is an optional consumer of the same C++ core used by production
# adapters.  Keep it OFF by default so installing or configuring the portable
# core has no Python/nanobind dependency.
option(BLOCKPOSE_CORE_BUILD_PYTHON "Build the optional nanobind Python extension" OFF)
if(BLOCKPOSE_CORE_BUILD_PYTHON)
  find_package(Python COMPONENTS Interpreter Development REQUIRED)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(blockpose_python_core bindings/blockpose_core_module.cpp)
  target_link_libraries(blockpose_python_core PRIVATE blockpose::core)
  if(BLOCKPOSE_VISION_ENABLED)
    target_link_libraries(blockpose_python_core PRIVATE blockpose::vision)
    target_compile_definitions(blockpose_python_core PRIVATE BLOCKPOSE_CORE_WITH_VISION=1)
  endif()
  target_compile_definitions(blockpose_python_core PRIVATE
    BLOCKPOSE_CORE_VERSION="${PROJECT_VERSION}")
  set_target_properties(blockpose_python_core PROPERTIES OUTPUT_NAME "_core")

  # The extension is deliberately installed inside the Python package.  It is
  # not part of the exported CMake target set: C++ consumers only need the
  # dependency-free blockpose::core interface above.
  install(TARGETS blockpose_python_core LIBRARY DESTINATION blockpose)
endif()

install(TARGETS blockpose_core EXPORT blockpose_coreTargets)
if(BLOCKPOSE_VISION_ENABLED)
  install(TARGETS blockpose_vision EXPORT blockpose_coreTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT blockpose_coreTargets
  FILE blockpose_coreTargets.cmake
  NAMESPACE blockpose::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/blockpose_core)

configure_package_config_file(
  cmake/blockpose_coreConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/blockpose_coreConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/blockpose_core)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/blockpose_coreConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/blockpose_coreConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/blockpose_coreConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/blockpose_core)
