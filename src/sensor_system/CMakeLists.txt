# Sensor System Module - CMake Configuration
#
# This module provides the core sensor framework with vtable-based polymorphism
# and batch-by-type processing for cache-efficient observation generation.
#
# Dependencies:
#   - foundation: Arena, Vec3, SIMD macros
#   - drone_state: DroneStateSOA
#
# Targets:
#   sensor_system        - Main library
#   test_sensor_system   - Unit tests (requires sensor_implementations)
#   bench_sensor_system  - Performance benchmarks

cmake_minimum_required(VERSION 3.16)

# Only set project if this is the top-level CMakeLists
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sensor_system C)
    set(CMAKE_C_STANDARD 11)
    set(CMAKE_C_STANDARD_REQUIRED ON)

    option(BUILD_TESTING "Build unit tests" ON)
    option(BUILD_BENCHMARKS "Build performance benchmarks" ON)

    if(BUILD_TESTING)
        enable_testing()
    endif()
endif()

# ============================================================================
# Dependencies (sibling directories for standalone builds)
# ============================================================================

if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET drone_state)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../drone_state
                     ${CMAKE_CURRENT_BINARY_DIR}/drone_state EXCLUDE_FROM_ALL)
endif()

# ============================================================================
# Library Target
# ============================================================================

add_library(sensor_system STATIC
    src/sensor_system.c
    src/noise.c
)

target_include_directories(sensor_system
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(sensor_system
    PUBLIC
        foundation
        drone_state
)

# Compiler settings
target_compile_features(sensor_system PUBLIC c_std_11)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sensor_system PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-g -O0 -DFOUNDATION_DEBUG>
    )
endif()

# SIMD flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(sensor_system PRIVATE -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # NEON is default on ARM64
endif()

# ============================================================================
# Test and Benchmark Targets
# ============================================================================
# NOTE: test_sensor_system and bench_sensor_system are built from the
# top-level CMakeLists.txt (after all add_subdirectory calls) because they
# depend on sensor_implementations which is processed at a later layer.

# ============================================================================
# Installation
# ============================================================================

install(TARGETS sensor_system
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
