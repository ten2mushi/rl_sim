# Sensor Implementations Module - CMake Configuration
#
# This module provides implementations for all 10 sensor types:
# - IMU, ToF, LiDAR 2D/3D, Camera RGB/Depth/Seg, Position, Velocity, Neighbor
#
# Dependencies:
#   - sensor_system: SensorVtable, SensorConfig
#   - foundation: Arena, Vec3, SIMD macros, PCG32
#   - drone_state: DroneStateSOA
#   - world_brick_map: SDF queries, raymarching
#   - collision_system: KNN queries
#
# Targets:
#   sensor_implementations        - Main library
#   test_sensor_implementations   - Unit tests
#   bench_sensor_implementations  - Performance benchmarks

cmake_minimum_required(VERSION 3.16)

# Only set project if this is the top-level CMakeLists
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(sensor_implementations C)
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

if(NOT TARGET world_brick_map)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../world_brick_map
                     ${CMAKE_CURRENT_BINARY_DIR}/world_brick_map EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET collision_system)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../collision_system
                     ${CMAKE_CURRENT_BINARY_DIR}/collision_system EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET sensor_system)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sensor_system
                     ${CMAKE_CURRENT_BINARY_DIR}/sensor_system EXCLUDE_FROM_ALL)
endif()

# ============================================================================
# Library Target
# ============================================================================

add_library(sensor_implementations STATIC
    src/sensor_imu.c
    src/sensor_tof.c
    src/sensor_lidar.c
    src/sensor_camera.c
    src/sensor_neighbor.c
    src/sensor_direct.c
    src/sensor_registry.c
)

target_include_directories(sensor_implementations
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link dependencies
target_link_libraries(sensor_implementations
    PUBLIC
        sensor_system
        foundation
        drone_state
    PRIVATE
        world_brick_map
        collision_system
        m
)

# Compiler settings
target_compile_features(sensor_implementations PUBLIC c_std_11)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(sensor_implementations PRIVATE
        -Wall -Wextra -Wpedantic
        -Wno-unused-parameter
        $<$<CONFIG:Release>:-O3>
        $<$<CONFIG:Debug>:-g -O0 -DFOUNDATION_DEBUG>
    )
endif()

# SIMD flags
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    target_compile_options(sensor_implementations PRIVATE -mavx2 -mfma)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
    # NEON is default on ARM64
endif()

# ============================================================================
# Test Target
# ============================================================================

if(BUILD_TESTING)
    add_executable(test_sensor_implementations
        tests/test_sensor_implementations.c
    )

    target_link_libraries(test_sensor_implementations
        PRIVATE
            sensor_implementations
            sensor_system
            world_brick_map
            collision_system
            foundation
            drone_state
            platform
            m
    )

    target_compile_features(test_sensor_implementations PRIVATE c_std_11)

    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_sensor_implementations PRIVATE
            -Wall -Wextra
            -g -O0 -DFOUNDATION_DEBUG
        )
    endif()

    add_test(NAME test_sensor_implementations COMMAND test_sensor_implementations)
endif()

# ============================================================================
# Benchmark Target
# ============================================================================

if(BUILD_BENCHMARKS)
    add_executable(bench_sensor_implementations
        tests/bench_sensor_implementations.c
    )

    target_link_libraries(bench_sensor_implementations
        PRIVATE
            sensor_implementations
            sensor_system
            world_brick_map
            collision_system
            foundation
            drone_state
            platform
            m
    )

    target_compile_features(bench_sensor_implementations PRIVATE c_std_11)

    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench_sensor_implementations PRIVATE
            -Wall -Wextra
            -O3
        )
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS sensor_implementations
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)
