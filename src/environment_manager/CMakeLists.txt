# Environment Manager Module CMake Configuration
#
# Build configuration for the environment manager module providing the top-level
# orchestration layer and PufferLib integration for batch drone RL training.

cmake_minimum_required(VERSION 3.16)
project(environment_manager C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(ENVIRONMENT_MANAGER_DEBUG "Enable debug assertions and logging" OFF)
option(ENVIRONMENT_MANAGER_BUILD_TESTS "Build unit tests" ON)
option(ENVIRONMENT_MANAGER_BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(ENVIRONMENT_MANAGER_ARCH_ARM TRUE)
    message(STATUS "Environment Manager: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(ENVIRONMENT_MANAGER_ARCH_X86_64 TRUE)
    message(STATUS "Environment Manager: Detected x86_64 architecture")
else()
    message(STATUS "Environment Manager: Unknown architecture, using scalar fallback")
endif()

# ============================================================================
# Dependencies (sibling directories)
# ============================================================================

# Foundation
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

# Drone State
if(NOT TARGET drone_state)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../drone_state
                     ${CMAKE_CURRENT_BINARY_DIR}/drone_state EXCLUDE_FROM_ALL)
endif()

# Physics
if(NOT TARGET physics)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../physics
                     ${CMAKE_CURRENT_BINARY_DIR}/physics EXCLUDE_FROM_ALL)
endif()

# World Brick Map
if(NOT TARGET world_brick_map)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../world_brick_map
                     ${CMAKE_CURRENT_BINARY_DIR}/world_brick_map EXCLUDE_FROM_ALL)
endif()

# Collision System
if(NOT TARGET collision_system)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../collision_system
                     ${CMAKE_CURRENT_BINARY_DIR}/collision_system EXCLUDE_FROM_ALL)
endif()

# Sensor System
if(NOT TARGET sensor_system)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sensor_system
                     ${CMAKE_CURRENT_BINARY_DIR}/sensor_system EXCLUDE_FROM_ALL)
endif()

# Sensor Implementations
if(NOT TARGET sensor_implementations)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../sensor_implementations
                     ${CMAKE_CURRENT_BINARY_DIR}/sensor_implementations EXCLUDE_FROM_ALL)
endif()

# Reward System
if(NOT TARGET reward_system)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../reward_system
                     ${CMAKE_CURRENT_BINARY_DIR}/reward_system EXCLUDE_FROM_ALL)
endif()

# Threading
if(NOT TARGET threading)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../threading
                     ${CMAKE_CURRENT_BINARY_DIR}/threading EXCLUDE_FROM_ALL)
endif()

# Configuration
if(NOT TARGET configuration)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../configuration
                     ${CMAKE_CURRENT_BINARY_DIR}/configuration EXCLUDE_FROM_ALL)
endif()

# OBJ I/O
if(NOT TARGET obj_io)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../obj_io
                     ${CMAKE_CURRENT_BINARY_DIR}/obj_io EXCLUDE_FROM_ALL)
endif()

# GPU
if(NOT TARGET gpu)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../gpu
                     ${CMAKE_CURRENT_BINARY_DIR}/gpu EXCLUDE_FROM_ALL)
endif()

# ============================================================================
# Environment Manager Library
# ============================================================================

set(ENVIRONMENT_MANAGER_SOURCES
    src/engine_config.c
    src/engine_lifecycle.c
    src/engine_step.c
    src/engine_reset.c
    src/engine_buffers.c
    src/engine_queries.c
    src/engine_stats.c
    src/engine_world.c
    src/puffer_env.c
)

set(ENVIRONMENT_MANAGER_HEADERS
    include/environment_manager.h
)

# Create static library
add_library(environment_manager STATIC ${ENVIRONMENT_MANAGER_SOURCES})

# Include directories
target_include_directories(environment_manager
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(environment_manager
    PUBLIC
        foundation
        drone_state
        physics
        platform
        world_brick_map
        collision_system
        sensor_system
        sensor_implementations
        reward_system
        threading
        configuration
        obj_io
        gpu
)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(environment_manager PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )

    # Architecture-specific flags
    if(ENVIRONMENT_MANAGER_ARCH_X86_64)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(environment_manager PUBLIC -mavx2)
            message(STATUS "Environment Manager: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(environment_manager PUBLIC -mfma)
            message(STATUS "Environment Manager: FMA enabled")
        endif()
    endif()

    if(ENVIRONMENT_MANAGER_ARCH_ARM)
        message(STATUS "Environment Manager: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(environment_manager PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(environment_manager PRIVATE
        /W4
    )

    if(ENVIRONMENT_MANAGER_ARCH_X86_64)
        target_compile_options(environment_manager PUBLIC /arch:AVX2)
        message(STATUS "Environment Manager: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(ENVIRONMENT_MANAGER_DEBUG)
    target_compile_definitions(environment_manager PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "Environment Manager: Debug mode enabled")
endif()

# Link math library on Unix
if(UNIX AND NOT APPLE)
    target_link_libraries(environment_manager PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(ENVIRONMENT_MANAGER_BUILD_TESTS)
    enable_testing()

    # Test executables
    set(TEST_SOURCES
        tests/test_engine_config.c
        tests/test_engine_lifecycle.c
        tests/test_engine_step.c
        tests/test_engine_reset.c
        tests/test_engine_buffers.c
        tests/test_engine_integration.c
        tests/test_engine_core.c
        tests/test_engine_memory.c
        tests/test_engine_determinism.c
        tests/test_determinism_deep.c
        tests/test_memory_deep.c
        tests/test_lidar_world_integration.c
        tests/test_camera_ground_truth.c
        tests/test_multi_engine.c
        tests/test_puffer_env.c
    )

    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE environment_manager)
        target_compile_definitions(${test_name} PRIVATE FOUNDATION_DEBUG=1)
        add_test(NAME ${test_name} COMMAND ${test_name}
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endforeach()

    # Extended timeout for heavy tests that need more time under parallel load
    set_tests_properties(test_determinism_deep test_memory_deep PROPERTIES TIMEOUT 120)

    # Custom target to run all tests
    add_custom_target(test_environment_manager_run
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        COMMENT "Running environment manager unit tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(ENVIRONMENT_MANAGER_BUILD_BENCHMARKS)
    add_executable(bench_environment tests/bench_environment.c)
    target_link_libraries(bench_environment PRIVATE environment_manager)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_environment PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_environment PRIVATE /O2)
    endif()

    # Custom target to run benchmarks
    add_custom_target(bench_environment_run
        COMMAND bench_environment
        DEPENDS bench_environment
        COMMENT "Running environment manager benchmarks"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS environment_manager
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${ENVIRONMENT_MANAGER_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/environment_manager
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Environment Manager Module Configuration:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${ENVIRONMENT_MANAGER_DEBUG}")
message(STATUS "  Build Tests: ${ENVIRONMENT_MANAGER_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${ENVIRONMENT_MANAGER_BUILD_BENCHMARKS}")
if(ENVIRONMENT_MANAGER_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(ENVIRONMENT_MANAGER_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "")
