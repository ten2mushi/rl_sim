# Collision System Module CMake Configuration
#
# Build configuration for the collision-system module providing spatial hash-based
# collision detection for drone simulation with SIMD-optimized batch operations.
#
# Dependencies:
# - foundation: Vec3, Arena, SIMD macros
# - drone_state: DroneStateSOA, DroneParamsSOA
# - world_brick_map: world_sdf_query_batch, world_sdf_gradient

cmake_minimum_required(VERSION 3.16)
project(collision_system C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(COLLISION_SYSTEM_DEBUG "Enable debug assertions and logging" OFF)
option(COLLISION_SYSTEM_BUILD_TESTS "Build unit tests" ON)
option(COLLISION_SYSTEM_BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(COLLISION_SYSTEM_ARCH_ARM TRUE)
    message(STATUS "CollisionSystem: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(COLLISION_SYSTEM_ARCH_X86_64 TRUE)
    message(STATUS "CollisionSystem: Detected x86_64 architecture")
else()
    message(STATUS "CollisionSystem: Unknown architecture, using scalar fallback")
endif()

# Foundation dependency (sibling directory)
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

# Drone state dependency (sibling directory)
if(NOT TARGET drone_state)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../drone_state
                     ${CMAKE_CURRENT_BINARY_DIR}/drone_state EXCLUDE_FROM_ALL)
endif()

# World brick map dependency (sibling directory)
if(NOT TARGET world_brick_map)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../world_brick_map
                     ${CMAKE_CURRENT_BINARY_DIR}/world_brick_map EXCLUDE_FROM_ALL)
endif()

# Source files
set(COLLISION_SYSTEM_SOURCES
    src/collision_system.c
)

set(COLLISION_SYSTEM_HEADERS
    include/collision_system.h
)

# Create static library
add_library(collision_system STATIC ${COLLISION_SYSTEM_SOURCES})

# Include directories
target_include_directories(collision_system
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(collision_system
    PUBLIC
        foundation
        drone_state
        world_brick_map
)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(collision_system PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -ffast-math
    )

    # Architecture-specific flags
    if(COLLISION_SYSTEM_ARCH_X86_64)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(collision_system PUBLIC -mavx2)
            message(STATUS "CollisionSystem: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(collision_system PUBLIC -mfma)
            message(STATUS "CollisionSystem: FMA enabled")
        endif()
    endif()

    if(COLLISION_SYSTEM_ARCH_ARM)
        # NEON is typically enabled by default on ARM64
        message(STATUS "CollisionSystem: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(collision_system PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(collision_system PRIVATE
        /W4
        /fp:fast
    )

    if(COLLISION_SYSTEM_ARCH_X86_64)
        target_compile_options(collision_system PUBLIC /arch:AVX2)
        message(STATUS "CollisionSystem: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(COLLISION_SYSTEM_DEBUG)
    target_compile_definitions(collision_system PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "CollisionSystem: Debug mode enabled")
endif()

# Link math library on Unix
if(UNIX AND NOT APPLE)
    target_link_libraries(collision_system PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(COLLISION_SYSTEM_BUILD_TESTS)
    enable_testing()

    add_executable(test_collision_system tests/test_collision_system.c)
    target_link_libraries(test_collision_system PRIVATE collision_system)

    # Enable debug assertions for tests
    target_compile_definitions(test_collision_system PRIVATE FOUNDATION_DEBUG=1)

    # Add to CTest
    add_test(NAME test_collision_system COMMAND test_collision_system)

    # Custom target to run tests with output
    add_custom_target(test_collision_system_run
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_collision_system
        COMMENT "Running collision_system unit tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(COLLISION_SYSTEM_BUILD_BENCHMARKS)
    add_executable(bench_collision_system tests/bench_collision_system.c)
    target_link_libraries(bench_collision_system PRIVATE collision_system)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_collision_system PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_collision_system PRIVATE /O2)
    endif()

    # Custom target to run benchmarks
    add_custom_target(bench_collision_system_run
        COMMAND bench_collision_system
        DEPENDS bench_collision_system
        COMMENT "Running collision_system benchmarks"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS collision_system
    EXPORT collision_systemTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${COLLISION_SYSTEM_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/collision_system
)

install(EXPORT collision_systemTargets
    FILE collision_systemTargets.cmake
    NAMESPACE collision_system::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/collision_system
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Collision System Module Configuration:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${COLLISION_SYSTEM_DEBUG}")
message(STATUS "  Build Tests: ${COLLISION_SYSTEM_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${COLLISION_SYSTEM_BUILD_BENCHMARKS}")
if(COLLISION_SYSTEM_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(COLLISION_SYSTEM_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "")
