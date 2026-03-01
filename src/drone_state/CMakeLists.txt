# Drone State Module CMake Configuration
#
# Build configuration for the drone-state module providing SoA data structures
# for drone state and parameters with SIMD-optimized batch operations.

cmake_minimum_required(VERSION 3.16)
project(drone_state C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(DRONE_STATE_DEBUG "Enable debug assertions and logging" OFF)
option(DRONE_STATE_BUILD_TESTS "Build unit tests" ON)
option(DRONE_STATE_BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(DRONE_STATE_ARCH_ARM TRUE)
    message(STATUS "DroneState: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(DRONE_STATE_ARCH_X86_64 TRUE)
    message(STATUS "DroneState: Detected x86_64 architecture")
else()
    message(STATUS "DroneState: Unknown architecture, using scalar fallback")
endif()

# Foundation dependency (sibling directory)
# Check if foundation is already defined (when built as part of larger project)
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

# Source files
set(DRONE_STATE_SOURCES
    src/drone_state.c
)

set(DRONE_STATE_HEADERS
    include/drone_state.h
)

# Create static library
add_library(drone_state STATIC ${DRONE_STATE_SOURCES})

# Include directories
target_include_directories(drone_state
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link to foundation and rigid_body_state
if(NOT TARGET rigid_body_state)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../rigid_body_state
                     ${CMAKE_CURRENT_BINARY_DIR}/rigid_body_state EXCLUDE_FROM_ALL)
endif()
target_link_libraries(drone_state PUBLIC foundation rigid_body_state)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(drone_state PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -ffast-math
    )

    # Architecture-specific flags (inherited from foundation, but ensure they're set)
    if(DRONE_STATE_ARCH_X86_64)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(drone_state PUBLIC -mavx2)
            message(STATUS "DroneState: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(drone_state PUBLIC -mfma)
            message(STATUS "DroneState: FMA enabled")
        endif()
    endif()

    if(DRONE_STATE_ARCH_ARM)
        # NEON is typically enabled by default on ARM64
        message(STATUS "DroneState: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(drone_state PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(drone_state PRIVATE
        /W4
        /fp:fast
    )

    if(DRONE_STATE_ARCH_X86_64)
        target_compile_options(drone_state PUBLIC /arch:AVX2)
        message(STATUS "DroneState: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(DRONE_STATE_DEBUG)
    target_compile_definitions(drone_state PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "DroneState: Debug mode enabled")
endif()

# Link math library on Unix (if not already linked through foundation)
if(UNIX AND NOT APPLE)
    target_link_libraries(drone_state PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(DRONE_STATE_BUILD_TESTS)
    enable_testing()

    add_executable(test_drone_state tests/test_drone_state.c)
    target_link_libraries(test_drone_state PRIVATE drone_state platform)

    # Enable debug assertions for tests
    target_compile_definitions(test_drone_state PRIVATE FOUNDATION_DEBUG=1)

    # Add to CTest
    add_test(NAME test_drone_state COMMAND test_drone_state)

    # Deep tests: scale, boundary, copy semantics (Yoneda philosophy)
    # NOTE: Compiled WITHOUT FOUNDATION_DEBUG because these tests exercise
    # out-of-bounds / NULL paths that rely on graceful fallback (not abort).
    add_executable(test_drone_state_deep tests/test_drone_state_deep.c)
    target_link_libraries(test_drone_state_deep PRIVATE drone_state platform)
    add_test(NAME test_drone_state_deep COMMAND test_drone_state_deep)

    # Custom target to run tests with output
    add_custom_target(test_drone_state_run
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_drone_state test_drone_state_deep
        COMMENT "Running drone_state unit tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(DRONE_STATE_BUILD_BENCHMARKS)
    add_executable(bench_drone_state tests/bench_drone_state.c)
    target_link_libraries(bench_drone_state PRIVATE drone_state platform)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_drone_state PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_drone_state PRIVATE /O2)
    endif()

    # Custom target to run benchmarks
    add_custom_target(bench_drone_state_run
        COMMAND bench_drone_state
        DEPENDS bench_drone_state
        COMMENT "Running drone_state benchmarks"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS drone_state
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${DRONE_STATE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/drone_state
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Drone State Module Configuration:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${DRONE_STATE_DEBUG}")
message(STATUS "  Build Tests: ${DRONE_STATE_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${DRONE_STATE_BUILD_BENCHMARKS}")
if(DRONE_STATE_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(DRONE_STATE_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "")
