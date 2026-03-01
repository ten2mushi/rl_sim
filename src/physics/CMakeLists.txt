# Physics Engine Module CMake Configuration
#
# Build configuration for the physics module providing RK4-integrated
# quadcopter physics with SIMD-optimized batch processing.

cmake_minimum_required(VERSION 3.16)
project(physics C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(PHYSICS_DEBUG "Enable debug assertions and logging" OFF)
option(PHYSICS_BUILD_TESTS "Build unit tests" ON)
option(PHYSICS_BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(PHYSICS_ARCH_ARM TRUE)
    message(STATUS "Physics: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(PHYSICS_ARCH_X86_64 TRUE)
    message(STATUS "Physics: Detected x86_64 architecture")
else()
    message(STATUS "Physics: Unknown architecture, using scalar fallback")
endif()

# Dependencies (sibling directories)
# Check if foundation and drone_state are already defined
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET drone_state)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../drone_state
                     ${CMAKE_CURRENT_BINARY_DIR}/drone_state EXCLUDE_FROM_ALL)
endif()

# Source files
set(PHYSICS_SOURCES
    src/physics.c
)

set(PHYSICS_HEADERS
    include/physics.h
)

# Create static library
add_library(physics STATIC ${PHYSICS_SOURCES})

# Include directories
target_include_directories(physics
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        # Platform include for PlatformVTable definition (used in physics.c)
        # Not a link dependency to avoid circular dep: platform -> physics -> platform
        ${CMAKE_CURRENT_SOURCE_DIR}/../platform/include
)

# Link to dependencies
target_link_libraries(physics PUBLIC foundation drone_state)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(physics PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -ffast-math
    )

    # Architecture-specific flags
    if(PHYSICS_ARCH_X86_64)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(physics PUBLIC -mavx2)
            message(STATUS "Physics: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(physics PUBLIC -mfma)
            message(STATUS "Physics: FMA enabled")
        endif()
    endif()

    if(PHYSICS_ARCH_ARM)
        # NEON is typically enabled by default on ARM64
        message(STATUS "Physics: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(physics PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(physics PRIVATE
        /W4
        /fp:fast
    )

    if(PHYSICS_ARCH_X86_64)
        target_compile_options(physics PUBLIC /arch:AVX2)
        message(STATUS "Physics: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(PHYSICS_DEBUG)
    target_compile_definitions(physics PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "Physics: Debug mode enabled")
endif()

# Link math library on Unix
if(UNIX AND NOT APPLE)
    target_link_libraries(physics PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(PHYSICS_BUILD_TESTS)
    enable_testing()

    add_executable(test_physics tests/test_physics.c)
    target_link_libraries(test_physics PRIVATE physics platform)

    # Enable debug assertions for tests
    target_compile_definitions(test_physics PRIVATE FOUNDATION_DEBUG=1)

    # Add to CTest
    add_test(NAME test_physics COMMAND test_physics)

    # Custom target to run tests with output
    add_custom_target(test_physics_run
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_physics
        COMMENT "Running physics unit tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(PHYSICS_BUILD_BENCHMARKS)
    add_executable(bench_physics tests/bench_physics.c)
    target_link_libraries(bench_physics PRIVATE physics platform)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_physics PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_physics PRIVATE /O2)
    endif()

    # Custom target to run benchmarks
    add_custom_target(bench_physics_run
        COMMAND bench_physics
        DEPENDS bench_physics
        COMMENT "Running physics benchmarks"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS physics
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${PHYSICS_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/physics
)


# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Physics Module Configuration:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${PHYSICS_DEBUG}")
message(STATUS "  Build Tests: ${PHYSICS_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${PHYSICS_BUILD_BENCHMARKS}")
if(PHYSICS_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(PHYSICS_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "")
