# Foundation Module CMake Configuration
#
# Build configuration for the foundation module providing core types,
# arena allocators, SIMD utilities, and math primitives.

cmake_minimum_required(VERSION 3.16)
project(foundation C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(FOUNDATION_DEBUG "Enable debug assertions and logging" OFF)
option(FOUNDATION_BUILD_TESTS "Build unit tests" ON)
option(FOUNDATION_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(FOUNDATION_ENABLE_AVX2 "Enable AVX2 SIMD instructions (x86_64)" ON)
option(FOUNDATION_ENABLE_NEON "Enable NEON SIMD instructions (ARM)" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(FOUNDATION_ARCH_ARM TRUE)
    message(STATUS "Foundation: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(FOUNDATION_ARCH_X86_64 TRUE)
    message(STATUS "Foundation: Detected x86_64 architecture")
else()
    message(STATUS "Foundation: Unknown architecture, using scalar fallback")
endif()

# Source files
set(FOUNDATION_SOURCES
    src/arena.c
    src/math.c
    src/pcg32.c
)

set(FOUNDATION_HEADERS
    include/foundation.h
)

# Create static library
add_library(foundation STATIC ${FOUNDATION_SOURCES})

# Include directories
target_include_directories(foundation
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(foundation PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -ffast-math
    )

    # Architecture-specific flags
    if(FOUNDATION_ARCH_X86_64 AND FOUNDATION_ENABLE_AVX2)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(foundation PUBLIC -mavx2)
            message(STATUS "Foundation: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(foundation PUBLIC -mfma)
            message(STATUS "Foundation: FMA enabled")
        endif()
    endif()

    if(FOUNDATION_ARCH_ARM AND FOUNDATION_ENABLE_NEON)
        # NEON is typically enabled by default on ARM64
        message(STATUS "Foundation: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(foundation PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(foundation PRIVATE
        /W4
        /fp:fast
    )

    if(FOUNDATION_ARCH_X86_64 AND FOUNDATION_ENABLE_AVX2)
        target_compile_options(foundation PUBLIC /arch:AVX2)
        message(STATUS "Foundation: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(FOUNDATION_DEBUG)
    target_compile_definitions(foundation PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "Foundation: Debug mode enabled")
endif()

# Link math library on Unix
if(UNIX AND NOT APPLE)
    target_link_libraries(foundation PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(FOUNDATION_BUILD_TESTS)
    enable_testing()

    # Test executables
    set(TEST_SOURCES
        tests/test_arena.c
        tests/test_math.c
        tests/test_pcg32.c
        tests/test_simd.c
    )

    foreach(TEST_SRC ${TEST_SOURCES})
        get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_SRC})
        target_link_libraries(${TEST_NAME} PRIVATE foundation)

        # Enable debug for tests
        target_compile_definitions(${TEST_NAME} PRIVATE FOUNDATION_DEBUG=1)

        # Add to CTest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    endforeach()

    # Custom target to run all tests
    add_custom_target(test_foundation
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_arena test_math test_pcg32 test_simd
        COMMENT "Running foundation unit tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(FOUNDATION_BUILD_BENCHMARKS)
    add_executable(bench_foundation tests/bench_foundation.c)
    target_link_libraries(bench_foundation PRIVATE foundation)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_foundation PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_foundation PRIVATE /O2)
    endif()
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS foundation
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${FOUNDATION_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/foundation
)


# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Foundation Module Configuration Summary:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${FOUNDATION_DEBUG}")
message(STATUS "  Build Tests: ${FOUNDATION_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${FOUNDATION_BUILD_BENCHMARKS}")
if(FOUNDATION_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(FOUNDATION_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "")
