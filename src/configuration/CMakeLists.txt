# Configuration Module CMake Configuration
#
# Build configuration for the configuration module providing TOML-based
# configuration parsing, validation, and conversion for the drone RL engine.

cmake_minimum_required(VERSION 3.16)
project(configuration C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(CONFIGURATION_DEBUG "Enable debug assertions and logging" OFF)
option(CONFIGURATION_BUILD_TESTS "Build unit tests" ON)
option(CONFIGURATION_BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(CONFIGURATION_ARCH_ARM TRUE)
    message(STATUS "Configuration: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(CONFIGURATION_ARCH_X86_64 TRUE)
    message(STATUS "Configuration: Detected x86_64 architecture")
else()
    message(STATUS "Configuration: Unknown architecture")
endif()

# ============================================================================
# Dependencies
# ============================================================================

# Foundation dependency (sibling directory)
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

# Drone State dependency (sibling directory)
if(NOT TARGET drone_state)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../drone_state
                     ${CMAKE_CURRENT_BINARY_DIR}/drone_state EXCLUDE_FROM_ALL)
endif()

# tomlc99 library - use local copy in external/
set(TOMLC99_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

if(NOT TARGET tomlc99)
    add_library(tomlc99 STATIC
        ${TOMLC99_DIR}/toml.c
    )
    target_include_directories(tomlc99
        PUBLIC
            $<BUILD_INTERFACE:${TOMLC99_DIR}>
            $<INSTALL_INTERFACE:include>
    )
    # Suppress warnings from external library
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(tomlc99 PRIVATE -w)
    endif()
endif()

set(tomlc99_SOURCE_DIR ${TOMLC99_DIR})

# yxml library - minimal XML parser for URDF
set(YXML_DIR ${CMAKE_CURRENT_SOURCE_DIR}/external)

if(NOT TARGET yxml)
    add_library(yxml STATIC
        ${YXML_DIR}/yxml.c
    )
    target_include_directories(yxml
        PUBLIC
            $<BUILD_INTERFACE:${YXML_DIR}>
            $<INSTALL_INTERFACE:include>
    )
    # Suppress warnings from external library
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(yxml PRIVATE -w)
    endif()
endif()

# ============================================================================
# Configuration Library
# ============================================================================

set(CONFIGURATION_SOURCES
    src/config_defaults.c
    src/config_parser.c
    src/config_validation.c
    src/config_conversion.c
    src/config_serialization.c
    src/urdf_parser.c
)

set(CONFIGURATION_HEADERS
    include/configuration.h
    include/urdf_parser.h
)

# Create static library
add_library(configuration STATIC ${CONFIGURATION_SOURCES})

# Include directories
target_include_directories(configuration
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${tomlc99_SOURCE_DIR}
)

# Link dependencies
target_link_libraries(configuration
    PUBLIC
        foundation
        drone_state
    PRIVATE
        tomlc99
        yxml
)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(configuration PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )

    # Architecture-specific flags
    if(CONFIGURATION_ARCH_X86_64)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(configuration PUBLIC -mavx2)
            message(STATUS "Configuration: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(configuration PUBLIC -mfma)
            message(STATUS "Configuration: FMA enabled")
        endif()
    endif()

    if(CONFIGURATION_ARCH_ARM)
        message(STATUS "Configuration: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(configuration PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(configuration PRIVATE
        /W4
    )

    if(CONFIGURATION_ARCH_X86_64)
        target_compile_options(configuration PUBLIC /arch:AVX2)
        message(STATUS "Configuration: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(CONFIGURATION_DEBUG)
    target_compile_definitions(configuration PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "Configuration: Debug mode enabled")
endif()

# Link math library on Unix
if(UNIX AND NOT APPLE)
    target_link_libraries(configuration PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(CONFIGURATION_BUILD_TESTS)
    enable_testing()

    # Main test executable
    add_executable(test_configuration tests/test_configuration.c)
    target_link_libraries(test_configuration PRIVATE configuration)

    # Enable debug assertions for tests
    target_compile_definitions(test_configuration PRIVATE FOUNDATION_DEBUG=1)

    # Add to CTest
    add_test(NAME test_configuration COMMAND test_configuration)

    # Set working directory for test data files
    set_tests_properties(test_configuration PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # URDF parser test executable
    add_executable(test_urdf tests/test_urdf.c)
    target_link_libraries(test_urdf PRIVATE configuration)
    target_compile_definitions(test_urdf PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_urdf COMMAND test_urdf)
    set_tests_properties(test_urdf PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # Custom target to run tests with output
    add_custom_target(test_configuration_run
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_configuration test_urdf
        COMMENT "Running configuration unit tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(CONFIGURATION_BUILD_BENCHMARKS)
    add_executable(bench_configuration tests/bench_configuration.c)
    target_link_libraries(bench_configuration PRIVATE configuration)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_configuration PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_configuration PRIVATE /O2)
    endif()

    # Custom target to run benchmarks
    add_custom_target(bench_configuration_run
        COMMAND bench_configuration
        DEPENDS bench_configuration
        COMMENT "Running configuration benchmarks"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS configuration tomlc99 yxml
    EXPORT configurationTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${CONFIGURATION_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/configuration
)

# Install toml.h header for tomlc99
install(FILES ${TOMLC99_DIR}/toml.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT configurationTargets
    FILE configurationTargets.cmake
    NAMESPACE configuration::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/configuration
)

# Install test data files
install(DIRECTORY test_data/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/configuration/test_data
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Configuration Module Configuration:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${CONFIGURATION_DEBUG}")
message(STATUS "  Build Tests: ${CONFIGURATION_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${CONFIGURATION_BUILD_BENCHMARKS}")
if(CONFIGURATION_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(CONFIGURATION_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "  tomlc99 source: ${tomlc99_SOURCE_DIR}")
message(STATUS "")
