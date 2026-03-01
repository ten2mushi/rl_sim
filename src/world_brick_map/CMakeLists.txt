# World Brick Map Module CMake Configuration
#
# Build configuration for the world-brick-map module providing sparse SDF storage
# with 8x8x8 brick atlas for efficient world representation and raymarching.

cmake_minimum_required(VERSION 3.16)
project(world_brick_map C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(WORLD_BRICK_MAP_DEBUG "Enable debug assertions and logging" OFF)
option(WORLD_BRICK_MAP_BUILD_TESTS "Build unit tests" ON)
option(WORLD_BRICK_MAP_BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(WORLD_BRICK_MAP_ARCH_ARM TRUE)
    message(STATUS "WorldBrickMap: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(WORLD_BRICK_MAP_ARCH_X86_64 TRUE)
    message(STATUS "WorldBrickMap: Detected x86_64 architecture")
else()
    message(STATUS "WorldBrickMap: Unknown architecture, using scalar fallback")
endif()

# Foundation dependency (sibling directory)
# Check if foundation is already defined (when built as part of larger project)
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

# Source files
set(WORLD_BRICK_MAP_SOURCES
    src/world_brick_map.c
    src/incremental.c
    src/clipmap.c
)

set(WORLD_BRICK_MAP_HEADERS
    include/world_brick_map.h
)

# Create static library
add_library(world_brick_map STATIC ${WORLD_BRICK_MAP_SOURCES})

# Include directories
target_include_directories(world_brick_map
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link to foundation
target_link_libraries(world_brick_map PUBLIC foundation)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(world_brick_map PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -ffast-math
    )

    # Architecture-specific flags
    if(WORLD_BRICK_MAP_ARCH_X86_64)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(world_brick_map PUBLIC -mavx2)
            message(STATUS "WorldBrickMap: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(world_brick_map PUBLIC -mfma)
            message(STATUS "WorldBrickMap: FMA enabled")
        endif()
    endif()

    if(WORLD_BRICK_MAP_ARCH_ARM)
        # NEON is typically enabled by default on ARM64
        message(STATUS "WorldBrickMap: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(world_brick_map PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(world_brick_map PRIVATE
        /W4
        /fp:fast
    )

    if(WORLD_BRICK_MAP_ARCH_X86_64)
        target_compile_options(world_brick_map PUBLIC /arch:AVX2)
        message(STATUS "WorldBrickMap: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(WORLD_BRICK_MAP_DEBUG)
    target_compile_definitions(world_brick_map PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "WorldBrickMap: Debug mode enabled")
endif()

# Link math library on Unix (if not already linked through foundation)
if(UNIX AND NOT APPLE)
    target_link_libraries(world_brick_map PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(WORLD_BRICK_MAP_BUILD_TESTS)
    enable_testing()

    add_executable(test_world_brick_map tests/test_world_brick_map.c)
    target_link_libraries(test_world_brick_map PRIVATE world_brick_map)

    # Enable debug assertions for tests
    target_compile_definitions(test_world_brick_map PRIVATE FOUNDATION_DEBUG=1)

    # Add to CTest
    add_test(NAME test_world_brick_map COMMAND test_world_brick_map)

    # Geometry primitives and mesh integrity tests (Yoneda philosophy)
    add_executable(test_geometry_primitives tests/test_geometry_primitives.c)
    target_link_libraries(test_geometry_primitives PRIVATE world_brick_map)
    target_compile_definitions(test_geometry_primitives PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_geometry_primitives COMMAND test_geometry_primitives)

    # Material registry tests
    add_executable(test_world_materials tests/test_world_materials.c)
    target_link_libraries(test_world_materials PRIVATE world_brick_map)
    target_compile_definitions(test_world_materials PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_world_materials COMMAND test_world_materials)

    # Advanced tests: clipmap, incremental, boundary queries (Yoneda philosophy)
    add_executable(test_world_advanced tests/test_world_advanced.c)
    target_link_libraries(test_world_advanced PRIVATE world_brick_map)
    target_compile_definitions(test_world_advanced PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_world_advanced COMMAND test_world_advanced)

    # Custom target to run tests with output
    add_custom_target(test_world_brick_map_run
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_world_brick_map test_geometry_primitives test_world_materials test_world_advanced
        COMMENT "Running world_brick_map unit tests"
    )

    # Custom target to run geometry primitives tests specifically
    add_custom_target(test_geometry_primitives_run
        COMMAND test_geometry_primitives
        DEPENDS test_geometry_primitives
        COMMENT "Running geometry primitives and mesh integrity tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(WORLD_BRICK_MAP_BUILD_BENCHMARKS)
    add_executable(bench_world_brick_map tests/bench_world_brick_map.c)
    target_link_libraries(bench_world_brick_map PRIVATE world_brick_map)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_world_brick_map PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_world_brick_map PRIVATE /O2)
    endif()

    # Custom target to run benchmarks
    add_custom_target(bench_world_brick_map_run
        COMMAND bench_world_brick_map
        DEPENDS bench_world_brick_map
        COMMENT "Running world_brick_map benchmarks"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS world_brick_map
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${WORLD_BRICK_MAP_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/world_brick_map
)


# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "World Brick Map Module Configuration:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${WORLD_BRICK_MAP_DEBUG}")
message(STATUS "  Build Tests: ${WORLD_BRICK_MAP_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${WORLD_BRICK_MAP_BUILD_BENCHMARKS}")
if(WORLD_BRICK_MAP_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(WORLD_BRICK_MAP_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "")
