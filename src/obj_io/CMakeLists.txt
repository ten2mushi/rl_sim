# OBJ I/O Module CMake Configuration
#
# Build configuration for the obj_io module providing bidirectional conversion
# between OBJ mesh files and WorldBrickMap SDF representation.
#
# Features:
# - Streaming OBJ/MTL parser for large files
# - Three-phase sparse voxelization with BVH acceleration
# - Marching Cubes mesh extraction
# - Material/color preservation

cmake_minimum_required(VERSION 3.16)
project(obj_io C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(OBJ_IO_DEBUG "Enable debug assertions and logging" OFF)
option(OBJ_IO_BUILD_TESTS "Build unit tests" ON)
option(OBJ_IO_BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(OBJ_IO_BUILD_TOOLS "Build CLI tools" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(OBJ_IO_ARCH_ARM TRUE)
    message(STATUS "ObjIO: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(OBJ_IO_ARCH_X86_64 TRUE)
    message(STATUS "ObjIO: Detected x86_64 architecture")
else()
    message(STATUS "ObjIO: Unknown architecture, using scalar fallback")
endif()

# Dependencies (sibling directories)
# Check if foundation is already defined (when built as part of larger project)
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

if(NOT TARGET world_brick_map)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../world_brick_map
                     ${CMAKE_CURRENT_BINARY_DIR}/world_brick_map EXCLUDE_FROM_ALL)
endif()

# Source files
set(OBJ_IO_SOURCES
    src/obj_parser.c
    src/mtl_parser.c
    src/mesh.c
    src/bvh.c
    src/voxelize.c
    src/marching_cubes.c
    src/obj_writer.c
)

set(OBJ_IO_HEADERS
    include/obj_io.h
)

# Create static library
add_library(obj_io STATIC ${OBJ_IO_SOURCES})

# Include directories
target_include_directories(obj_io
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link to dependencies
target_link_libraries(obj_io PUBLIC foundation world_brick_map)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(obj_io PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -ffast-math
    )

    # Architecture-specific flags
    if(OBJ_IO_ARCH_X86_64)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(obj_io PUBLIC -mavx2)
            message(STATUS "ObjIO: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(obj_io PUBLIC -mfma)
            message(STATUS "ObjIO: FMA enabled")
        endif()
    endif()

    if(OBJ_IO_ARCH_ARM)
        # NEON is typically enabled by default on ARM64
        message(STATUS "ObjIO: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(obj_io PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(obj_io PRIVATE
        /W4
        /fp:fast
    )

    if(OBJ_IO_ARCH_X86_64)
        target_compile_options(obj_io PUBLIC /arch:AVX2)
        message(STATUS "ObjIO: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(OBJ_IO_DEBUG)
    target_compile_definitions(obj_io PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "ObjIO: Debug mode enabled")
endif()

# Link math library on Unix (if not already linked through foundation)
if(UNIX AND NOT APPLE)
    target_link_libraries(obj_io PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(OBJ_IO_BUILD_TESTS)
    enable_testing()

    # OBJ Parser tests
    add_executable(test_obj_parser tests/test_obj_parser.c)
    target_link_libraries(test_obj_parser PRIVATE obj_io)
    target_compile_definitions(test_obj_parser PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_obj_parser COMMAND test_obj_parser)

    # MTL Parser tests
    add_executable(test_mtl_parser tests/test_mtl_parser.c)
    target_link_libraries(test_mtl_parser PRIVATE obj_io)
    target_compile_definitions(test_mtl_parser PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_mtl_parser COMMAND test_mtl_parser)

    # Mesh operations tests
    add_executable(test_mesh tests/test_mesh.c)
    target_link_libraries(test_mesh PRIVATE obj_io)
    target_compile_definitions(test_mesh PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_mesh COMMAND test_mesh)

    # Voxelization tests
    add_executable(test_voxelize tests/test_voxelize.c)
    target_link_libraries(test_voxelize PRIVATE obj_io)
    target_compile_definitions(test_voxelize PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_voxelize COMMAND test_voxelize)

    # Marching Cubes tests
    add_executable(test_marching_cubes tests/test_marching_cubes.c)
    target_link_libraries(test_marching_cubes PRIVATE obj_io)
    target_compile_definitions(test_marching_cubes PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_marching_cubes COMMAND test_marching_cubes)

    # Material handling tests
    add_executable(test_materials tests/test_materials.c)
    target_link_libraries(test_materials PRIVATE obj_io)
    target_compile_definitions(test_materials PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_materials COMMAND test_materials)

    # Roundtrip validation tests
    add_executable(test_roundtrip tests/test_roundtrip.c)
    target_link_libraries(test_roundtrip PRIVATE obj_io)
    target_compile_definitions(test_roundtrip PRIVATE FOUNDATION_DEBUG=1)
    add_test(NAME test_roundtrip COMMAND test_roundtrip)

    # Custom target to run all tests with output
    add_custom_target(test_obj_io_run
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_obj_parser test_mtl_parser test_mesh test_voxelize
                test_marching_cubes test_materials test_roundtrip
        COMMENT "Running obj_io unit tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(OBJ_IO_BUILD_BENCHMARKS)
    add_executable(bench_obj_io tests/bench_obj_io.c)
    target_link_libraries(bench_obj_io PRIVATE obj_io)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_obj_io PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_obj_io PRIVATE /O2)
    endif()

    # Custom target to run benchmarks
    add_custom_target(bench_obj_io_run
        COMMAND bench_obj_io
        DEPENDS bench_obj_io
        COMMENT "Running obj_io benchmarks"
    )
endif()

# ============================================================================
# CLI Tools
# ============================================================================

if(OBJ_IO_BUILD_TOOLS)
    add_executable(obj_io_tool tools/obj_io_tool.c)
    target_link_libraries(obj_io_tool PRIVATE obj_io)

    # Optimize tools
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(obj_io_tool PRIVATE -O2)
    elseif(MSVC)
        target_compile_options(obj_io_tool PRIVATE /O2)
    endif()

    # Install tool
    install(TARGETS obj_io_tool
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS obj_io
    EXPORT obj_ioTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${OBJ_IO_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/obj_io
)

install(EXPORT obj_ioTargets
    FILE obj_ioTargets.cmake
    NAMESPACE obj_io::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/obj_io
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "OBJ I/O Module Configuration:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${OBJ_IO_DEBUG}")
message(STATUS "  Build Tests: ${OBJ_IO_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${OBJ_IO_BUILD_BENCHMARKS}")
message(STATUS "  Build Tools: ${OBJ_IO_BUILD_TOOLS}")
if(OBJ_IO_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(OBJ_IO_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "")
