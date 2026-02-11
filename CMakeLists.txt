# RL Engine - Top Level CMake Configuration
#
# Unified build system for all drone RL engine modules.
# Builds all modules in correct dependency order with full test coverage.
#
# Modules (in dependency order):
#   00-foundation          - Core utilities: Arena, Vec3, Quat, PCG32, SIMD
#   01-drone_state         - SoA data structures for drone state/params
#   02-physics             - RK4 quadcopter physics integration
#   03-world_brick_map     - Sparse SDF world representation
#   04-collision_system    - Spatial hash collision detection
#   05-sensor_system       - Sensor framework with vtable polymorphism
#   06-sensor_impl         - Concrete sensor implementations
#   07-reward_system       - Task rewards and termination conditions
#   08-threading           - Thread pool and work-stealing scheduler
#   10-configuration       - TOML config parsing
#   09-environment_manager - Top-level orchestration (PufferLib API)
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_BUILD_TYPE=Release
#   make -j8
#   ctest --output-on-failure

cmake_minimum_required(VERSION 3.16)
project(rl_engine C)

# ============================================================================
# Global Settings
# ============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(BUILD_TESTING "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build performance benchmarks" ON)
option(RL_ENGINE_ENABLE_SIMD "Enable SIMD optimizations" ON)

# Sanitizer options
option(SANITIZE_ADDRESS "Enable AddressSanitizer" OFF)
option(SANITIZE_THREAD "Enable ThreadSanitizer" OFF)
option(SANITIZE_UNDEFINED "Enable UndefinedBehaviorSanitizer" OFF)

if(SANITIZE_ADDRESS)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
    message(STATUS "AddressSanitizer ENABLED")
endif()

if(SANITIZE_THREAD)
    add_compile_options(-fsanitize=thread)
    add_link_options(-fsanitize=thread)
    message(STATUS "ThreadSanitizer ENABLED")
endif()

if(SANITIZE_UNDEFINED)
    add_compile_options(-fsanitize=undefined -fno-sanitize-recover=all)
    add_link_options(-fsanitize=undefined)
    message(STATUS "UndefinedBehaviorSanitizer ENABLED")
endif()

# Enable testing
if(BUILD_TESTING)
    enable_testing()
endif()

# ============================================================================
# Architecture Detection
# ============================================================================

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64|arm64")
    set(RL_ENGINE_ARCH_ARM TRUE)
    message(STATUS "RL Engine: Detected ARM architecture (NEON)")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(RL_ENGINE_ARCH_X86_64 TRUE)
    message(STATUS "RL Engine: Detected x86_64 architecture (AVX2)")
else()
    message(STATUS "RL Engine: Unknown architecture, using scalar fallback")
endif()

# ============================================================================
# Modules (in dependency order)
# ============================================================================

# Layer 0: No dependencies
add_subdirectory(src/foundation)

# Layer 1: Depends on foundation only
add_subdirectory(src/drone_state)
add_subdirectory(src/threading)
add_subdirectory(src/world_brick_map)
add_subdirectory(src/gpu)

# Layer 2: Depends on foundation + drone_state
add_subdirectory(src/physics)
add_subdirectory(src/sensor_system)

# Layer 3: Depends on foundation + drone_state + world_brick_map
add_subdirectory(src/collision_system)
add_subdirectory(src/obj_io)

# Layer 4: Depends on collision_system
add_subdirectory(src/reward_system)

# Layer 5: Depends on sensor_system + collision_system + world_brick_map
add_subdirectory(src/sensor_implementations)

# Layer 6: TOML configuration (foundation + drone_state)
add_subdirectory(src/configuration)

# Layer 7: Top-level orchestration (all modules)
add_subdirectory(src/environment_manager)

# ============================================================================
# Unified Library: libdronerl.a
# ============================================================================
option(BUILD_UNIFIED_LIBRARY "Build unified libdronerl.a" ON)

if(BUILD_UNIFIED_LIBRARY)
    # Collect all source files from each module
    file(GLOB_RECURSE DRONERL_SOURCES
        "src/foundation/src/*.c"
        "src/drone_state/src/*.c"
        "src/physics/src/*.c"
        "src/world_brick_map/src/*.c"
        "src/collision_system/src/*.c"
        "src/sensor_system/src/*.c"
        "src/sensor_implementations/src/*.c"
        "src/reward_system/src/*.c"
        "src/threading/src/*.c"
        "src/configuration/src/*.c"
        "src/configuration/external/toml.c"
        "src/configuration/external/yxml.c"
        "src/obj_io/src/*.c"
        "src/environment_manager/src/*.c"
        "src/gpu/src/*.c"
    )

    # On Apple, include Objective-C Metal backend
    if(APPLE AND GPU_AVAILABLE)
        file(GLOB GPU_OBJC_SOURCES "src/gpu/src/*.m")
        list(APPEND DRONERL_SOURCES ${GPU_OBJC_SOURCES})
    endif()

    add_library(dronerl_static STATIC ${DRONERL_SOURCES})

    target_include_directories(dronerl_static PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/foundation/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/drone_state/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/physics/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/world_brick_map/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/collision_system/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/sensor_system/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/sensor_implementations/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/reward_system/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/threading/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/configuration/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/configuration/external>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/obj_io/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/environment_manager/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders>
    )

    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(dronerl_static PRIVATE -Wall -Wextra -Wpedantic)
        if(RL_ENGINE_ARCH_X86_64)
            target_compile_options(dronerl_static PUBLIC -mavx2 -mfma)
        endif()
    endif()

    if(UNIX)
        target_link_libraries(dronerl_static PUBLIC pthread)
        if(NOT APPLE)
            target_link_libraries(dronerl_static PUBLIC m)
        endif()
    endif()

    # GPU: Metal framework linking and ObjC ARC
    if(APPLE AND GPU_AVAILABLE)
        target_link_libraries(dronerl_static PUBLIC
            "-framework Metal"
            "-framework Foundation"
        )
        # Enable ARC for .m files
        foreach(objc_src ${GPU_OBJC_SOURCES})
            set_source_files_properties(${objc_src} PROPERTIES
                COMPILE_FLAGS "-fobjc-arc"
            )
        endforeach()
    endif()

    set_target_properties(dronerl_static PROPERTIES
        OUTPUT_NAME dronerl
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )

    message(STATUS "Building unified libdronerl.a")
endif()

# ============================================================================
# Cross-Layer Tests (depend on targets from multiple layers)
# ============================================================================

# test_sensor_system depends on sensor_system (Layer 2) + sensor_implementations (Layer 5)
if(BUILD_TESTING)
    add_executable(test_sensor_system
        src/sensor_system/tests/test_sensor_system.c)
    target_link_libraries(test_sensor_system PRIVATE
        sensor_system sensor_implementations)
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_sensor_system PRIVATE m)
    endif()
    target_compile_features(test_sensor_system PRIVATE c_std_11)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_sensor_system PRIVATE
            -Wall -Wextra -g -O0 -DFOUNDATION_DEBUG)
    endif()
    add_test(NAME test_sensor_system COMMAND test_sensor_system)

    add_executable(test_noise
        src/sensor_system/tests/test_noise.c)
    target_link_libraries(test_noise PRIVATE sensor_system)
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_noise PRIVATE m)
    endif()
    target_compile_features(test_noise PRIVATE c_std_11)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(test_noise PRIVATE
            -Wall -Wextra -g -O0 -DFOUNDATION_DEBUG)
    endif()
    add_test(NAME test_noise COMMAND test_noise)
endif()

if(BUILD_BENCHMARKS)
    add_executable(bench_sensor_system
        src/sensor_system/tests/bench_sensor_system.c)
    target_link_libraries(bench_sensor_system PRIVATE
        sensor_system sensor_implementations)
    if(UNIX AND NOT APPLE)
        target_link_libraries(bench_sensor_system PRIVATE m)
    endif()
    target_compile_features(bench_sensor_system PRIVATE c_std_11)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(bench_sensor_system PRIVATE -Wall -Wextra -O3)
    endif()
endif()

# ============================================================================
# Diagnostic Tools
# ============================================================================
if(BUILD_UNIFIED_LIBRARY)
    add_executable(diag_mountain_roundtrip
        src/obj_io/tests/diag_mountain_roundtrip.c)
    target_link_libraries(diag_mountain_roundtrip PRIVATE dronerl_static)
    set_target_properties(diag_mountain_roundtrip PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    add_executable(diag_ridge_artifacts
        src/obj_io/tests/diag_ridge_artifacts.c)
    target_link_libraries(diag_ridge_artifacts PRIVATE dronerl_static)
    set_target_properties(diag_ridge_artifacts PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    add_executable(diag_gyroid_signs
        src/obj_io/tests/diag_gyroid_signs.c)
    target_link_libraries(diag_gyroid_signs PRIVATE dronerl_static)
    set_target_properties(diag_gyroid_signs PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    add_executable(diag_mc_welding
        src/obj_io/tests/diag_mc_welding.c)
    target_link_libraries(diag_mc_welding PRIVATE dronerl_static)
    set_target_properties(diag_mc_welding PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

    add_executable(diag_shell_artifacts
        src/obj_io/tests/diag_shell_artifacts.c)
    target_link_libraries(diag_shell_artifacts PRIVATE dronerl_static)
    set_target_properties(diag_shell_artifacts PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
endif()

# ============================================================================
# Comprehensive Benchmark Suite
# ============================================================================
if(BUILD_BENCHMARKS AND BUILD_UNIFIED_LIBRARY)
    add_subdirectory(benchmarks)
endif()

# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════╗")
message(STATUS "║              RL Engine Build Configuration               ║")
message(STATUS "╠══════════════════════════════════════════════════════════╣")
message(STATUS "║  Build Type:        ${CMAKE_BUILD_TYPE}")
message(STATUS "║  Build Tests:       ${BUILD_TESTING}")
message(STATUS "║  Build Benchmarks:  ${BUILD_BENCHMARKS}")
message(STATUS "║  SIMD Enabled:      ${RL_ENGINE_ENABLE_SIMD}")
if(GPU_AVAILABLE)
message(STATUS "║  GPU Backend:       METAL")
else()
message(STATUS "║  GPU Backend:       NONE")
endif()
if(RL_ENGINE_ARCH_ARM)
message(STATUS "║  Architecture:      ARM64 (NEON)")
elseif(RL_ENGINE_ARCH_X86_64)
message(STATUS "║  Architecture:      x86_64 (AVX2)")
else()
message(STATUS "║  Architecture:      Scalar")
endif()
message(STATUS "╚══════════════════════════════════════════════════════════╝")
message(STATUS "")
