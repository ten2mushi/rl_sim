# Reward System Module CMake Configuration
#
# Build configuration for the reward-system module providing task-specific
# reward computation and termination conditions for drone RL training.
#
# Dependencies:
# - foundation: Vec3, Arena, SIMD macros, PCG32
# - drone_state: DroneStateSOA, DroneParamsSOA
# - collision_system: CollisionResults

cmake_minimum_required(VERSION 3.16)
project(reward_system C)

# C standard
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Options
option(REWARD_SYSTEM_DEBUG "Enable debug assertions and logging" OFF)
option(REWARD_SYSTEM_BUILD_TESTS "Build unit tests" ON)
option(REWARD_SYSTEM_BUILD_BENCHMARKS "Build performance benchmarks" ON)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm|ARM|aarch64|AARCH64")
    set(REWARD_SYSTEM_ARCH_ARM TRUE)
    message(STATUS "RewardSystem: Detected ARM architecture")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    set(REWARD_SYSTEM_ARCH_X86_64 TRUE)
    message(STATUS "RewardSystem: Detected x86_64 architecture")
else()
    message(STATUS "RewardSystem: Unknown architecture, using scalar fallback")
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

# Collision system dependency (sibling directory)
if(NOT TARGET collision_system)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../collision_system
                     ${CMAKE_CURRENT_BINARY_DIR}/collision_system EXCLUDE_FROM_ALL)
endif()

# Source files
set(REWARD_SYSTEM_SOURCES
    src/reward_system.c
)

set(REWARD_SYSTEM_HEADERS
    include/reward_system.h
)

# Create static library
add_library(reward_system STATIC ${REWARD_SYSTEM_SOURCES})

# Include directories
target_include_directories(reward_system
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link dependencies
target_link_libraries(reward_system
    PUBLIC
        foundation
        drone_state
        collision_system
)

# Compiler flags
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(reward_system PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
        -ffast-math
    )

    # Architecture-specific flags
    if(REWARD_SYSTEM_ARCH_X86_64)
        include(CheckCCompilerFlag)
        check_c_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
        check_c_compiler_flag("-mfma" COMPILER_SUPPORTS_FMA)

        if(COMPILER_SUPPORTS_AVX2)
            target_compile_options(reward_system PUBLIC -mavx2)
            message(STATUS "RewardSystem: AVX2 enabled")
        endif()
        if(COMPILER_SUPPORTS_FMA)
            target_compile_options(reward_system PUBLIC -mfma)
            message(STATUS "RewardSystem: FMA enabled")
        endif()
    endif()

    if(REWARD_SYSTEM_ARCH_ARM)
        # NEON is typically enabled by default on ARM64
        message(STATUS "RewardSystem: NEON enabled (default on ARM64)")
    endif()

    # Release optimizations
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        target_compile_options(reward_system PRIVATE -O3 -DNDEBUG)
    endif()

elseif(MSVC)
    target_compile_options(reward_system PRIVATE
        /W4
        /fp:fast
    )

    if(REWARD_SYSTEM_ARCH_X86_64)
        target_compile_options(reward_system PUBLIC /arch:AVX2)
        message(STATUS "RewardSystem: AVX2 enabled (MSVC)")
    endif()
endif()

# Debug mode
if(REWARD_SYSTEM_DEBUG)
    target_compile_definitions(reward_system PUBLIC FOUNDATION_DEBUG=1)
    message(STATUS "RewardSystem: Debug mode enabled")
endif()

# Link math library on Unix
if(UNIX AND NOT APPLE)
    target_link_libraries(reward_system PUBLIC m)
endif()

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# Tests
# ============================================================================

if(REWARD_SYSTEM_BUILD_TESTS)
    enable_testing()

    add_executable(test_reward_system tests/test_reward_system.c)
    target_link_libraries(test_reward_system PRIVATE reward_system)

    # Enable debug assertions for tests
    target_compile_definitions(test_reward_system PRIVATE FOUNDATION_DEBUG=1)

    # Add to CTest
    add_test(NAME test_reward_system COMMAND test_reward_system)

    # Custom target to run tests with output
    add_custom_target(test_reward_system_run
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
        DEPENDS test_reward_system
        COMMENT "Running reward_system unit tests"
    )
endif()

# ============================================================================
# Benchmarks
# ============================================================================

if(REWARD_SYSTEM_BUILD_BENCHMARKS)
    add_executable(bench_reward_system tests/bench_reward_system.c)
    target_link_libraries(bench_reward_system PRIVATE reward_system)

    # Optimize benchmarks
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(bench_reward_system PRIVATE -O3)
    elseif(MSVC)
        target_compile_options(bench_reward_system PRIVATE /O2)
    endif()

    # Custom target to run benchmarks
    add_custom_target(bench_reward_system_run
        COMMAND bench_reward_system
        DEPENDS bench_reward_system
        COMMENT "Running reward_system benchmarks"
    )
endif()

# ============================================================================
# Installation
# ============================================================================

include(GNUInstallDirs)

install(TARGETS reward_system
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(FILES ${REWARD_SYSTEM_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/reward_system
)


# ============================================================================
# Summary
# ============================================================================

message(STATUS "")
message(STATUS "Reward System Module Configuration:")
message(STATUS "  C Standard: C${CMAKE_C_STANDARD}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Debug Mode: ${REWARD_SYSTEM_DEBUG}")
message(STATUS "  Build Tests: ${REWARD_SYSTEM_BUILD_TESTS}")
message(STATUS "  Build Benchmarks: ${REWARD_SYSTEM_BUILD_BENCHMARKS}")
if(REWARD_SYSTEM_ARCH_X86_64)
    message(STATUS "  SIMD: AVX2 (x86_64)")
elseif(REWARD_SYSTEM_ARCH_ARM)
    message(STATUS "  SIMD: NEON (ARM)")
else()
    message(STATUS "  SIMD: Scalar fallback")
endif()
message(STATUS "")
