cmake_minimum_required(VERSION 3.16)
project(threading C)

# ============================================================================
# Configuration
# ============================================================================

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

# Thread library
find_package(Threads REQUIRED)

# ============================================================================
# Source Files
# ============================================================================

set(THREADING_SOURCES
    src/deque.c
    src/barrier.c
    src/spsc_queue.c
    src/mpmc_queue.c
    src/threadpool.c
    src/scheduler.c
)

set(THREADING_HEADERS
    include/threading.h
)

# ============================================================================
# Library Target
# ============================================================================

add_library(threading STATIC ${THREADING_SOURCES} ${THREADING_HEADERS})

target_include_directories(threading
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/include
)

# Find foundation library
find_library(FOUNDATION_LIBRARY
    NAMES foundation
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/build
    NO_DEFAULT_PATH
)

if(NOT FOUNDATION_LIBRARY)
    message(WARNING "Foundation library not found at expected path. Tests may fail to link.")
    set(FOUNDATION_LIBRARY foundation)
endif()

target_link_libraries(threading
    PUBLIC
        Threads::Threads
    PRIVATE
        ${FOUNDATION_LIBRARY}
)

# ============================================================================
# Compiler Flags
# ============================================================================

# Base flags for all configurations
target_compile_options(threading PRIVATE
    -Wall
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wcast-align
    -Wformat=2
    -Wstrict-prototypes
    -Wmissing-prototypes
    -Wno-unused-function
)

# Release optimizations
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(threading PRIVATE
        -O3
        -fno-strict-aliasing
    )
    # Architecture-specific optimizations (optional)
    if(NOT CMAKE_CROSSCOMPILING)
        target_compile_options(threading PRIVATE -march=native)
    endif()
endif()

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(threading PRIVATE FOUNDATION_DEBUG)
    target_compile_options(threading PRIVATE
        -g3
        -O0
        -fno-omit-frame-pointer
    )
endif()

# Thread sanitizer support (use with Debug_TSan build type)
option(THREADING_ENABLE_TSAN "Enable ThreadSanitizer" OFF)
if(THREADING_ENABLE_TSAN)
    target_compile_options(threading PUBLIC -fsanitize=thread)
    target_link_options(threading PUBLIC -fsanitize=thread)
endif()

# Address sanitizer support
option(THREADING_ENABLE_ASAN "Enable AddressSanitizer" OFF)
if(THREADING_ENABLE_ASAN)
    target_compile_options(threading PUBLIC -fsanitize=address -fno-omit-frame-pointer)
    target_link_options(threading PUBLIC -fsanitize=address)
endif()

# ============================================================================
# Tests
# ============================================================================

option(THREADING_BUILD_TESTS "Build threading tests" ON)

if(THREADING_BUILD_TESTS)
    enable_testing()

    set(TEST_SOURCES
        tests/test_deque.c
        tests/test_barrier.c
        tests/test_spsc.c
        tests/test_mpmc.c
        tests/test_threadpool.c
        tests/test_scheduler.c
    )

    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)

        add_executable(${test_name} ${test_src})

        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/include
        )

        target_link_libraries(${test_name} PRIVATE
            threading
            ${FOUNDATION_LIBRARY}
            Threads::Threads
        )

        target_compile_options(${test_name} PRIVATE
            -Wall -Wextra
        )

        # Test-specific debug flags
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            target_compile_definitions(${test_name} PRIVATE FOUNDATION_DEBUG)
        endif()

        add_test(NAME ${test_name} COMMAND ${test_name})

        # Set test timeout (60 seconds for stress tests)
        set_tests_properties(${test_name} PROPERTIES TIMEOUT 60)
    endforeach()

    # ========================================================================
    # Benchmarks
    # ========================================================================

    add_executable(bench_threading tests/bench_threading.c)

    target_include_directories(bench_threading PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/include
    )

    target_link_libraries(bench_threading PRIVATE
        threading
        ${FOUNDATION_LIBRARY}
        Threads::Threads
    )

    # Benchmarks should always be optimized
    target_compile_options(bench_threading PRIVATE
        -O3
        -DNDEBUG
    )
    if(NOT CMAKE_CROSSCOMPILING)
        target_compile_options(bench_threading PRIVATE -march=native)
    endif()

    # Custom target to run benchmarks
    add_custom_target(run_benchmarks
        COMMAND bench_threading
        DEPENDS bench_threading
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Running threading benchmarks..."
    )
endif()

# ============================================================================
# Installation
# ============================================================================

install(TARGETS threading
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)

install(FILES ${THREADING_HEADERS}
    DESTINATION include
)

# ============================================================================
# Summary
# ============================================================================

message(STATUS "Threading module configuration:")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Tests: ${THREADING_BUILD_TESTS}")
message(STATUS "  TSan: ${THREADING_ENABLE_TSAN}")
message(STATUS "  ASan: ${THREADING_ENABLE_ASAN}")
