# Rigid Body State Module CMake Configuration
#
# Robot-agnostic SoA data structures for the platform abstraction layer.

cmake_minimum_required(VERSION 3.16)
project(rigid_body_state C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Foundation dependency
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()

# Source files
set(RIGID_BODY_STATE_SOURCES
    src/rigid_body_state.c
)

# Create static library
add_library(rigid_body_state STATIC ${RIGID_BODY_STATE_SOURCES})

target_include_directories(rigid_body_state
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(rigid_body_state PUBLIC foundation)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(rigid_body_state PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter -ffast-math)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(rigid_body_state PUBLIC m)
endif()

# Tests
if(BUILD_TESTING)
    enable_testing()

    add_executable(test_rigid_body_state tests/test_rigid_body_state.c)
    target_link_libraries(test_rigid_body_state PRIVATE rigid_body_state)
    target_compile_definitions(test_rigid_body_state PRIVATE FOUNDATION_DEBUG=1)
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_rigid_body_state PRIVATE m)
    endif()
    add_test(NAME test_rigid_body_state COMMAND test_rigid_body_state)
endif()
