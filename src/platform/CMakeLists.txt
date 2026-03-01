# Platform Module CMake Configuration
#
# VTable-based robot platform abstraction with quadcopter implementation.

cmake_minimum_required(VERSION 3.16)
project(platform C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Dependencies
if(NOT TARGET foundation)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../foundation
                     ${CMAKE_CURRENT_BINARY_DIR}/foundation EXCLUDE_FROM_ALL)
endif()
if(NOT TARGET rigid_body_state)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../rigid_body_state
                     ${CMAKE_CURRENT_BINARY_DIR}/rigid_body_state EXCLUDE_FROM_ALL)
endif()
if(NOT TARGET physics)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../physics
                     ${CMAKE_CURRENT_BINARY_DIR}/physics EXCLUDE_FROM_ALL)
endif()

# Source files
set(PLATFORM_SOURCES
    src/platform_quadcopter.c
    src/platform_diff_drive.c
    src/platform_registry.c
)

# Create static library
add_library(platform STATIC ${PLATFORM_SOURCES})

target_include_directories(platform
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(platform PUBLIC foundation rigid_body_state physics)

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(platform PRIVATE
        -Wall -Wextra -Wpedantic -Wno-unused-parameter -ffast-math)
endif()

if(UNIX AND NOT APPLE)
    target_link_libraries(platform PUBLIC m)
endif()

# Tests
if(BUILD_TESTING)
    enable_testing()

    add_executable(test_platform_quadcopter tests/test_platform_quadcopter.c)
    target_link_libraries(test_platform_quadcopter PRIVATE platform drone_state)
    target_compile_definitions(test_platform_quadcopter PRIVATE FOUNDATION_DEBUG=1)
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_platform_quadcopter PRIVATE m)
    endif()
    add_test(NAME test_platform_quadcopter COMMAND test_platform_quadcopter)

    add_executable(test_platform_diff_drive tests/test_platform_diff_drive.c)
    target_link_libraries(test_platform_diff_drive PRIVATE platform drone_state)
    target_compile_definitions(test_platform_diff_drive PRIVATE FOUNDATION_DEBUG=1)
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_platform_diff_drive PRIVATE m)
    endif()
    add_test(NAME test_platform_diff_drive COMMAND test_platform_diff_drive)
endif()
