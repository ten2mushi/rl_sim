# GPU Module CMake Configuration
#
# Provides GPU compute acceleration for sensor raymarching.
# Backend selection: -DGPU_BACKEND=METAL (default on Apple) or NONE
#
# Shaders are compiled at runtime via MTLDevice::newLibraryWithSource
# (no Xcode needed). Pre-compiled .metallib is used if available.

# Backend selection
if(NOT DEFINED GPU_BACKEND)
    if(APPLE)
        set(GPU_BACKEND "METAL")
    else()
        set(GPU_BACKEND "NONE")
    endif()
endif()

message(STATUS "GPU Backend: ${GPU_BACKEND}")

# Set compile definitions based on backend
if(GPU_BACKEND STREQUAL "METAL")
    set(GPU_AVAILABLE TRUE)
    add_compile_definitions(GPU_BACKEND_METAL=1)
else()
    set(GPU_AVAILABLE FALSE)
    add_compile_definitions(GPU_BACKEND_NONE=1)
endif()

# Export to parent scope
set(GPU_AVAILABLE ${GPU_AVAILABLE} PARENT_SCOPE)

# ============================================================================
# Source Files
# ============================================================================

set(GPU_SOURCES
    src/gpu_common.c
)

if(GPU_BACKEND STREQUAL "METAL")
    list(APPEND GPU_SOURCES
        src/gpu_metal.m
        src/gpu_sensor_dispatch.c
        src/gpu_sensor_context.c
        src/gpu_bvh.c
        src/gpu_voxelize.c
    )
else()
    list(APPEND GPU_SOURCES
        src/gpu_none.c
    )
endif()

# ============================================================================
# Library Target
# ============================================================================

add_library(gpu STATIC ${GPU_SOURCES})

target_include_directories(gpu PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/shaders>
)

target_include_directories(gpu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../foundation/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../drone_state/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../world_brick_map/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../sensor_system/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../sensor_implementations/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../collision_system/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../obj_io/include
)

# Link foundation
target_link_libraries(gpu PUBLIC foundation)

if(GPU_BACKEND STREQUAL "METAL")
    # Enable ARC for Objective-C files
    set_source_files_properties(src/gpu_metal.m PROPERTIES
        COMPILE_FLAGS "-fobjc-arc"
    )

    # Link Metal and Foundation frameworks
    target_link_libraries(gpu PUBLIC
        "-framework Metal"
        "-framework Foundation"
    )

    # ============================================================================
    # Metal Shader Compilation (optional - runtime compilation is the fallback)
    # ============================================================================

    # Try to find Metal offline compiler (requires Xcode, not just CommandLineTools)
    execute_process(
        COMMAND xcrun --sdk macosx --find metal
        RESULT_VARIABLE METAL_COMPILER_RESULT
        OUTPUT_VARIABLE METAL_COMPILER_PATH
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    if(METAL_COMPILER_RESULT EQUAL 0)
        message(STATUS "Metal offline compiler found: ${METAL_COMPILER_PATH}")
        set(SHADER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/shaders)
        set(METALLIB_OUTPUT ${CMAKE_BINARY_DIR}/default.metallib)

        file(GLOB METAL_SHADER_SOURCES "${SHADER_DIR}/*.metal")

        if(METAL_SHADER_SOURCES)
            set(AIR_FILES "")
            foreach(metal_src ${METAL_SHADER_SOURCES})
                get_filename_component(metal_name ${metal_src} NAME_WE)
                set(air_file ${CMAKE_BINARY_DIR}/${metal_name}.air)
                list(APPEND AIR_FILES ${air_file})

                add_custom_command(
                    OUTPUT ${air_file}
                    COMMAND xcrun -sdk macosx metal
                        -c ${metal_src}
                        -o ${air_file}
                        -I ${SHADER_DIR}
                        -std=metal3.0
                    DEPENDS ${metal_src}
                            ${SHADER_DIR}/sdf_types.h
                    COMMENT "Compiling Metal shader: ${metal_name}.metal"
                )
            endforeach()

            add_custom_command(
                OUTPUT ${METALLIB_OUTPUT}
                COMMAND xcrun -sdk macosx metallib
                    ${AIR_FILES}
                    -o ${METALLIB_OUTPUT}
                DEPENDS ${AIR_FILES}
                COMMENT "Linking Metal library: default.metallib"
            )

            add_custom_target(metal_shaders ALL
                DEPENDS ${METALLIB_OUTPUT}
            )

            add_custom_command(
                TARGET metal_shaders POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${METALLIB_OUTPUT}
                    ${CMAKE_BINARY_DIR}/src/gpu/default.metallib
                COMMENT "Copying metallib to test directory"
            )
        endif()
    else()
        message(STATUS "Metal offline compiler not found - using runtime compilation")
    endif()
endif()

# ============================================================================
# Tests
# ============================================================================

if(BUILD_TESTING)
    # HAL test (works with any backend)
    add_executable(test_gpu_hal tests/test_gpu_hal.c)
    target_link_libraries(test_gpu_hal PRIVATE gpu dronerl_static)
    target_include_directories(test_gpu_hal PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    set_target_properties(test_gpu_hal PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/gpu
    )
    add_test(NAME test_gpu_hal COMMAND test_gpu_hal
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    if(GPU_AVAILABLE)
        # Raymarch test (uses runtime compilation if no metallib)
        add_executable(test_gpu_raymarch tests/test_gpu_raymarch.c)
        target_link_libraries(test_gpu_raymarch PRIVATE gpu dronerl_static)
        target_include_directories(test_gpu_raymarch PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        )
        set_target_properties(test_gpu_raymarch PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/gpu
        )
        if(TARGET metal_shaders)
            add_dependencies(test_gpu_raymarch metal_shaders)
        endif()
        add_test(NAME test_gpu_raymarch COMMAND test_gpu_raymarch
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

        # All sensors test
        add_executable(test_gpu_all_sensors tests/test_gpu_all_sensors.c)
        target_link_libraries(test_gpu_all_sensors PRIVATE gpu dronerl_static)
        target_include_directories(test_gpu_all_sensors PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        set_target_properties(test_gpu_all_sensors PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/gpu
        )
        if(TARGET metal_shaders)
            add_dependencies(test_gpu_all_sensors metal_shaders)
        endif()
        add_test(NAME test_gpu_all_sensors COMMAND test_gpu_all_sensors
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

        # Pipeline integration test
        add_executable(test_gpu_pipeline tests/test_gpu_pipeline.c)
        target_link_libraries(test_gpu_pipeline PRIVATE gpu dronerl_static)
        target_include_directories(test_gpu_pipeline PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        set_target_properties(test_gpu_pipeline PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/gpu
        )
        if(TARGET metal_shaders)
            add_dependencies(test_gpu_pipeline metal_shaders)
        endif()
        add_test(NAME test_gpu_pipeline COMMAND test_gpu_pipeline
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

        # GPU determinism test
        add_executable(test_gpu_determinism tests/test_gpu_determinism.c)
        target_link_libraries(test_gpu_determinism PRIVATE gpu dronerl_static)
        target_include_directories(test_gpu_determinism PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        set_target_properties(test_gpu_determinism PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/gpu
        )
        if(TARGET metal_shaders)
            add_dependencies(test_gpu_determinism metal_shaders)
        endif()
        add_test(NAME test_gpu_determinism COMMAND test_gpu_determinism
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

        # GPU voxelization test
        add_executable(test_gpu_voxelize tests/test_gpu_voxelize.c)
        target_link_libraries(test_gpu_voxelize PRIVATE gpu dronerl_static)
        target_include_directories(test_gpu_voxelize PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders
        )
        set_target_properties(test_gpu_voxelize PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/src/gpu
        )
        if(TARGET metal_shaders)
            add_dependencies(test_gpu_voxelize metal_shaders)
        endif()
        add_test(NAME test_gpu_voxelize COMMAND test_gpu_voxelize
                 WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    endif()
endif()
