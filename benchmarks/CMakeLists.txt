# Comprehensive Benchmark Suite for RL Engine
#
# Builds individual benchmark executables linked against dronerl_static.
# All benchmarks compiled with -O3 -DNDEBUG regardless of build type.
# Target names prefixed with "suite_" to avoid conflicts with per-module benchmarks.
#
# Usage:
#   cmake --build . --target suite_bench_all          # Build all benchmarks
#   cmake --build . --target suite_bench_sensors_run  # Build and run sensors bench

set(BENCH_SOURCES
    bench_world.c
    bench_collision.c
    bench_physics.c
    bench_rewards.c
    bench_sensors.c
    bench_engine.c
)

# Common include path for bench_harness.h
set(BENCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

foreach(src ${BENCH_SOURCES})
    # Extract base name: bench_world.c -> bench_world
    get_filename_component(base_name ${src} NAME_WE)
    # Prefixed target name to avoid collisions with per-module benchmarks
    set(target_name "suite_${base_name}")

    add_executable(${target_name} ${src})

    target_link_libraries(${target_name} PRIVATE dronerl_static)

    target_include_directories(${target_name} PRIVATE ${BENCH_INCLUDE_DIR})

    # Force -O3 -DNDEBUG regardless of build type
    target_compile_options(${target_name} PRIVATE -O3)
    target_compile_definitions(${target_name} PRIVATE NDEBUG)

    # Suppress warnings for benchmark code (may use unused variables for timing)
    if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wno-unused-variable)
    endif()

    # Output executable without prefix for clean naming
    set_target_properties(${target_name} PROPERTIES
        OUTPUT_NAME ${base_name}
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
    )

    # Per-benchmark run target
    add_custom_target(${target_name}_run
        COMMAND ${CMAKE_BINARY_DIR}/benchmarks/${base_name}
        DEPENDS ${target_name}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Running ${base_name}..."
    )
endforeach()

# Convenience target to build all suite benchmarks
add_custom_target(suite_bench_all
    DEPENDS suite_bench_world suite_bench_collision suite_bench_physics
            suite_bench_rewards suite_bench_sensors suite_bench_engine
    COMMENT "All suite benchmarks built"
)

# Run-all target
add_custom_target(suite_bench_run_all
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_world
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_collision
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_physics
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_rewards
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_sensors
    COMMAND ${CMAKE_BINARY_DIR}/benchmarks/bench_engine
    DEPENDS suite_bench_all
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running all suite benchmarks..."
)

# ============================================================================
# GPU Benchmarks (only when GPU backend is available)
# ============================================================================

if(GPU_AVAILABLE)
    set(GPU_BENCH_SOURCES
        bench_sensors_gpu.c
        bench_engine_gpu.c
        bench_pipeline_comprehensive.c
    )

    foreach(src ${GPU_BENCH_SOURCES})
        get_filename_component(base_name ${src} NAME_WE)
        set(target_name "suite_${base_name}")

        add_executable(${target_name} ${src})
        target_link_libraries(${target_name} PRIVATE dronerl_static)
        target_include_directories(${target_name} PRIVATE ${BENCH_INCLUDE_DIR})
        target_compile_options(${target_name} PRIVATE -O3)
        target_compile_definitions(${target_name} PRIVATE NDEBUG)

        if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
            target_compile_options(${target_name} PRIVATE -Wall -Wextra -Wno-unused-variable)
        endif()

        set_target_properties(${target_name} PROPERTIES
            OUTPUT_NAME ${base_name}
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks
        )

        add_custom_target(${target_name}_run
            COMMAND ${CMAKE_BINARY_DIR}/benchmarks/${base_name}
            DEPENDS ${target_name}
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running ${base_name}..."
        )
    endforeach()
endif()
