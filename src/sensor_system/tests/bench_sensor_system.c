/**
 * Sensor System Benchmarks
 *
 * Performance benchmarks for the sensor system overhead.
 * Measures grouping, dispatch, and scatter operations.
 *
 * Target Performance:
 * - Vtable dispatch: <5ns
 * - Drone grouping: <100us (1024 drones)
 * - Scatter to obs buffer: <200us
 * - Total system overhead: <500us
 */

#include "sensor_system.h"
#include "sensor_implementations.h"
#include "drone_state.h"
#include "platform_quadcopter.h"
#include <stdio.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

static double get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

#define BENCH_ITERATIONS 100

/* ============================================================================
 * Mock Sensor for Benchmarking
 * ============================================================================ */

static void bench_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    (void)config; (void)arena;
    sensor->impl = NULL;
}

static size_t bench_get_output_size(const Sensor* sensor) {
    (void)sensor;
    return 6;  /* IMU-like output */
}

static const char* bench_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t bench_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    (void)sensor;
    shape[0] = 6;
    return 1;
}

static void bench_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    (void)sensor;
    /* Minimal work - just fill with zeros */
    memset(output_buffer, 0, ctx->agent_count * 6 * sizeof(float));
}

static void bench_reset(Sensor* sensor, uint32_t agent_index) {
    (void)sensor; (void)agent_index;
}

static void bench_destroy(Sensor* sensor) {
    (void)sensor;
}

static const SensorVTable BENCH_VTABLE = {
    .name = "Bench",
    .type = SENSOR_TYPE_IMU,
    .init = bench_init,
    .get_output_size = bench_get_output_size,
    .get_output_dtype = bench_get_output_dtype,
    .get_output_shape = bench_get_output_shape,
    .batch_sample = bench_batch_sample,
    .reset = bench_reset,
    .destroy = bench_destroy
};

/* ============================================================================
 * Benchmarks
 * ============================================================================ */

static void bench_system_overhead(uint32_t num_agents, uint32_t num_sensors) {
    printf("Benchmarking %u drones, %u sensors:\n", num_agents, num_sensors);

    Arena* arena = arena_create(16 * 1024 * 1024);
    if (arena == NULL) {
        printf("  Failed to create arena\n");
        return;
    }

    SensorSystem* sys = sensor_system_create(arena, num_agents, num_sensors, 128);
    if (sys == NULL) {
        printf("  Failed to create sensor system\n");
        arena_destroy(arena);
        return;
    }

    /* Register and create sensors */
    sensor_registry_register(&sys->registry, SENSOR_TYPE_IMU, &BENCH_VTABLE);

    SensorConfig config = sensor_config_default(SENSOR_TYPE_IMU);

    for (uint32_t s = 0; s < num_sensors; s++) {
        sensor_system_create_sensor(sys, &config);
    }

    /* Attach sensors to drones (varying attachments) */
    for (uint32_t d = 0; d < num_agents; d++) {
        /* Attach 1-4 sensors per drone */
        uint32_t attach_count = 1 + (d % 4);
        for (uint32_t a = 0; a < attach_count && a < num_sensors; a++) {
            sensor_system_attach(sys, d, a % num_sensors);
        }
    }

    /* Create drone state */
    PlatformStateSOA* drones = platform_state_create(arena, num_agents, QUAD_STATE_EXT_COUNT);
    for (uint32_t d = 0; d < num_agents; d++) {
        platform_state_init(drones, d);
    }
    drones->rigid_body.count = num_agents;

    /* Warmup */
    for (int i = 0; i < 10; i++) {
        sensor_system_sample_all(sys, drones, NULL, NULL, num_agents);
    }

    /* Benchmark */
    double total_time = 0.0;
    double min_time = 1e9;
    double max_time = 0.0;

    for (int i = 0; i < BENCH_ITERATIONS; i++) {
        double start = get_time_ms();
        sensor_system_sample_all(sys, drones, NULL, NULL, num_agents);
        double end = get_time_ms();

        double elapsed = end - start;
        total_time += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
    }

    double avg_time = total_time / BENCH_ITERATIONS;

    printf("  Average: %.3f ms\n", avg_time);
    printf("  Min: %.3f ms, Max: %.3f ms\n", min_time, max_time);
    printf("  Per-drone overhead: %.3f us\n", avg_time * 1000.0 / num_agents);

    if (num_agents == 1024 && num_sensors >= 4) {
        if (avg_time < 0.5) {
            printf("  [PASS] Under 500us target\n");
        } else {
            printf("  [FAIL] Over 500us target\n");
        }
    }

    sensor_system_destroy(sys);
    arena_destroy(arena);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("Sensor System Benchmarks\n");
    printf("========================\n\n");

    bench_system_overhead(64, 4);
    printf("\n");

    bench_system_overhead(256, 8);
    printf("\n");

    bench_system_overhead(1024, 16);
    printf("\n");

    bench_system_overhead(1024, 64);
    printf("\n");

    printf("========================\n");
    printf("Benchmarks complete\n");

    return 0;
}
