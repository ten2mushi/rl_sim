/**
 * Configuration Module Benchmarks
 *
 * Performance benchmarks for:
 * - Config loading and parsing
 * - Validation
 * - Hashing
 * - Conversion to PlatformParamsSOA
 */

#include "configuration.h"
#include "drone_state.h"
#include "platform_quadcopter.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ============================================================================
 * Timing Utilities
 * ============================================================================ */

#if defined(__APPLE__)
#include <mach/mach_time.h>

static uint64_t get_time_ns(void) {
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    return mach_absolute_time() * timebase.numer / timebase.denom;
}

#elif defined(__linux__)
static uint64_t get_time_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

#else
static uint64_t get_time_ns(void) {
    return (uint64_t)clock() * (1000000000ULL / CLOCKS_PER_SEC);
}
#endif

/* ============================================================================
 * Test Config String
 * ============================================================================ */

static const char* BENCH_CONFIG =
    "[platform]\n"
    "name = \"bench_drone\"\n"
    "mass = 0.5\n"
    "ixx = 1e-4\n"
    "iyy = 1e-4\n"
    "izz = 2e-4\n"
    "\n"
    "[platform.quadcopter]\n"
    "arm_length = 0.1\n"
    "k_thrust = 3e-8\n"
    "k_torque = 8e-10\n"
    "motor_tau = 0.03\n"
    "max_rpm = 3000.0\n"
    "\n"
    "[environment]\n"
    "num_envs = 1024\n"
    "agents_per_env = 8\n"
    "world_size = [30.0, 30.0, 15.0]\n"
    "seed = 12345\n"
    "\n"
    "[physics]\n"
    "timestep = 0.01\n"
    "substeps = 2\n"
    "gravity = 9.81\n"
    "integrator = \"rk4\"\n"
    "\n"
    "[reward]\n"
    "task = \"hover\"\n"
    "distance_scale = 2.0\n"
    "\n"
    "[training]\n"
    "algorithm = \"ppo\"\n"
    "learning_rate = 3e-4\n"
    "\n"
    "[[sensors]]\n"
    "type = \"imu\"\n"
    "name = \"main_imu\"\n"
    "sample_rate = 500.0\n"
    "\n"
    "[[sensors]]\n"
    "type = \"camera_depth\"\n"
    "name = \"front_camera\"\n"
    "width = 84\n"
    "height = 84\n";

/* ============================================================================
 * Benchmarks
 * ============================================================================ */

static void bench_config_load_string(void) {
    const int iterations = 1000;
    Config cfg;
    char error_msg[256];

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        config_load_string(BENCH_CONFIG, &cfg, error_msg);
        config_free(&cfg);
    }
    uint64_t end = get_time_ns();

    double total_ms = (double)(end - start) / 1e6;
    double per_load_us = total_ms * 1000.0 / iterations;

    printf("config_load_string: %.2f us/call (%.2f ms total for %d iterations)\n",
           per_load_us, total_ms, iterations);
    printf("  Target: <10 ms/call - %s\n", per_load_us < 10000 ? "PASS" : "FAIL");
}

static void bench_config_validate(void) {
    const int iterations = 100000;
    Config cfg;
    config_set_defaults(&cfg);
    ConfigError errors[CONFIG_MAX_ERRORS];

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        config_validate(&cfg, errors, CONFIG_MAX_ERRORS);
    }
    uint64_t end = get_time_ns();

    double total_ms = (double)(end - start) / 1e6;
    double per_validate_us = total_ms * 1000.0 / iterations;

    printf("config_validate: %.3f us/call (%.2f ms total for %d iterations)\n",
           per_validate_us, total_ms, iterations);
    printf("  Target: <1 ms/call - %s\n", per_validate_us < 1000 ? "PASS" : "FAIL");
}

static void bench_config_hash(void) {
    const int iterations = 1000000;
    Config cfg;
    config_set_defaults(&cfg);
    volatile uint64_t h = 0;

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        h = config_hash(&cfg);
    }
    uint64_t end = get_time_ns();
    (void)h;

    double total_ms = (double)(end - start) / 1e6;
    double per_hash_us = total_ms * 1000.0 / iterations;

    printf("config_hash: %.3f us/call (%.2f ms total for %d iterations)\n",
           per_hash_us, total_ms, iterations);
    printf("  Target: <100 us/call - %s\n", per_hash_us < 100 ? "PASS" : "FAIL");
}

static void bench_config_to_params_1024(void) {
    const int iterations = 10000;
    Config cfg;
    config_set_defaults(&cfg);
    Arena* arena = arena_create(2 * 1024 * 1024);
    PlatformParamsSOA* params = platform_params_create(arena, 1024, QUAD_PARAMS_EXT_COUNT);

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        platform_config_to_params(&cfg.platform, &cfg.physics, params, 0, 1024);
    }
    uint64_t end = get_time_ns();

    double total_ms = (double)(end - start) / 1e6;
    double per_convert_us = total_ms * 1000.0 / iterations;

    printf("platform_config_to_params(1024): %.2f us/call (%.2f ms total for %d iterations)\n",
           per_convert_us, total_ms, iterations);
    printf("  Target: <1 ms/call - %s\n", per_convert_us < 1000 ? "PASS" : "FAIL");

    arena_destroy(arena);
}

static void bench_config_compare(void) {
    const int iterations = 1000000;
    Config cfg1, cfg2;
    config_set_defaults(&cfg1);
    config_set_defaults(&cfg2);
    volatile int result = 0;

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        result = config_compare(&cfg1, &cfg2);
    }
    uint64_t end = get_time_ns();
    (void)result;

    double total_ms = (double)(end - start) / 1e6;
    double per_compare_us = total_ms * 1000.0 / iterations;

    printf("config_compare: %.3f us/call (%.2f ms total for %d iterations)\n",
           per_compare_us, total_ms, iterations);

    config_free(&cfg1);
    config_free(&cfg2);
}

static void bench_config_clone(void) {
    const int iterations = 100000;
    Config src;
    config_set_defaults(&src);
    Arena* arena = arena_create(1024 * 1024);

    uint64_t start = get_time_ns();
    for (int i = 0; i < iterations; i++) {
        Config dst;
        config_clone(&src, &dst, arena);
        /* Free the malloc'd platform_specific from clone (arena handles sensors) */
        if (dst.platform.platform_specific) {
            free(dst.platform.platform_specific);
        }
        arena_reset(arena);
    }
    uint64_t end = get_time_ns();

    double total_ms = (double)(end - start) / 1e6;
    double per_clone_us = total_ms * 1000.0 / iterations;

    printf("config_clone: %.3f us/call (%.2f ms total for %d iterations)\n",
           per_clone_us, total_ms, iterations);

    config_free(&src);
    arena_destroy(arena);
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    printf("\n=== Configuration Module Benchmarks ===\n\n");

    bench_config_load_string();
    bench_config_validate();
    bench_config_hash();
    bench_config_to_params_1024();
    bench_config_compare();
    bench_config_clone();

    printf("\n=== Done ===\n");
    return 0;
}
