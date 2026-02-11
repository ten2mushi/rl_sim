/**
 * Engine Statistics Implementation
 *
 * Provides statistics collection and reporting.
 */

#include "environment_manager.h"
#include <stdio.h>

/* ============================================================================
 * Statistics Collection
 * ============================================================================ */

void engine_get_stats(const BatchDroneEngine* engine, EngineStats* stats) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(stats != NULL, "assertion failed");

    /* Timing statistics */
    stats->physics_time_ms = engine->physics_time_ms;
    stats->collision_time_ms = engine->collision_time_ms;
    stats->sensor_time_ms = engine->sensor_time_ms;
    stats->reward_time_ms = engine->reward_time_ms;
    stats->reset_time_ms = engine->reset_time_ms;

    stats->avg_step_time_ms = stats->physics_time_ms +
                               stats->collision_time_ms +
                               stats->sensor_time_ms +
                               stats->reward_time_ms +
                               stats->reset_time_ms;

    /* Count statistics */
    stats->total_steps = engine->total_steps;
    stats->total_episodes = engine->total_episodes;

    /* Episode statistics (compute averages) */
    uint32_t total_drones = engine->config.total_drones;
    float total_return = 0.0f;
    uint32_t total_length = 0;

    for (uint32_t i = 0; i < total_drones; i++) {
        total_return += engine->episode_returns[i];
        total_length += engine->episode_lengths[i];
    }

    stats->avg_episode_return = total_return / (float)total_drones;
    stats->avg_episode_length = (float)total_length / (float)total_drones;

    /* Performance metrics */
    if (stats->avg_step_time_ms > 0.0) {
        stats->steps_per_second = 1000.0 / stats->avg_step_time_ms;
        stats->drones_per_second = stats->steps_per_second * total_drones;
    } else {
        stats->steps_per_second = 0.0;
        stats->drones_per_second = 0.0;
    }

    /* Memory usage */
    stats->persistent_memory_used = engine->persistent_arena->used;
    stats->frame_memory_used = engine->frame_arena->used;
}

/* ============================================================================
 * Statistics Reset
 * ============================================================================ */

void engine_reset_stats(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");

    engine->total_steps = 0;
    engine->total_episodes = 0;
    engine->physics_time_ms = 0.0;
    engine->collision_time_ms = 0.0;
    engine->sensor_time_ms = 0.0;
    engine->reward_time_ms = 0.0;
    engine->reset_time_ms = 0.0;
}

/* ============================================================================
 * Statistics Printing
 * ============================================================================ */

void engine_print_stats(const BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");

    EngineStats stats;
    engine_get_stats(engine, &stats);

    printf("\n=== Engine Statistics ===\n");
    printf("Configuration:\n");
    printf("  Environments: %u\n", engine->config.num_envs);
    printf("  Drones/env:   %u\n", engine->config.drones_per_env);
    printf("  Total drones: %u\n", engine->config.total_drones);
    printf("  Obs dim:      %u\n", engine->obs_dim);
    printf("  Action dim:   %u\n", engine->action_dim);

    printf("\nPerformance:\n");
    printf("  Avg step time:    %.3f ms\n", stats.avg_step_time_ms);
    printf("    Physics:        %.3f ms\n", stats.physics_time_ms);
    printf("    Collision:      %.3f ms\n", stats.collision_time_ms);
    printf("    Sensors:        %.3f ms\n", stats.sensor_time_ms);
    printf("    Rewards:        %.3f ms\n", stats.reward_time_ms);
    printf("    Reset:          %.3f ms\n", stats.reset_time_ms);
    printf("  Steps/second:     %.1f\n", stats.steps_per_second);
    printf("  Drones/second:    %.1f\n", stats.drones_per_second);

    printf("\nTraining:\n");
    printf("  Total steps:      %llu\n", (unsigned long long)stats.total_steps);
    printf("  Total episodes:   %llu\n", (unsigned long long)stats.total_episodes);
    printf("  Avg ep return:    %.4f\n", stats.avg_episode_return);
    printf("  Avg ep length:    %.1f\n", stats.avg_episode_length);

    printf("\nMemory:\n");
    printf("  Persistent used:  %.2f MB\n",
           stats.persistent_memory_used / (1024.0 * 1024.0));
    printf("  Frame used:       %.2f MB\n",
           stats.frame_memory_used / (1024.0 * 1024.0));

    printf("========================\n\n");
}
