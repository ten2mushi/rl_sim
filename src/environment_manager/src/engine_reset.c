/**
 * Engine Reset Implementation
 *
 * Provides reset logic for full reset, partial reset, and auto-reset.
 */

#include "environment_manager.h"
#include "noise.h"
#include <string.h>

/* ============================================================================
 * Internal Helpers
 * ============================================================================ */

/**
 * Generate a random spawn position within configured bounds.
 */
static Vec3 generate_spawn_position(BatchDroneEngine* engine, uint32_t drone_idx) {
    (void)drone_idx;  /* Can be used for env-specific spawning */

    Vec3 spawn_min, spawn_max;

    if (engine->config.use_custom_spawn) {
        /* Use user-provided spawn region directly */
        spawn_min = engine->config.spawn_min;
        spawn_max = engine->config.spawn_max;
    } else {
        /* Derive from world bounds with uniform 5m margin on all axes */
        float margin = 5.0f;
        spawn_min = engine->config.world_min;
        spawn_max = engine->config.world_max;
        spawn_min.x += margin;
        spawn_min.y += margin;
        spawn_min.z += margin;
        spawn_max.x -= margin;
        spawn_max.y -= margin;
        spawn_max.z -= margin;

        /* Constrain Z to lower 50% of remaining range */
        float z_mid = (spawn_min.z + spawn_max.z) * 0.5f;
        spawn_max.z = z_mid;
    }

    /* Apply domain randomization if enabled */
    float rand_scale = engine->config.domain_randomization;
    if (rand_scale > 0.0f) {
        float range_x = (spawn_max.x - spawn_min.x) * rand_scale;
        float range_y = (spawn_max.y - spawn_min.y) * rand_scale;
        float range_z = (spawn_max.z - spawn_min.z) * rand_scale;
        spawn_min.x += (1.0f - rand_scale) * range_x * 0.5f;
        spawn_max.x -= (1.0f - rand_scale) * range_x * 0.5f;
        spawn_min.y += (1.0f - rand_scale) * range_y * 0.5f;
        spawn_max.y -= (1.0f - rand_scale) * range_y * 0.5f;
        spawn_min.z += (1.0f - rand_scale) * range_z * 0.5f;
        spawn_max.z -= (1.0f - rand_scale) * range_z * 0.5f;
    }

    return (Vec3){
        pcg32_range(&engine->rng, spawn_min.x, spawn_max.x),
        pcg32_range(&engine->rng, spawn_min.y, spawn_max.y),
        pcg32_range(&engine->rng, spawn_min.z, spawn_max.z),
        0.0f
    };
}

/**
 * Reset a single drone to given position and orientation.
 */
static void reset_single_drone_internal(BatchDroneEngine* engine, uint32_t drone_idx,
                                        Vec3 position, Quat orientation) {
    DroneStateSOA* states = engine->states;

    /* Set position */
    states->pos_x[drone_idx] = position.x;
    states->pos_y[drone_idx] = position.y;
    states->pos_z[drone_idx] = position.z;

    /* Zero velocity */
    states->vel_x[drone_idx] = 0.0f;
    states->vel_y[drone_idx] = 0.0f;
    states->vel_z[drone_idx] = 0.0f;

    /* Set orientation */
    states->quat_w[drone_idx] = orientation.w;
    states->quat_x[drone_idx] = orientation.x;
    states->quat_y[drone_idx] = orientation.y;
    states->quat_z[drone_idx] = orientation.z;

    /* Zero angular velocity */
    states->omega_x[drone_idx] = 0.0f;
    states->omega_y[drone_idx] = 0.0f;
    states->omega_z[drone_idx] = 0.0f;

    /* Set hover RPMs (approximately counteract gravity) */
    float hover_rpm = engine->params->max_rpm[drone_idx] * 0.5f;
    states->rpm_0[drone_idx] = hover_rpm;
    states->rpm_1[drone_idx] = hover_rpm;
    states->rpm_2[drone_idx] = hover_rpm;
    states->rpm_3[drone_idx] = hover_rpm;

    /* Reset episode tracking for this drone */
    engine->episode_returns[drone_idx] = 0.0f;
    engine->episode_lengths[drone_idx] = 0;

    /* Clear termination flags */
    engine->dones[drone_idx] = 0;
    engine->truncations[drone_idx] = 0;
}

/* ============================================================================
 * Full Reset
 * ============================================================================ */

void engine_reset(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->initialized, "assertion failed");

    uint32_t total_drones = engine->config.total_drones;

    /* Re-seed RNG to original seed for deterministic resets */
    pcg32_seed(&engine->rng, engine->config.seed);

    /* Reset all drones with randomized positions */
    for (uint32_t i = 0; i < total_drones; i++) {
        Vec3 position = generate_spawn_position(engine, i);
        Quat orientation = QUAT_IDENTITY;
        reset_single_drone_internal(engine, i, position, orientation);
    }

    /* Reset collision system (clears stale penetration_depth used by ground effect) */
    collision_reset(engine->collision);

    /* Reset reward system */
    for (uint32_t i = 0; i < total_drones; i++) {
        reward_reset(engine->rewards, i);
    }

    /* Reset sensor system */
    sensor_system_reset(engine->sensors);

    /* Clear all done/truncation flags (redundant but explicit) */
    memset(engine->dones, 0, total_drones);
    memset(engine->truncations, 0, total_drones);

    /* Targets must be set externally via engine_set_target()/engine_set_targets() */

    /* Compute initial observations */
    sensor_system_sample_all(engine->sensors, engine->states,
                             engine->world, engine->collision,
                             total_drones);

    /* Sensor observations: zero-copy (sensor system writes directly to engine->observations) */

    /* Clear needs_reset flag */
    engine->needs_reset = false;
}

/* ============================================================================
 * Partial Reset
 * ============================================================================ */

void engine_reset_envs(BatchDroneEngine* engine, const uint32_t* env_indices, uint32_t count) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->initialized, "assertion failed");
    FOUNDATION_ASSERT(env_indices != NULL || count == 0, "assertion failed");

    uint32_t drones_per_env = engine->config.drones_per_env;

    for (uint32_t e = 0; e < count; e++) {
        uint32_t env_id = env_indices[e];
        FOUNDATION_ASSERT(env_id < engine->config.num_envs, "assertion failed");

        /* Reset all drones in this environment */
        for (uint32_t d = 0; d < drones_per_env; d++) {
            uint32_t drone_idx = env_id * drones_per_env + d;
            Vec3 position = generate_spawn_position(engine, drone_idx);
            Quat orientation = QUAT_IDENTITY;
            reset_single_drone_internal(engine, drone_idx, position, orientation);

            /* Reset reward state */
            reward_reset(engine->rewards, drone_idx);
        }
    }
}

/* ============================================================================
 * Single Drone Reset
 * ============================================================================ */

void engine_reset_drone(BatchDroneEngine* engine, uint32_t drone_idx,
                        Vec3 position, Quat orientation) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->initialized, "assertion failed");
    FOUNDATION_ASSERT(drone_idx < engine->config.total_drones, "assertion failed");

    reset_single_drone_internal(engine, drone_idx, position, orientation);
    reward_reset(engine->rewards, drone_idx);
}

/* ============================================================================
 * Auto-Reset Terminated Drones
 * ============================================================================ */

void engine_step_reset_terminated(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");

    uint32_t total_drones = engine->config.total_drones;

    /* Scratch arrays for batch reset */
    /* Using a reasonable max size - in practice, not all drones terminate at once */
    uint32_t max_reset = total_drones;
    uint32_t* done_indices = (uint32_t*)arena_alloc(engine->frame_arena,
                                                     max_reset * sizeof(uint32_t));
    Vec3* reset_positions = (Vec3*)arena_alloc(engine->frame_arena,
                                                max_reset * sizeof(Vec3));
    Quat* reset_orientations = (Quat*)arena_alloc(engine->frame_arena,
                                                   max_reset * sizeof(Quat));

    uint32_t done_count = 0;

    /* Collect terminated drones */
    for (uint32_t i = 0; i < total_drones; i++) {
        if (engine->dones[i] || engine->truncations[i]) {
            /* Log episode completion */
            engine->total_episodes++;

            /* Store reset info */
            done_indices[done_count] = i;
            reset_positions[done_count] = generate_spawn_position(engine, i);
            reset_orientations[done_count] = QUAT_IDENTITY;
            done_count++;
        }
    }

    /* Batch reset terminated drones */
    if (done_count > 0) {
        /* Use batch reset if available, otherwise loop */
        drone_state_reset_batch(engine->states, done_indices,
                                reset_positions, reset_orientations, done_count);

        /* Reset rewards and episode tracking for terminated drones */
        reward_reset_batch(engine->rewards, done_indices, done_count);

        /* Reset episode tracking and noise state */
        for (uint32_t i = 0; i < done_count; i++) {
            uint32_t idx = done_indices[i];
            engine->episode_returns[idx] = 0.0f;
            engine->episode_lengths[idx] = 0;
            engine->dones[idx] = 0;
            engine->truncations[idx] = 0;

            /* Reset per-drone noise state for deterministic episodes */
            for (uint32_t s = 0; s < engine->sensors->sensor_count; s++) {
                if (engine->sensors->sensors[s].noise_state != NULL) {
                    noise_state_reset_drone(engine->sensors->sensors[s].noise_state, idx);
                }
            }
        }

        /* Targets stay at previous values. Training code sets new targets. */
    }
}
