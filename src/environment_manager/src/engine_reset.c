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
static Vec3 generate_spawn_position(BatchEngine* engine) {

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

    /* Apply domain randomization: shrink spawn region by (1-scale) on each side.
     * When rand_scale == 0 (default), this is a no-op — skip entirely.
     * When rand_scale == 1, full range is used (no shrinkage). */
    float rand_scale = engine->config.domain_randomization;
    if (rand_scale > 0.0f && rand_scale < 1.0f) {
        float shrink = (1.0f - rand_scale) * 0.5f;
        float dx = (spawn_max.x - spawn_min.x) * shrink;
        float dy = (spawn_max.y - spawn_min.y) * shrink;
        float dz = (spawn_max.z - spawn_min.z) * shrink;
        spawn_min.x += dx;  spawn_max.x -= dx;
        spawn_min.y += dy;  spawn_max.y -= dy;
        spawn_min.z += dz;  spawn_max.z -= dz;
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
static void reset_single_drone_internal(BatchEngine* engine, uint32_t agent_idx,
                                        Vec3 position, Quat orientation) {
    RigidBodyStateSOA* rb = &engine->states->rigid_body;

    /* Set position */
    rb->pos_x[agent_idx] = position.x;
    rb->pos_y[agent_idx] = position.y;
    rb->pos_z[agent_idx] = position.z;

    /* Zero velocity */
    rb->vel_x[agent_idx] = 0.0f;
    rb->vel_y[agent_idx] = 0.0f;
    rb->vel_z[agent_idx] = 0.0f;

    /* Set orientation */
    rb->quat_w[agent_idx] = orientation.w;
    rb->quat_x[agent_idx] = orientation.x;
    rb->quat_y[agent_idx] = orientation.y;
    rb->quat_z[agent_idx] = orientation.z;

    /* Zero angular velocity */
    rb->omega_x[agent_idx] = 0.0f;
    rb->omega_y[agent_idx] = 0.0f;
    rb->omega_z[agent_idx] = 0.0f;

    /* Reset platform-specific extensions via vtable */
    const PlatformVTable* vtable = engine->config.platform_vtable;
    if (vtable && vtable->reset_state) {
        vtable->reset_state(engine->states->extension,
                            engine->states->extension_count, agent_idx);
    }

    /* Reset episode tracking for this drone */
    engine->episode_returns[agent_idx] = 0.0f;
    engine->episode_lengths[agent_idx] = 0;

    /* Clear termination flags */
    engine->dones[agent_idx] = 0;
    engine->truncations[agent_idx] = 0;
}

/* ============================================================================
 * Full Reset
 * ============================================================================ */

void engine_reset(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");

    uint32_t total_agents = engine->config.total_agents;

    /* Re-seed RNG to original seed for deterministic resets */
    pcg32_seed(&engine->rng, engine->config.seed);

    /* Reset all drones with randomized positions */
    for (uint32_t i = 0; i < total_agents; i++) {
        Vec3 position = generate_spawn_position(engine);
        Quat orientation = QUAT_IDENTITY;
        reset_single_drone_internal(engine, i, position, orientation);
    }

    /* Reset collision system (clears stale penetration_depth used by ground effect) */
    collision_reset(engine->collision);

    /* Reset reward system */
    for (uint32_t i = 0; i < total_agents; i++) {
        reward_reset(engine->rewards, i);
    }

    /* Reset sensor system */
    sensor_system_reset(engine->sensors);

    /* dones/truncations already cleared per-drone by reset_single_drone_internal */

    /* Targets must be set externally via engine_set_target()/engine_set_targets() */

    /* Compute initial observations */
    sensor_system_sample_all(engine->sensors, &engine->states->rigid_body,
                             engine->world, engine->collision,
                             total_agents);

    /* Sensor observations: zero-copy (sensor system writes directly to engine->observations) */

    /* Clear needs_reset flag */
    engine->needs_reset = false;
}

/* ============================================================================
 * Partial Reset
 * ============================================================================ */

void engine_reset_envs(BatchEngine* engine, const uint32_t* env_indices, uint32_t count) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");
    FOUNDATION_ASSERT(env_indices != NULL || count == 0, "env_indices is NULL with non-zero count");

    uint32_t agents_per_env = engine->config.agents_per_env;

    for (uint32_t e = 0; e < count; e++) {
        uint32_t env_id = env_indices[e];
        FOUNDATION_ASSERT(env_id < engine->config.num_envs, "env_id out of bounds");

        /* Reset all drones in this environment */
        for (uint32_t d = 0; d < agents_per_env; d++) {
            uint32_t agent_idx = env_id * agents_per_env + d;
            Vec3 position = generate_spawn_position(engine);
            Quat orientation = QUAT_IDENTITY;
            reset_single_drone_internal(engine, agent_idx, position, orientation);

            /* Reset reward state */
            reward_reset(engine->rewards, agent_idx);
        }
    }
}

/* ============================================================================
 * Single Drone Reset
 * ============================================================================ */

void engine_reset_agent(BatchEngine* engine, uint32_t agent_idx,
                        Vec3 position, Quat orientation) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");
    FOUNDATION_ASSERT(agent_idx < engine->config.total_agents, "drone index out of bounds");

    reset_single_drone_internal(engine, agent_idx, position, orientation);
    reward_reset(engine->rewards, agent_idx);
}

/* ============================================================================
 * Auto-Reset Terminated Drones
 * ============================================================================ */

void engine_step_reset_terminated(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");

    uint32_t total_agents = engine->config.total_agents;

    /* Scratch arrays for batch reset */
    /* Using a reasonable max size - in practice, not all drones terminate at once */
    uint32_t max_reset = total_agents;
    uint32_t* done_indices = (uint32_t*)arena_alloc(engine->frame_arena,
                                                     max_reset * sizeof(uint32_t));
    Vec3* reset_positions = (Vec3*)arena_alloc(engine->frame_arena,
                                                max_reset * sizeof(Vec3));
    Quat* reset_orientations = (Quat*)arena_alloc(engine->frame_arena,
                                                   max_reset * sizeof(Quat));

    uint32_t done_count = 0;

    /* Collect terminated drones */
    for (uint32_t i = 0; i < total_agents; i++) {
        if (engine->dones[i] || engine->truncations[i]) {
            /* Log episode completion */
            engine->total_episodes++;

            /* Store reset info */
            done_indices[done_count] = i;
            reset_positions[done_count] = generate_spawn_position(engine);
            reset_orientations[done_count] = QUAT_IDENTITY;
            done_count++;
        }
    }

    /* Batch reset terminated drones */
    if (done_count > 0) {
        /* Reset rigid body state (position, velocity, quaternion, omega) */
        rigid_body_state_reset_batch(&engine->states->rigid_body, done_indices,
                                reset_positions, reset_orientations, done_count);

        /* Reset platform-specific extension state (e.g. RPMs) via vtable */
        const PlatformVTable* vt = engine->config.platform_vtable;
        if (vt && vt->reset_state) {
            for (uint32_t i = 0; i < done_count; i++) {
                vt->reset_state(
                    engine->states->extension,
                    engine->states->extension_count,
                    done_indices[i]);
            }
        }

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
