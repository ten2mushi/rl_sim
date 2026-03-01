/**
 * Engine Step Implementation
 *
 * Implements the step pipeline: physics -> collision -> sensors -> rewards -> reset.
 */

#include "environment_manager.h"
#include "noise.h"
#include <string.h>

/* ============================================================================
 * Timing Utility
 * ============================================================================ */

#if defined(__APPLE__)
#include <mach/mach_time.h>

double engine_get_time_ms(void) {
    static mach_timebase_info_data_t timebase = {0};
    if (timebase.denom == 0) {
        mach_timebase_info(&timebase);
    }
    uint64_t time = mach_absolute_time();
    return (double)(time * timebase.numer) / (timebase.denom * 1e6);
}

#elif defined(__linux__)
#include <time.h>

double engine_get_time_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

#else
#include <time.h>

double engine_get_time_ms(void) {
    return (double)clock() / (CLOCKS_PER_SEC / 1000.0);
}

#endif

/* ============================================================================
 * Individual Step Phases
 * ============================================================================ */

void engine_step_physics(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");

    double start = engine_get_time_ms();

    /* Provide SDF distances from previous collision frame for ground effect */
    engine->physics->sdf_distances = engine->collision->penetration_depth;

    /* Physics step expects actions in [total_agents × 4] layout */
    physics_step(engine->physics, engine->states, engine->params,
                 engine->actions, engine->config.total_agents);

    engine->physics_time_ms = engine_get_time_ms() - start;
}

void engine_step_collision(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");

    double start = engine_get_time_ms();

    /* Detect all collisions */
    collision_detect_all(engine->collision, &engine->states->rigid_body,
                         engine->world, engine->config.total_agents);

    /* Apply collision response */
    collision_apply_response(engine->collision, &engine->states->rigid_body,
                             &engine->params->rigid_body,
                             0.5f,  /* restitution */
                             1.0f,  /* separation_force */
                             engine->config.total_agents);

    engine->collision_time_ms = engine_get_time_ms() - start;
}

void engine_step_sensors(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");

    double start = engine_get_time_ms();

    if (engine->gpu_sensor_ctx != NULL) {
        /* GPU-accelerated pipeline:
         * a. Sync SDF atlas + drone poses to GPU
         * b. Dispatch GPU sensors (async, returns immediately)
         * c. Sample CPU-only sensors while GPU works
         * d. Wait for GPU completion and scatter results */
        gpu_sensor_context_sync_frame(engine->gpu_sensor_ctx,
                                       engine->world, &engine->states->rigid_body,
                                       engine->config.total_agents);

        gpu_sensors_dispatch(engine->gpu_sensor_ctx, engine->sensors,
                              engine->world, engine->config.total_agents);

        sensor_system_sample_cpu_only(engine->sensors, &engine->states->rigid_body,
                                       engine->world, engine->collision,
                                       engine->config.total_agents);

        gpu_sensors_wait(engine->gpu_sensor_ctx);
        gpu_sensors_scatter_results(engine->gpu_sensor_ctx, engine->sensors,
                                     engine->config.total_agents);

        /* Apply noise to GPU sensor outputs (post-scatter, in obs buffer) */
        for (uint32_t s = 0; s < engine->sensors->sensor_count; s++) {
            Sensor* sensor = &engine->sensors->sensors[s];
            if (sensor->vtable->batch_sample_gpu == NULL) continue;
            if (sensor->noise_state == NULL) continue;

            uint32_t count = engine->sensors->drones_per_sensor[s];
            if (count == 0) continue;

            const uint32_t* indices = engine->sensors->drones_by_sensor[s];
            uint32_t out_size = (uint32_t)sensor->output_size;
            float dt = engine->sensors->dt;

            /* Pre-compute output offsets for noise gather/scatter */
            uint32_t* output_offsets = (uint32_t*)arena_alloc_aligned(
                engine->frame_arena, count * sizeof(uint32_t), 32);
            if (output_offsets == NULL) continue;

            for (uint32_t i = 0; i < count; i++) {
                uint32_t agent_idx = indices[i];
                uint32_t base = agent_idx * MAX_SENSORS_PER_DRONE;
                uint32_t ac = engine->sensors->attachment_counts[agent_idx];
                output_offsets[i] = 0;
                for (uint32_t a = 0; a < ac; a++) {
                    if (engine->sensors->attachments[base + a].sensor_idx == s) {
                        output_offsets[i] = engine->sensors->attachments[base + a].output_offset;
                        break;
                    }
                }
            }

            /* Allocate scratch batch buffer, copy from obs, apply noise, copy back */
            size_t batch_bytes = (size_t)count * out_size * sizeof(float);
            float* batch = (float*)arena_alloc_aligned(
                engine->frame_arena, batch_bytes, SENSOR_OBS_ALIGNMENT);
            if (batch == NULL) continue;

            /* Gather from obs buffer */
            for (uint32_t i = 0; i < count; i++) {
                uint32_t agent_idx = indices[i];
                float* src = engine->sensors->observation_buffer +
                             agent_idx * engine->sensors->obs_dim +
                             output_offsets[i];
                memcpy(batch + (size_t)i * out_size, src, out_size * sizeof(float));
            }

            /* Apply noise */
            noise_apply(&sensor->noise_config, sensor->noise_state,
                        batch, indices, count, out_size, dt);

            /* Scatter back to obs buffer */
            for (uint32_t i = 0; i < count; i++) {
                uint32_t agent_idx = indices[i];
                float* dst = engine->sensors->observation_buffer +
                             agent_idx * engine->sensors->obs_dim +
                             output_offsets[i];
                memcpy(dst, batch + (size_t)i * out_size, out_size * sizeof(float));
            }
        }
    } else {
        /* CPU-only pipeline */
        sensor_system_sample_all(engine->sensors, &engine->states->rigid_body,
                                  engine->world, engine->collision,
                                  engine->config.total_agents);
    }

    engine->sensor_time_ms = engine_get_time_ms() - start;
}

void engine_step_rewards(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");

    double start = engine_get_time_ms();

    /* Get collision results for reward computation */
    CollisionResults col_results = collision_get_results(engine->collision);

    /* Compute rewards */
    reward_compute(engine->rewards, &engine->states->rigid_body,
                   &engine->params->rigid_body,
                   engine->actions, &col_results,
                   engine->rewards_buffer, engine->config.total_agents);

    /* Compute termination conditions */
    TerminationFlags term_flags = {
        .done = engine->dones,
        .truncated = engine->truncations,
        .success = engine->term_success,
        .collision = engine->term_collision,
        .out_of_bounds = engine->term_out_of_bounds,
        .timeout = engine->term_timeout
    };

    Vec3 term_min = engine->config.use_custom_termination
        ? engine->config.termination_min : engine->config.world_min;
    Vec3 term_max = engine->config.use_custom_termination
        ? engine->config.termination_max : engine->config.world_max;

    reward_compute_terminations(engine->rewards, &engine->states->rigid_body,
                                &col_results,
                                term_min, term_max,
                                engine->config.max_episode_steps, &term_flags,
                                engine->config.total_agents);

    engine->reward_time_ms = engine_get_time_ms() - start;
}

/* ============================================================================
 * Core Step Pipeline (shared by engine_step and engine_step_no_reset)
 * ============================================================================ */

static void engine_step_core(BatchEngine* engine) {
    /* Reset frame arena */
    arena_reset(engine->frame_arena);

    /* Physics integration */
    engine_step_physics(engine);

    /* Collision detection and response */
    engine_step_collision(engine);

    /* Sensor sampling (includes GPU wait+scatter if applicable) */
    engine_step_sensors(engine);

    /* Reward computation and termination checking */
    engine_step_rewards(engine);
}

/* ============================================================================
 * Full Step Pipeline
 * ============================================================================ */

void engine_step(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");
    FOUNDATION_ASSERT(!engine->needs_reset, "engine needs reset before step");

    engine_step_core(engine);

    /* Update episode tracking */
    uint32_t total_agents = engine->config.total_agents;
    for (uint32_t i = 0; i < total_agents; i++) {
        engine->episode_returns[i] += engine->rewards_buffer[i];
        engine->episode_lengths[i]++;
    }

    /* Auto-reset terminated environments */
    double reset_start = engine_get_time_ms();
    engine_step_reset_terminated(engine);
    engine->reset_time_ms = engine_get_time_ms() - reset_start;

    /* Update statistics */
    engine->total_steps++;
}

void engine_step_no_reset(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(engine->initialized, "engine not initialized");
    FOUNDATION_ASSERT(!engine->needs_reset, "engine needs reset before step");

    engine_step_core(engine);

    /* Update episode tracking (skip terminated drones) */
    uint32_t total_agents = engine->config.total_agents;
    for (uint32_t i = 0; i < total_agents; i++) {
        if (!engine->dones[i] && !engine->truncations[i]) {
            engine->episode_returns[i] += engine->rewards_buffer[i];
            engine->episode_lengths[i]++;
        }
    }

    /* Update statistics (no reset) */
    engine->total_steps++;
}
