/**
 * Engine Buffers Implementation
 *
 * Provides zero-copy buffer access and dimension getters.
 */

#include "environment_manager.h"

/* ============================================================================
 * Buffer Getters
 * ============================================================================ */

float* engine_get_observations(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->observations;
}

float* engine_get_actions(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->actions;
}

float* engine_get_rewards(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->rewards_buffer;
}

uint8_t* engine_get_dones(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->dones;
}

uint8_t* engine_get_truncations(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->truncations;
}

/* ============================================================================
 * Dimension Getters
 * ============================================================================ */

uint32_t engine_get_num_envs(const BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->config.num_envs;
}

uint32_t engine_get_drones_per_env(const BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->config.drones_per_env;
}

uint32_t engine_get_total_drones(const BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->config.total_drones;
}

uint32_t engine_get_obs_dim(const BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->obs_dim;
}

uint32_t engine_get_action_dim(const BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    return engine->action_dim;
}

/* ============================================================================
 * Index Conversion Utilities
 * ============================================================================ */

void engine_drone_idx_to_env(const BatchDroneEngine* engine, uint32_t drone_idx,
                             uint32_t* env_id, uint32_t* local_id) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(drone_idx < engine->config.total_drones, "assertion failed");

    uint32_t drones_per_env = engine->config.drones_per_env;
    *env_id = drone_idx / drones_per_env;
    *local_id = drone_idx % drones_per_env;
}

uint32_t engine_env_to_drone_idx(const BatchDroneEngine* engine,
                                 uint32_t env_id, uint32_t local_id) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(env_id < engine->config.num_envs, "assertion failed");
    FOUNDATION_ASSERT(local_id < engine->config.drones_per_env, "assertion failed");

    return env_id * engine->config.drones_per_env + local_id;
}
