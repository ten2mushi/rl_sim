/**
 * Engine Buffers Implementation
 *
 * Provides zero-copy buffer access and dimension getters.
 */

#include "environment_manager.h"

/* ============================================================================
 * Buffer Getters
 * ============================================================================ */

float* engine_get_observations(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->observations;
}

float* engine_get_actions(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->actions;
}

float* engine_get_rewards(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->rewards_buffer;
}

uint8_t* engine_get_dones(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->dones;
}

uint8_t* engine_get_truncations(BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->truncations;
}

/* ============================================================================
 * Dimension Getters
 * ============================================================================ */

uint32_t engine_get_num_envs(const BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->config.num_envs;
}

uint32_t engine_get_agents_per_env(const BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->config.agents_per_env;
}

uint32_t engine_get_total_agents(const BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->config.total_agents;
}

uint32_t engine_get_obs_dim(const BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->obs_dim;
}

uint32_t engine_get_action_dim(const BatchEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    return engine->action_dim;
}

/* ============================================================================
 * Index Conversion Utilities
 * ============================================================================ */

void engine_agent_idx_to_env(const BatchEngine* engine, uint32_t agent_idx,
                             uint32_t* env_id, uint32_t* local_id) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(agent_idx < engine->config.total_agents, "drone index out of bounds");

    uint32_t agents_per_env = engine->config.agents_per_env;
    *env_id = agent_idx / agents_per_env;
    *local_id = agent_idx % agents_per_env;
}

uint32_t engine_env_to_agent_idx(const BatchEngine* engine,
                                 uint32_t env_id, uint32_t local_id) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(env_id < engine->config.num_envs, "env_id out of bounds");
    FOUNDATION_ASSERT(local_id < engine->config.agents_per_env, "local_id out of bounds");

    return env_id * engine->config.agents_per_env + local_id;
}
