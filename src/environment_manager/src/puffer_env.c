/**
 * PufferLib Integration Implementation
 *
 * Provides PufferEnv wrapper for standard RL environment interface.
 */

#include "environment_manager.h"
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * PufferEnv Creation
 * ============================================================================ */

PufferEnv* puffer_env_create(const char* config_path) {
    EngineConfig config = engine_config_default();

    /* Load config from file if provided */
    if (config_path != NULL) {
        char error_msg[ENGINE_ERROR_MSG_SIZE];
        if (engine_config_load(config_path, &config, error_msg) != 0) {
            return NULL;
        }
    }

    return puffer_env_create_from_config(&config, NULL);
}

PufferEnv* puffer_env_create_from_config(const EngineConfig* config, char* error_msg) {
    if (config == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE, "config is NULL");
        return NULL;
    }

    /* Create engine */
    char local_error[ENGINE_ERROR_MSG_SIZE];
    char* err = error_msg ? error_msg : local_error;
    BatchDroneEngine* engine = engine_create(config, err);
    if (engine == NULL) {
        return NULL;
    }

    /* Allocate PufferEnv from persistent arena */
    PufferEnv* env = (PufferEnv*)arena_alloc(engine->persistent_arena,
                                              sizeof(PufferEnv));
    if (env == NULL) {
        engine_destroy(engine);
        return NULL;
    }

    /* Initialize PufferEnv */
    env->engine = engine;

    /* Alias buffers (zero-copy) */
    env->observations = engine->observations;
    env->actions = engine->actions;
    env->rewards = engine->rewards_buffer;
    env->terminals = engine->dones;
    env->truncations = engine->truncations;

    /* Set dimensions */
    env->num_envs = (int)engine->config.num_envs;
    env->num_agents = (int)engine->config.drones_per_env;
    env->obs_size = (int)engine->obs_dim;
    env->action_size = (int)engine->action_dim;

    /* Metadata */
    env->name = "DroneSwarm";
    env->version = "1.0.0";

    return env;
}

/* ============================================================================
 * PufferEnv Reset
 * ============================================================================ */

void puffer_env_reset(PufferEnv* env) {
    if (env == NULL || env->engine == NULL) {
        return;
    }

    engine_reset(env->engine);
}

/* ============================================================================
 * PufferEnv Step
 * ============================================================================ */

void puffer_env_step(PufferEnv* env) {
    if (env == NULL || env->engine == NULL) {
        return;
    }

    engine_step(env->engine);
}

/* ============================================================================
 * PufferEnv Close
 * ============================================================================ */

void puffer_env_close(PufferEnv* env) {
    if (env == NULL) {
        return;
    }

    if (env->engine != NULL) {
        /* PufferEnv itself is allocated from engine's persistent arena,
         * so engine_destroy frees this struct. Save the engine pointer
         * and do NOT write to env after engine_destroy returns. */
        BatchDroneEngine* engine = env->engine;
        engine_destroy(engine);
    }
}

/* ============================================================================
 * Space Queries
 * ============================================================================ */

void puffer_env_get_observation_space(PufferEnv* env, int* shape, int* ndim) {
    if (env == NULL || shape == NULL || ndim == NULL) {
        return;
    }

    *ndim = 2;
    shape[0] = env->num_envs * env->num_agents;
    shape[1] = env->obs_size;
}

void puffer_env_get_action_space(PufferEnv* env, int* shape, int* ndim) {
    if (env == NULL || shape == NULL || ndim == NULL) {
        return;
    }

    *ndim = 2;
    shape[0] = env->num_envs * env->num_agents;
    shape[1] = env->action_size;
}

/* ============================================================================
 * Render (Placeholder)
 * ============================================================================ */

void puffer_env_render(PufferEnv* env, const char* mode) {
    (void)env;
    (void)mode;
    /* Placeholder - actual rendering would use OpenGL/Vulkan */
}
