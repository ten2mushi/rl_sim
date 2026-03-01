/**
 * PufferLib Binding for Robot RL Engine
 *
 * This file implements the C extension binding layer that connects the
 * high-performance robot simulation engine to PufferLib's Python interface.
 *
 * Key Features:
 * - Zero-copy observation/action buffers shared with Python/NumPy
 * - Automatic log aggregation across vectorized environments
 * - Compatible with PufferLib's env_binding.h template
 *
 * Usage:
 *   from rl_engine import binding
 *   env = binding.env_init(obs, actions, rewards, terminals, truncations, seed,
 *                          num_envs=4, agents_per_env=16, config_path="config.toml")
 */

#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>

/* Include rl_engine headers */
#include "environment_manager.h"
#include "platform.h"
#include "sensor_implementations.h"
#include "sensor_system.h"
#include "world_brick_map.h"

/* ============================================================================
 * Log Structure (MUST be all floats, 'n' MUST be last)
 * ============================================================================ */

typedef struct Log {
    float episode_return;        /* Cumulative reward per episode */
    float episode_length;        /* Steps per episode */
    float collision_rate;        /* Fraction of episodes ending in collision */
    float out_of_bounds;         /* Fraction of episodes ending out of bounds */
    float timeout;               /* Fraction of episodes ending in timeout */
    float success_rate;          /* Fraction of episodes reaching goal */
    float avg_velocity;          /* Average velocity magnitude */
    float avg_distance_to_goal;  /* Average distance to goal at episode end */
    float n;                     /* Episode count - MUST be last */
} Log;

/* ============================================================================
 * RobotEnv Structure
 * ============================================================================ */

typedef struct RobotEnv {
    /* REQUIRED: Log MUST be first field for PufferLib aggregation */
    Log log;

    /* REQUIRED: Buffer pointers (shared with Python/NumPy) */
    float* observations;
    float* actions;
    float* rewards;
    unsigned char* terminals;
    unsigned char* truncations;

    /* Engine handle */
    PufferEnv* puffer_env;

    /* Configuration */
    int num_envs;
    int agents_per_env;
    int total_agents;
    int tick;

    /* Per-episode tracking (for logging) */
    float* episode_returns;
    int* episode_lengths;

    /* Saved GPU context when GPU is temporarily disabled */
    struct GpuSensorContext* saved_gpu_ctx;
} RobotEnv;

/* Define Env type for env_binding.h template */
#define Env RobotEnv

/* ============================================================================
 * Core Functions (called by env_binding.h)
 * ============================================================================ */

/**
 * Reset environment to initial state
 */
static void c_reset(RobotEnv* env) {
    if (env == NULL || env->puffer_env == NULL) return;

    env->tick = 0;
    puffer_env_reset(env->puffer_env);

    /* Copy engine observations to numpy buffer after reset */
    BatchEngine* engine = env->puffer_env->engine;
    if (engine != NULL && engine->observations != NULL && env->observations != NULL) {
        memcpy(env->observations, engine->observations,
               (size_t)env->total_agents * engine->obs_dim * sizeof(float));
    }

    /* Reset episode tracking */
    if (env->episode_returns) {
        memset(env->episode_returns, 0, env->total_agents * sizeof(float));
    }
    if (env->episode_lengths) {
        memset(env->episode_lengths, 0, env->total_agents * sizeof(int));
    }
}

/**
 * Step environment forward by one timestep
 */
static void c_step(RobotEnv* env) {
    if (env == NULL || env->puffer_env == NULL) return;

    env->tick++;

    /* Copy actions from numpy buffer to engine before stepping */
    BatchEngine* engine = env->puffer_env->engine;
    if (engine != NULL && env->actions != NULL && engine->actions != NULL) {
        memcpy(engine->actions, env->actions,
               (size_t)env->total_agents * engine->action_dim * sizeof(float));
    }

    /* Step the underlying engine */
    puffer_env_step(env->puffer_env);

    /* Copy engine buffers to numpy buffers */
    if (engine != NULL) {
        if (engine->observations != NULL && env->observations != NULL) {
            memcpy(env->observations, engine->observations,
                   (size_t)env->total_agents * engine->obs_dim * sizeof(float));
        }
        if (engine->rewards_buffer != NULL && env->rewards != NULL) {
            memcpy(env->rewards, engine->rewards_buffer,
                   (size_t)env->total_agents * sizeof(float));
        }
        if (engine->dones != NULL && env->terminals != NULL) {
            memcpy(env->terminals, engine->dones,
                   (size_t)env->total_agents * sizeof(unsigned char));
        }
        if (engine->truncations != NULL && env->truncations != NULL) {
            memcpy(env->truncations, engine->truncations,
                   (size_t)env->total_agents * sizeof(unsigned char));
        }
    }

    /* Update episode tracking and logging */
    for (int i = 0; i < env->total_agents; i++) {
        /* Accumulate episode return */
        if (env->episode_returns) {
            env->episode_returns[i] += env->rewards[i];
        }
        if (env->episode_lengths) {
            env->episode_lengths[i]++;
        }

        /* Check for episode termination or truncation */
        if (env->terminals[i] || (env->truncations && env->truncations[i])) {
            /* Record episode stats to log */
            if (env->episode_returns) {
                env->log.episode_return += env->episode_returns[i];
            }
            if (env->episode_lengths) {
                env->log.episode_length += (float)env->episode_lengths[i];
            }

            /* Termination reason tracking */
            if (engine != NULL) {
                if (engine->term_collision && engine->term_collision[i]) {
                    env->log.collision_rate += 1.0f;
                }
                if (engine->term_out_of_bounds && engine->term_out_of_bounds[i]) {
                    env->log.out_of_bounds += 1.0f;
                }
                if (engine->term_timeout && engine->term_timeout[i]) {
                    env->log.timeout += 1.0f;
                }
                if (engine->term_success && engine->term_success[i]) {
                    env->log.success_rate += 1.0f;
                }
            }

            env->log.n += 1.0f;

            /* Reset episode tracking for this drone (auto-reset handled by engine) */
            if (env->episode_returns) {
                env->episode_returns[i] = 0.0f;
            }
            if (env->episode_lengths) {
                env->episode_lengths[i] = 0;
            }
        }
    }
}

/**
 * Render environment (currently a no-op, placeholder for future visualization)
 */
static void c_render(RobotEnv* env) {
    if (env == NULL || env->puffer_env == NULL) return;
    puffer_env_render(env->puffer_env, "human");
}

/**
 * Close and cleanup environment resources
 */
static void c_close(RobotEnv* env) {
    if (env == NULL) return;

    /* Restore GPU context before closing so engine_destroy can clean it up */
    if (env->saved_gpu_ctx != NULL && env->puffer_env != NULL &&
        env->puffer_env->engine != NULL) {
        env->puffer_env->engine->gpu_sensor_ctx = env->saved_gpu_ctx;
        env->saved_gpu_ctx = NULL;
    }

    if (env->puffer_env != NULL) {
        puffer_env_close(env->puffer_env);
        env->puffer_env = NULL;
    }

    if (env->episode_returns) {
        free(env->episode_returns);
        env->episode_returns = NULL;
    }
    if (env->episode_lengths) {
        free(env->episode_lengths);
        env->episode_lengths = NULL;
    }

    /* Note: observations, actions, rewards, terminals are owned by Python
     * (NumPy arrays), so we don't free them here */
}

/* ============================================================================
 * PufferLib Binding Callbacks
 * ============================================================================ */

/**
 * Initialize environment from Python keyword arguments
 */
static int my_init(RobotEnv* env, PyObject* args, PyObject* kwargs) {
    /* Extract configuration from kwargs */
    PyObject* num_envs_obj = PyDict_GetItemString(kwargs, "num_envs");
    PyObject* agents_per_env_obj = PyDict_GetItemString(kwargs, "agents_per_env");
    PyObject* config_path_obj = PyDict_GetItemString(kwargs, "config_path");
    PyObject* seed_obj = PyDict_GetItemString(kwargs, "seed");

    /* Set defaults */
    env->num_envs = num_envs_obj ? (int)PyLong_AsLong(num_envs_obj) : 64;
    env->agents_per_env = agents_per_env_obj ? (int)PyLong_AsLong(agents_per_env_obj) : 16;
    env->total_agents = env->num_envs * env->agents_per_env;
    env->tick = 0;

    /* Get config path if provided */
    const char* config_path = NULL;
    if (config_path_obj && PyUnicode_Check(config_path_obj)) {
        config_path = PyUnicode_AsUTF8(config_path_obj);
    }

    /* Parse platform type (default: quadcopter) */
    const char* platform_name = "quadcopter";
    PyObject* platform_obj = PyDict_GetItemString(kwargs, "platform");
    if (platform_obj && PyUnicode_Check(platform_obj)) {
        platform_name = PyUnicode_AsUTF8(platform_obj);
    }

    /* Create engine config — TOML first (if provided), then kwargs override */
    EngineConfig config = engine_config_default();

    if (config_path != NULL) {
        char error_msg[ENGINE_ERROR_MSG_SIZE];
        if (engine_config_load(config_path, &config, error_msg) != 0) {
            PyErr_SetString(PyExc_RuntimeError, error_msg);
            return -1;
        }
    }

    /* Kwargs override TOML values (or set defaults if no TOML) */
    config.num_envs = (uint32_t)env->num_envs;
    config.agents_per_env = (uint32_t)env->agents_per_env;
    config.total_agents = (uint32_t)env->total_agents;

    if (seed_obj) {
        config.seed = (uint64_t)PyLong_AsUnsignedLongLong(seed_obj);
    }

    /* Lookup platform vtable */
    PlatformRegistry registry;
    platform_registry_init(&registry);
    const PlatformVTable* vtable = platform_registry_find(&registry, platform_name);
    if (vtable == NULL) {
        PyErr_Format(PyExc_ValueError, "Unknown platform: '%s'", platform_name);
        return -1;
    }
    config.platform_vtable = vtable;

    /* Parse OBJ file path */
    PyObject* obj_path_obj = PyDict_GetItemString(kwargs, "obj_path");
    if (obj_path_obj && PyUnicode_Check(obj_path_obj)) {
        config.obj_path = PyUnicode_AsUTF8(obj_path_obj);
    }

    /* Parse GPU voxelization preference (default: true) */
    PyObject* gpu_vox_obj = PyDict_GetItemString(kwargs, "use_gpu_voxelization");
    if (gpu_vox_obj) {
        config.use_gpu_voxelization = PyObject_IsTrue(gpu_vox_obj);
    }

    /* Parse world bounds */
    PyObject* voxel_size_obj = PyDict_GetItemString(kwargs, "voxel_size");
    if (voxel_size_obj) config.voxel_size = (float)PyFloat_AsDouble(voxel_size_obj);

    PyObject* max_bricks_obj = PyDict_GetItemString(kwargs, "max_bricks");
    if (max_bricks_obj) config.max_bricks = (uint32_t)PyLong_AsUnsignedLong(max_bricks_obj);

    PyObject* world_min_obj = PyDict_GetItemString(kwargs, "world_min");
    PyObject* world_max_obj = PyDict_GetItemString(kwargs, "world_max");
    if (world_min_obj && PyTuple_Check(world_min_obj) && PyTuple_Size(world_min_obj) == 3) {
        config.world_min = (Vec3){
            (float)PyFloat_AsDouble(PyTuple_GetItem(world_min_obj, 0)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(world_min_obj, 1)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(world_min_obj, 2)),
            0.0f
        };
        config.use_custom_bounds = true;
    }
    if (world_max_obj && PyTuple_Check(world_max_obj) && PyTuple_Size(world_max_obj) == 3) {
        config.world_max = (Vec3){
            (float)PyFloat_AsDouble(PyTuple_GetItem(world_max_obj, 0)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(world_max_obj, 1)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(world_max_obj, 2)),
            0.0f
        };
        config.use_custom_bounds = true;
    }

    /* Parse spawn region */
    PyObject* spawn_min_obj = PyDict_GetItemString(kwargs, "spawn_min");
    PyObject* spawn_max_obj = PyDict_GetItemString(kwargs, "spawn_max");
    if (spawn_min_obj && spawn_max_obj && PyTuple_Check(spawn_min_obj) && PyTuple_Check(spawn_max_obj)) {
        config.spawn_min = (Vec3){
            (float)PyFloat_AsDouble(PyTuple_GetItem(spawn_min_obj, 0)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(spawn_min_obj, 1)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(spawn_min_obj, 2)),
            0.0f
        };
        config.spawn_max = (Vec3){
            (float)PyFloat_AsDouble(PyTuple_GetItem(spawn_max_obj, 0)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(spawn_max_obj, 1)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(spawn_max_obj, 2)),
            0.0f
        };
        config.use_custom_spawn = true;
    }

    /* Parse termination bounds */
    PyObject* term_min_obj = PyDict_GetItemString(kwargs, "termination_min");
    PyObject* term_max_obj = PyDict_GetItemString(kwargs, "termination_max");
    if (term_min_obj && term_max_obj && PyTuple_Check(term_min_obj) && PyTuple_Check(term_max_obj)) {
        config.termination_min = (Vec3){
            (float)PyFloat_AsDouble(PyTuple_GetItem(term_min_obj, 0)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(term_min_obj, 1)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(term_min_obj, 2)),
            0.0f
        };
        config.termination_max = (Vec3){
            (float)PyFloat_AsDouble(PyTuple_GetItem(term_max_obj, 0)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(term_max_obj, 1)),
            (float)PyFloat_AsDouble(PyTuple_GetItem(term_max_obj, 2)),
            0.0f
        };
        config.use_custom_termination = true;
    }

    /* Parse physics tunables */
    PyObject* collision_radius_obj = PyDict_GetItemString(kwargs, "collision_radius");
    if (collision_radius_obj) config.collision_radius = (float)PyFloat_AsDouble(collision_radius_obj);

    PyObject* air_density_obj = PyDict_GetItemString(kwargs, "air_density");
    if (air_density_obj) config.air_density = (float)PyFloat_AsDouble(air_density_obj);

    PyObject* enable_ground_effect_obj = PyDict_GetItemString(kwargs, "enable_ground_effect");
    if (enable_ground_effect_obj) config.enable_ground_effect = PyObject_IsTrue(enable_ground_effect_obj);

    PyObject* enable_drag_obj = PyDict_GetItemString(kwargs, "enable_drag");
    if (enable_drag_obj) config.enable_drag = PyObject_IsTrue(enable_drag_obj);

    PyObject* enable_motor_dynamics_obj = PyDict_GetItemString(kwargs, "enable_motor_dynamics");
    if (enable_motor_dynamics_obj) config.enable_motor_dynamics = PyObject_IsTrue(enable_motor_dynamics_obj);

    PyObject* ground_effect_height_obj = PyDict_GetItemString(kwargs, "ground_effect_height");
    if (ground_effect_height_obj) config.ground_effect_height = (float)PyFloat_AsDouble(ground_effect_height_obj);

    PyObject* ground_effect_coeff_obj = PyDict_GetItemString(kwargs, "ground_effect_coeff");
    if (ground_effect_coeff_obj) config.ground_effect_coeff = (float)PyFloat_AsDouble(ground_effect_coeff_obj);

    PyObject* collision_cell_size_obj = PyDict_GetItemString(kwargs, "collision_cell_size");
    if (collision_cell_size_obj) config.collision_cell_size = (float)PyFloat_AsDouble(collision_cell_size_obj);

    /* Parse sensor configurations (only add kwargs sensors if no TOML sensors) */
    if (config_path == NULL || config.num_sensor_configs == 0) {
        PyObject* camera_width_obj = PyDict_GetItemString(kwargs, "camera_width");
        PyObject* camera_height_obj = PyDict_GetItemString(kwargs, "camera_height");
        if (camera_width_obj && camera_height_obj) {
            uint32_t cam_w = (uint32_t)PyLong_AsUnsignedLong(camera_width_obj);
            uint32_t cam_h = (uint32_t)PyLong_AsUnsignedLong(camera_height_obj);
            float cam_fov = 1.5708f;  /* 90 degrees default */
            float cam_far = 30.0f;

            PyObject* camera_fov_obj = PyDict_GetItemString(kwargs, "camera_fov");
            if (camera_fov_obj) cam_fov = (float)PyFloat_AsDouble(camera_fov_obj);

            PyObject* camera_far_obj = PyDict_GetItemString(kwargs, "camera_far");
            if (camera_far_obj) cam_far = (float)PyFloat_AsDouble(camera_far_obj);

            /* Add RGB camera */
            SensorConfig rgb_cfg = sensor_config_camera(cam_w, cam_h, cam_fov, cam_far);
            engine_config_add_sensor(&config, &rgb_cfg);

            /* Add Depth camera (same params, different type) */
            SensorConfig depth_cfg = rgb_cfg;
            depth_cfg.type = SENSOR_TYPE_CAMERA_DEPTH;
            engine_config_add_sensor(&config, &depth_cfg);
        }

        /* Add position sensor if requested */
        PyObject* add_pos_obj = PyDict_GetItemString(kwargs, "add_position_sensor");
        if (add_pos_obj && PyObject_IsTrue(add_pos_obj)) {
            SensorConfig pos_cfg = sensor_config_position();
            engine_config_add_sensor(&config, &pos_cfg);
        }

        /* Add velocity sensor if requested */
        PyObject* add_vel_obj = PyDict_GetItemString(kwargs, "add_velocity_sensor");
        if (add_vel_obj && PyObject_IsTrue(add_vel_obj)) {
            SensorConfig vel_cfg = sensor_config_velocity();
            engine_config_add_sensor(&config, &vel_cfg);
        }
    }

    /* Add IMU sensor if requested (works with both TOML and kwargs paths) */
    PyObject* add_imu_obj = PyDict_GetItemString(kwargs, "add_imu_sensor");
    if (add_imu_obj && PyObject_IsTrue(add_imu_obj)) {
        SensorConfig imu_cfg = sensor_config_imu();
        engine_config_add_sensor(&config, &imu_cfg);
    }

    /* Create PufferEnv wrapper (arena auto-sized by engine_create) */
    char engine_error[ENGINE_ERROR_MSG_SIZE];
    engine_error[0] = '\0';
    env->puffer_env = puffer_env_create_from_config(&config, engine_error);
    if (env->puffer_env == NULL) {
        if (engine_error[0] != '\0') {
            PyErr_SetString(PyExc_RuntimeError, engine_error);
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create PufferEnv");
        }
        return -1;
    }

    /* Free sensor configs allocated by engine_config_add_sensor */
    if (config.sensor_configs != NULL) {
        free(config.sensor_configs);
        config.sensor_configs = NULL;
    }

    /* Verify buffer compatibility */
    BatchEngine* engine = env->puffer_env->engine;
    if (engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Engine is NULL");
        return -1;
    }

    /* Point our buffers to engine's zero-copy buffers */
    /* Note: These should match the numpy arrays passed from Python */
    /* The engine's buffers are 32-byte aligned */

    /* Allocate episode tracking arrays */
    env->episode_returns = (float*)calloc(env->total_agents, sizeof(float));
    env->episode_lengths = (int*)calloc(env->total_agents, sizeof(int));

    if (!env->episode_returns || !env->episode_lengths) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate episode tracking arrays");
        return -1;
    }

    /* Initialize log */
    memset(&env->log, 0, sizeof(Log));

    env->saved_gpu_ctx = NULL;

    return 0;
}

/**
 * Populate log dictionary with aggregated metrics
 */
/* Forward declaration of assign_to_dict (defined as static in env_binding.h) */
static int assign_to_dict(PyObject* dict, char* key, float value);

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "collision_rate", log->collision_rate);
    assign_to_dict(dict, "out_of_bounds", log->out_of_bounds);
    assign_to_dict(dict, "timeout", log->timeout);
    assign_to_dict(dict, "success_rate", log->success_rate);
    assign_to_dict(dict, "avg_velocity", log->avg_velocity);
    assign_to_dict(dict, "avg_distance_to_goal", log->avg_distance_to_goal);

    return 0;
}

/* ============================================================================
 * Custom Python Methods (set_agent_state, get_agent_state, step_sensors, etc.)
 * ============================================================================ */

/**
 * set_agent_state(env_handle, agent_idx, px, py, pz, qw, qx, qy, qz)
 * Teleport an agent: set position and orientation, zero velocities.
 */
static PyObject* py_set_agent_state(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int agent_idx;
    float px, py, pz, qw, qx, qy, qz;

    if (!PyArg_ParseTuple(args, "Oifffffff", &handle_obj, &agent_idx,
                          &px, &py, &pz, &qw, &qx, &qy, &qz)) {
        return NULL;
    }

    RobotEnv* env = (RobotEnv*)PyLong_AsVoidPtr(handle_obj);
    if (env == NULL || env->puffer_env == NULL || env->puffer_env->engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid environment handle");
        return NULL;
    }

    BatchEngine* engine = env->puffer_env->engine;
    uint32_t idx = (uint32_t)agent_idx;

    if (idx >= engine->config.total_agents) {
        PyErr_SetString(PyExc_IndexError, "agent_idx out of range");
        return NULL;
    }

    RigidBodyStateSOA* rb = &engine->states->rigid_body;

    /* Set position */
    rb->pos_x[idx] = px;
    rb->pos_y[idx] = py;
    rb->pos_z[idx] = pz;

    /* Set orientation */
    rb->quat_w[idx] = qw;
    rb->quat_x[idx] = qx;
    rb->quat_y[idx] = qy;
    rb->quat_z[idx] = qz;

    /* Zero velocities (teleportation = no momentum) */
    rb->vel_x[idx] = 0.0f;
    rb->vel_y[idx] = 0.0f;
    rb->vel_z[idx] = 0.0f;
    rb->omega_x[idx] = 0.0f;
    rb->omega_y[idx] = 0.0f;
    rb->omega_z[idx] = 0.0f;

    Py_RETURN_NONE;
}

/**
 * get_agent_state(env_handle, agent_idx) -> dict
 */
static PyObject* py_get_agent_state(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int agent_idx;

    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &agent_idx)) {
        return NULL;
    }

    RobotEnv* env = (RobotEnv*)PyLong_AsVoidPtr(handle_obj);
    if (env == NULL || env->puffer_env == NULL || env->puffer_env->engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid environment handle");
        return NULL;
    }

    BatchEngine* engine = env->puffer_env->engine;
    AgentStateQuery query;
    engine_get_agent_state(engine, (uint32_t)agent_idx, &query);

    PyObject* dict = PyDict_New();
    PyObject* pos = Py_BuildValue("(fff)", query.position.x, query.position.y, query.position.z);
    PyObject* orient = Py_BuildValue("(ffff)", query.orientation.w, query.orientation.x,
                                     query.orientation.y, query.orientation.z);
    PyObject* vel = Py_BuildValue("(fff)", query.velocity.x, query.velocity.y, query.velocity.z);
    PyObject* omega = Py_BuildValue("(fff)", query.angular_velocity.x,
                                    query.angular_velocity.y, query.angular_velocity.z);

    PyDict_SetItemString(dict, "position", pos);
    PyDict_SetItemString(dict, "orientation", orient);
    PyDict_SetItemString(dict, "velocity", vel);
    PyDict_SetItemString(dict, "angular_velocity", omega);

    Py_DECREF(pos);
    Py_DECREF(orient);
    Py_DECREF(vel);
    Py_DECREF(omega);

    return dict;
}

/**
 * step_sensors(env_handle)
 * Run sensor-only step and copy observations to numpy buffer.
 * Handles both CPU-only and GPU-accelerated paths correctly.
 */
static PyObject* py_step_sensors(PyObject* self, PyObject* args) {
    PyObject* handle_obj;

    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    RobotEnv* env = (RobotEnv*)PyLong_AsVoidPtr(handle_obj);
    if (env == NULL || env->puffer_env == NULL || env->puffer_env->engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid environment handle");
        return NULL;
    }

    BatchEngine* engine = env->puffer_env->engine;

    /* Run sensor sampling (dispatches GPU async + samples CPU-only sensors) */
    engine_step_sensors(engine);

    /* GPU path: wait for GPU to finish and scatter results to sensor obs buffer */
    if (engine->gpu_sensor_ctx != NULL) {
        gpu_sensors_wait(engine->gpu_sensor_ctx);
        gpu_sensors_scatter_results(engine->gpu_sensor_ctx, engine->sensors,
                                     engine->config.total_agents);
    }

    /* Copy sensor system observations to engine->observations */
    float* sensor_obs = sensor_system_get_observations(engine->sensors);
    if (sensor_obs != NULL && engine->observations != NULL) {
        memcpy(engine->observations, sensor_obs,
               (size_t)engine->config.total_agents * engine->obs_dim * sizeof(float));
    }

    /* Copy engine observations to numpy buffer */
    if (engine->observations != NULL && env->observations != NULL) {
        size_t obs_bytes = (size_t)engine->config.total_agents * engine->obs_dim * sizeof(float);
        memcpy(env->observations, engine->observations, obs_bytes);
    }

    Py_RETURN_NONE;
}

/**
 * set_gpu_enabled(env_handle, enabled)
 * Enable or disable GPU sensor acceleration.
 * When disabled, the GPU context is saved and NULLed out so the engine
 * falls back to the CPU-only sensor path. Re-enabling restores the context.
 */
static PyObject* py_set_gpu_enabled(PyObject* self, PyObject* args) {
    PyObject* handle_obj;
    int enabled;

    if (!PyArg_ParseTuple(args, "Oi", &handle_obj, &enabled)) {
        return NULL;
    }

    RobotEnv* env = (RobotEnv*)PyLong_AsVoidPtr(handle_obj);
    if (env == NULL || env->puffer_env == NULL || env->puffer_env->engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid environment handle");
        return NULL;
    }

    BatchEngine* engine = env->puffer_env->engine;

    if (enabled) {
        /* Restore saved GPU context if we previously disabled it */
        if (engine->gpu_sensor_ctx == NULL && env->saved_gpu_ctx != NULL) {
            engine->gpu_sensor_ctx = env->saved_gpu_ctx;
            env->saved_gpu_ctx = NULL;
        }
    } else {
        /* Save and NULL out the GPU context to force CPU path */
        if (engine->gpu_sensor_ctx != NULL) {
            env->saved_gpu_ctx = engine->gpu_sensor_ctx;
            engine->gpu_sensor_ctx = NULL;
        }
    }

    Py_RETURN_NONE;
}

/**
 * is_gpu_enabled(env_handle) -> bool
 */
static PyObject* py_is_gpu_enabled(PyObject* self, PyObject* args) {
    PyObject* handle_obj;

    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    RobotEnv* env = (RobotEnv*)PyLong_AsVoidPtr(handle_obj);
    if (env == NULL || env->puffer_env == NULL || env->puffer_env->engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid environment handle");
        return NULL;
    }

    if (env->puffer_env->engine->gpu_sensor_ctx != NULL) {
        Py_RETURN_TRUE;
    }
    Py_RETURN_FALSE;
}

/**
 * get_obs_dim(env_handle) -> int
 */
static PyObject* py_get_obs_dim(PyObject* self, PyObject* args) {
    PyObject* handle_obj;

    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    RobotEnv* env = (RobotEnv*)PyLong_AsVoidPtr(handle_obj);
    if (env == NULL || env->puffer_env == NULL || env->puffer_env->engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid environment handle");
        return NULL;
    }

    return PyLong_FromLong((long)env->puffer_env->engine->obs_dim);
}

/**
 * get_action_dim(env_handle) -> int
 */
static PyObject* py_get_action_dim(PyObject* self, PyObject* args) {
    PyObject* handle_obj;

    if (!PyArg_ParseTuple(args, "O", &handle_obj)) {
        return NULL;
    }

    RobotEnv* env = (RobotEnv*)PyLong_AsVoidPtr(handle_obj);
    if (env == NULL || env->puffer_env == NULL || env->puffer_env->engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid environment handle");
        return NULL;
    }

    return PyLong_FromLong((long)env->puffer_env->engine->action_dim);
}

/* ============================================================================
 * Include PufferLib Template (defines env_init, env_step, vec_*, etc.)
 * ============================================================================
 *
 * The env_binding.h header provides:
 * - env_init: Python callable that creates an Env
 * - env_reset, env_step, env_render, env_close: Python callables
 * - vec_init, vec_reset, vec_step, vec_render, vec_close: Vectorized versions
 * - vec_log: Aggregated logging across all environments
 * - vectorize: Creates VecEnv from multiple Env handles
 *
 * These functions call c_reset, c_step, c_render, c_close defined above.
 */

/**
 * debug_world(env_handle) -> dict
 * Returns diagnostic info about the world brick map.
 */
static PyObject* py_debug_world(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* handle_obj;
    if (!PyArg_ParseTuple(args, "O", &handle_obj)) return NULL;

    RobotEnv* env = (RobotEnv*)PyLong_AsVoidPtr(handle_obj);
    if (env == NULL || env->puffer_env == NULL || env->puffer_env->engine == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Invalid environment handle");
        return NULL;
    }

    BatchEngine* engine = env->puffer_env->engine;
    const WorldBrickMap* w = engine->world;

    PyObject* d = PyDict_New();
    if (w == NULL) {
        PyDict_SetItemString(d, "world_is_null", Py_True);
        return d;
    }
    PyDict_SetItemString(d, "world_is_null", Py_False);
    PyDict_SetItemString(d, "grid_x", PyLong_FromLong(w->grid_x));
    PyDict_SetItemString(d, "grid_y", PyLong_FromLong(w->grid_y));
    PyDict_SetItemString(d, "grid_z", PyLong_FromLong(w->grid_z));
    PyDict_SetItemString(d, "voxel_size", PyFloat_FromDouble(w->voxel_size));
    PyDict_SetItemString(d, "sdf_scale", PyFloat_FromDouble(w->sdf_scale));
    PyDict_SetItemString(d, "atlas_count", PyLong_FromLong(w->atlas_count));
    PyDict_SetItemString(d, "max_bricks", PyLong_FromLong(w->max_bricks));

    /* Count non-empty bricks */
    uint32_t total_bricks = w->grid_x * w->grid_y * w->grid_z;
    uint32_t empty = 0, surface = 0, uniform_in = 0, uniform_out = 0;
    for (uint32_t i = 0; i < total_bricks; i++) {
        int32_t idx = w->brick_indices[i];
        if (idx == BRICK_EMPTY_INDEX) empty++;
        else if (idx == BRICK_UNIFORM_INSIDE) uniform_in++;
        else if (idx == BRICK_UNIFORM_OUTSIDE) uniform_out++;
        else surface++;
    }
    PyDict_SetItemString(d, "total_grid_slots", PyLong_FromLong(total_bricks));
    PyDict_SetItemString(d, "empty_bricks", PyLong_FromLong(empty));
    PyDict_SetItemString(d, "surface_bricks", PyLong_FromLong(surface));
    PyDict_SetItemString(d, "uniform_inside", PyLong_FromLong(uniform_in));
    PyDict_SetItemString(d, "uniform_outside", PyLong_FromLong(uniform_out));

    /* Query SDF at a few test points */
    Vec3 p0 = VEC3(0, 0, 0);
    Vec3 p1 = VEC3(5, 0, 0);
    Vec3 p2 = VEC3(10, 0, 0);
    Vec3 p3 = VEC3(15, 0, 0);
    PyDict_SetItemString(d, "sdf_at_origin", PyFloat_FromDouble(world_sdf_query(w, p0)));
    PyDict_SetItemString(d, "sdf_at_5_0_0", PyFloat_FromDouble(world_sdf_query(w, p1)));
    PyDict_SetItemString(d, "sdf_at_10_0_0", PyFloat_FromDouble(world_sdf_query(w, p2)));
    PyDict_SetItemString(d, "sdf_at_15_0_0", PyFloat_FromDouble(world_sdf_query(w, p3)));

    /* World bounds */
    PyDict_SetItemString(d, "world_min_x", PyFloat_FromDouble(w->world_min.x));
    PyDict_SetItemString(d, "world_min_y", PyFloat_FromDouble(w->world_min.y));
    PyDict_SetItemString(d, "world_min_z", PyFloat_FromDouble(w->world_min.z));
    PyDict_SetItemString(d, "world_max_x", PyFloat_FromDouble(w->world_max.x));
    PyDict_SetItemString(d, "world_max_y", PyFloat_FromDouble(w->world_max.y));
    PyDict_SetItemString(d, "world_max_z", PyFloat_FromDouble(w->world_max.z));

    return d;
}

/* Define MY_METHODS before including env_binding.h */
#define MY_METHODS \
    {"set_agent_state", py_set_agent_state, METH_VARARGS, "Teleport agent"}, \
    {"get_agent_state", py_get_agent_state, METH_VARARGS, "Get agent state"}, \
    {"step_sensors", py_step_sensors, METH_VARARGS, "Step sensors only"}, \
    {"get_obs_dim", py_get_obs_dim, METH_VARARGS, "Get obs dimension"}, \
    {"get_action_dim", py_get_action_dim, METH_VARARGS, "Get action dimension"}, \
    {"set_gpu_enabled", py_set_gpu_enabled, METH_VARARGS, "Enable/disable GPU sensors"}, \
    {"is_gpu_enabled", py_is_gpu_enabled, METH_VARARGS, "Check if GPU sensors enabled"}, \
    {"debug_world", py_debug_world, METH_VARARGS, "Debug world state"}

/* Path relative to PufferLib source */
#include "env_binding.h"
