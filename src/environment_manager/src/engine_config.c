/**
 * Engine Configuration Implementation
 *
 * Provides default configuration values, validation, and loading from files.
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* noise.h needed for noise_type_from_string in convert_noise_config */
#include "noise.h"

#define LOCAL_MAX_SENSORS 64

/* ============================================================================
 * Default Configuration
 * ============================================================================ */

EngineConfig engine_config_default(void) {
    EngineConfig config = {0};

    /* Environment dimensions */
    config.num_envs = ENGINE_DEFAULT_NUM_ENVS;
    config.drones_per_env = ENGINE_DEFAULT_DRONES_PER_ENV;
    config.total_drones = config.num_envs * config.drones_per_env;

    /* World bounds - 100m × 100m × 60m volume (Z-up ENU) */
    config.world_min = (Vec3){-50.0f, -50.0f, -10.0f, 0.0f};
    config.world_max = (Vec3){50.0f, 50.0f, 50.0f, 0.0f};
    config.voxel_size = ENGINE_DEFAULT_VOXEL_SIZE;
    config.max_bricks = ENGINE_DEFAULT_MAX_BRICKS;

    /* Physics */
    config.timestep = ENGINE_DEFAULT_TIMESTEP;
    config.physics_substeps = ENGINE_DEFAULT_PHYSICS_SUBSTEPS;
    config.gravity = ENGINE_DEFAULT_GRAVITY;

    /* Sensors - none by default */
    config.sensor_configs = NULL;
    config.num_sensor_configs = 0;

    /* Rewards - hover task default */
    config.reward_config = reward_config_default(TASK_HOVER);

    /* Episode limits */
    config.max_episode_steps = ENGINE_DEFAULT_MAX_EPISODE_STEPS;

    /* Threading - auto-detect */
    config.num_threads = 0;

    /* Memory */
    config.persistent_arena_size = (size_t)ENGINE_DEFAULT_PERSISTENT_ARENA_MB * 1024 * 1024;
    config.frame_arena_size = (size_t)ENGINE_DEFAULT_FRAME_ARENA_MB * 1024 * 1024;

    /* Random seed */
    config.seed = ENGINE_DEFAULT_SEED;

    /* Domain randomization */
    config.domain_randomization = 0.0f;

    /* Termination bounds - default to world bounds */
    config.termination_min = config.world_min;
    config.termination_max = config.world_max;
    config.use_custom_termination = false;

    /* Spawn region - default to world bounds (margin applied at runtime) */
    config.spawn_min = (Vec3){0.0f, 0.0f, 0.0f, 0.0f};
    config.spawn_max = (Vec3){0.0f, 0.0f, 0.0f, 0.0f};
    config.use_custom_spawn = false;

    /* OBJ file path */
    config.obj_path = NULL;
    config.use_gpu_voxelization = true;

    /* Physics tunables */
    config.air_density = 1.225f;
    config.enable_drag = true;
    config.enable_ground_effect = true;
    config.enable_motor_dynamics = true;
    config.enable_gyroscopic = false;
    config.ground_effect_height = 0.5f;
    config.ground_effect_coeff = 1.15f;
    config.max_linear_accel = 100.0f;
    config.max_angular_accel = 200.0f;

    /* Collision tunables */
    config.drone_radius = 0.1f;
    config.collision_cell_size = 1.0f;

    /* Debug options */
    config.enable_profiling = false;
    config.verbose_logging = false;

    return config;
}

/* ============================================================================
 * Configuration Validation
 * ============================================================================ */

int engine_config_validate(const EngineConfig* config, char* error_msg) {
    if (config == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE, "config is NULL");
        return -1;
    }

    /* Validate environment dimensions */
    if (config->num_envs == 0) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "num_envs must be > 0 (got 0)");
        return -2;
    }

    if (config->drones_per_env == 0) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "drones_per_env must be > 0 (got 0)");
        return -3;
    }

    /* Check for overflow */
    uint64_t total = (uint64_t)config->num_envs * (uint64_t)config->drones_per_env;
    if (total > UINT32_MAX) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "total_drones exceeds uint32 max (%u * %u)",
                               config->num_envs, config->drones_per_env);
        return -4;
    }

    /* Validate world bounds */
    if (config->world_min.x >= config->world_max.x ||
        config->world_min.y >= config->world_max.y ||
        config->world_min.z >= config->world_max.z) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "world_min must be < world_max in all dimensions");
        return -5;
    }

    /* Validate voxel size */
    if (config->voxel_size <= 0.0f) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "voxel_size must be > 0 (got %.4f)", config->voxel_size);
        return -6;
    }

    /* Validate physics parameters */
    if (config->timestep <= 0.0f) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "timestep must be > 0 (got %.6f)", config->timestep);
        return -7;
    }

    if (config->physics_substeps == 0) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "physics_substeps must be > 0 (got 0)");
        return -8;
    }

    if (config->gravity < 0.0f) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "gravity must be >= 0 (got %.4f)", config->gravity);
        return -9;
    }

    /* Validate episode limits */
    if (config->max_episode_steps == 0) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "max_episode_steps must be > 0 (got 0)");
        return -10;
    }

    /* Validate arena sizes */
    size_t min_persistent = 64 * 1024 * 1024; /* 64 MB minimum */
    if (config->persistent_arena_size < min_persistent) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "persistent_arena_size must be >= 64 MB (got %zu MB)",
                               config->persistent_arena_size / (1024 * 1024));
        return -11;
    }

    size_t min_frame = 16 * 1024 * 1024; /* 16 MB minimum */
    if (config->frame_arena_size < min_frame) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "frame_arena_size must be >= 16 MB (got %zu MB)",
                               config->frame_arena_size / (1024 * 1024));
        return -12;
    }

    /* Validate thread count */
    if (config->num_threads > THREADING_MAX_THREADS) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "num_threads exceeds maximum (%u > %u)",
                               config->num_threads, THREADING_MAX_THREADS);
        return -13;
    }

    /* Validate domain randomization */
    if (config->domain_randomization < 0.0f || config->domain_randomization > 1.0f) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "domain_randomization must be in [0, 1] (got %.4f)",
                               config->domain_randomization);
        return -14;
    }

    /* Validate sensor configs (if provided) */
    if (config->num_sensor_configs > 0 && config->sensor_configs == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "sensor_configs is NULL but num_sensor_configs = %u",
                               config->num_sensor_configs);
        return -15;
    }

    /* Validate custom spawn region */
    if (config->use_custom_spawn) {
        if (config->spawn_min.x >= config->spawn_max.x ||
            config->spawn_min.y >= config->spawn_max.y ||
            config->spawn_min.z >= config->spawn_max.z) {
            if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                                   "spawn_min must be < spawn_max in all dimensions");
            return -16;
        }
    }

    /* Validate custom termination bounds */
    if (config->use_custom_termination) {
        if (config->termination_min.x >= config->termination_max.x ||
            config->termination_min.y >= config->termination_max.y ||
            config->termination_min.z >= config->termination_max.z) {
            if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                                   "termination_min must be < termination_max in all dimensions");
            return -17;
        }
    }

    /* Validate collision tunables */
    if (config->drone_radius <= 0.0f) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "drone_radius must be > 0 (got %.4f)", config->drone_radius);
        return -18;
    }

    if (config->collision_cell_size <= 0.0f) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "collision_cell_size must be > 0 (got %.4f)",
                               config->collision_cell_size);
        return -19;
    }

    /* All checks passed */
    if (error_msg) error_msg[0] = '\0';
    return 0;
}

/* ============================================================================
 * Configuration Loading
 * ============================================================================ */

/**
 * Convert NoiseGroupEntry (TOML-facing) → NoiseConfig (runtime).
 */
static NoiseConfig convert_noise_config(const SensorConfigEntry* entry) {
    NoiseConfig nc;
    memset(&nc, 0, sizeof(nc));
    nc.group_count = entry->num_noise_groups;
    if (nc.group_count > MAX_NOISE_GROUPS) nc.group_count = MAX_NOISE_GROUPS;

    for (uint32_t g = 0; g < nc.group_count; g++) {
        NoisePipeline* pipe = &nc.groups[g];
        const NoiseGroupEntry* ge = &entry->noise_groups[g];
        pipe->channel_start = ge->channels[0];
        pipe->channel_count = ge->channels[1];
        pipe->stage_count = ge->num_stages;
        if (pipe->stage_count > MAX_NOISE_STAGES) pipe->stage_count = MAX_NOISE_STAGES;

        for (uint32_t s = 0; s < pipe->stage_count; s++) {
            const NoiseStageEntry* se = &ge->stages[s];
            NoiseStage* stage = &pipe->stages[s];
            stage->type = noise_type_from_string(se->type);
            memset(&stage->params, 0, sizeof(NoiseParams));

            switch (stage->type) {
                case NOISE_WHITE_GAUSSIAN:
                    stage->params.white.stddev = se->stddev;
                    break;
                case NOISE_CONSTANT_BIAS:
                    stage->params.bias.count = se->value_count;
                    if (stage->params.bias.count > MAX_NOISE_BIAS_VALUES)
                        stage->params.bias.count = MAX_NOISE_BIAS_VALUES;
                    for (uint32_t v = 0; v < stage->params.bias.count; v++)
                        stage->params.bias.values[v] = se->values[v];
                    break;
                case NOISE_BIAS_DRIFT:
                    stage->params.drift.tau = se->tau;
                    stage->params.drift.sigma = se->sigma;
                    break;
                case NOISE_SCALE_FACTOR:
                    stage->params.scale.error = se->error;
                    break;
                case NOISE_DISTANCE_DEPENDENT:
                    stage->params.distance.coeff = se->coeff;
                    stage->params.distance.power = se->power;
                    break;
                case NOISE_QUANTIZATION:
                    stage->params.quantize.step = se->step;
                    break;
                case NOISE_DROPOUT:
                    stage->params.dropout.probability = se->probability;
                    stage->params.dropout.replacement = se->replacement;
                    break;
                case NOISE_SATURATION:
                    stage->params.saturate.min_val = se->min_val;
                    stage->params.saturate.max_val = se->max_val;
                    break;
                default:
                    stage->type = NOISE_NONE;
                    break;
            }
        }
    }
    return nc;
}

/**
 * Convert SensorConfigEntry (TOML-facing) → SensorConfig (runtime).
 */
static SensorConfig sensor_config_from_entry(const SensorConfigEntry* entry) {
    SensorConfig sc;
    memset(&sc, 0, sizeof(sc));

    /* Map type string to enum */
    if (strcmp(entry->type, "imu") == 0) sc.type = SENSOR_TYPE_IMU;
    else if (strcmp(entry->type, "tof") == 0) sc.type = SENSOR_TYPE_TOF;
    else if (strcmp(entry->type, "lidar_2d") == 0) sc.type = SENSOR_TYPE_LIDAR_2D;
    else if (strcmp(entry->type, "lidar_3d") == 0) sc.type = SENSOR_TYPE_LIDAR_3D;
    else if (strcmp(entry->type, "camera_rgb") == 0) sc.type = SENSOR_TYPE_CAMERA_RGB;
    else if (strcmp(entry->type, "camera_depth") == 0) sc.type = SENSOR_TYPE_CAMERA_DEPTH;
    else if (strcmp(entry->type, "camera_segmentation") == 0) sc.type = SENSOR_TYPE_CAMERA_SEGMENTATION;
    else if (strcmp(entry->type, "position") == 0) sc.type = SENSOR_TYPE_POSITION;
    else if (strcmp(entry->type, "velocity") == 0) sc.type = SENSOR_TYPE_VELOCITY;
    else if (strcmp(entry->type, "neighbor") == 0) sc.type = SENSOR_TYPE_NEIGHBOR;
    else sc.type = SENSOR_TYPE_IMU; /* fallback */

    /* Common fields */
    sc.position_offset = (Vec3){entry->position[0], entry->position[1], entry->position[2], 0.0f};
    sc.orientation_offset = (Quat){entry->orientation[0], entry->orientation[1],
                                    entry->orientation[2], entry->orientation[3]};
    sc.sample_rate = entry->sample_rate;

    /* Convert noise config */
    sc.noise_config = convert_noise_config(entry);

    /* Type-specific fields */
    switch (sc.type) {
        case SENSOR_TYPE_IMU:
            break;
        case SENSOR_TYPE_TOF:
            sc.tof.direction = (Vec3){0.0f, 0.0f, -1.0f, 0.0f}; /* Default: down */
            sc.tof.max_range = entry->max_range;
            break;
        case SENSOR_TYPE_LIDAR_2D:
            sc.lidar_2d.num_rays = entry->num_rays;
            sc.lidar_2d.fov = entry->fov;
            sc.lidar_2d.max_range = entry->max_range;
            break;
        case SENSOR_TYPE_LIDAR_3D:
            sc.lidar_3d.horizontal_rays = entry->num_rays;
            sc.lidar_3d.vertical_layers = entry->vertical_layers;
            sc.lidar_3d.horizontal_fov = entry->fov;
            sc.lidar_3d.vertical_fov = entry->fov_vertical;
            sc.lidar_3d.max_range = entry->max_range;
            break;
        case SENSOR_TYPE_CAMERA_RGB:
        case SENSOR_TYPE_CAMERA_DEPTH:
        case SENSOR_TYPE_CAMERA_SEGMENTATION:
            sc.camera.width = entry->width;
            sc.camera.height = entry->height;
            sc.camera.fov_horizontal = entry->fov;
            sc.camera.fov_vertical = entry->fov_vertical;
            sc.camera.near_clip = entry->near_clip;
            sc.camera.far_clip = entry->far_clip;
            sc.camera.num_classes = entry->num_classes;
            break;
        case SENSOR_TYPE_NEIGHBOR:
            sc.neighbor.k = entry->k_neighbors;
            sc.neighbor.max_range = entry->max_range;
            break;
        default:
            break;
    }

    return sc;
}

/**
 * Convert Config (TOML) → EngineConfig (runtime).
 */
static int config_to_engine_config(const Config* cfg, EngineConfig* ec) {
    ec->num_envs = cfg->environment.num_envs;
    ec->drones_per_env = cfg->environment.drones_per_env;
    ec->total_drones = ec->num_envs * ec->drones_per_env;
    ec->timestep = cfg->physics.timestep;
    ec->physics_substeps = cfg->physics.substeps;
    ec->gravity = cfg->physics.gravity;
    ec->max_episode_steps = cfg->environment.max_episode_steps;
    ec->seed = (uint64_t)cfg->environment.seed;
    ec->voxel_size = cfg->environment.voxel_size;
    ec->max_bricks = cfg->environment.max_bricks;

    /* World bounds from environment config */
    float half_w = cfg->environment.world_size[0] * 0.5f;
    float half_d = cfg->environment.world_size[1] * 0.5f;
    float half_h = cfg->environment.world_size[2] * 0.5f;
    ec->world_min = (Vec3){
        cfg->environment.world_origin[0] - half_w,
        cfg->environment.world_origin[1] - half_d,
        cfg->environment.world_origin[2] - half_h, 0.0f};
    ec->world_max = (Vec3){
        cfg->environment.world_origin[0] + half_w,
        cfg->environment.world_origin[1] + half_d,
        cfg->environment.world_origin[2] + half_h, 0.0f};

    /* Physics tunables from ConfigPhysics */
    ec->enable_ground_effect = cfg->physics.enable_ground_effect;
    ec->ground_effect_height = cfg->physics.ground_effect_height;
    ec->ground_effect_coeff = cfg->physics.ground_effect_strength;

    /* Convert sensors */
    for (uint32_t i = 0; i < cfg->num_sensors; i++) {
        SensorConfig sc = sensor_config_from_entry(&cfg->sensors[i]);
        engine_config_add_sensor(ec, &sc);
    }

    return 0;
}

int engine_config_load(const char* path, EngineConfig* config, char* error_msg) {
    if (path == NULL || config == NULL) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "path or config is NULL");
        return -1;
    }

    /* Start with defaults */
    *config = engine_config_default();

    /* Parse TOML file via configuration module */
    Config cfg;
    int result = config_load(path, &cfg, error_msg);
    if (result != 0) {
        return result;
    }

    /* Convert Config → EngineConfig */
    result = config_to_engine_config(&cfg, config);
    config_free(&cfg);

    if (result != 0) {
        if (error_msg) snprintf(error_msg, ENGINE_ERROR_MSG_SIZE,
                               "Failed to convert config");
        return result;
    }

    /* Validate the resulting config */
    return engine_config_validate(config, error_msg);
}

/* ============================================================================
 * Sensor Configuration Management
 * ============================================================================ */

int engine_config_add_sensor(EngineConfig* config, const SensorConfig* sensor_config) {
    if (config == NULL || sensor_config == NULL) {
        return -1;
    }

    /* Check limit */
    if (config->num_sensor_configs >= LOCAL_MAX_SENSORS) {
        return -1;
    }

    /* Allocate or reallocate sensor array */
    if (config->sensor_configs == NULL) {
        config->sensor_configs = (SensorConfig*)malloc(LOCAL_MAX_SENSORS * sizeof(SensorConfig));
        if (config->sensor_configs == NULL) {
            return -1;
        }
    }

    /* Add the sensor config */
    config->sensor_configs[config->num_sensor_configs] = *sensor_config;
    config->num_sensor_configs++;

    return 0;
}
