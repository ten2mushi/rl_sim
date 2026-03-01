/**
 * Configuration Parser - TOML parsing with tomlc99
 *
 * Parses TOML configuration files into Config structures.
 * Uses tomlc99 library for parsing (https://github.com/cktan/tomlc99).
 */

#include "configuration.h"
#include "platform_quadcopter.h"
#include "platform_diff_drive.h"
#include "toml.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ============================================================================
 * Helper Functions for Type-Safe TOML Value Extraction
 * ============================================================================ */

/**
 * Get float value from TOML table with default.
 */
static float get_float(toml_table_t* table, const char* key, float def) {
    if (!table) return def;
    toml_datum_t d = toml_double_in(table, key);
    return d.ok ? (float)d.u.d : def;
}

/**
 * Get unsigned integer value from TOML table with default.
 */
static uint32_t get_uint(toml_table_t* table, const char* key, uint32_t def) {
    if (!table) return def;
    toml_datum_t d = toml_int_in(table, key);
    return d.ok ? (uint32_t)d.u.i : def;
}

/**
 * Get boolean value from TOML table with default.
 */
static bool get_bool(toml_table_t* table, const char* key, bool def) {
    if (!table) return def;
    toml_datum_t d = toml_bool_in(table, key);
    return d.ok ? (bool)d.u.b : def;
}

/**
 * Get string value from TOML table with default.
 * Copies into provided buffer.
 */
static void get_string(toml_table_t* table, const char* key,
                       char* out, size_t out_size, const char* def) {
    /* Handle case where out and def are the same (avoid self-copy) */
    if (out == def && !table) {
        return;  /* Already contains the default */
    }

    if (!table) {
        if (out != def) {
            snprintf(out, out_size, "%s", def);
        }
        return;
    }
    toml_datum_t d = toml_string_in(table, key);
    if (d.ok) {
        snprintf(out, out_size, "%s", d.u.s);
        free(d.u.s);
    } else if (out != def) {
        snprintf(out, out_size, "%s", def);
    }
    /* If key not found and out == def, do nothing (already has default) */
}

/**
 * Get float array from TOML table with default.
 */
static void get_float_array(toml_table_t* table, const char* key,
                            float* out, size_t count, const float* def) {
    /* Handle case where out and def overlap (avoid self-copy) */
    if (!table) {
        if (out != def) {
            memcpy(out, def, count * sizeof(float));
        }
        return;
    }
    toml_array_t* arr = toml_array_in(table, key);
    if (arr && (size_t)toml_array_nelem(arr) >= count) {
        for (size_t i = 0; i < count; i++) {
            toml_datum_t d = toml_double_at(arr, (int)i);
            out[i] = d.ok ? (float)d.u.d : def[i];
        }
    } else if (out != def) {
        memcpy(out, def, count * sizeof(float));
    }
    /* If key not found and out == def, do nothing (already has default) */
}

/* ============================================================================
 * Section Parsers
 * ============================================================================ */

/**
 * Parse [platform] section.
 * Common fields live at [platform] level, quad-specific in [platform.quadcopter].
 */
static void parse_platform(toml_table_t* section, PlatformConfig* platform) {
    if (!section) return;

    get_string(section, "type", platform->type, sizeof(platform->type), platform->type);
    get_string(section, "name", platform->name, sizeof(platform->name), platform->name);
    get_string(section, "model_path", platform->model_path, sizeof(platform->model_path), platform->model_path);

    platform->mass = get_float(section, "mass", platform->mass);
    platform->ixx = get_float(section, "ixx", platform->ixx);
    platform->iyy = get_float(section, "iyy", platform->iyy);
    platform->izz = get_float(section, "izz", platform->izz);
    platform->collision_radius = get_float(section, "collision_radius", platform->collision_radius);
    platform->max_velocity = get_float(section, "max_velocity", platform->max_velocity);
    platform->max_angular_velocity = get_float(section, "max_angular_velocity", platform->max_angular_velocity);
    platform->max_tilt_angle = get_float(section, "max_tilt_angle", platform->max_tilt_angle);
    platform->scale = get_float(section, "scale", platform->scale);

    float def_color[3] = {platform->color[0], platform->color[1], platform->color[2]};
    get_float_array(section, "color", platform->color, 3, def_color);

    /* Parse [platform.quadcopter] sub-table */
    toml_table_t* quad_section = toml_table_in(section, "quadcopter");
    if (quad_section && platform->platform_specific) {
        QuadcopterConfig* quad = (QuadcopterConfig*)platform->platform_specific;
        quad->arm_length = get_float(quad_section, "arm_length", quad->arm_length);
        quad->k_thrust = get_float(quad_section, "k_thrust", quad->k_thrust);
        quad->k_torque = get_float(quad_section, "k_torque", quad->k_torque);
        quad->motor_tau = get_float(quad_section, "motor_tau", quad->motor_tau);
        quad->max_rpm = get_float(quad_section, "max_rpm", quad->max_rpm);
        quad->k_drag = get_float(quad_section, "k_drag", quad->k_drag);
        /* TOML key k_drag_angular maps to QuadcopterConfig.k_ang_damp */
        quad->k_ang_damp = get_float(quad_section, "k_drag_angular", quad->k_ang_damp);
    }

    /* Also parse quad fields at top level for backward compatibility with
     * old [drone] configs that put arm_length etc directly under the section */
    if (platform->platform_specific && strcmp(platform->type, "quadcopter") == 0 && !quad_section) {
        QuadcopterConfig* quad = (QuadcopterConfig*)platform->platform_specific;
        quad->arm_length = get_float(section, "arm_length", quad->arm_length);
        quad->k_thrust = get_float(section, "k_thrust", quad->k_thrust);
        quad->k_torque = get_float(section, "k_torque", quad->k_torque);
        quad->motor_tau = get_float(section, "motor_tau", quad->motor_tau);
        quad->max_rpm = get_float(section, "max_rpm", quad->max_rpm);
        quad->k_drag = get_float(section, "k_drag", quad->k_drag);
        quad->k_ang_damp = get_float(section, "k_drag_angular", quad->k_ang_damp);
    }

    /* Handle diff_drive platform type.
     * If type changed to "diff_drive", swap platform_specific from QuadcopterConfig
     * to DiffDriveConfig and parse [platform.diff_drive] sub-table. */
    if (strcmp(platform->type, "diff_drive") == 0) {
        /* Replace quadcopter config with diff_drive config */
        if (platform->platform_specific) {
            free(platform->platform_specific);
            platform->platform_specific = NULL;
        }

        DiffDriveConfig* dd = (DiffDriveConfig*)malloc(sizeof(DiffDriveConfig));
        if (dd) {
            /* Set defaults */
            memset(dd, 0, sizeof(DiffDriveConfig));
            dd->wheel_radius = 0.033f;
            dd->axle_length = 0.16f;
            dd->max_wheel_vel = 6.67f;

            /* Parse [platform.diff_drive] sub-table */
            toml_table_t* dd_section = toml_table_in(section, "diff_drive");
            if (dd_section) {
                dd->wheel_radius = get_float(dd_section, "wheel_radius", dd->wheel_radius);
                dd->axle_length = get_float(dd_section, "axle_length", dd->axle_length);
                dd->max_wheel_vel = get_float(dd_section, "max_wheel_vel", dd->max_wheel_vel);
            }

            platform->platform_specific = dd;
            platform->platform_specific_size = sizeof(DiffDriveConfig);
        }
    }
}

/**
 * Parse [environment] section.
 */
static void parse_environment(toml_table_t* section, EnvironmentConfig* env) {
    if (!section) return;

    env->num_envs = get_uint(section, "num_envs", env->num_envs);
    env->agents_per_env = get_uint(section, "agents_per_env", env->agents_per_env);

    float def_size[3] = {env->world_size[0], env->world_size[1], env->world_size[2]};
    get_float_array(section, "world_size", env->world_size, 3, def_size);

    float def_origin[3] = {env->world_origin[0], env->world_origin[1], env->world_origin[2]};
    get_float_array(section, "world_origin", env->world_origin, 3, def_origin);

    env->voxel_size = get_float(section, "voxel_size", env->voxel_size);
    env->max_bricks = get_uint(section, "max_bricks", env->max_bricks);
    env->spawn_radius = get_float(section, "spawn_radius", env->spawn_radius);
    env->spawn_height_min = get_float(section, "spawn_height_min", env->spawn_height_min);
    env->spawn_height_max = get_float(section, "spawn_height_max", env->spawn_height_max);
    env->min_separation = get_float(section, "min_separation", env->min_separation);
    env->max_episode_steps = get_uint(section, "max_episode_steps", env->max_episode_steps);
    env->auto_reset = get_bool(section, "auto_reset", env->auto_reset);
    get_string(section, "world_type", env->world_type, sizeof(env->world_type), env->world_type);
    env->num_obstacles = get_uint(section, "num_obstacles", env->num_obstacles);
    env->seed = get_uint(section, "seed", env->seed);
}

/**
 * Parse [physics] section.
 */
static void parse_physics(toml_table_t* section, ConfigPhysics* physics) {
    if (!section) return;

    physics->timestep = get_float(section, "timestep", physics->timestep);
    physics->substeps = get_uint(section, "substeps", physics->substeps);
    physics->gravity = get_float(section, "gravity", physics->gravity);
    get_string(section, "integrator", physics->integrator, sizeof(physics->integrator), physics->integrator);
    physics->velocity_clamp = get_float(section, "velocity_clamp", physics->velocity_clamp);
    physics->angular_velocity_clamp = get_float(section, "angular_velocity_clamp", physics->angular_velocity_clamp);
    physics->normalize_quaternions = get_bool(section, "normalize_quaternions", physics->normalize_quaternions);
    physics->enable_ground_effect = get_bool(section, "enable_ground_effect", physics->enable_ground_effect);
    physics->ground_effect_height = get_float(section, "ground_effect_height", physics->ground_effect_height);
    physics->ground_effect_strength = get_float(section, "ground_effect_strength", physics->ground_effect_strength);
    physics->dt_variance = get_float(section, "dt_variance", physics->dt_variance);
    physics->mass_variance = get_float(section, "mass_variance", physics->mass_variance);
    physics->thrust_variance = get_float(section, "thrust_variance", physics->thrust_variance);
}

/**
 * Parse [reward] section.
 */
static void parse_reward(toml_table_t* section, RewardConfigData* reward) {
    if (!section) return;

    get_string(section, "task", reward->task, sizeof(reward->task), reward->task);
    reward->distance_scale = get_float(section, "distance_scale", reward->distance_scale);
    reward->distance_exp = get_float(section, "distance_exp", reward->distance_exp);
    reward->reach_bonus = get_float(section, "reach_bonus", reward->reach_bonus);
    reward->reach_radius = get_float(section, "reach_radius", reward->reach_radius);
    reward->velocity_match_scale = get_float(section, "velocity_match_scale", reward->velocity_match_scale);
    reward->uprightness_scale = get_float(section, "uprightness_scale", reward->uprightness_scale);
    reward->energy_scale = get_float(section, "energy_scale", reward->energy_scale);
    reward->jerk_scale = get_float(section, "jerk_scale", reward->jerk_scale);
    reward->collision_penalty = get_float(section, "collision_penalty", reward->collision_penalty);
    reward->world_collision_penalty = get_float(section, "world_collision_penalty", reward->world_collision_penalty);
    reward->drone_collision_penalty = get_float(section, "drone_collision_penalty", reward->drone_collision_penalty);
    reward->alive_bonus = get_float(section, "alive_bonus", reward->alive_bonus);
    reward->success_bonus = get_float(section, "success_bonus", reward->success_bonus);
    reward->reward_min = get_float(section, "reward_min", reward->reward_min);
    reward->reward_max = get_float(section, "reward_max", reward->reward_max);
}

/**
 * Parse [training] section.
 */
static void parse_training(toml_table_t* section, TrainingConfig* training) {
    if (!section) return;

    get_string(section, "algorithm", training->algorithm, sizeof(training->algorithm), training->algorithm);
    training->learning_rate = get_float(section, "learning_rate", training->learning_rate);
    training->gamma = get_float(section, "gamma", training->gamma);
    training->gae_lambda = get_float(section, "gae_lambda", training->gae_lambda);
    training->clip_range = get_float(section, "clip_range", training->clip_range);
    training->entropy_coef = get_float(section, "entropy_coef", training->entropy_coef);
    training->value_coef = get_float(section, "value_coef", training->value_coef);
    training->max_grad_norm = get_float(section, "max_grad_norm", training->max_grad_norm);
    training->batch_size = get_uint(section, "batch_size", training->batch_size);
    training->num_epochs = get_uint(section, "num_epochs", training->num_epochs);
    training->rollout_length = get_uint(section, "rollout_length", training->rollout_length);
    training->log_interval = get_uint(section, "log_interval", training->log_interval);
    training->save_interval = get_uint(section, "save_interval", training->save_interval);
    get_string(section, "checkpoint_dir", training->checkpoint_dir, sizeof(training->checkpoint_dir), training->checkpoint_dir);
}

/**
 * Parse single [[sensors]] entry.
 */
static void parse_sensor(toml_table_t* sensor, SensorConfigEntry* entry) {
    if (!sensor) return;

    /* Get type first to set type-specific defaults */
    char type[32] = "imu";
    get_string(sensor, "type", type, sizeof(type), "imu");

    /* Initialize with defaults for this type */
    *entry = sensor_config_entry_default(type);

    /* Override with TOML values */
    get_string(sensor, "type", entry->type, sizeof(entry->type), entry->type);
    get_string(sensor, "name", entry->name, sizeof(entry->name), entry->name);

    float def_pos[3] = {entry->position[0], entry->position[1], entry->position[2]};
    get_float_array(sensor, "position", entry->position, 3, def_pos);

    float def_quat[4] = {entry->orientation[0], entry->orientation[1],
                         entry->orientation[2], entry->orientation[3]};
    get_float_array(sensor, "orientation", entry->orientation, 4, def_quat);

    entry->sample_rate = get_float(sensor, "sample_rate", entry->sample_rate);

    /* Parse noise groups */
    toml_array_t* noise_groups = toml_array_in(sensor, "noise_groups");
    if (noise_groups) {
        int ng = toml_array_nelem(noise_groups);
        if (ng > CONFIG_MAX_NOISE_GROUPS) ng = CONFIG_MAX_NOISE_GROUPS;
        entry->num_noise_groups = (uint32_t)ng;
        for (int g = 0; g < ng; g++) {
            toml_table_t* gtab = toml_table_at(noise_groups, g);
            if (!gtab) continue;
            NoiseGroupEntry* ge = &entry->noise_groups[g];
            memset(ge, 0, sizeof(NoiseGroupEntry));

            /* Parse channels = [start, count] */
            toml_array_t* ch = toml_array_in(gtab, "channels");
            if (ch && toml_array_nelem(ch) == 2) {
                toml_datum_t d0 = toml_int_at(ch, 0);
                toml_datum_t d1 = toml_int_at(ch, 1);
                if (d0.ok) ge->channels[0] = (uint32_t)d0.u.i;
                if (d1.ok) ge->channels[1] = (uint32_t)d1.u.i;
            }

            /* Parse [[sensors.noise_groups.stages]] */
            toml_array_t* stages = toml_array_in(gtab, "stages");
            if (!stages) continue;
            int ns = toml_array_nelem(stages);
            if (ns > CONFIG_MAX_NOISE_STAGES) ns = CONFIG_MAX_NOISE_STAGES;
            ge->num_stages = (uint32_t)ns;
            for (int s = 0; s < ns; s++) {
                toml_table_t* stab = toml_table_at(stages, s);
                if (!stab) continue;
                NoiseStageEntry* se = &ge->stages[s];
                memset(se, 0, sizeof(NoiseStageEntry));
                get_string(stab, "type", se->type, sizeof(se->type), "none");
                se->stddev = get_float(stab, "stddev", 0.0f);
                se->tau = get_float(stab, "tau", 0.0f);
                se->sigma = get_float(stab, "sigma", 0.0f);
                se->error = get_float(stab, "error", 0.0f);
                se->coeff = get_float(stab, "coeff", 0.0f);
                se->power = get_float(stab, "power", 1.0f);
                se->step = get_float(stab, "step", 0.0f);
                se->probability = get_float(stab, "probability", 0.0f);
                se->replacement = get_float(stab, "replacement", 0.0f);
                se->min_val = get_float(stab, "min_val", 0.0f);
                se->max_val = get_float(stab, "max_val", 0.0f);

                /* Parse values array for constant_bias */
                toml_array_t* vals = toml_array_in(stab, "values");
                if (vals) {
                    int nv = toml_array_nelem(vals);
                    if (nv > CONFIG_MAX_NOISE_BIAS_VALUES) nv = CONFIG_MAX_NOISE_BIAS_VALUES;
                    se->value_count = (uint32_t)nv;
                    for (int v = 0; v < nv; v++) {
                        toml_datum_t dv = toml_double_at(vals, v);
                        if (dv.ok) se->values[v] = (float)dv.u.d;
                    }
                }
            }
        }
    }

    /* Type-specific fields */
    entry->max_range = get_float(sensor, "max_range", entry->max_range);
    entry->num_rays = get_uint(sensor, "num_rays", entry->num_rays);
    entry->fov = get_float(sensor, "fov", entry->fov);
    entry->fov_vertical = get_float(sensor, "fov_vertical", entry->fov_vertical);
    entry->vertical_layers = get_uint(sensor, "vertical_layers", entry->vertical_layers);
    entry->width = get_uint(sensor, "width", entry->width);
    entry->height = get_uint(sensor, "height", entry->height);
    entry->near_clip = get_float(sensor, "near_clip", entry->near_clip);
    entry->far_clip = get_float(sensor, "far_clip", entry->far_clip);
    entry->num_classes = get_uint(sensor, "num_classes", entry->num_classes);
    entry->k_neighbors = get_uint(sensor, "k_neighbors", entry->k_neighbors);
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

int config_load_string(const char* toml_str, Config* config, char* error_msg) {
    if (!toml_str || !config || !error_msg) {
        if (error_msg) {
            snprintf(error_msg, CONFIG_ERROR_MSG_MAX, "NULL parameter");
        }
        return -1;
    }

    /* Set defaults first */
    config_set_defaults(config);

    /* Handle empty string - just use defaults */
    if (toml_str[0] == '\0') {
        config->config_hash = config_hash(config);
        return 0;
    }

    /* Parse TOML string */
    char errbuf[256] = {0};
    toml_table_t* root = toml_parse((char*)toml_str, errbuf, sizeof(errbuf));

    if (!root) {
        snprintf(error_msg, CONFIG_ERROR_MSG_MAX, "TOML parse error: %s", errbuf);
        return -2;
    }

    /* Parse each section */
    /* Support both [platform] (new) and [drone] (legacy) */
    toml_table_t* platform = toml_table_in(root, "platform");
    if (platform) {
        parse_platform(platform, &config->platform);
    } else {
        /* Legacy: try [drone] section */
        toml_table_t* drone = toml_table_in(root, "drone");
        if (drone) parse_platform(drone, &config->platform);
    }

    toml_table_t* env = toml_table_in(root, "environment");
    if (env) parse_environment(env, &config->environment);

    toml_table_t* physics = toml_table_in(root, "physics");
    if (physics) parse_physics(physics, &config->physics);

    toml_table_t* reward = toml_table_in(root, "reward");
    if (reward) parse_reward(reward, &config->reward);

    toml_table_t* training = toml_table_in(root, "training");
    if (training) parse_training(training, &config->training);

    /* Parse [[sensors]] array */
    toml_array_t* sensors = toml_array_in(root, "sensors");
    if (sensors) {
        int count = toml_array_nelem(sensors);
        if (count > 0) {
            if ((uint32_t)count > CONFIG_MAX_SENSORS) {
                count = CONFIG_MAX_SENSORS;
            }
            config->sensors = (SensorConfigEntry*)malloc((size_t)count * sizeof(SensorConfigEntry));
            if (!config->sensors) {
                toml_free(root);
                snprintf(error_msg, CONFIG_ERROR_MSG_MAX, "Failed to allocate sensors array");
                return -3;
            }
            config->num_sensors = (uint32_t)count;
            for (int i = 0; i < count; i++) {
                toml_table_t* sensor = toml_table_at(sensors, i);
                parse_sensor(sensor, &config->sensors[i]);
            }
        }
    }

    toml_free(root);

    /* Validate */
    ConfigError errors[CONFIG_MAX_ERRORS];
    int num_errors = config_validate(config, errors, CONFIG_MAX_ERRORS);
    if (num_errors > 0) {
        snprintf(error_msg, CONFIG_ERROR_MSG_MAX, "Validation error: %s - %s",
                 errors[0].field, errors[0].message);
        config_free(config);
        config_set_defaults(config);
        return -4;
    }

    config->config_hash = config_hash(config);
    error_msg[0] = '\0';
    return 0;
}

int config_load(const char* path, Config* config, char* error_msg) {
    if (!path || !config || !error_msg) {
        if (error_msg) {
            snprintf(error_msg, CONFIG_ERROR_MSG_MAX, "NULL parameter");
        }
        return -1;
    }

    /* Open file */
    FILE* fp = fopen(path, "r");
    if (!fp) {
        snprintf(error_msg, CONFIG_ERROR_MSG_MAX, "Cannot open file: %s", path);
        return -1;
    }

    /* Get file size */
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    if (fsize < 0 || fsize > 1024 * 1024) {  /* Max 1MB config file */
        fclose(fp);
        snprintf(error_msg, CONFIG_ERROR_MSG_MAX, "Invalid file size: %ld", fsize);
        return -1;
    }

    /* Read file content */
    char* content = (char*)malloc((size_t)fsize + 1);
    if (!content) {
        fclose(fp);
        snprintf(error_msg, CONFIG_ERROR_MSG_MAX, "Failed to allocate memory for file");
        return -1;
    }

    size_t bytes_read = fread(content, 1, (size_t)fsize, fp);
    fclose(fp);
    content[bytes_read] = '\0';

    /* Parse content (config_load_string calls config_set_defaults internally) */
    int result = config_load_string(content, config, error_msg);

    /* Restore path after parsing (config_load_string clears it via set_defaults) */
    strncpy(config->config_path, path, CONFIG_PATH_MAX - 1);
    config->config_path[CONFIG_PATH_MAX - 1] = '\0';

    free(content);
    return result;
}

void config_free(Config* config) {
    if (!config) return;

    if (config->sensors) {
        free(config->sensors);
        config->sensors = NULL;
    }
    config->num_sensors = 0;

    /* Free platform_specific */
    if (config->platform.platform_specific) {
        free(config->platform.platform_specific);
        config->platform.platform_specific = NULL;
    }
    config->platform.platform_specific_size = 0;
}
