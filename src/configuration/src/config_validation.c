/**
 * Configuration Validation - Schema and Semantic Validation
 *
 * Validates configuration parameters for:
 * - Physical plausibility (positive mass, reasonable values)
 * - Schema constraints (required fields, valid ranges)
 * - Semantic consistency (spawn_height_min < spawn_height_max)
 */

#include "configuration.h"
#include <string.h>
#include <math.h>

/* ============================================================================
 * Helper Macros
 * ============================================================================ */

#define ADD_ERROR(errors, count, max, fld, msg) \
    do { \
        if ((count) < (max)) { \
            strncpy((errors)[count].field, (fld), CONFIG_NAME_MAX - 1); \
            (errors)[count].field[CONFIG_NAME_MAX - 1] = '\0'; \
            strncpy((errors)[count].message, (msg), CONFIG_ERROR_MSG_MAX - 1); \
            (errors)[count].message[CONFIG_ERROR_MSG_MAX - 1] = '\0'; \
            (errors)[count].line_number = -1; \
            (count)++; \
        } \
    } while(0)

/* ============================================================================
 * Drone Validation
 * ============================================================================ */

int config_validate_drone(const DroneConfig* config,
                          ConfigError* errors, uint32_t max_errors) {
    if (!config || !errors) return 0;

    uint32_t err_count = 0;

    /* Mass must be positive */
    if (config->mass <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.mass", "Mass must be positive");
    }

    /* Arm length must be positive */
    if (config->arm_length <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.arm_length", "Arm length must be positive");
    }

    /* Inertia tensor must be positive */
    if (config->ixx <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.ixx", "Moment of inertia ixx must be positive");
    }
    if (config->iyy <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.iyy", "Moment of inertia iyy must be positive");
    }
    if (config->izz <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.izz", "Moment of inertia izz must be positive");
    }

    /* Thrust coefficient must be positive */
    if (config->k_thrust <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.k_thrust", "Thrust coefficient must be positive");
    }

    /* Motor time constant must be positive */
    if (config->motor_tau <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.motor_tau", "Motor time constant must be positive");
    }

    /* Max RPM must be positive */
    if (config->max_rpm <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.max_rpm", "Max RPM must be positive");
    }

    /* Collision radius must be positive */
    if (config->collision_radius <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.collision_radius", "Collision radius must be positive");
    }

    /* Drag coefficients must be non-negative */
    if (config->k_drag < 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.k_drag", "Drag coefficient must be non-negative");
    }
    if (config->k_drag_angular < 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.k_drag_angular", "Angular drag coefficient must be non-negative");
    }

    /* Velocity limits must be positive */
    if (config->max_velocity <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.max_velocity", "Max velocity must be positive");
    }
    if (config->max_angular_velocity <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.max_angular_velocity", "Max angular velocity must be positive");
    }

    /* Color values should be in [0, 1] */
    for (int i = 0; i < 3; i++) {
        if (config->color[i] < 0.0f || config->color[i] > 1.0f) {
            ADD_ERROR(errors, err_count, max_errors,
                      "drone.color", "Color values must be in range [0, 1]");
            break;
        }
    }

    /* Scale must be positive */
    if (config->scale <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "drone.scale", "Scale must be positive");
    }

    return (int)err_count;
}

/* ============================================================================
 * Environment Validation
 * ============================================================================ */

int config_validate_environment(const EnvironmentConfig* config,
                                ConfigError* errors, uint32_t max_errors) {
    if (!config || !errors) return 0;

    uint32_t err_count = 0;

    /* num_envs must be positive */
    if (config->num_envs == 0) {
        ADD_ERROR(errors, err_count, max_errors,
                  "environment.num_envs", "num_envs must be > 0");
    }

    /* drones_per_env can be zero (empty environments) but typically > 0 */
    /* No validation needed */

    /* World size must be positive */
    if (config->world_size[0] <= 0.0f || config->world_size[1] <= 0.0f ||
        config->world_size[2] <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "environment.world_size", "World size dimensions must be positive");
    }

    /* Voxel size must be positive */
    if (config->voxel_size <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "environment.voxel_size", "Voxel size must be positive");
    }

    /* Spawn radius must be non-negative */
    if (config->spawn_radius < 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "environment.spawn_radius", "Spawn radius must be non-negative");
    }

    /* Spawn height range must be valid */
    if (config->spawn_height_min >= config->spawn_height_max) {
        ADD_ERROR(errors, err_count, max_errors,
                  "environment.spawn_height", "spawn_height_min must be < spawn_height_max");
    }

    /* Min separation must be non-negative */
    if (config->min_separation < 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "environment.min_separation", "Min separation must be non-negative");
    }

    /* max_episode_steps must be positive */
    if (config->max_episode_steps == 0) {
        ADD_ERROR(errors, err_count, max_errors,
                  "environment.max_episode_steps", "max_episode_steps must be > 0");
    }

    /* World type must be recognized */
    if (strcmp(config->world_type, "empty") != 0 &&
        strcmp(config->world_type, "obstacles") != 0 &&
        strcmp(config->world_type, "maze") != 0 &&
        strcmp(config->world_type, "race") != 0 &&
        strcmp(config->world_type, "custom") != 0) {
        ADD_ERROR(errors, err_count, max_errors,
                  "environment.world_type", "Unknown world type (use: empty, obstacles, maze, race, custom)");
    }

    return (int)err_count;
}

/* ============================================================================
 * Physics Validation
 * ============================================================================ */

int config_validate_physics(const ConfigPhysics* config,
                            ConfigError* errors, uint32_t max_errors) {
    if (!config || !errors) return 0;

    uint32_t err_count = 0;

    /* Timestep must be positive and reasonable */
    if (config->timestep <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.timestep", "Timestep must be positive");
    } else if (config->timestep > 0.1f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.timestep", "Timestep must be <= 0.1s for stability");
    }

    /* Substeps must be at least 1 */
    if (config->substeps < 1) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.substeps", "Substeps must be >= 1");
    }

    /* Gravity must be positive (magnitude) */
    if (config->gravity < 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.gravity", "Gravity magnitude must be non-negative");
    }

    /* Check integrator type */
    if (strcmp(config->integrator, "euler") != 0 &&
        strcmp(config->integrator, "rk4") != 0) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.integrator", "Integrator must be 'euler' or 'rk4'");
    }

    /* Velocity clamps must be positive */
    if (config->velocity_clamp <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.velocity_clamp", "Velocity clamp must be positive");
    }
    if (config->angular_velocity_clamp <= 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.angular_velocity_clamp", "Angular velocity clamp must be positive");
    }

    /* Ground effect parameters */
    if (config->enable_ground_effect) {
        if (config->ground_effect_height <= 0.0f) {
            ADD_ERROR(errors, err_count, max_errors,
                      "physics.ground_effect_height", "Ground effect height must be positive");
        }
        if (config->ground_effect_strength < 1.0f) {
            ADD_ERROR(errors, err_count, max_errors,
                      "physics.ground_effect_strength", "Ground effect strength must be >= 1.0");
        }
    }

    /* Variance parameters must be non-negative */
    if (config->dt_variance < 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.dt_variance", "dt_variance must be non-negative");
    }
    if (config->mass_variance < 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.mass_variance", "mass_variance must be non-negative");
    }
    if (config->thrust_variance < 0.0f) {
        ADD_ERROR(errors, err_count, max_errors,
                  "physics.thrust_variance", "thrust_variance must be non-negative");
    }

    return (int)err_count;
}

/* ============================================================================
 * Sensor Validation
 * ============================================================================ */

int config_validate_sensors(const SensorConfigEntry* sensors, uint32_t count,
                            ConfigError* errors, uint32_t max_errors) {
    if (!errors) return 0;
    if (!sensors || count == 0) return 0;

    uint32_t err_count = 0;

    for (uint32_t i = 0; i < count && err_count < max_errors; i++) {
        const SensorConfigEntry* s = &sensors[i];

        /* Check sensor type is recognized */
        if (strcmp(s->type, "imu") != 0 &&
            strcmp(s->type, "tof") != 0 &&
            strcmp(s->type, "lidar_2d") != 0 &&
            strcmp(s->type, "lidar_3d") != 0 &&
            strcmp(s->type, "camera_rgb") != 0 &&
            strcmp(s->type, "camera_depth") != 0 &&
            strcmp(s->type, "camera_segmentation") != 0 &&
            strcmp(s->type, "position") != 0 &&
            strcmp(s->type, "velocity") != 0 &&
            strcmp(s->type, "neighbor") != 0) {
            char msg[CONFIG_ERROR_MSG_MAX];
            snprintf(msg, sizeof(msg), "Unknown sensor type: %s", s->type);
            ADD_ERROR(errors, err_count, max_errors, "sensors.type", msg);
        }

        /* Orientation quaternion should be normalized */
        float qnorm = s->orientation[0] * s->orientation[0] +
                      s->orientation[1] * s->orientation[1] +
                      s->orientation[2] * s->orientation[2] +
                      s->orientation[3] * s->orientation[3];
        if (fabsf(qnorm - 1.0f) > 0.01f) {
            ADD_ERROR(errors, err_count, max_errors,
                      "sensors.orientation", "Quaternion must be normalized (|q| = 1)");
        }

        /* Sample rate must be non-negative */
        if (s->sample_rate < 0.0f) {
            ADD_ERROR(errors, err_count, max_errors,
                      "sensors.sample_rate", "Sample rate must be non-negative");
        }

        /* Type-specific validation */
        if (strcmp(s->type, "tof") == 0 ||
            strcmp(s->type, "lidar_2d") == 0 ||
            strcmp(s->type, "lidar_3d") == 0) {
            if (s->max_range <= 0.0f) {
                ADD_ERROR(errors, err_count, max_errors,
                          "sensors.max_range", "Max range must be positive");
            }
        }

        if (strcmp(s->type, "lidar_2d") == 0 ||
            strcmp(s->type, "lidar_3d") == 0) {
            if (s->num_rays == 0) {
                ADD_ERROR(errors, err_count, max_errors,
                          "sensors.num_rays", "Number of rays must be > 0");
            }
            if (s->fov <= 0.0f) {
                ADD_ERROR(errors, err_count, max_errors,
                          "sensors.fov", "FOV must be positive");
            }
        }

        if (strcmp(s->type, "camera_rgb") == 0 ||
            strcmp(s->type, "camera_depth") == 0 ||
            strcmp(s->type, "camera_segmentation") == 0) {
            if (s->width == 0 || s->height == 0) {
                ADD_ERROR(errors, err_count, max_errors,
                          "sensors.width/height", "Camera dimensions must be > 0");
            }
            if (s->near_clip >= s->far_clip) {
                ADD_ERROR(errors, err_count, max_errors,
                          "sensors.near_clip/far_clip", "near_clip must be < far_clip");
            }
        }

        if (strcmp(s->type, "neighbor") == 0) {
            if (s->k_neighbors == 0) {
                ADD_ERROR(errors, err_count, max_errors,
                          "sensors.k_neighbors", "k_neighbors must be > 0");
            }
        }
    }

    return (int)err_count;
}

/* ============================================================================
 * Complete Configuration Validation
 * ============================================================================ */

int config_validate(const Config* config,
                    ConfigError* errors, uint32_t max_errors) {
    if (!config || !errors) return 0;

    uint32_t total_errors = 0;

    /* Validate each section */
    int drone_errors = config_validate_drone(&config->drone,
                                             errors + total_errors,
                                             max_errors - total_errors);
    total_errors += (uint32_t)drone_errors;

    if (total_errors < max_errors) {
        int physics_errors = config_validate_physics(&config->physics,
                                                     errors + total_errors,
                                                     max_errors - total_errors);
        total_errors += (uint32_t)physics_errors;
    }

    if (total_errors < max_errors) {
        int env_errors = config_validate_environment(&config->environment,
                                                     errors + total_errors,
                                                     max_errors - total_errors);
        total_errors += (uint32_t)env_errors;
    }

    if (total_errors < max_errors) {
        int sensor_errors = config_validate_sensors(config->sensors,
                                                    config->num_sensors,
                                                    errors + total_errors,
                                                    max_errors - total_errors);
        total_errors += (uint32_t)sensor_errors;
    }

    return (int)total_errors;
}
