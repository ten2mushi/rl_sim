/**
 * Configuration Defaults - Crazyflie 2.0 and sensible default values
 *
 * Default values are based on the well-documented Crazyflie 2.0 quadcopter.
 * Reference: https://www.bitcraze.io/products/crazyflie-2-1/
 */

#include "configuration.h"
#include "platform_quadcopter.h"
#include "platform_diff_drive.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

/* ============================================================================
 * Platform Configuration Defaults (Crazyflie 2.0)
 * ============================================================================ */

void platform_config_set_defaults(PlatformConfig* config) {
    /* Zero entire struct first to ensure consistent padding */
    memset(config, 0, sizeof(PlatformConfig));

    /* Platform type */
    snprintf(config->type, CONFIG_NAME_MAX, "%s", "quadcopter");

    /* Crazyflie 2.0 defaults (well-documented reference drone) */
    snprintf(config->name, CONFIG_NAME_MAX, "%s", "crazyflie2");

    /* Physical properties */
    config->mass = 0.027f;           /* 27 grams */

    /* Inertia tensor (diagonal) - from Crazyflie documentation */
    config->ixx = 1.4e-5f;           /* kg*m^2 */
    config->iyy = 1.4e-5f;           /* kg*m^2 */
    config->izz = 2.17e-5f;          /* kg*m^2 */

    /* Geometry */
    config->collision_radius = 0.056f; /* arm_length + 0.01 safety margin */

    /* Limits */
    config->max_velocity = 10.0f;    /* m/s */
    config->max_angular_velocity = 20.0f; /* rad/s */
    config->max_tilt_angle = 1.0f;   /* ~57 degrees */

    /* Visual */
    config->color[0] = 0.2f;         /* R */
    config->color[1] = 0.6f;         /* G */
    config->color[2] = 1.0f;         /* B - Crazyflie blue */
    config->scale = 1.0f;

    /* Allocate and set quadcopter-specific defaults */
    QuadcopterConfig* quad = (QuadcopterConfig*)malloc(sizeof(QuadcopterConfig));
    if (quad) {
        memset(quad, 0, sizeof(QuadcopterConfig));
        quad->arm_length = 0.046f;       /* 46mm arm length */
        quad->k_thrust = 2.88e-8f;       /* N/(rad/s)^2 */
        quad->k_torque = 7.24e-10f;      /* N*m/(rad/s)^2 */
        quad->motor_tau = 0.02f;         /* 20ms motor response time */
        quad->max_rpm = 21702.0f;        /* Maximum motor RPM */
        quad->k_drag = 0.0f;             /* Linear drag coefficient */
        quad->k_ang_damp = 0.0f;         /* Angular drag coefficient */
        config->platform_specific = quad;
        config->platform_specific_size = sizeof(QuadcopterConfig);
    }
}

/* ============================================================================
 * Environment Configuration Defaults
 * ============================================================================ */

void environment_config_set_defaults(EnvironmentConfig* config) {
    /* Zero entire struct first to ensure consistent padding */
    memset(config, 0, sizeof(EnvironmentConfig));

    /* Dimensions */
    config->num_envs = 64;
    config->agents_per_env = 16;

    /* World bounds */
    config->world_size[0] = 20.0f;   /* Width (m) */
    config->world_size[1] = 20.0f;   /* Depth (m) */
    config->world_size[2] = 10.0f;   /* Height (m) */
    config->world_origin[0] = 0.0f;
    config->world_origin[1] = 0.0f;
    config->world_origin[2] = 5.0f;  /* Center height */

    /* Voxel settings */
    config->voxel_size = 0.1f;       /* 10cm voxels */
    config->max_bricks = 8192;

    /* Spawning */
    config->spawn_radius = 5.0f;
    config->spawn_height_min = 2.0f;
    config->spawn_height_max = 8.0f;
    config->min_separation = 1.0f;

    /* Episode */
    config->max_episode_steps = 1000;
    config->auto_reset = true;

    /* World generation */
    snprintf(config->world_type, sizeof(config->world_type), "%s", "obstacles");
    config->num_obstacles = 20;
    config->seed = 42;
}

/* ============================================================================
 * Physics Configuration Defaults
 * ============================================================================ */

void physics_config_set_defaults(ConfigPhysics* config) {
    /* Zero entire struct first to ensure consistent padding */
    memset(config, 0, sizeof(ConfigPhysics));

    /* Timing */
    config->timestep = 0.02f;        /* 50 Hz frame rate */
    config->substeps = 4;            /* Effective 200 Hz physics */
    config->gravity = 9.81f;         /* m/s^2 */

    /* Integration */
    snprintf(config->integrator, sizeof(config->integrator), "%s", "rk4");

    /* Stability clamps */
    config->velocity_clamp = 20.0f;
    config->angular_velocity_clamp = 30.0f;
    config->normalize_quaternions = true;

    /* Ground effect */
    config->enable_ground_effect = true;
    config->ground_effect_height = 0.5f;   /* Effect starts below 0.5m */
    config->ground_effect_strength = 1.5f; /* 50% thrust increase at ground */

    /* Domain randomization (disabled by default) */
    config->dt_variance = 0.0f;
    config->mass_variance = 0.0f;
    config->thrust_variance = 0.0f;
}

/* ============================================================================
 * Reward Configuration Defaults
 * ============================================================================ */

void reward_config_data_set_defaults(RewardConfigData* config) {
    /* Zero entire struct first to ensure consistent padding */
    memset(config, 0, sizeof(RewardConfigData));

    /* Task */
    snprintf(config->task, sizeof(config->task), "%s", "hover");

    /* Distance rewards */
    config->distance_scale = 1.0f;
    config->distance_exp = 1.0f;     /* Linear distance penalty */
    config->reach_bonus = 10.0f;
    config->reach_radius = 0.5f;

    /* Velocity */
    config->velocity_match_scale = 0.0f; /* Disabled for hover */

    /* Orientation */
    config->uprightness_scale = 0.1f;

    /* Energy/efficiency */
    config->energy_scale = 0.001f;
    config->jerk_scale = 0.0001f;

    /* Collisions */
    config->collision_penalty = 10.0f;
    config->world_collision_penalty = 5.0f;
    config->drone_collision_penalty = 2.0f;

    /* Survival */
    config->alive_bonus = 0.01f;
    config->success_bonus = 100.0f;

    /* Clipping */
    config->reward_min = -10.0f;
    config->reward_max = 10.0f;
}

/* ============================================================================
 * Training Configuration Defaults
 * ============================================================================ */

void training_config_set_defaults(TrainingConfig* config) {
    /* Zero entire struct first to ensure consistent padding */
    memset(config, 0, sizeof(TrainingConfig));

    /* Algorithm */
    snprintf(config->algorithm, sizeof(config->algorithm), "%s", "ppo");

    /* Hyperparameters (PPO defaults) */
    config->learning_rate = 3.0e-4f;
    config->gamma = 0.99f;
    config->gae_lambda = 0.95f;
    config->clip_range = 0.2f;
    config->entropy_coef = 0.01f;
    config->value_coef = 0.5f;
    config->max_grad_norm = 0.5f;

    /* Batch settings */
    config->batch_size = 2048;
    config->num_epochs = 10;
    config->rollout_length = 128;

    /* Logging */
    config->log_interval = 10;
    config->save_interval = 100;
    snprintf(config->checkpoint_dir, CONFIG_PATH_MAX, "%s", "checkpoints");
}

/* ============================================================================
 * Sensor Configuration Defaults
 * ============================================================================ */

SensorConfigEntry sensor_config_entry_default(const char* type) {
    SensorConfigEntry entry;
    memset(&entry, 0, sizeof(entry));

    /* Common defaults */
    snprintf(entry.type, sizeof(entry.type), "%s", type);
    snprintf(entry.name, sizeof(entry.name), "%s", "sensor");

    /* Identity quaternion (w=1) — all other fields zeroed by memset */
    entry.orientation[0] = 1.0f;

    /* Type-specific defaults */
    if (strcmp(type, "imu") == 0) {
        /* Noise configured via noise_groups */
    } else if (strcmp(type, "tof") == 0) {
        entry.max_range = 4.0f;    /* 4m typical ToF range */
        entry.num_rays = 1;
    } else if (strcmp(type, "lidar_2d") == 0) {
        entry.max_range = 10.0f;
        entry.num_rays = 64;
        entry.fov = 6.28318f;      /* 2*PI = 360 degrees */
    } else if (strcmp(type, "lidar_3d") == 0) {
        entry.max_range = 20.0f;
        entry.num_rays = 64;
        entry.fov = 6.28318f;
        entry.fov_vertical = 0.524f; /* 30 degrees */
        entry.vertical_layers = 16;
    } else if (strcmp(type, "camera_rgb") == 0 ||
               strcmp(type, "camera_depth") == 0 ||
               strcmp(type, "camera_segmentation") == 0) {
        entry.width = 84;
        entry.height = 84;
        entry.fov = 1.57f;         /* 90 degrees horizontal */
        entry.fov_vertical = 1.18f; /* ~67 degrees vertical (4:3) */
        entry.near_clip = 0.1f;
        entry.far_clip = 100.0f;
        entry.num_classes = 10;    /* Segmentation classes */
    } else if (strcmp(type, "position") == 0 ||
               strcmp(type, "velocity") == 0) {
        /* Oracle sensors - no specific params */
    } else if (strcmp(type, "neighbor") == 0) {
        entry.k_neighbors = 5;
        entry.max_range = 10.0f;
    }

    return entry;
}

/* ============================================================================
 * Complete Configuration Defaults
 * ============================================================================ */

void config_set_defaults(Config* config) {
    /* Zero entire struct first to ensure consistent padding in outer struct */
    memset(config, 0, sizeof(Config));

    /* Now set defaults for each section (which also zeroes their internals) */
    platform_config_set_defaults(&config->platform);
    environment_config_set_defaults(&config->environment);
    physics_config_set_defaults(&config->physics);
    reward_config_data_set_defaults(&config->reward);
    training_config_set_defaults(&config->training);

    /* No sensors by default (already zeroed) */
    config->sensors = NULL;
    config->num_sensors = 0;

    /* Metadata is already zeroed */
}
