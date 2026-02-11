/**
 * Configuration Defaults - Crazyflie 2.0 and sensible default values
 *
 * Default values are based on the well-documented Crazyflie 2.0 quadcopter.
 * Reference: https://www.bitcraze.io/products/crazyflie-2-1/
 */

#include "configuration.h"
#include <string.h>

/* ============================================================================
 * Drone Configuration Defaults (Crazyflie 2.0)
 * ============================================================================ */

void drone_config_set_defaults(DroneConfig* config) {
    /* Zero entire struct first to ensure consistent padding */
    memset(config, 0, sizeof(DroneConfig));

    /* Crazyflie 2.0 defaults (well-documented reference drone) */
    strncpy(config->name, "crazyflie2", CONFIG_NAME_MAX - 1);
    config->name[CONFIG_NAME_MAX - 1] = '\0';

    /* Physical properties */
    config->mass = 0.027f;           /* 27 grams */
    config->arm_length = 0.046f;     /* 46mm arm length */

    /* Inertia tensor (diagonal) - from Crazyflie documentation */
    config->ixx = 1.4e-5f;           /* kg*m^2 */
    config->iyy = 1.4e-5f;           /* kg*m^2 */
    config->izz = 2.17e-5f;          /* kg*m^2 */

    /* Motor parameters - derived from motor characterization */
    config->k_thrust = 2.88e-8f;     /* N/(rad/s)^2 */
    config->k_torque = 7.24e-10f;    /* N*m/(rad/s)^2 */
    config->motor_tau = 0.02f;       /* 20ms motor response time */
    config->max_rpm = 21702.0f;      /* Maximum motor RPM */

    /* Geometry */
    config->collision_radius = 0.056f; /* arm_length + 0.01 safety margin */

    /* Aerodynamics */
    config->k_drag = 0.0f;           /* Linear drag coefficient */
    config->k_drag_angular = 0.0f;   /* Angular drag coefficient */

    /* Limits */
    config->max_velocity = 10.0f;    /* m/s */
    config->max_angular_velocity = 20.0f; /* rad/s */
    config->max_tilt_angle = 1.0f;   /* ~57 degrees */

    /* Visual */
    config->color[0] = 0.2f;         /* R */
    config->color[1] = 0.6f;         /* G */
    config->color[2] = 1.0f;         /* B - Crazyflie blue */
    config->scale = 1.0f;
}

/* ============================================================================
 * Environment Configuration Defaults
 * ============================================================================ */

void environment_config_set_defaults(EnvironmentConfig* config) {
    /* Zero entire struct first to ensure consistent padding */
    memset(config, 0, sizeof(EnvironmentConfig));

    /* Dimensions */
    config->num_envs = 64;
    config->drones_per_env = 16;

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
    strncpy(config->world_type, "obstacles", sizeof(config->world_type) - 1);
    config->world_type[sizeof(config->world_type) - 1] = '\0';
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
    strncpy(config->integrator, "rk4", sizeof(config->integrator) - 1);
    config->integrator[sizeof(config->integrator) - 1] = '\0';

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
    strncpy(config->task, "hover", sizeof(config->task) - 1);
    config->task[sizeof(config->task) - 1] = '\0';

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
    strncpy(config->algorithm, "ppo", sizeof(config->algorithm) - 1);
    config->algorithm[sizeof(config->algorithm) - 1] = '\0';

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
    strncpy(config->checkpoint_dir, "checkpoints", CONFIG_PATH_MAX - 1);
    config->checkpoint_dir[CONFIG_PATH_MAX - 1] = '\0';
}

/* ============================================================================
 * Sensor Configuration Defaults
 * ============================================================================ */

SensorConfigEntry sensor_config_entry_default(const char* type) {
    SensorConfigEntry entry;
    memset(&entry, 0, sizeof(entry));

    /* Common defaults */
    strncpy(entry.type, type, sizeof(entry.type) - 1);
    entry.type[sizeof(entry.type) - 1] = '\0';
    strncpy(entry.name, "sensor", sizeof(entry.name) - 1);
    entry.name[sizeof(entry.name) - 1] = '\0';

    /* Default mounting - centered, no rotation */
    entry.position[0] = 0.0f;
    entry.position[1] = 0.0f;
    entry.position[2] = 0.0f;
    entry.orientation[0] = 1.0f;  /* w */
    entry.orientation[1] = 0.0f;  /* x */
    entry.orientation[2] = 0.0f;  /* y */
    entry.orientation[3] = 0.0f;  /* z */

    entry.sample_rate = 0.0f;     /* Every frame */
    entry.num_noise_groups = 0;   /* No noise by default */

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
    drone_config_set_defaults(&config->drone);
    environment_config_set_defaults(&config->environment);
    physics_config_set_defaults(&config->physics);
    reward_config_data_set_defaults(&config->reward);
    training_config_set_defaults(&config->training);

    /* No sensors by default (already zeroed) */
    config->sensors = NULL;
    config->num_sensors = 0;

    /* Metadata is already zeroed */
}
