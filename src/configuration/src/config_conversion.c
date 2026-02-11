/**
 * Configuration Conversion - DroneConfig to/from DroneParamsSOA
 *
 * Converts configuration structures to the SoA data structures used
 * by the physics simulation.
 *
 * Field Name Mappings (DroneConfig -> DroneParamsSOA):
 *   k_drag_angular    -> k_ang_damp
 *   max_velocity      -> max_vel
 *   max_angular_velocity -> max_omega
 *   physics.gravity   -> gravity (from ConfigPhysics)
 */

#include "configuration.h"
#include <string.h>

/* ============================================================================
 * DroneConfig to DroneParamsSOA Conversion
 * ============================================================================ */

void drone_config_to_params(const DroneConfig* drone_cfg,
                            const ConfigPhysics* physics_cfg,
                            DroneParamsSOA* params,
                            uint32_t start_index,
                            uint32_t count) {
    if (!drone_cfg || !physics_cfg || !params) return;
    if (start_index + count > params->capacity) return;

    /* Broadcast config to all specified indices */
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = start_index + i;

        /* Mass and inertia */
        params->mass[idx] = drone_cfg->mass;
        params->ixx[idx] = drone_cfg->ixx;
        params->iyy[idx] = drone_cfg->iyy;
        params->izz[idx] = drone_cfg->izz;

        /* Geometry */
        params->arm_length[idx] = drone_cfg->arm_length;
        params->collision_radius[idx] = drone_cfg->collision_radius;

        /* Thrust and torque */
        params->k_thrust[idx] = drone_cfg->k_thrust;
        params->k_torque[idx] = drone_cfg->k_torque;

        /* Aerodynamics (note: k_drag_angular -> k_ang_damp) */
        params->k_drag[idx] = drone_cfg->k_drag;
        params->k_ang_damp[idx] = drone_cfg->k_drag_angular;

        /* Motor dynamics */
        params->motor_tau[idx] = drone_cfg->motor_tau;
        params->max_rpm[idx] = drone_cfg->max_rpm;

        /* Limits (note: max_velocity -> max_vel, max_angular_velocity -> max_omega) */
        params->max_vel[idx] = drone_cfg->max_velocity;
        params->max_omega[idx] = drone_cfg->max_angular_velocity;

        /* Environment (from ConfigPhysics) */
        params->gravity[idx] = physics_cfg->gravity;
    }

    /* Update count if extending */
    if (start_index + count > params->count) {
        params->count = start_index + count;
    }
}

/* ============================================================================
 * DroneParamsSOA to DroneConfig Conversion
 * ============================================================================ */

DroneConfig drone_params_to_config(const DroneParamsSOA* params, uint32_t index) {
    DroneConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    if (!params || index >= params->count) {
        /* Return empty config if invalid */
        drone_config_set_defaults(&cfg);
        return cfg;
    }

    /* Set default name and path */
    strncpy(cfg.name, "extracted", CONFIG_NAME_MAX - 1);
    cfg.name[CONFIG_NAME_MAX - 1] = '\0';
    cfg.model_path[0] = '\0';

    /* Mass and inertia */
    cfg.mass = params->mass[index];
    cfg.ixx = params->ixx[index];
    cfg.iyy = params->iyy[index];
    cfg.izz = params->izz[index];

    /* Geometry */
    cfg.arm_length = params->arm_length[index];
    cfg.collision_radius = params->collision_radius[index];

    /* Thrust and torque */
    cfg.k_thrust = params->k_thrust[index];
    cfg.k_torque = params->k_torque[index];

    /* Aerodynamics (note: k_ang_damp -> k_drag_angular) */
    cfg.k_drag = params->k_drag[index];
    cfg.k_drag_angular = params->k_ang_damp[index];

    /* Motor dynamics */
    cfg.motor_tau = params->motor_tau[index];
    cfg.max_rpm = params->max_rpm[index];

    /* Limits (note: max_vel -> max_velocity, max_omega -> max_angular_velocity) */
    cfg.max_velocity = params->max_vel[index];
    cfg.max_angular_velocity = params->max_omega[index];

    /* Note: gravity is NOT extracted (belongs to ConfigPhysics) */

    /* Set reasonable defaults for fields not in SoA */
    cfg.max_tilt_angle = 1.0f;
    cfg.color[0] = 0.2f;
    cfg.color[1] = 0.6f;
    cfg.color[2] = 1.0f;
    cfg.scale = 1.0f;

    return cfg;
}

/* ============================================================================
 * Convenience Wrapper
 * ============================================================================ */

void config_init_drone_params(const Config* config,
                              DroneParamsSOA* params,
                              uint32_t num_drones) {
    if (!config || !params) return;

    drone_config_to_params(&config->drone, &config->physics,
                           params, 0, num_drones);
}

/* ============================================================================
 * Memory Size Calculation
 * ============================================================================ */

size_t config_memory_size(uint32_t num_sensors) {
    /* Base Config struct */
    size_t size = sizeof(Config);

    /* Sensors array */
    size += num_sensors * sizeof(SensorConfigEntry);

    return size;
}
