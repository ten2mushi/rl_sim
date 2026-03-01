/**
 * Configuration Conversion - PlatformConfig to/from PlatformParamsSOA
 *
 * Converts configuration structures to the composite SoA data structures
 * used by the physics simulation. Rigid body params go in the
 * RigidBodyParamsSOA core, quadcopter-specific params go in extension arrays.
 *
 * Field Name Mappings (PlatformConfig -> RigidBodyParamsSOA):
 *   max_velocity      -> max_vel
 *   max_angular_velocity -> max_omega
 *   physics.gravity   -> gravity (from ConfigPhysics)
 *
 * Field Name Mappings (QuadcopterConfig -> extension[QUAD_PEXT_*]):
 *   k_ang_damp        -> extension[QUAD_PEXT_K_ANG_DAMP]
 *   arm_length        -> extension[QUAD_PEXT_ARM_LENGTH]
 *   k_thrust          -> extension[QUAD_PEXT_K_THRUST]
 *   etc.
 */

#include "configuration.h"
#include "platform_quadcopter.h"
#include "platform_diff_drive.h"
#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * PlatformConfig to PlatformParamsSOA Conversion
 * ============================================================================ */

void platform_config_to_params(const PlatformConfig* platform_cfg,
                                const ConfigPhysics* physics_cfg,
                                PlatformParamsSOA* params,
                                uint32_t start_index,
                                uint32_t count) {
    if (!platform_cfg || !physics_cfg || !params) return;
    if (start_index + count > params->rigid_body.capacity) return;

    RigidBodyParamsSOA* rb = &params->rigid_body;

    /* Get platform-specific config if available */
    const QuadcopterConfig* quad = NULL;
    const DiffDriveConfig* dd = NULL;
    if (platform_cfg->platform_specific) {
        if (strcmp(platform_cfg->type, "quadcopter") == 0) {
            quad = (const QuadcopterConfig*)platform_cfg->platform_specific;
        } else if (strcmp(platform_cfg->type, "diff_drive") == 0) {
            dd = (const DiffDriveConfig*)platform_cfg->platform_specific;
        }
    }

    /* Broadcast config to all specified indices */
    for (uint32_t i = 0; i < count; i++) {
        uint32_t idx = start_index + i;

        /* Rigid body params */
        rb->mass[idx] = platform_cfg->mass;
        rb->ixx[idx] = platform_cfg->ixx;
        rb->iyy[idx] = platform_cfg->iyy;
        rb->izz[idx] = platform_cfg->izz;
        rb->collision_radius[idx] = platform_cfg->collision_radius;
        rb->max_vel[idx] = platform_cfg->max_velocity;
        rb->max_omega[idx] = platform_cfg->max_angular_velocity;
        rb->gravity[idx] = physics_cfg->gravity;

        /* Platform-specific (quadcopter) extension params */
        if (quad) {
            if (params->extension_count > QUAD_PEXT_ARM_LENGTH)
                params->extension[QUAD_PEXT_ARM_LENGTH][idx] = quad->arm_length;
            if (params->extension_count > QUAD_PEXT_K_THRUST)
                params->extension[QUAD_PEXT_K_THRUST][idx] = quad->k_thrust;
            if (params->extension_count > QUAD_PEXT_K_TORQUE)
                params->extension[QUAD_PEXT_K_TORQUE][idx] = quad->k_torque;
            if (params->extension_count > QUAD_PEXT_K_DRAG)
                params->extension[QUAD_PEXT_K_DRAG][idx] = quad->k_drag;
            if (params->extension_count > QUAD_PEXT_MOTOR_TAU)
                params->extension[QUAD_PEXT_MOTOR_TAU][idx] = quad->motor_tau;
            if (params->extension_count > QUAD_PEXT_MAX_RPM)
                params->extension[QUAD_PEXT_MAX_RPM][idx] = quad->max_rpm;
            if (params->extension_count > QUAD_PEXT_K_ANG_DAMP)
                params->extension[QUAD_PEXT_K_ANG_DAMP][idx] = quad->k_ang_damp;
        }

        /* Platform-specific (diff_drive) extension params */
        if (dd) {
            if (params->extension_count > DD_PEXT_WHEEL_RADIUS)
                params->extension[DD_PEXT_WHEEL_RADIUS][idx] = dd->wheel_radius;
            if (params->extension_count > DD_PEXT_AXLE_LENGTH)
                params->extension[DD_PEXT_AXLE_LENGTH][idx] = dd->axle_length;
            if (params->extension_count > DD_PEXT_MAX_WHEEL_VEL)
                params->extension[DD_PEXT_MAX_WHEEL_VEL][idx] = dd->max_wheel_vel;
        }
    }

    /* Update count if extending */
    if (start_index + count > rb->count) {
        rb->count = start_index + count;
    }
}

/* ============================================================================
 * PlatformParamsSOA to PlatformConfig Conversion
 * ============================================================================ */

PlatformConfig platform_params_to_config(const PlatformParamsSOA* params, uint32_t index) {
    PlatformConfig cfg;
    memset(&cfg, 0, sizeof(cfg));

    if (!params || index >= params->rigid_body.count) {
        /* Return empty config if invalid */
        platform_config_set_defaults(&cfg);
        return cfg;
    }

    const RigidBodyParamsSOA* rb = &params->rigid_body;

    /* Set type and default name */
    snprintf(cfg.type, CONFIG_NAME_MAX, "%s", "quadcopter");
    strncpy(cfg.name, "extracted", CONFIG_NAME_MAX - 1);
    cfg.name[CONFIG_NAME_MAX - 1] = '\0';
    cfg.model_path[0] = '\0';

    /* Mass and inertia (from rigid body) */
    cfg.mass = rb->mass[index];
    cfg.ixx = rb->ixx[index];
    cfg.iyy = rb->iyy[index];
    cfg.izz = rb->izz[index];

    /* Geometry (rigid body) */
    cfg.collision_radius = rb->collision_radius[index];

    /* Limits (rigid body, note: max_vel -> max_velocity, max_omega -> max_angular_velocity) */
    cfg.max_velocity = rb->max_vel[index];
    cfg.max_angular_velocity = rb->max_omega[index];

    /* Set reasonable defaults for fields not in SoA */
    cfg.max_tilt_angle = 1.0f;
    cfg.color[0] = 0.2f;
    cfg.color[1] = 0.6f;
    cfg.color[2] = 1.0f;
    cfg.scale = 1.0f;

    /* Allocate and extract quadcopter-specific extension params */
    QuadcopterConfig* quad = (QuadcopterConfig*)malloc(sizeof(QuadcopterConfig));
    if (quad) {
        memset(quad, 0, sizeof(QuadcopterConfig));
        if (params->extension_count > QUAD_PEXT_ARM_LENGTH)
            quad->arm_length = params->extension[QUAD_PEXT_ARM_LENGTH][index];
        if (params->extension_count > QUAD_PEXT_K_THRUST)
            quad->k_thrust = params->extension[QUAD_PEXT_K_THRUST][index];
        if (params->extension_count > QUAD_PEXT_K_TORQUE)
            quad->k_torque = params->extension[QUAD_PEXT_K_TORQUE][index];
        if (params->extension_count > QUAD_PEXT_K_DRAG)
            quad->k_drag = params->extension[QUAD_PEXT_K_DRAG][index];
        if (params->extension_count > QUAD_PEXT_K_ANG_DAMP)
            quad->k_ang_damp = params->extension[QUAD_PEXT_K_ANG_DAMP][index];
        if (params->extension_count > QUAD_PEXT_MOTOR_TAU)
            quad->motor_tau = params->extension[QUAD_PEXT_MOTOR_TAU][index];
        if (params->extension_count > QUAD_PEXT_MAX_RPM)
            quad->max_rpm = params->extension[QUAD_PEXT_MAX_RPM][index];
        cfg.platform_specific = quad;
        cfg.platform_specific_size = sizeof(QuadcopterConfig);
    }

    /* Note: gravity is NOT extracted (belongs to ConfigPhysics) */

    return cfg;
}

/* ============================================================================
 * Convenience Wrapper
 * ============================================================================ */

void config_init_platform_params(const Config* config,
                                  PlatformParamsSOA* params,
                                  uint32_t num_agents) {
    if (!config || !params) return;

    platform_config_to_params(&config->platform, &config->physics,
                               params, 0, num_agents);
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
