/**
 * Engine Queries Implementation
 *
 * Provides state query functions for debugging and visualization.
 */

#include "environment_manager.h"

/* ============================================================================
 * Single Drone State Query
 * ============================================================================ */

void engine_get_drone_state(const BatchDroneEngine* engine, uint32_t drone_idx,
                            DroneStateQuery* out) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(out != NULL, "assertion failed");
    FOUNDATION_ASSERT(drone_idx < engine->config.total_drones, "assertion failed");

    const DroneStateSOA* states = engine->states;

    /* Position */
    out->position.x = states->pos_x[drone_idx];
    out->position.y = states->pos_y[drone_idx];
    out->position.z = states->pos_z[drone_idx];

    /* Velocity */
    out->velocity.x = states->vel_x[drone_idx];
    out->velocity.y = states->vel_y[drone_idx];
    out->velocity.z = states->vel_z[drone_idx];

    /* Orientation */
    out->orientation.w = states->quat_w[drone_idx];
    out->orientation.x = states->quat_x[drone_idx];
    out->orientation.y = states->quat_y[drone_idx];
    out->orientation.z = states->quat_z[drone_idx];

    /* Angular velocity */
    out->angular_velocity.x = states->omega_x[drone_idx];
    out->angular_velocity.y = states->omega_y[drone_idx];
    out->angular_velocity.z = states->omega_z[drone_idx];

    /* Motor RPMs */
    out->rpms[0] = states->rpm_0[drone_idx];
    out->rpms[1] = states->rpm_1[drone_idx];
    out->rpms[2] = states->rpm_2[drone_idx];
    out->rpms[3] = states->rpm_3[drone_idx];

    /* Environment info */
    out->env_id = engine->env_ids[drone_idx];
    out->drone_id = drone_idx % engine->config.drones_per_env;

    /* Termination flags */
    out->is_done = engine->dones[drone_idx] != 0;
    out->is_truncated = engine->truncations[drone_idx] != 0;
}

/* ============================================================================
 * Batch Position Export
 * ============================================================================ */

void engine_get_all_positions(const BatchDroneEngine* engine, float* positions_xyz) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(positions_xyz != NULL, "assertion failed");

    const DroneStateSOA* states = engine->states;
    uint32_t total = engine->config.total_drones;

    for (uint32_t i = 0; i < total; i++) {
        positions_xyz[i * 3 + 0] = states->pos_x[i];
        positions_xyz[i * 3 + 1] = states->pos_y[i];
        positions_xyz[i * 3 + 2] = states->pos_z[i];
    }
}

/* ============================================================================
 * Batch Orientation Export
 * ============================================================================ */

void engine_get_all_orientations(const BatchDroneEngine* engine, float* quats_wxyz) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(quats_wxyz != NULL, "assertion failed");

    const DroneStateSOA* states = engine->states;
    uint32_t total = engine->config.total_drones;

    for (uint32_t i = 0; i < total; i++) {
        quats_wxyz[i * 4 + 0] = states->quat_w[i];
        quats_wxyz[i * 4 + 1] = states->quat_x[i];
        quats_wxyz[i * 4 + 2] = states->quat_y[i];
        quats_wxyz[i * 4 + 3] = states->quat_z[i];
    }
}

/* ============================================================================
 * Batch Velocity Export
 * ============================================================================ */

void engine_get_all_velocities(const BatchDroneEngine* engine, float* velocities_xyz) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(velocities_xyz != NULL, "assertion failed");

    const DroneStateSOA* states = engine->states;
    uint32_t total = engine->config.total_drones;

    for (uint32_t i = 0; i < total; i++) {
        velocities_xyz[i * 3 + 0] = states->vel_x[i];
        velocities_xyz[i * 3 + 1] = states->vel_y[i];
        velocities_xyz[i * 3 + 2] = states->vel_z[i];
    }
}
