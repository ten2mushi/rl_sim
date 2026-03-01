/**
 * Engine Queries Implementation
 *
 * Provides state query functions for debugging and visualization.
 */

#include "environment_manager.h"

/* ============================================================================
 * Single Drone State Query
 * ============================================================================ */

void engine_get_agent_state(const BatchEngine* engine, uint32_t agent_idx,
                            AgentStateQuery* out) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(out != NULL, "query output is NULL");
    FOUNDATION_ASSERT(agent_idx < engine->config.total_agents, "drone index out of bounds");

    const RigidBodyStateSOA* rb = &engine->states->rigid_body;

    /* Position */
    out->position.x = rb->pos_x[agent_idx];
    out->position.y = rb->pos_y[agent_idx];
    out->position.z = rb->pos_z[agent_idx];

    /* Velocity */
    out->velocity.x = rb->vel_x[agent_idx];
    out->velocity.y = rb->vel_y[agent_idx];
    out->velocity.z = rb->vel_z[agent_idx];

    /* Orientation */
    out->orientation.w = rb->quat_w[agent_idx];
    out->orientation.x = rb->quat_x[agent_idx];
    out->orientation.y = rb->quat_y[agent_idx];
    out->orientation.z = rb->quat_z[agent_idx];

    /* Angular velocity */
    out->angular_velocity.x = rb->omega_x[agent_idx];
    out->angular_velocity.y = rb->omega_y[agent_idx];
    out->angular_velocity.z = rb->omega_z[agent_idx];

    /* Environment info */
    out->env_id = engine->env_ids[agent_idx];
    out->agent_id = agent_idx % engine->config.agents_per_env;

    /* Termination flags */
    out->is_done = engine->dones[agent_idx] != 0;
    out->is_truncated = engine->truncations[agent_idx] != 0;
}

/* ============================================================================
 * Batch Position Export
 * ============================================================================ */

void engine_get_all_positions(const BatchEngine* engine, float* positions_xyz) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(positions_xyz != NULL, "positions output buffer is NULL");

    const RigidBodyStateSOA* rb = &engine->states->rigid_body;
    uint32_t total = engine->config.total_agents;

    for (uint32_t i = 0; i < total; i++) {
        positions_xyz[i * 3 + 0] = rb->pos_x[i];
        positions_xyz[i * 3 + 1] = rb->pos_y[i];
        positions_xyz[i * 3 + 2] = rb->pos_z[i];
    }
}

/* ============================================================================
 * Batch Orientation Export
 * ============================================================================ */

void engine_get_all_orientations(const BatchEngine* engine, float* quats_wxyz) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(quats_wxyz != NULL, "orientations output buffer is NULL");

    const RigidBodyStateSOA* rb = &engine->states->rigid_body;
    uint32_t total = engine->config.total_agents;

    for (uint32_t i = 0; i < total; i++) {
        quats_wxyz[i * 4 + 0] = rb->quat_w[i];
        quats_wxyz[i * 4 + 1] = rb->quat_x[i];
        quats_wxyz[i * 4 + 2] = rb->quat_y[i];
        quats_wxyz[i * 4 + 3] = rb->quat_z[i];
    }
}

/* ============================================================================
 * Batch Velocity Export
 * ============================================================================ */

void engine_get_all_velocities(const BatchEngine* engine, float* velocities_xyz) {
    FOUNDATION_ASSERT(engine != NULL, "engine handle is NULL");
    FOUNDATION_ASSERT(velocities_xyz != NULL, "velocities output buffer is NULL");

    const RigidBodyStateSOA* rb = &engine->states->rigid_body;
    uint32_t total = engine->config.total_agents;

    for (uint32_t i = 0; i < total; i++) {
        velocities_xyz[i * 3 + 0] = rb->vel_x[i];
        velocities_xyz[i * 3 + 1] = rb->vel_y[i];
        velocities_xyz[i * 3 + 2] = rb->vel_z[i];
    }
}
