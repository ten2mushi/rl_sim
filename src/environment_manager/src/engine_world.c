/**
 * Engine World Implementation
 *
 * Provides world manipulation helpers for adding obstacles and setting targets.
 */

#include "environment_manager.h"
#include "obj_io.h"

/* ============================================================================
 * Obstacle Addition
 * ============================================================================ */

void engine_add_box(BatchDroneEngine* engine, Vec3 min_corner, Vec3 max_corner,
                    uint8_t material) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->world != NULL, "assertion failed");

    Vec3 center = vec3_scale(vec3_add(min_corner, max_corner), 0.5f);
    Vec3 half_size = vec3_scale(vec3_sub(max_corner, min_corner), 0.5f);

    world_set_box(engine->world, center, half_size, material);
}

void engine_add_sphere(BatchDroneEngine* engine, Vec3 center, float radius,
                       uint8_t material) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->world != NULL, "assertion failed");

    world_set_sphere(engine->world, center, radius, material);
}

void engine_add_cylinder(BatchDroneEngine* engine, Vec3 center,
                         float radius, float half_height, uint8_t material) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->world != NULL, "assertion failed");

    world_set_cylinder(engine->world, center, radius, half_height, material);
}

/* ============================================================================
 * OBJ Loading
 * ============================================================================ */

int engine_load_obj(BatchDroneEngine* engine, const char* obj_path) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(obj_path != NULL, "assertion failed");

    char obj_error[256] = {0};
    WorldBrickMap* new_world = NULL;
    VoxelizeOptions vox_opts = VOXELIZE_DEFAULTS;
    vox_opts.voxel_size = engine->config.voxel_size;
    ObjIOResult result = obj_to_world(engine->persistent_arena, obj_path,
                                       &vox_opts, &new_world, obj_error);
    if (result != OBJ_IO_SUCCESS || new_world == NULL) {
        return -1;
    }

    /* Replace world (old world leaks in arena -- freed on engine destroy) */
    engine->world = new_world;

    /* Update config bounds from loaded world */
    engine->config.world_min = new_world->world_min;
    engine->config.world_max = new_world->world_max;

    /* Sync termination bounds if not custom */
    if (!engine->config.use_custom_termination) {
        engine->config.termination_min = new_world->world_min;
        engine->config.termination_max = new_world->world_max;
    }

    engine->needs_reset = true;
    return 0;
}

/* ============================================================================
 * World Reset
 * ============================================================================ */

void engine_clear_world(BatchDroneEngine* engine) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->world != NULL, "assertion failed");

    world_clear(engine->world);
}

/* ============================================================================
 * Target Management
 * ============================================================================ */

void engine_set_target(BatchDroneEngine* engine, uint32_t drone_idx, Vec3 target) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->rewards != NULL, "assertion failed");
    FOUNDATION_ASSERT(drone_idx < engine->config.total_drones, "assertion failed");

    reward_set_target(engine->rewards, drone_idx, target, VEC3_ZERO, 0.5f);
}

void engine_set_targets(BatchDroneEngine* engine, const Vec3* targets) {
    FOUNDATION_ASSERT(engine != NULL, "assertion failed");
    FOUNDATION_ASSERT(engine->rewards != NULL, "assertion failed");
    FOUNDATION_ASSERT(targets != NULL, "assertion failed");

    uint32_t total_drones = engine->config.total_drones;
    for (uint32_t i = 0; i < total_drones; i++) {
        reward_set_target(engine->rewards, i, targets[i], VEC3_ZERO, 0.5f);
    }
}
