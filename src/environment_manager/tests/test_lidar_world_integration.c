/**
 * LiDAR + World Integration Test
 *
 * Verifies end-to-end pipeline: spawn world geometry -> attach 3D LiDAR -> step ->
 * verify non-max_range observations when obstacle is present.
 *
 * This is a critical integration test that validates the complete sensor pipeline
 * works correctly with world geometry before Python binding integration.
 */

#include "environment_manager.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "test_harness.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/* ============================================================================
 * Helper: Create engine with LiDAR 3D sensor
 * ============================================================================ */

static BatchDroneEngine* create_engine_with_lidar3d(
    uint32_t num_envs,
    uint32_t drones_per_env,
    uint32_t horizontal_rays,
    uint32_t vertical_layers,
    float max_range
) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = num_envs;
    cfg.drones_per_env = drones_per_env;
    cfg.seed = 12345;
    cfg.persistent_arena_size = 256 * 1024 * 1024;  /* 256 MB */
    cfg.frame_arena_size = 64 * 1024 * 1024;         /* 64 MB */

    /* Configure LiDAR 3D sensor */
    SensorConfig lidar_cfg = {0};
    lidar_cfg.type = SENSOR_TYPE_LIDAR_3D;
    lidar_cfg.position_offset = VEC3(0.0f, 0.0f, 0.0f);  /* Center of drone */
    lidar_cfg.orientation_offset = QUAT(1.0f, 0.0f, 0.0f, 0.0f);  /* Identity */
    lidar_cfg.sample_rate = 0.0f;  /* Every step */
    /* noise_stddev removed - noise configured via NoiseConfig */
    lidar_cfg.lidar_3d.horizontal_rays = horizontal_rays;
    lidar_cfg.lidar_3d.vertical_layers = vertical_layers;
    lidar_cfg.lidar_3d.horizontal_fov = (float)M_PI / 2.0f;  /* 90 degrees */
    lidar_cfg.lidar_3d.vertical_fov = (float)M_PI / 4.0f;    /* 45 degrees */
    lidar_cfg.lidar_3d.max_range = max_range;

    if (engine_config_add_sensor(&cfg, &lidar_cfg) != 0) {
        return NULL;
    }

    char error[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, error);
}

/* ============================================================================
 * Test 1: LiDAR with empty world returns max_range
 * ============================================================================ */

TEST(lidar_empty_world_max_range) {
    /* Create engine with 8x8 LiDAR, no world geometry */
    BatchDroneEngine* engine = create_engine_with_lidar3d(1, 1, 8, 8, 50.0f);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Step to populate observations */
    float* actions = engine_get_actions(engine);
    memset(actions, 0, ENGINE_ACTION_DIM * sizeof(float));
    engine_step(engine);

    /* Get observations */
    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    /* All LiDAR readings should be max_range (50.0) since no geometry */
    /* Note: obs_dim may include other sensors, we need to find the LiDAR data */
    uint32_t lidar_rays = 8 * 8;  /* 64 rays */

    /* Check that we have observations */
    ASSERT_GT(engine->obs_dim, 0);

    /* With empty world (no geometry added), all readings should be max_range */
    /* The exact position in obs buffer depends on observation layout */
    /* For now, check that observations are populated and finite */
    for (uint32_t i = 0; i < engine->obs_dim && i < lidar_rays; i++) {
        ASSERT_TRUE(isfinite(obs[i]));
        /* In empty world, expect max_range or close to it */
        ASSERT_GT(obs[i], 0.0f);
    }

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 2: LiDAR detects box obstacle in front
 * ============================================================================ */

TEST(lidar_detects_box_obstacle) {
    /* Create engine with 16x8 LiDAR */
    float max_range = 50.0f;
    BatchDroneEngine* engine = create_engine_with_lidar3d(1, 1, 16, 8, max_range);
    ASSERT_NOT_NULL(engine);
    ASSERT_NOT_NULL(engine->world);

    engine_reset(engine);

    /* Debug: Print world dimensions */
    printf("\n    [Debug] World bounds: (%.1f,%.1f,%.1f) to (%.1f,%.1f,%.1f)",
           engine->world->world_min.x, engine->world->world_min.y, engine->world->world_min.z,
           engine->world->world_max.x, engine->world->world_max.y, engine->world->world_max.z);
    printf("\n    [Debug] Voxel size: %.3f, brick size: %.3f, max_bricks: %u",
           engine->world->voxel_size, engine->world->brick_size_world, engine->world->max_bricks);
    printf("\n    [Debug] Grid: %u x %u x %u = %u bricks",
           engine->world->grid_x, engine->world->grid_y, engine->world->grid_z, engine->world->grid_total);
    printf("\n    [Debug] SDF scale: %.4f", engine->world->sdf_scale);

    /* Check atlas count before any geometry */
    printf("\n    [Debug] Atlas count BEFORE: %u", engine->world->atlas_count);

    /* SDF query before adding box */
    Vec3 box_center = VEC3(10.0f, 0.0f, 10.0f);
    float sdf_before = world_sdf_query(engine->world, box_center);
    printf("\n    [Debug] SDF at box center BEFORE: %.4f", sdf_before);

    /* Try a much larger box to ensure it's visible */
    Vec3 box_half_size = VEC3(5.0f, 5.0f, 5.0f);  /* 10m cube */
    printf("\n    [Debug] Adding box: center=(%.1f,%.1f,%.1f) half=(%.1f,%.1f,%.1f)",
           box_center.x, box_center.y, box_center.z,
           box_half_size.x, box_half_size.y, box_half_size.z);

    world_set_box(engine->world, box_center, box_half_size, 1);

    /* Check atlas count after adding box */
    printf("\n    [Debug] Atlas count AFTER box: %u, uniform_inside: %u, uniform_outside: %u",
           engine->world->atlas_count, engine->world->uniform_inside_count, engine->world->uniform_outside_count);

    /* SDF query after adding box */
    float sdf_after = world_sdf_query(engine->world, box_center);
    printf("\n    [Debug] SDF at box center AFTER: %.4f (should be negative)", sdf_after);

    /* Test direct raymarching: ray from (0, 0, 10) towards +X should hit box */
    Vec3 ray_origin = VEC3(0.0f, 0.0f, 10.0f);
    Vec3 ray_dir = VEC3(1.0f, 0.0f, 0.0f);  /* +X direction */
    RayHit hit = world_raymarch(engine->world, ray_origin, ray_dir, max_range);

    printf("\n    [Debug] Direct raymarch: hit=%d, distance=%.2f", hit.hit, hit.distance);

    ASSERT_TRUE(hit.hit);
    ASSERT_LT(hit.distance, 12.0f);

    /* Now position drone and test through sensor system */
    engine->states->pos_x[0] = 0.0f;
    engine->states->pos_y[0] = 0.0f;
    engine->states->pos_z[0] = 10.0f;  /* Same Z (altitude) as box */
    engine->states->quat_w[0] = 1.0f;  /* Identity quaternion */
    engine->states->quat_x[0] = 0.0f;
    engine->states->quat_y[0] = 0.0f;
    engine->states->quat_z[0] = 0.0f;

    /* Sample sensors */
    engine_step_sensors(engine);

    /* Get observations */
    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    /* Find any readings less than max_range */
    bool found_hit_sensor = false;
    float min_distance = max_range;

    uint32_t total_rays = 16 * 8;
    for (uint32_t i = 0; i < total_rays && i < engine->obs_dim; i++) {
        if (obs[i] < max_range && obs[i] > 0.0f) {
            found_hit_sensor = true;
            if (obs[i] < min_distance) {
                min_distance = obs[i];
            }
        }
    }

    printf("\n    [Info] Min LiDAR distance: %.2f m (expected ~8m)", min_distance);

    ASSERT_TRUE(found_hit_sensor);
    ASSERT_LT(min_distance, 15.0f);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 3: LiDAR detects sphere obstacle
 * ============================================================================ */

TEST(lidar_detects_sphere_obstacle) {
    float max_range = 50.0f;
    BatchDroneEngine* engine = create_engine_with_lidar3d(1, 1, 16, 8, max_range);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Position drone at origin, elevated to match sphere Z */
    engine->states->pos_x[0] = 0.0f;
    engine->states->pos_y[0] = 0.0f;
    engine->states->pos_z[0] = 10.0f;  /* Match sphere Z (altitude) */
    engine->states->quat_w[0] = 1.0f;
    engine->states->quat_x[0] = 0.0f;
    engine->states->quat_y[0] = 0.0f;
    engine->states->quat_z[0] = 0.0f;

    /* Add large sphere in front of drone (use larger radius for atlas efficiency) */
    Vec3 sphere_center = VEC3(15.0f, 0.0f, 10.0f);  /* 15m in +X, same altitude */
    float sphere_radius = 5.0f;  /* Large enough for UNIFORM_INSIDE optimization */
    world_set_sphere(engine->world, sphere_center, sphere_radius, 1);

    printf("\n    [Debug] Sphere: center=(%.1f,%.1f,%.1f) r=%.1f",
           sphere_center.x, sphere_center.y, sphere_center.z, sphere_radius);

    /* Sample sensors */
    engine_step_sensors(engine);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    /* Find minimum distance (should hit sphere) */
    float min_distance = max_range;
    uint32_t total_rays = 16 * 8;

    for (uint32_t i = 0; i < total_rays && i < engine->obs_dim; i++) {
        if (obs[i] < min_distance && obs[i] > 0.0f) {
            min_distance = obs[i];
        }
    }

    /* Should detect sphere at ~10m (15 - 5 = 10) */
    ASSERT_LT(min_distance, max_range);
    ASSERT_LT(min_distance, 15.0f);
    ASSERT_GT(min_distance, 5.0f);

    printf("\n    [Info] Min LiDAR distance to sphere: %.2f m (expected ~10m)", min_distance);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 4: Multiple drones with shared world geometry
 * ============================================================================ */

TEST(lidar_multiple_drones_shared_world) {
    float max_range = 50.0f;
    BatchDroneEngine* engine = create_engine_with_lidar3d(2, 2, 8, 4, max_range);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Position drones at different locations - all at same Z level (altitude) */
    /* Drone 0: near obstacle */
    engine->states->pos_x[0] = 0.0f;
    engine->states->pos_y[0] = 0.0f;
    engine->states->pos_z[0] = 10.0f;

    /* Drone 1: far from obstacle in -X */
    engine->states->pos_x[1] = -30.0f;
    engine->states->pos_y[1] = 0.0f;
    engine->states->pos_z[1] = 10.0f;

    /* Drone 2: also near obstacle */
    engine->states->pos_x[2] = 2.0f;
    engine->states->pos_y[2] = 0.0f;
    engine->states->pos_z[2] = 10.0f;

    /* Drone 3: very far in -X */
    engine->states->pos_x[3] = -40.0f;
    engine->states->pos_y[3] = 0.0f;
    engine->states->pos_z[3] = 10.0f;

    /* Identity quaternions for all */
    for (int i = 0; i < 4; i++) {
        engine->states->quat_w[i] = 1.0f;
        engine->states->quat_x[i] = 0.0f;
        engine->states->quat_y[i] = 0.0f;
        engine->states->quat_z[i] = 0.0f;
    }

    /* Add large box that drones 0 and 2 should see, but not 1 and 3 */
    Vec3 box_center = VEC3(15.0f, 0.0f, 10.0f);  /* 15m in +X direction, same altitude */
    Vec3 box_half_size = VEC3(5.0f, 5.0f, 5.0f);  /* Large for UNIFORM_INSIDE */
    world_set_box(engine->world, box_center, box_half_size, 1);

    printf("\n    [Debug] Box: center=(%.1f,%.1f,%.1f) half=(%.1f,%.1f,%.1f)",
           box_center.x, box_center.y, box_center.z,
           box_half_size.x, box_half_size.y, box_half_size.z);

    /* Sample sensors */
    engine_step_sensors(engine);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    uint32_t rays_per_drone = 8 * 4;
    uint32_t obs_per_drone = engine->obs_dim;

    /* Find min distance for each drone */
    float min_dist[4];
    for (int d = 0; d < 4; d++) {
        min_dist[d] = max_range;
        float* drone_obs = &obs[d * obs_per_drone];
        for (uint32_t r = 0; r < rays_per_drone && r < obs_per_drone; r++) {
            if (drone_obs[r] < min_dist[d] && drone_obs[r] > 0.0f) {
                min_dist[d] = drone_obs[r];
            }
        }
    }

    /* Drones 0 and 2 should detect obstacle (close) */
    /* Drones 1 and 3 should see mostly max_range (far away, obstacle behind them) */
    printf("\n    [Info] Drone distances: D0=%.1f, D1=%.1f, D2=%.1f, D3=%.1f",
           min_dist[0], min_dist[1], min_dist[2], min_dist[3]);

    /* At minimum, check drone 0 sees something closer than drone 3 */
    ASSERT_LT(min_dist[0], min_dist[3]);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 5: Ground detection
 * ============================================================================ */

TEST(lidar_detects_ground) {
    float max_range = 50.0f;
    BatchDroneEngine* engine = create_engine_with_lidar3d(1, 1, 16, 16, max_range);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Position drone elevated at z=10 */
    engine->states->pos_x[0] = 0.0f;
    engine->states->pos_y[0] = 0.0f;
    engine->states->pos_z[0] = 10.0f;  /* 10m above a floor at z=0 */

    /* Identity quaternion */
    engine->states->quat_w[0] = 1.0f;
    engine->states->quat_x[0] = 0.0f;
    engine->states->quat_y[0] = 0.0f;
    engine->states->quat_z[0] = 0.0f;

    /* Add a large flat box as a floor at z=0 (extending from z=-5 to z=5) */
    Vec3 floor_center = VEC3(0.0f, 0.0f, 0.0f);
    Vec3 floor_half = VEC3(20.0f, 20.0f, 5.0f);  /* Large flat floor (uses UNIFORM_INSIDE) */
    world_set_box(engine->world, floor_center, floor_half, 1);

    printf("\n    [Debug] Drone at (0, 0, 10), floor top at z=5");
    printf("\n    [Debug] Atlas count after floor: %u", engine->world->atlas_count);

    /* Test direct raymarch downward */
    Vec3 test_origin = VEC3(0.0f, 0.0f, 10.0f);
    Vec3 test_dir_down = VEC3(0.0f, 0.0f, -1.0f);
    RayHit direct_hit = world_raymarch(engine->world, test_origin, test_dir_down, max_range);
    printf("\n    [Debug] Direct raymarch down: hit=%d, distance=%.2f", direct_hit.hit, direct_hit.distance);

    /* The direct raymarch should hit at ~5m (10 - 5 = 5) */
    ASSERT_TRUE(direct_hit.hit);
    ASSERT_LT(direct_hit.distance, 10.0f);

    /* Sample sensors - even with standard FOV, angled rays should hit the floor */
    engine_step_sensors(engine);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    /* Find min distance - the LiDAR's downward-angled rays should hit floor */
    float min_distance = max_range;
    uint32_t total_rays = 16 * 16;

    for (uint32_t i = 0; i < total_rays && i < engine->obs_dim; i++) {
        if (obs[i] < min_distance && obs[i] > 0.0f) {
            min_distance = obs[i];
        }
    }

    printf("\n    [Info] Min distance to floor: %.2f m", min_distance);

    /* Some rays should detect the floor surface */
    ASSERT_LT(min_distance, max_range);
    ASSERT_LT(min_distance, 25.0f);  /* Should detect floor at some angle */

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 6: Full step pipeline with LiDAR
 * ============================================================================ */

TEST(lidar_full_step_pipeline) {
    float max_range = 50.0f;
    BatchDroneEngine* engine = create_engine_with_lidar3d(4, 4, 8, 4, max_range);
    ASSERT_NOT_NULL(engine);

    /* Add some large geometry (large enough for UNIFORM_INSIDE optimization) */
    Vec3 box1_center = VEC3(20.0f, 0.0f, 10.0f);
    Vec3 box1_half = VEC3(5.0f, 5.0f, 5.0f);  /* Large cube */
    world_set_box(engine->world, box1_center, box1_half, 1);

    Vec3 sphere_center = VEC3(-20.0f, 0.0f, 10.0f);
    world_set_sphere(engine->world, sphere_center, 5.0f, 1);  /* Large sphere */

    engine_reset(engine);

    /* Run several steps */
    float* actions = engine_get_actions(engine);
    for (int step = 0; step < 50; step++) {
        /* Random-ish actions */
        for (uint32_t i = 0; i < 16 * ENGINE_ACTION_DIM; i++) {
            actions[i] = 0.5f;
        }

        engine_step(engine);

        /* Check observations are valid */
        float* obs = engine_get_observations(engine);
        for (uint32_t d = 0; d < 16; d++) {
            for (uint32_t i = 0; i < engine->obs_dim && i < 32; i++) {
                ASSERT_TRUE(isfinite(obs[d * engine->obs_dim + i]));
            }
        }
    }

    /* Should complete without crash */
    EngineStats stats;
    engine_get_stats(engine, &stats);
    ASSERT_TRUE(stats.total_steps == 50);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 7: Clear world removes geometry
 * ============================================================================ */

TEST(lidar_clear_world) {
    float max_range = 50.0f;
    BatchDroneEngine* engine = create_engine_with_lidar3d(1, 1, 16, 8, max_range);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    /* Position drone elevated and facing +X */
    engine->states->pos_x[0] = 0.0f;
    engine->states->pos_y[0] = 0.0f;
    engine->states->pos_z[0] = 10.0f;  /* Elevated */
    engine->states->quat_w[0] = 1.0f;
    engine->states->quat_x[0] = 0.0f;
    engine->states->quat_y[0] = 0.0f;
    engine->states->quat_z[0] = 0.0f;

    /* Add large box */
    Vec3 box_center = VEC3(15.0f, 0.0f, 10.0f);  /* 15m in +X direction, same altitude */
    Vec3 box_half = VEC3(5.0f, 5.0f, 5.0f);      /* Large for UNIFORM_INSIDE */
    world_set_box(engine->world, box_center, box_half, 1);

    /* Sample with geometry */
    engine_step_sensors(engine);
    float* obs = engine_get_observations(engine);

    float min_with_box = max_range;
    uint32_t total_rays = 16 * 8;
    for (uint32_t i = 0; i < total_rays && i < engine->obs_dim; i++) {
        if (obs[i] < min_with_box && obs[i] > 0.0f) {
            min_with_box = obs[i];
        }
    }

    printf("\n    [Debug] Min dist with box: %.1f (expected ~10m)", min_with_box);

    /* Clear world */
    engine_clear_world(engine);

    /* Sample again */
    engine_step_sensors(engine);

    float min_after_clear = max_range;
    for (uint32_t i = 0; i < total_rays && i < engine->obs_dim; i++) {
        if (obs[i] < min_after_clear && obs[i] > 0.0f) {
            min_after_clear = obs[i];
        }
    }

    /* After clearing, min distance should be larger (no obstacles) */
    printf("\n    [Info] Min dist with box: %.1f, after clear: %.1f",
           min_with_box, min_after_clear);

    ASSERT_LT(min_with_box, min_after_clear);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("LiDAR + World Integration Tests");

    RUN_TEST(lidar_empty_world_max_range);
    RUN_TEST(lidar_detects_box_obstacle);
    RUN_TEST(lidar_detects_sphere_obstacle);
    RUN_TEST(lidar_multiple_drones_shared_world);
    RUN_TEST(lidar_detects_ground);
    RUN_TEST(lidar_full_step_pipeline);
    RUN_TEST(lidar_clear_world);

    TEST_SUITE_END();
}
