/**
 * Camera Sensor Ground Truth Test
 *
 * Quantitative validation of camera sensor pipeline against analytical
 * ground truth. Places cameras at known positions facing known geometry
 * (spheres/boxes via world_set_sphere/world_set_box), computes analytical
 * ray-sphere/ray-box intersections, and compares pixel-by-pixel against
 * the sensor output.
 *
 * With null noise config, the sensor pipeline should produce a faithful
 * reconstruction of the world. Any deviation beyond SDF quantization
 * tolerance is a bug.
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
 * Constants
 * ============================================================================ */

#define CAM_W 32
#define CAM_H 32
#define CAM_FOV ((float)M_PI / 2.0f)  /* 90 degrees */
#define CAM_MAX_RANGE 100.0f
#define CAM_NEAR_CLIP 0.1f

/* Geometry placement */
#define SPHERE_CX 20.0f
#define SPHERE_CY 0.0f
#define SPHERE_CZ 10.0f
#define SPHERE_R  5.0f
#define SPHERE_MAT 1

#define BOX_CX 20.0f
#define BOX_CY 0.0f
#define BOX_CZ 10.0f
#define BOX_HALF 5.0f
#define BOX_MAT 3

/* Camera default position */
#define CAM_PX 0.0f
#define CAM_PY 0.0f
#define CAM_PZ 10.0f

/* Tolerances */
#define DEPTH_NORM_TOL 0.003f   /* normalized depth (~0.3m at 100m range) */
#define RGB_TOL        0.10f    /* per-channel (SDF normal via finite differences) */
#define SKY_TOL        1e-5f    /* sky values are exact */
#define EDGE_TOLERANCE 4        /* max phantom/missed hits at silhouette edges */

/* Sky color (must match sensor_camera.c) */
static const float SKY_COLOR[3] = {0.5f, 0.7f, 0.9f};

/* Material colors (must match sensor_camera.c) */
static const float MATERIAL_COLORS[16][3] = {
    {0.5f, 0.5f, 0.5f},   /* 0 */
    {0.4f, 0.3f, 0.2f},   /* 1 */
    {0.2f, 0.6f, 0.2f},   /* 2 */
    {0.5f, 0.5f, 0.5f},   /* 3 */
    {0.3f, 0.2f, 0.1f},   /* 4 */
    {0.7f, 0.1f, 0.1f},   /* 5 */
    {0.1f, 0.7f, 0.1f},   /* 6 */
    {0.1f, 0.1f, 0.7f},   /* 7 */
    {0.9f, 0.9f, 0.1f},   /* 8 */
    {0.9f, 0.5f, 0.1f},   /* 9 */
    {0.6f, 0.1f, 0.6f},   /* 10 */
    {0.1f, 0.6f, 0.6f},   /* 11 */
    {0.9f, 0.9f, 0.9f},   /* 12 */
    {0.1f, 0.1f, 0.1f},   /* 13 */
    {0.8f, 0.6f, 0.4f},   /* 14 */
    {0.3f, 0.3f, 0.8f},   /* 15 */
};

/* Light direction (must match sensor_camera.c) */
static const Vec3 LIGHT_DIR = {0.43193421f, 0.25916053f, 0.86386843f, 0.0f};

/* ============================================================================
 * Stats
 * ============================================================================ */

typedef struct {
    uint32_t total_pixels;
    uint32_t expected_hits;
    uint32_t actual_hits;
    uint32_t phantom_hits;   /* sensor hit where analytical ray misses */
    uint32_t missed_hits;    /* sensor miss where analytical ray hits */
    float    max_depth_error; /* max |observed - expected| (normalized) */
    double   sum_depth_error;
    float    max_rgb_error;   /* max per-channel |observed - expected| */
} GroundTruthStats;

static void stats_print(const GroundTruthStats* s, const char* label) {
    float mean_depth_err = (s->expected_hits > 0)
        ? (float)(s->sum_depth_error / s->expected_hits) : 0.0f;
    printf("\n    [%s] pixels=%u hits_expected=%u hits_actual=%u "
           "phantom=%u missed=%u max_depth_err=%.6f mean_depth_err=%.6f max_rgb_err=%.4f",
           label, s->total_pixels, s->expected_hits, s->actual_hits,
           s->phantom_hits, s->missed_hits, (double)s->max_depth_error,
           (double)mean_depth_err, (double)s->max_rgb_error);
}

/* ============================================================================
 * Analytical Intersection Functions
 * ============================================================================ */

static bool ray_sphere_intersect(Vec3 origin, Vec3 dir,
                                  Vec3 center, float radius, float* t) {
    Vec3 oc = vec3_sub(origin, center);
    float b = vec3_dot(oc, dir);
    float c = vec3_dot(oc, oc) - radius * radius;
    float disc = b * b - c;
    if (disc < 0.0f) return false;
    float sq = sqrtf(disc);
    float t0 = -b - sq;
    float t1 = -b + sq;
    if (t0 > 0.001f) { *t = t0; return true; }
    if (t1 > 0.001f) { *t = t1; return true; }
    return false;
}

static bool ray_box_intersect(Vec3 origin, Vec3 dir,
                               Vec3 box_min, Vec3 box_max, float* t) {
    float tmin = -1e30f, tmax = 1e30f;

    /* X slab */
    if (fabsf(dir.x) > 1e-8f) {
        float inv = 1.0f / dir.x;
        float t1 = (box_min.x - origin.x) * inv;
        float t2 = (box_max.x - origin.x) * inv;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
    } else {
        if (origin.x < box_min.x || origin.x > box_max.x) return false;
    }

    /* Y slab */
    if (fabsf(dir.y) > 1e-8f) {
        float inv = 1.0f / dir.y;
        float t1 = (box_min.y - origin.y) * inv;
        float t2 = (box_max.y - origin.y) * inv;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
    } else {
        if (origin.y < box_min.y || origin.y > box_max.y) return false;
    }

    /* Z slab */
    if (fabsf(dir.z) > 1e-8f) {
        float inv = 1.0f / dir.z;
        float t1 = (box_min.z - origin.z) * inv;
        float t2 = (box_max.z - origin.z) * inv;
        if (t1 > t2) { float tmp = t1; t1 = t2; t2 = tmp; }
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
    } else {
        if (origin.z < box_min.z || origin.z > box_max.z) return false;
    }

    if (tmin > tmax) return false;
    if (tmax < 0.001f) return false;
    *t = (tmin > 0.001f) ? tmin : tmax;
    return true;
}

/* ============================================================================
 * Ray Direction Computation (matches precompute_camera_rays in sensor_camera.c)
 * ============================================================================ */

static Vec3 compute_ray_dir(uint32_t px, uint32_t py,
                             uint32_t W, uint32_t H,
                             float fov_h, float fov_v) {
    float focal_x = 1.0f / tanf(fov_h * 0.5f);
    float focal_y = 1.0f / tanf(fov_v * 0.5f);
    float half_w = (float)W * 0.5f;
    float half_h = (float)H * 0.5f;

    float u = ((float)px - half_w + 0.5f) / half_w;
    float v = ((float)py - half_h + 0.5f) / half_h;

    Vec3 dir = VEC3(focal_x, u, -v * focal_x / focal_y);
    return vec3_normalize(dir);
}

/* ============================================================================
 * Expected RGB Shading (matches camera_rgb_batch_sample in sensor_camera.c)
 * ============================================================================ */

static void expected_rgb(uint8_t material, Vec3 normal, float out[3]) {
    uint8_t mat = (material > 15) ? 0 : material;
    float diffuse = 0.3f + 0.7f * clampf(vec3_dot(normal, LIGHT_DIR), 0.0f, 1.0f);
    out[0] = MATERIAL_COLORS[mat][0] * diffuse;
    out[1] = MATERIAL_COLORS[mat][1] * diffuse;
    out[2] = MATERIAL_COLORS[mat][2] * diffuse;
}

/* ============================================================================
 * Helper: Create engine with camera sensor
 * ============================================================================ */

static BatchEngine* create_engine_with_camera(SensorType type) {
    EngineConfig cfg = engine_config_default();
    cfg.num_envs = 1;
    cfg.agents_per_env = 1;
    cfg.seed = 42;
    cfg.persistent_arena_size = 256 * 1024 * 1024;
    cfg.frame_arena_size = 64 * 1024 * 1024;

    SensorConfig cam = sensor_config_camera(CAM_W, CAM_H, CAM_FOV, CAM_MAX_RANGE);
    cam.type = type;

    if (engine_config_add_sensor(&cfg, &cam) != 0) {
        return NULL;
    }

    char err[ENGINE_ERROR_MSG_SIZE];
    return engine_create(&cfg, err);
}

/* ============================================================================
 * Helper: Position drone and sample sensors
 * ============================================================================ */

static void position_drone_and_sample(BatchEngine* engine,
                                       Vec3 pos, Quat quat) {
    engine->states->rigid_body.pos_x[0] = pos.x;
    engine->states->rigid_body.pos_y[0] = pos.y;
    engine->states->rigid_body.pos_z[0] = pos.z;
    engine->states->rigid_body.quat_w[0] = quat.w;
    engine->states->rigid_body.quat_x[0] = quat.x;
    engine->states->rigid_body.quat_y[0] = quat.y;
    engine->states->rigid_body.quat_z[0] = quat.z;

    engine_step_sensors(engine);
}

/* ============================================================================
 * Helper: Compute FOV_V (must match sensor_config_camera)
 * ============================================================================ */

static float compute_fov_v(float fov_h, uint32_t W, uint32_t H) {
    float aspect = (float)H / (float)W;
    return 2.0f * atanf(aspect * tanf(fov_h * 0.5f));
}

/* ============================================================================
 * Helper: Normalize depth (must match camera_depth_batch_sample)
 * ============================================================================ */

static float normalize_depth(float distance) {
    float norm = (distance - CAM_NEAR_CLIP) / (CAM_MAX_RANGE - CAM_NEAR_CLIP);
    if (norm < 0.0f) norm = 0.0f;
    if (norm > 1.0f) norm = 1.0f;
    return norm;
}

/* ============================================================================
 * Test 1: depth_sphere_front_view
 *
 * Camera at (0,0,10) identity quat, sphere at (20,0,10) r=5.
 * Center pixel depth ~ 15m -> normalized ~ 0.149.
 * ============================================================================ */

TEST(depth_sphere_front_view) {
    BatchEngine* engine = create_engine_with_camera(SENSOR_TYPE_CAMERA_DEPTH);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    Vec3 sphere_center = VEC3(SPHERE_CX, SPHERE_CY, SPHERE_CZ);
    world_set_sphere(engine->world, sphere_center, SPHERE_R, SPHERE_MAT);

    Vec3 cam_pos = VEC3(CAM_PX, CAM_PY, CAM_PZ);
    Quat cam_quat = QUAT(1, 0, 0, 0);
    position_drone_and_sample(engine, cam_pos, cam_quat);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    float fov_v = compute_fov_v(CAM_FOV, CAM_W, CAM_H);
    GroundTruthStats stats = {0};
    stats.total_pixels = CAM_W * CAM_H;

    for (uint32_t py = 0; py < CAM_H; py++) {
        for (uint32_t px = 0; px < CAM_W; px++) {
            uint32_t idx = py * CAM_W + px;

            /* Compute analytical ray (body frame = world frame, identity quat) */
            Vec3 ray_dir = compute_ray_dir(px, py, CAM_W, CAM_H, CAM_FOV, fov_v);

            float t_analytical;
            bool analytical_hit = ray_sphere_intersect(
                cam_pos, ray_dir, sphere_center, SPHERE_R, &t_analytical);

            float observed = obs[idx];
            bool sensor_hit = (observed < 1.0f - SKY_TOL);

            if (analytical_hit) {
                stats.expected_hits++;
                float expected_norm = normalize_depth(t_analytical);

                if (sensor_hit) {
                    stats.actual_hits++;
                    float err = fabsf(observed - expected_norm);
                    if (err > stats.max_depth_error) stats.max_depth_error = err;
                    stats.sum_depth_error += err;
                } else {
                    stats.missed_hits++;
                }
            } else {
                if (sensor_hit) {
                    stats.phantom_hits++;
                    stats.actual_hits++;
                }
                /* Sky pixel: must be 1.0 */
                if (!sensor_hit) {
                    ASSERT_FLOAT_NEAR(observed, 1.0f, SKY_TOL);
                }
            }
        }
    }

    stats_print(&stats, "depth_sphere_front");

    ASSERT_LE(stats.phantom_hits, (uint32_t)EDGE_TOLERANCE);
    ASSERT_LE(stats.missed_hits, (uint32_t)EDGE_TOLERANCE);
    ASSERT_LT(stats.max_depth_error, DEPTH_NORM_TOL);
    ASSERT_GT(stats.actual_hits, 50u);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 2: rgb_sphere_shading
 *
 * Same geometry, CAMERA_RGB. For hit pixels: compare analytical shading.
 * Sky pixels must be (0.5, 0.7, 0.9).
 * ============================================================================ */

TEST(rgb_sphere_shading) {
    BatchEngine* engine = create_engine_with_camera(SENSOR_TYPE_CAMERA_RGB);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    Vec3 sphere_center = VEC3(SPHERE_CX, SPHERE_CY, SPHERE_CZ);
    world_set_sphere(engine->world, sphere_center, SPHERE_R, SPHERE_MAT);

    Vec3 cam_pos = VEC3(CAM_PX, CAM_PY, CAM_PZ);
    Quat cam_quat = QUAT(1, 0, 0, 0);
    position_drone_and_sample(engine, cam_pos, cam_quat);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    float fov_v = compute_fov_v(CAM_FOV, CAM_W, CAM_H);
    GroundTruthStats stats = {0};
    stats.total_pixels = CAM_W * CAM_H;

    for (uint32_t py = 0; py < CAM_H; py++) {
        for (uint32_t px = 0; px < CAM_W; px++) {
            uint32_t idx = py * CAM_W + px;

            Vec3 ray_dir = compute_ray_dir(px, py, CAM_W, CAM_H, CAM_FOV, fov_v);

            float t_analytical;
            bool analytical_hit = ray_sphere_intersect(
                cam_pos, ray_dir, sphere_center, SPHERE_R, &t_analytical);

            float obs_r = obs[idx * 3 + 0];
            float obs_g = obs[idx * 3 + 1];
            float obs_b = obs[idx * 3 + 2];

            /* Detect sensor hit: any channel differs from sky */
            bool sensor_hit = (fabsf(obs_r - SKY_COLOR[0]) > SKY_TOL ||
                               fabsf(obs_g - SKY_COLOR[1]) > SKY_TOL ||
                               fabsf(obs_b - SKY_COLOR[2]) > SKY_TOL);

            if (analytical_hit) {
                stats.expected_hits++;

                if (sensor_hit) {
                    stats.actual_hits++;

                    /* Compute expected shading */
                    Vec3 hit_pos = vec3_add(cam_pos, vec3_scale(ray_dir, t_analytical));
                    Vec3 normal = vec3_normalize(vec3_sub(hit_pos, sphere_center));
                    float exp_rgb[3];
                    expected_rgb(SPHERE_MAT, normal, exp_rgb);

                    float err_r = fabsf(obs_r - exp_rgb[0]);
                    float err_g = fabsf(obs_g - exp_rgb[1]);
                    float err_b = fabsf(obs_b - exp_rgb[2]);
                    float max_ch = err_r;
                    if (err_g > max_ch) max_ch = err_g;
                    if (err_b > max_ch) max_ch = err_b;

                    if (max_ch > stats.max_rgb_error)
                        stats.max_rgb_error = max_ch;
                } else {
                    stats.missed_hits++;
                }
            } else {
                if (sensor_hit) {
                    stats.phantom_hits++;
                    stats.actual_hits++;
                } else {
                    /* Sky pixel: must be exact */
                    ASSERT_FLOAT_NEAR(obs_r, SKY_COLOR[0], SKY_TOL);
                    ASSERT_FLOAT_NEAR(obs_g, SKY_COLOR[1], SKY_TOL);
                    ASSERT_FLOAT_NEAR(obs_b, SKY_COLOR[2], SKY_TOL);
                }
            }
        }
    }

    stats_print(&stats, "rgb_sphere_shading");

    ASSERT_LE(stats.phantom_hits, (uint32_t)EDGE_TOLERANCE);
    ASSERT_LE(stats.missed_hits, (uint32_t)EDGE_TOLERANCE);
    ASSERT_LT(stats.max_rgb_error, RGB_TOL);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 3: sky_no_false_hits
 *
 * Camera at (0,0,10) facing -X. Sphere at +X. All 1024 pixels = sky.
 * ============================================================================ */

TEST(sky_no_false_hits) {
    /* Test depth camera */
    {
        BatchEngine* engine = create_engine_with_camera(SENSOR_TYPE_CAMERA_DEPTH);
        ASSERT_NOT_NULL(engine);

        engine_reset(engine);

        Vec3 sphere_center = VEC3(SPHERE_CX, SPHERE_CY, SPHERE_CZ);
        world_set_sphere(engine->world, sphere_center, SPHERE_R, SPHERE_MAT);

        /* Face -X via 180 rotation around Z */
        Vec3 cam_pos = VEC3(CAM_PX, CAM_PY, CAM_PZ);
        Quat cam_quat = quat_from_axis_angle(VEC3(0, 0, 1), (float)M_PI);
        position_drone_and_sample(engine, cam_pos, cam_quat);

        float* obs = engine_get_observations(engine);
        ASSERT_NOT_NULL(obs);

        uint32_t phantom_count = 0;
        for (uint32_t i = 0; i < CAM_W * CAM_H; i++) {
            if (fabsf(obs[i] - 1.0f) > SKY_TOL) {
                phantom_count++;
            }
        }

        printf("\n    [sky_depth] phantoms=%u / %u", phantom_count, CAM_W * CAM_H);
        ASSERT_EQ(phantom_count, 0u);

        engine_destroy(engine);
    }

    /* Test RGB camera */
    {
        BatchEngine* engine = create_engine_with_camera(SENSOR_TYPE_CAMERA_RGB);
        ASSERT_NOT_NULL(engine);

        engine_reset(engine);

        Vec3 sphere_center = VEC3(SPHERE_CX, SPHERE_CY, SPHERE_CZ);
        world_set_sphere(engine->world, sphere_center, SPHERE_R, SPHERE_MAT);

        Vec3 cam_pos = VEC3(CAM_PX, CAM_PY, CAM_PZ);
        Quat cam_quat = quat_from_axis_angle(VEC3(0, 0, 1), (float)M_PI);
        position_drone_and_sample(engine, cam_pos, cam_quat);

        float* obs = engine_get_observations(engine);
        ASSERT_NOT_NULL(obs);

        uint32_t phantom_count = 0;
        for (uint32_t i = 0; i < CAM_W * CAM_H; i++) {
            if (fabsf(obs[i * 3 + 0] - SKY_COLOR[0]) > SKY_TOL ||
                fabsf(obs[i * 3 + 1] - SKY_COLOR[1]) > SKY_TOL ||
                fabsf(obs[i * 3 + 2] - SKY_COLOR[2]) > SKY_TOL) {
                phantom_count++;
            }
        }

        printf("\n    [sky_rgb] phantoms=%u / %u", phantom_count, CAM_W * CAM_H);
        ASSERT_EQ(phantom_count, 0u);

        engine_destroy(engine);
    }

    return 0;
}

/* ============================================================================
 * Test 4: depth_sphere_oblique_view
 *
 * Camera at (0,20,10) facing sphere via quat_from_axis_angle(Z, -pi/4).
 * Distance to sphere center = 20*sqrt(2) ~ 28.28, center ray hits ~23.28m.
 * ============================================================================ */

TEST(depth_sphere_oblique_view) {
    BatchEngine* engine = create_engine_with_camera(SENSOR_TYPE_CAMERA_DEPTH);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    Vec3 sphere_center = VEC3(SPHERE_CX, SPHERE_CY, SPHERE_CZ);
    world_set_sphere(engine->world, sphere_center, SPHERE_R, SPHERE_MAT);

    Vec3 cam_pos = VEC3(0.0f, 20.0f, 10.0f);
    Quat cam_quat = quat_from_axis_angle(VEC3(0, 0, 1), -(float)M_PI / 4.0f);
    position_drone_and_sample(engine, cam_pos, cam_quat);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    float fov_v = compute_fov_v(CAM_FOV, CAM_W, CAM_H);
    GroundTruthStats stats = {0};
    stats.total_pixels = CAM_W * CAM_H;

    for (uint32_t py = 0; py < CAM_H; py++) {
        for (uint32_t px = 0; px < CAM_W; px++) {
            uint32_t idx = py * CAM_W + px;

            Vec3 body_ray = compute_ray_dir(px, py, CAM_W, CAM_H, CAM_FOV, fov_v);
            Vec3 world_ray = quat_rotate(cam_quat, body_ray);

            float t_analytical;
            bool analytical_hit = ray_sphere_intersect(
                cam_pos, world_ray, sphere_center, SPHERE_R, &t_analytical);

            float observed = obs[idx];
            bool sensor_hit = (observed < 1.0f - SKY_TOL);

            if (analytical_hit) {
                stats.expected_hits++;
                float expected_norm = normalize_depth(t_analytical);

                if (sensor_hit) {
                    stats.actual_hits++;
                    float err = fabsf(observed - expected_norm);
                    if (err > stats.max_depth_error) stats.max_depth_error = err;
                    stats.sum_depth_error += err;
                } else {
                    stats.missed_hits++;
                }
            } else {
                if (sensor_hit) {
                    stats.phantom_hits++;
                    stats.actual_hits++;
                }
            }
        }
    }

    stats_print(&stats, "depth_sphere_oblique");

    ASSERT_LE(stats.phantom_hits, (uint32_t)EDGE_TOLERANCE);
    ASSERT_LE(stats.missed_hits, (uint32_t)EDGE_TOLERANCE);
    ASSERT_LT(stats.max_depth_error, DEPTH_NORM_TOL);
    ASSERT_GT(stats.actual_hits, 20u); /* fewer hits at oblique angle */

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 5: depth_box_front_view
 *
 * Camera at (0,0,10) identity quat, box at (20,0,10) half (5,5,5).
 * Box front face at x=15, distance 15m. Exercises axis-aligned SDF paths.
 * ============================================================================ */

TEST(depth_box_front_view) {
    BatchEngine* engine = create_engine_with_camera(SENSOR_TYPE_CAMERA_DEPTH);
    ASSERT_NOT_NULL(engine);

    engine_reset(engine);

    Vec3 box_center = VEC3(BOX_CX, BOX_CY, BOX_CZ);
    Vec3 box_half = VEC3(BOX_HALF, BOX_HALF, BOX_HALF);
    world_set_box(engine->world, box_center, box_half, BOX_MAT);

    Vec3 cam_pos = VEC3(CAM_PX, CAM_PY, CAM_PZ);
    Quat cam_quat = QUAT(1, 0, 0, 0);
    position_drone_and_sample(engine, cam_pos, cam_quat);

    float* obs = engine_get_observations(engine);
    ASSERT_NOT_NULL(obs);

    float fov_v = compute_fov_v(CAM_FOV, CAM_W, CAM_H);
    Vec3 box_min = VEC3(BOX_CX - BOX_HALF, BOX_CY - BOX_HALF, BOX_CZ - BOX_HALF);
    Vec3 box_max = VEC3(BOX_CX + BOX_HALF, BOX_CY + BOX_HALF, BOX_CZ + BOX_HALF);

    GroundTruthStats stats = {0};
    stats.total_pixels = CAM_W * CAM_H;

    for (uint32_t py = 0; py < CAM_H; py++) {
        for (uint32_t px = 0; px < CAM_W; px++) {
            uint32_t idx = py * CAM_W + px;

            Vec3 ray_dir = compute_ray_dir(px, py, CAM_W, CAM_H, CAM_FOV, fov_v);

            float t_analytical;
            bool analytical_hit = ray_box_intersect(
                cam_pos, ray_dir, box_min, box_max, &t_analytical);

            float observed = obs[idx];
            bool sensor_hit = (observed < 1.0f - SKY_TOL);

            if (analytical_hit) {
                stats.expected_hits++;
                float expected_norm = normalize_depth(t_analytical);

                if (sensor_hit) {
                    stats.actual_hits++;
                    float err = fabsf(observed - expected_norm);
                    if (err > stats.max_depth_error) stats.max_depth_error = err;
                    stats.sum_depth_error += err;
                } else {
                    stats.missed_hits++;
                }
            } else {
                if (sensor_hit) {
                    stats.phantom_hits++;
                    stats.actual_hits++;
                }
                if (!sensor_hit) {
                    ASSERT_FLOAT_NEAR(observed, 1.0f, SKY_TOL);
                }
            }
        }
    }

    stats_print(&stats, "depth_box_front");

    ASSERT_LE(stats.phantom_hits, (uint32_t)EDGE_TOLERANCE);
    ASSERT_LE(stats.missed_hits, (uint32_t)EDGE_TOLERANCE);
    ASSERT_LT(stats.max_depth_error, DEPTH_NORM_TOL);
    ASSERT_GT(stats.actual_hits, 50u);

    engine_destroy(engine);
    return 0;
}

/* ============================================================================
 * Test 6: multi_view_consistency
 *
 * 4 viewpoints orbiting sphere center at distance 20. By symmetry, all 4
 * views see the same sphere profile. Validate hit counts and mean depths
 * are consistent, plus full per-pixel validation.
 * ============================================================================ */

TEST(multi_view_consistency) {
    float fov_v = compute_fov_v(CAM_FOV, CAM_W, CAM_H);

    Vec3 sphere_center = VEC3(SPHERE_CX, SPHERE_CY, SPHERE_CZ);

    /* 4 viewpoints, each 20m from sphere center */
    struct {
        Vec3 pos;
        Quat quat;
        const char* label;
    } views[4] = {
        { VEC3(0.0f, 0.0f, 10.0f),   QUAT(1, 0, 0, 0),  "+X" },
        { VEC3(20.0f, 20.0f, 10.0f),
          quat_from_axis_angle(VEC3(0, 0, 1), -(float)M_PI / 2.0f), "-Y" },
        { VEC3(40.0f, 0.0f, 10.0f),
          quat_from_axis_angle(VEC3(0, 0, 1), (float)M_PI), "-X" },
        { VEC3(20.0f, -20.0f, 10.0f),
          quat_from_axis_angle(VEC3(0, 0, 1), (float)M_PI / 2.0f), "+Y" },
    };

    uint32_t hit_counts[4] = {0};
    double   mean_depths[4] = {0};
    float    max_errors[4] = {0};

    for (int v = 0; v < 4; v++) {
        BatchEngine* engine = create_engine_with_camera(SENSOR_TYPE_CAMERA_DEPTH);
        ASSERT_NOT_NULL(engine);

        engine_reset(engine);
        world_set_sphere(engine->world, sphere_center, SPHERE_R, SPHERE_MAT);
        position_drone_and_sample(engine, views[v].pos, views[v].quat);

        float* obs = engine_get_observations(engine);
        ASSERT_NOT_NULL(obs);

        GroundTruthStats stats = {0};
        stats.total_pixels = CAM_W * CAM_H;

        for (uint32_t py = 0; py < CAM_H; py++) {
            for (uint32_t px = 0; px < CAM_W; px++) {
                uint32_t idx = py * CAM_W + px;

                Vec3 body_ray = compute_ray_dir(px, py, CAM_W, CAM_H, CAM_FOV, fov_v);
                Vec3 world_ray = quat_rotate(views[v].quat, body_ray);

                float t_analytical;
                bool analytical_hit = ray_sphere_intersect(
                    views[v].pos, world_ray, sphere_center, SPHERE_R, &t_analytical);

                float observed = obs[idx];
                bool sensor_hit = (observed < 1.0f - SKY_TOL);

                if (analytical_hit) {
                    stats.expected_hits++;
                    float expected_norm = normalize_depth(t_analytical);

                    if (sensor_hit) {
                        stats.actual_hits++;
                        float err = fabsf(observed - expected_norm);
                        if (err > stats.max_depth_error) stats.max_depth_error = err;
                        stats.sum_depth_error += err;
                    } else {
                        stats.missed_hits++;
                    }
                } else {
                    if (sensor_hit) {
                        stats.phantom_hits++;
                        stats.actual_hits++;
                    }
                }
            }
        }

        stats_print(&stats, views[v].label);

        /* Per-view validation */
        ASSERT_LE(stats.phantom_hits, (uint32_t)EDGE_TOLERANCE);
        ASSERT_LE(stats.missed_hits, (uint32_t)EDGE_TOLERANCE);
        ASSERT_LT(stats.max_depth_error, DEPTH_NORM_TOL);

        hit_counts[v] = stats.actual_hits;
        mean_depths[v] = (stats.actual_hits > 0)
            ? stats.sum_depth_error / stats.actual_hits : 0.0;
        max_errors[v] = stats.max_depth_error;

        engine_destroy(engine);
    }

    /* Cross-view consistency */
    for (int v = 1; v < 4; v++) {
        int diff = (int)hit_counts[v] - (int)hit_counts[0];
        if (diff < 0) diff = -diff;
        printf("\n    [consistency] view %d vs 0: hit_diff=%d mean_depth_diff=%.6f",
               v, diff, fabs(mean_depths[v] - mean_depths[0]));
        ASSERT_LE((uint32_t)diff, 4u);
        ASSERT_LT((float)fabs(mean_depths[v] - mean_depths[0]), 0.005f);
    }

    return 0;
}

/* ============================================================================
 * Main
 * ============================================================================ */

int main(void) {
    TEST_SUITE_BEGIN("Camera Ground Truth Tests");

    RUN_TEST(depth_sphere_front_view);
    RUN_TEST(rgb_sphere_shading);
    RUN_TEST(sky_no_false_hits);
    RUN_TEST(depth_sphere_oblique_view);
    RUN_TEST(depth_box_front_view);
    RUN_TEST(multi_view_consistency);

    TEST_SUITE_END();
}
