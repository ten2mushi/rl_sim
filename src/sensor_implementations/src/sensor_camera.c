/**
 * Camera Sensor Implementations (RGB, Depth, Segmentation)
 *
 * Raymarched camera sensors for visual observation generation.
 *
 * Camera RGB: W*H*3 floats (normalized RGB colors [0,1])
 * Camera Depth: W*H floats (normalized depth [0,1])
 * Camera Segmentation: W*H floats (material IDs as floats)
 *
 * All cameras use precomputed ray directions for efficient batch processing.
 *
 * Performance Targets (64x64):
 * - Camera RGB: <20ms for 1024 drones (with AO)
 * - Camera Depth: <15ms for 1024 drones
 * - Camera Seg: <15ms for 1024 drones
 */

#include "sensor_implementations.h"
#include "world_brick_map.h"
#include <string.h>
#include <math.h>

/* Constants */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Default camera dimensions used when impl is NULL */
#define CAMERA_DEFAULT_WIDTH  64
#define CAMERA_DEFAULT_HEIGHT 64

/* Material colors for RGB rendering (placeholder palette) */
static const float MATERIAL_COLORS[16][3] = {
    {0.5f, 0.5f, 0.5f},   /* 0: Air/sky (gray) */
    {0.4f, 0.3f, 0.2f},   /* 1: Ground (brown) */
    {0.2f, 0.6f, 0.2f},   /* 2: Grass (green) */
    {0.5f, 0.5f, 0.5f},   /* 3: Concrete (gray) */
    {0.3f, 0.2f, 0.1f},   /* 4: Wood (dark brown) */
    {0.7f, 0.1f, 0.1f},   /* 5: Red */
    {0.1f, 0.7f, 0.1f},   /* 6: Green */
    {0.1f, 0.1f, 0.7f},   /* 7: Blue */
    {0.9f, 0.9f, 0.1f},   /* 8: Yellow */
    {0.9f, 0.5f, 0.1f},   /* 9: Orange */
    {0.6f, 0.1f, 0.6f},   /* 10: Purple */
    {0.1f, 0.6f, 0.6f},   /* 11: Cyan */
    {0.9f, 0.9f, 0.9f},   /* 12: White */
    {0.1f, 0.1f, 0.1f},   /* 13: Black */
    {0.8f, 0.6f, 0.4f},   /* 14: Sand */
    {0.3f, 0.3f, 0.8f},   /* 15: Water */
};

/* Sky color for misses */
static const float SKY_COLOR[3] = {0.5f, 0.7f, 0.9f};

/* Precomputed light direction: normalize(0.5, 0.3, 1.0) */
static const Vec3 LIGHT_DIR = {0.43193421f, 0.25916053f, 0.86386843f, 0.0f};

/* ============================================================================
 * Camera Ray Precomputation
 * ============================================================================ */

Vec3* precompute_camera_rays(Arena* arena, uint32_t width, uint32_t height,
                             float fov_horizontal, float fov_vertical) {
    if (arena == NULL || width == 0 || height == 0) {
        return NULL;
    }

    uint32_t total_pixels = width * height;

    /* Allocate with 32-byte alignment */
    Vec3* rays = arena_alloc_aligned(arena, sizeof(Vec3) * total_pixels, 32);
    if (rays == NULL) {
        return NULL;
    }

    /* Compute focal lengths from FOV */
    float focal_x = 1.0f / tanf(fov_horizontal * 0.5f);
    float focal_y = 1.0f / tanf(fov_vertical * 0.5f);

    /* Generate rays for pinhole camera model
     * Camera looks along +X axis in body frame
     * Y is right, Z is up */
    float half_w = (float)width * 0.5f;
    float half_h = (float)height * 0.5f;

    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            uint32_t idx = y * width + x;

            /* Compute normalized image coordinates centered at (0,0) */
            float u = ((float)x - half_w + 0.5f) / half_w;
            float v = ((float)y - half_h + 0.5f) / half_h;

            /* Generate ray direction in camera/body frame */
            /* Camera looks along +X, Y is horizontal, Z is vertical */
            Vec3 dir = VEC3(focal_x, u, -v * focal_x / focal_y);
            rays[idx] = vec3_normalize(dir);
        }
    }

    return rays;
}

/* ============================================================================
 * Shared Camera Init Helper
 * ============================================================================ */

static CameraImpl* camera_init_common(Sensor* sensor, const SensorConfig* config,
                                      Arena* arena) {
    (void)sensor;
    CameraImpl* impl = arena_alloc_type(arena, CameraImpl);
    if (impl == NULL) {
        return NULL;
    }

    impl->width = config->camera.width;
    impl->height = config->camera.height;
    impl->total_pixels = impl->width * impl->height;
    impl->fov_horizontal = config->camera.fov_horizontal;
    impl->fov_vertical = config->camera.fov_vertical;
    impl->near_clip = config->camera.near_clip;
    impl->far_clip = config->camera.far_clip;
    impl->inv_depth_range = 1.0f / (impl->far_clip - impl->near_clip);
    impl->num_classes = config->camera.num_classes;

    /* Precompute ray directions */
    impl->ray_directions = precompute_camera_rays(arena, impl->width, impl->height,
                                                  impl->fov_horizontal, impl->fov_vertical);
    if (impl->ray_directions == NULL) {
        return NULL;
    }

    return impl;
}

/* ============================================================================
 * Camera RGB Implementation
 * ============================================================================ */

static void camera_rgb_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    sensor->impl = camera_init_common(sensor, config, arena);
}

static size_t camera_rgb_get_output_size(const Sensor* sensor) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL) return CAMERA_DEFAULT_WIDTH * CAMERA_DEFAULT_HEIGHT * 3;
    return impl->total_pixels * 3;
}

static const char* camera_rgb_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t camera_rgb_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL) {
        shape[0] = CAMERA_DEFAULT_HEIGHT;
        shape[1] = CAMERA_DEFAULT_WIDTH;
        shape[2] = 3;
    } else {
        shape[0] = impl->height;
        shape[1] = impl->width;
        shape[2] = 3;
    }
    return 3;  /* 3D tensor [H, W, 3] */
}

static void camera_rgb_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL || impl->ray_directions == NULL) {
        memset(output_buffer, 0, ctx->agent_count * CAMERA_DEFAULT_WIDTH * CAMERA_DEFAULT_HEIGHT * 3 * sizeof(float));
        return;
    }

    const RigidBodyStateSOA* drones = ctx->agents;
    const uint32_t* indices = ctx->agent_indices;
    uint32_t count = ctx->agent_count;
    const struct WorldBrickMap* world = ctx->world;
    uint32_t total_pixels = impl->total_pixels;

    /* Fast path: no world - just fill with sky color */
    if (world == NULL) {
        for (uint32_t i = 0; i < count; i++) {
            float* out = &output_buffer[i * total_pixels * 3];
            for (uint32_t p = 0; p < total_pixels; p++) {
                out[p * 3 + 0] = SKY_COLOR[0];
                out[p * 3 + 1] = SKY_COLOR[1];
                out[p * 3 + 2] = SKY_COLOR[2];
            }
        }
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * total_pixels * 3];

        /* Get drone position and orientation */
        Vec3 pos = VEC3(drones->pos_x[d], drones->pos_y[d], drones->pos_z[d]);
        Quat q = QUAT(drones->quat_w[d], drones->quat_x[d],
                      drones->quat_y[d], drones->quat_z[d]);

        /* Apply sensor mounting offset */
        Vec3 sensor_pos = vec3_add(pos, quat_rotate(q, sensor->position_offset));

        /* Process each pixel */
        for (uint32_t p = 0; p < total_pixels; p++) {
            float* pixel = &out[p * 3];

            /* Transform ray direction from body to world frame */
            Vec3 world_dir = quat_rotate(q, impl->ray_directions[p]);

            RayHit hit = world_raymarch(world, sensor_pos, world_dir, impl->far_clip);

            if (hit.hit && hit.distance >= impl->near_clip) {
                /* Get material color */
                uint8_t mat = hit.material;
                if (mat > 15) mat = 0;

                /* Simple diffuse lighting based on normal */
                float diffuse = 0.3f + 0.7f * clampf(vec3_dot(hit.normal, LIGHT_DIR), 0.0f, 1.0f);

                pixel[0] = MATERIAL_COLORS[mat][0] * diffuse;
                pixel[1] = MATERIAL_COLORS[mat][1] * diffuse;
                pixel[2] = MATERIAL_COLORS[mat][2] * diffuse;
            } else {
                /* Sky color */
                pixel[0] = SKY_COLOR[0];
                pixel[1] = SKY_COLOR[1];
                pixel[2] = SKY_COLOR[2];
            }
        }
    }
}

static void camera_rgb_reset(Sensor* sensor, uint32_t agent_index) {
    (void)sensor;
    (void)agent_index;
}

static void camera_rgb_destroy(Sensor* sensor) {
    (void)sensor;
}

const SensorVTable SENSOR_VTABLE_CAMERA_RGB = {
    .name = "Camera-RGB",
    .type = SENSOR_TYPE_CAMERA_RGB,
    .init = camera_rgb_init,
    .get_output_size = camera_rgb_get_output_size,
    .get_output_dtype = camera_rgb_get_output_dtype,
    .get_output_shape = camera_rgb_get_output_shape,
    .batch_sample = camera_rgb_batch_sample,
    .reset = camera_rgb_reset,
    .destroy = camera_rgb_destroy
};

/* ============================================================================
 * Camera Depth Implementation
 * ============================================================================ */

static void camera_depth_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    sensor->impl = camera_init_common(sensor, config, arena);
}

static size_t camera_depth_get_output_size(const Sensor* sensor) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL) return CAMERA_DEFAULT_WIDTH * CAMERA_DEFAULT_HEIGHT;
    return impl->total_pixels;
}

static const char* camera_depth_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t camera_depth_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL) {
        shape[0] = CAMERA_DEFAULT_HEIGHT;
        shape[1] = CAMERA_DEFAULT_WIDTH;
    } else {
        shape[0] = impl->height;
        shape[1] = impl->width;
    }
    return 2;  /* 2D tensor [H, W] */
}

static void camera_depth_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL || impl->ray_directions == NULL) {
        /* Fill with 1.0 (max depth) */
        for (uint32_t i = 0; i < ctx->agent_count * CAMERA_DEFAULT_WIDTH * CAMERA_DEFAULT_HEIGHT; i++) {
            output_buffer[i] = 1.0f;
        }
        return;
    }

    const RigidBodyStateSOA* drones = ctx->agents;
    const uint32_t* indices = ctx->agent_indices;
    uint32_t count = ctx->agent_count;
    const struct WorldBrickMap* world = ctx->world;
    uint32_t total_pixels = impl->total_pixels;

    /* Fast path: no world - fill with max depth (1.0) */
    if (world == NULL) {
        for (uint32_t i = 0; i < count * total_pixels; i++) {
            output_buffer[i] = 1.0f;
        }
        return;
    }

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * total_pixels];

        /* Get drone position and orientation */
        Vec3 pos = VEC3(drones->pos_x[d], drones->pos_y[d], drones->pos_z[d]);
        Quat q = QUAT(drones->quat_w[d], drones->quat_x[d],
                      drones->quat_y[d], drones->quat_z[d]);

        /* Apply sensor mounting offset */
        Vec3 sensor_pos = vec3_add(pos, quat_rotate(q, sensor->position_offset));

        /* Process each pixel */
        for (uint32_t p = 0; p < total_pixels; p++) {
            /* Transform ray direction from body to world frame */
            Vec3 world_dir = quat_rotate(q, impl->ray_directions[p]);

            RayHit hit = world_raymarch(world, sensor_pos, world_dir, impl->far_clip);

            float normalized_depth = 1.0f;  /* Default: max depth */
            if (hit.hit && hit.distance >= impl->near_clip) {
                /* Normalize depth to [0, 1] */
                normalized_depth = (hit.distance - impl->near_clip) * impl->inv_depth_range;
                normalized_depth = clampf(normalized_depth, 0.0f, 1.0f);
            }

            out[p] = normalized_depth;
        }
    }
}

static void camera_depth_reset(Sensor* sensor, uint32_t agent_index) {
    (void)sensor;
    (void)agent_index;
}

static void camera_depth_destroy(Sensor* sensor) {
    (void)sensor;
}

const SensorVTable SENSOR_VTABLE_CAMERA_DEPTH = {
    .name = "Camera-Depth",
    .type = SENSOR_TYPE_CAMERA_DEPTH,
    .init = camera_depth_init,
    .get_output_size = camera_depth_get_output_size,
    .get_output_dtype = camera_depth_get_output_dtype,
    .get_output_shape = camera_depth_get_output_shape,
    .batch_sample = camera_depth_batch_sample,
    .reset = camera_depth_reset,
    .destroy = camera_depth_destroy
};

/* ============================================================================
 * Camera Segmentation Implementation
 * ============================================================================ */

static void camera_seg_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    sensor->impl = camera_init_common(sensor, config, arena);
}

static size_t camera_seg_get_output_size(const Sensor* sensor) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL) return CAMERA_DEFAULT_WIDTH * CAMERA_DEFAULT_HEIGHT;
    return impl->total_pixels;
}

static const char* camera_seg_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t camera_seg_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL) {
        shape[0] = CAMERA_DEFAULT_HEIGHT;
        shape[1] = CAMERA_DEFAULT_WIDTH;
    } else {
        shape[0] = impl->height;
        shape[1] = impl->width;
    }
    return 2;  /* 2D tensor [H, W] */
}

static void camera_seg_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    CameraImpl* impl = (CameraImpl*)sensor->impl;
    if (impl == NULL || impl->ray_directions == NULL) {
        memset(output_buffer, 0, ctx->agent_count * CAMERA_DEFAULT_WIDTH * CAMERA_DEFAULT_HEIGHT * sizeof(float));
        return;
    }

    const RigidBodyStateSOA* drones = ctx->agents;
    const uint32_t* indices = ctx->agent_indices;
    uint32_t count = ctx->agent_count;
    const struct WorldBrickMap* world = ctx->world;
    uint32_t total_pixels = impl->total_pixels;

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * total_pixels];

        /* Get drone position and orientation */
        Vec3 pos = VEC3(drones->pos_x[d], drones->pos_y[d], drones->pos_z[d]);
        Quat q = QUAT(drones->quat_w[d], drones->quat_x[d],
                      drones->quat_y[d], drones->quat_z[d]);

        /* Apply sensor mounting offset */
        Vec3 sensor_pos = vec3_add(pos, quat_rotate(q, sensor->position_offset));

        /* Process each pixel */
        for (uint32_t p = 0; p < total_pixels; p++) {
            /* Transform ray direction from body to world frame */
            Vec3 world_dir = quat_rotate(q, impl->ray_directions[p]);

            float material_id = 0.0f;  /* Default: sky/air (class 0) */

            if (world != NULL) {
                RayHit hit = world_raymarch(world, sensor_pos, world_dir, impl->far_clip);

                if (hit.hit && hit.distance >= impl->near_clip) {
                    /* Output material ID as float */
                    material_id = (float)hit.material;
                }
            }

            out[p] = material_id;
        }
    }
}

static void camera_seg_reset(Sensor* sensor, uint32_t agent_index) {
    (void)sensor;
    (void)agent_index;
}

static void camera_seg_destroy(Sensor* sensor) {
    (void)sensor;
}

const SensorVTable SENSOR_VTABLE_CAMERA_SEGMENTATION = {
    .name = "Camera-Seg",
    .type = SENSOR_TYPE_CAMERA_SEGMENTATION,
    .init = camera_seg_init,
    .get_output_size = camera_seg_get_output_size,
    .get_output_dtype = camera_seg_get_output_dtype,
    .get_output_shape = camera_seg_get_output_shape,
    .batch_sample = camera_seg_batch_sample,
    .reset = camera_seg_reset,
    .destroy = camera_seg_destroy
};
