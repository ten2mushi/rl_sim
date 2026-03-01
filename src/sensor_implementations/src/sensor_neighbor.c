/**
 * Neighbor Sensor Implementation
 *
 * Detects K nearest neighbor drones and outputs their relative positions
 * and distances. Useful for swarm behaviors and collision avoidance.
 *
 * Output: K*4 floats (dx, dy, dz, distance) for each neighbor
 *
 * If fewer than K neighbors are found within max_range, remaining slots
 * are filled with zeros and distance = max_range.
 *
 * Performance Target: <1ms for 1024 drones (K=5)
 */

#include "sensor_implementations.h"
#include "collision_system.h"
#include <string.h>
#include <math.h>

/* Default neighbor count used when impl is NULL */
#define NEIGHBOR_DEFAULT_K 5

/* ============================================================================
 * Neighbor Sensor Implementation
 * ============================================================================ */

static void neighbor_init(Sensor* sensor, const SensorConfig* config, Arena* arena) {
    NeighborImpl* impl = arena_alloc_type(arena, NeighborImpl);
    if (impl == NULL) {
        sensor->impl = NULL;
        return;
    }

    impl->k = config->neighbor.k;
    impl->max_range = config->neighbor.max_range;
    impl->max_range_sq = impl->max_range * impl->max_range;
    impl->include_self = config->neighbor.include_self;

    sensor->impl = impl;
}

static size_t neighbor_get_output_size(const Sensor* sensor) {
    NeighborImpl* impl = (NeighborImpl*)sensor->impl;
    if (impl == NULL) return NEIGHBOR_DEFAULT_K * 4;
    return impl->k * 4;  /* dx, dy, dz, distance per neighbor */
}

static const char* neighbor_get_output_dtype(const Sensor* sensor) {
    (void)sensor;
    return "float32";
}

static uint32_t neighbor_get_output_shape(const Sensor* sensor, uint32_t* shape) {
    NeighborImpl* impl = (NeighborImpl*)sensor->impl;
    shape[0] = impl ? impl->k : NEIGHBOR_DEFAULT_K;
    shape[1] = 4;
    return 2;  /* 2D tensor [K, 4] */
}

static void neighbor_batch_sample(Sensor* sensor, const SensorContext* ctx, float* output_buffer) {
    NeighborImpl* impl = (NeighborImpl*)sensor->impl;
    if (impl == NULL) {
        memset(output_buffer, 0, ctx->agent_count * NEIGHBOR_DEFAULT_K * 4 * sizeof(float));
        return;
    }

    const RigidBodyStateSOA* drones = ctx->agents;
    const uint32_t* indices = ctx->agent_indices;
    uint32_t count = ctx->agent_count;
    const struct CollisionSystem* collision = ctx->collision;
    uint32_t k = impl->k;
    float max_range = impl->max_range;

    /* Temporary storage for k-nearest query results */
    uint32_t* neighbor_indices = NULL;
    float* neighbor_distances = NULL;
    uint32_t found_count = 0;

    /* Allocate from scratch arena if available */
    if (ctx->scratch != NULL) {
        neighbor_indices = arena_alloc_array(ctx->scratch, uint32_t, k);
        neighbor_distances = arena_alloc_array(ctx->scratch, float, k);
    }

    for (uint32_t i = 0; i < count; i++) {
        uint32_t d = indices[i];
        float* out = &output_buffer[i * k * 4];

        /* Get drone position */
        Vec3 pos = VEC3(drones->pos_x[d], drones->pos_y[d], drones->pos_z[d]);

        /* Initialize output with zeros */
        for (uint32_t n = 0; n < k; n++) {
            out[n * 4 + 0] = 0.0f;      /* dx */
            out[n * 4 + 1] = 0.0f;      /* dy */
            out[n * 4 + 2] = 0.0f;      /* dz */
            out[n * 4 + 3] = max_range; /* distance (max = no neighbor found) */
        }

        /* Query k-nearest neighbors if collision system available */
        if (collision != NULL && neighbor_indices != NULL && neighbor_distances != NULL) {
            collision_find_k_nearest(collision, drones, pos, k,
                                     neighbor_indices, neighbor_distances, &found_count);

            /* Fill in found neighbors */
            for (uint32_t n = 0; n < found_count && n < k; n++) {
                uint32_t neighbor_idx = neighbor_indices[n];

                /* Skip self if not including */
                if (!impl->include_self && neighbor_idx == d) {
                    continue;
                }

                /* Get relative position */
                Vec3 neighbor_pos = VEC3(
                    drones->pos_x[neighbor_idx],
                    drones->pos_y[neighbor_idx],
                    drones->pos_z[neighbor_idx]
                );
                Vec3 delta = vec3_sub(neighbor_pos, pos);

                /* Distance is returned squared, take sqrt */
                float dist = sqrtf(neighbor_distances[n]);

                /* Skip if beyond max range */
                if (dist > max_range) {
                    continue;
                }

                out[n * 4 + 0] = delta.x;
                out[n * 4 + 1] = delta.y;
                out[n * 4 + 2] = delta.z;
                out[n * 4 + 3] = dist;
            }
        } else if (collision == NULL) {
            /* No collision system - do brute force search within drone state */
            /* This is a fallback for testing without collision system */

            /* Simple approach: sort all drones by distance */
            /* For production, collision_find_k_nearest should be used */

            /* Temporary array for distances (small k, stack allocation is fine) */
            float best_dists[16];
            uint32_t best_indices[16];
            uint32_t found = 0;

            for (uint32_t n = 0; n < 16; n++) {
                best_dists[n] = max_range * max_range;
                best_indices[n] = UINT32_MAX;
            }

            /* Find k nearest (brute force) */
            uint32_t search_k = (k < 16) ? k : 16;
            for (uint32_t j = 0; j < drones->count; j++) {
                if (!impl->include_self && j == d) continue;

                Vec3 other_pos = VEC3(drones->pos_x[j], drones->pos_y[j], drones->pos_z[j]);
                Vec3 delta = vec3_sub(other_pos, pos);
                float dist_sq = vec3_length_sq(delta);

                if (dist_sq < impl->max_range_sq) {
                    /* Insert into sorted list */
                    for (uint32_t slot = 0; slot < search_k; slot++) {
                        if (dist_sq < best_dists[slot]) {
                            /* Shift down */
                            for (uint32_t shift = search_k - 1; shift > slot; shift--) {
                                best_dists[shift] = best_dists[shift - 1];
                                best_indices[shift] = best_indices[shift - 1];
                            }
                            best_dists[slot] = dist_sq;
                            best_indices[slot] = j;
                            if (found < search_k) found++;
                            break;
                        }
                    }
                }
            }

            /* Output results */
            for (uint32_t n = 0; n < found && n < k; n++) {
                uint32_t neighbor_idx = best_indices[n];
                if (neighbor_idx == UINT32_MAX) continue;

                Vec3 neighbor_pos = VEC3(
                    drones->pos_x[neighbor_idx],
                    drones->pos_y[neighbor_idx],
                    drones->pos_z[neighbor_idx]
                );
                Vec3 delta = vec3_sub(neighbor_pos, pos);
                float dist = sqrtf(best_dists[n]);

                out[n * 4 + 0] = delta.x;
                out[n * 4 + 1] = delta.y;
                out[n * 4 + 2] = delta.z;
                out[n * 4 + 3] = dist;
            }
        }
    }
}

static void neighbor_reset(Sensor* sensor, uint32_t agent_index) {
    (void)sensor;
    (void)agent_index;
}

static void neighbor_destroy(Sensor* sensor) {
    (void)sensor;
}

const SensorVTable SENSOR_VTABLE_NEIGHBOR = {
    .name = "Neighbor",
    .type = SENSOR_TYPE_NEIGHBOR,
    .init = neighbor_init,
    .get_output_size = neighbor_get_output_size,
    .get_output_dtype = neighbor_get_output_dtype,
    .get_output_shape = neighbor_get_output_shape,
    .batch_sample = neighbor_batch_sample,
    .reset = neighbor_reset,
    .destroy = neighbor_destroy
};
