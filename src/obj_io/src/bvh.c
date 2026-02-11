/**
 * BVH (Bounding Volume Hierarchy) Implementation
 *
 * SAH-based BVH construction with:
 * - Surface Area Heuristic for optimal splits
 * - Ray-triangle intersection using Moller-Trumbore
 * - Closest point queries via recursive traversal
 * - AABB-mesh intersection tests
 * - Inside/outside determination via ray counting
 */

#include "../include/obj_io.h"
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define BVH_MAX_LEAF_FACES 4         /* Max faces per leaf node */
#define BVH_SAH_BINS 12              /* Number of SAH binning buckets */
#define BVH_TRAVERSAL_COST 1.0f      /* Cost of traversing a node */
#define BVH_INTERSECTION_COST 1.0f   /* Cost of ray-triangle test */

/* ============================================================================
 * Helper Functions
 * ============================================================================ */

/**
 * Compute AABB for a range of faces
 */
static void compute_face_bounds(const TriangleMesh* mesh, const uint32_t* face_indices,
                                uint32_t start, uint32_t count,
                                Vec3* out_min, Vec3* out_max) {
    *out_min = VEC3(FLT_MAX, FLT_MAX, FLT_MAX);
    *out_max = VEC3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (uint32_t i = start; i < start + count; i++) {
        uint32_t f = face_indices[i];
        uint32_t base = f * 3;

        for (int v = 0; v < 3; v++) {
            uint32_t vi = mesh->face_v[base + v];
            float x = mesh->vx[vi];
            float y = mesh->vy[vi];
            float z = mesh->vz[vi];

            if (x < out_min->x) out_min->x = x;
            if (y < out_min->y) out_min->y = y;
            if (z < out_min->z) out_min->z = z;

            if (x > out_max->x) out_max->x = x;
            if (y > out_max->y) out_max->y = y;
            if (z > out_max->z) out_max->z = z;
        }
    }
}

/**
 * Compute centroid of a face
 */
static Vec3 face_centroid(const TriangleMesh* mesh, uint32_t face_idx) {
    uint32_t base = face_idx * 3;
    uint32_t i0 = mesh->face_v[base + 0];
    uint32_t i1 = mesh->face_v[base + 1];
    uint32_t i2 = mesh->face_v[base + 2];

    float cx = (mesh->vx[i0] + mesh->vx[i1] + mesh->vx[i2]) / 3.0f;
    float cy = (mesh->vy[i0] + mesh->vy[i1] + mesh->vy[i2]) / 3.0f;
    float cz = (mesh->vz[i0] + mesh->vz[i1] + mesh->vz[i2]) / 3.0f;

    return VEC3(cx, cy, cz);
}

/**
 * Compute surface area of AABB
 */
static float aabb_surface_area(Vec3 min, Vec3 max) {
    Vec3 d = vec3_sub(max, min);
    return 2.0f * (d.x * d.y + d.y * d.z + d.z * d.x);
}

/* ============================================================================
 * BVH Construction
 * ============================================================================ */

typedef struct BuildContext {
    Arena* arena;
    const TriangleMesh* mesh;
    uint32_t* face_indices;
    BVHNode* nodes;
    uint32_t node_count;
    uint32_t node_capacity;
} BuildContext;

/**
 * Allocate a new BVH node
 */
static uint32_t alloc_node(BuildContext* ctx) {
    if (ctx->node_count >= ctx->node_capacity) {
        /* Grow node array */
        uint32_t new_capacity = ctx->node_capacity * 2;
        BVHNode* new_nodes = arena_alloc_array(ctx->arena, BVHNode, new_capacity);
        if (!new_nodes) return UINT32_MAX;

        memcpy(new_nodes, ctx->nodes, ctx->node_count * sizeof(BVHNode));
        ctx->nodes = new_nodes;
        ctx->node_capacity = new_capacity;
    }

    return ctx->node_count++;
}

/**
 * Find best split using SAH
 * Returns split axis (0=x, 1=y, 2=z) and position, or -1 if no good split
 */
static int find_best_split(BuildContext* ctx, uint32_t start, uint32_t count,
                           Vec3 bounds_min, Vec3 bounds_max,
                           int* out_axis, float* out_pos, uint32_t* out_left_count) {
    if (count <= BVH_MAX_LEAF_FACES) {
        return -1; /* Make leaf */
    }

    float best_cost = FLT_MAX;
    int best_axis = -1;
    float best_pos = 0.0f;
    uint32_t best_left = 0;

    float parent_area = aabb_surface_area(bounds_min, bounds_max);
    if (parent_area < 1e-10f) return -1;

    /* Try each axis */
    for (int axis = 0; axis < 3; axis++) {
        float axis_min = (axis == 0) ? bounds_min.x : (axis == 1) ? bounds_min.y : bounds_min.z;
        float axis_max = (axis == 0) ? bounds_max.x : (axis == 1) ? bounds_max.y : bounds_max.z;

        if (axis_max - axis_min < 1e-6f) continue;

        /* Bin centroids */
        float bin_width = (axis_max - axis_min) / BVH_SAH_BINS;
        if (bin_width < 1e-6f) continue;

        uint32_t bin_count[BVH_SAH_BINS] = {0};
        Vec3 bin_min[BVH_SAH_BINS];
        Vec3 bin_max[BVH_SAH_BINS];

        for (int b = 0; b < BVH_SAH_BINS; b++) {
            bin_min[b] = VEC3(FLT_MAX, FLT_MAX, FLT_MAX);
            bin_max[b] = VEC3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
        }

        /* Assign faces to bins */
        for (uint32_t i = start; i < start + count; i++) {
            Vec3 centroid = face_centroid(ctx->mesh, ctx->face_indices[i]);
            float c = (axis == 0) ? centroid.x : (axis == 1) ? centroid.y : centroid.z;

            int bin = (int)((c - axis_min) / bin_width);
            if (bin < 0) bin = 0;
            if (bin >= BVH_SAH_BINS) bin = BVH_SAH_BINS - 1;

            bin_count[bin]++;

            /* Expand bin bounds */
            uint32_t f = ctx->face_indices[i];
            uint32_t base = f * 3;
            for (int v = 0; v < 3; v++) {
                uint32_t vi = ctx->mesh->face_v[base + v];
                float x = ctx->mesh->vx[vi];
                float y = ctx->mesh->vy[vi];
                float z = ctx->mesh->vz[vi];

                if (x < bin_min[bin].x) bin_min[bin].x = x;
                if (y < bin_min[bin].y) bin_min[bin].y = y;
                if (z < bin_min[bin].z) bin_min[bin].z = z;
                if (x > bin_max[bin].x) bin_max[bin].x = x;
                if (y > bin_max[bin].y) bin_max[bin].y = y;
                if (z > bin_max[bin].z) bin_max[bin].z = z;
            }
        }

        /* Evaluate splits between bins */
        uint32_t left_count = 0;
        Vec3 left_min = VEC3(FLT_MAX, FLT_MAX, FLT_MAX);
        Vec3 left_max = VEC3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

        for (int split = 0; split < BVH_SAH_BINS - 1; split++) {
            /* Accumulate left side */
            left_count += bin_count[split];
            if (bin_count[split] > 0) {
                left_min = vec3_min(left_min, bin_min[split]);
                left_max = vec3_max(left_max, bin_max[split]);
            }

            if (left_count == 0 || left_count == count) continue;

            /* Compute right side */
            uint32_t right_count = count - left_count;
            Vec3 right_min = VEC3(FLT_MAX, FLT_MAX, FLT_MAX);
            Vec3 right_max = VEC3(-FLT_MAX, -FLT_MAX, -FLT_MAX);

            for (int b = split + 1; b < BVH_SAH_BINS; b++) {
                if (bin_count[b] > 0) {
                    right_min = vec3_min(right_min, bin_min[b]);
                    right_max = vec3_max(right_max, bin_max[b]);
                }
            }

            /* SAH cost */
            float left_area = aabb_surface_area(left_min, left_max);
            float right_area = aabb_surface_area(right_min, right_max);

            float cost = BVH_TRAVERSAL_COST +
                         (left_area * left_count + right_area * right_count) *
                         BVH_INTERSECTION_COST / parent_area;

            if (cost < best_cost) {
                best_cost = cost;
                best_axis = axis;
                best_pos = axis_min + (split + 1) * bin_width;
                best_left = left_count;
            }
        }
    }

    /* Check if split is worthwhile */
    float leaf_cost = BVH_INTERSECTION_COST * count;
    if (best_cost >= leaf_cost || best_axis < 0) {
        return -1;
    }

    *out_axis = best_axis;
    *out_pos = best_pos;
    *out_left_count = best_left;
    return best_axis;
}

/**
 * Partition faces around split plane
 */
static uint32_t partition_faces(BuildContext* ctx, uint32_t start, uint32_t count,
                                int axis, float pos) {
    uint32_t left = start;
    uint32_t right = start + count - 1;

    while (left <= right && right < start + count) {
        Vec3 centroid = face_centroid(ctx->mesh, ctx->face_indices[left]);
        float c = (axis == 0) ? centroid.x : (axis == 1) ? centroid.y : centroid.z;

        if (c < pos) {
            left++;
        } else {
            /* Swap left and right */
            uint32_t tmp = ctx->face_indices[left];
            ctx->face_indices[left] = ctx->face_indices[right];
            ctx->face_indices[right] = tmp;
            right--;
        }
    }

    return left - start; /* Number of faces on left side */
}

/**
 * Recursive BVH build
 */
static uint32_t build_recursive(BuildContext* ctx, uint32_t start, uint32_t count) {
    uint32_t node_idx = alloc_node(ctx);
    if (node_idx == UINT32_MAX) return UINT32_MAX;

    BVHNode* node = &ctx->nodes[node_idx];

    /* Compute bounds */
    compute_face_bounds(ctx->mesh, ctx->face_indices, start, count,
                        &node->bbox_min, &node->bbox_max);

    /* Try to find good split */
    int axis;
    float pos;
    uint32_t left_count;

    if (find_best_split(ctx, start, count, node->bbox_min, node->bbox_max,
                        &axis, &pos, &left_count) >= 0) {
        /* Partition faces */
        uint32_t actual_left = partition_faces(ctx, start, count, axis, pos);

        /* Handle degenerate partitions */
        if (actual_left == 0 || actual_left == count) {
            actual_left = count / 2;
        }

        /* Build children */
        node->left = build_recursive(ctx, start, actual_left);
        node->right = build_recursive(ctx, start + actual_left, count - actual_left);
        node->face_start = 0;
        node->face_count = 0;
    } else {
        /* Make leaf */
        node->left = node_idx;  /* Self-reference for leaf */
        node->right = node_idx;
        node->face_start = start;
        node->face_count = count;
    }

    return node_idx;
}

/* ============================================================================
 * BVH Build
 * ============================================================================ */

MeshBVH* bvh_build(Arena* arena, const TriangleMesh* mesh) {
    if (!arena || !mesh || mesh->face_count == 0) {
        return NULL;
    }

    /* Allocate BVH structure */
    MeshBVH* bvh = arena_alloc_type(arena, MeshBVH);
    if (!bvh) return NULL;

    /* Estimate node count (roughly 2N-1 for N faces) */
    uint32_t estimated_nodes = mesh->face_count * 2;
    bvh->nodes = arena_alloc_array(arena, BVHNode, estimated_nodes);
    if (!bvh->nodes) return NULL;

    /* Allocate face indices (permutation array) */
    bvh->face_indices = arena_alloc_array(arena, uint32_t, mesh->face_count);
    if (!bvh->face_indices) return NULL;

    /* Initialize face indices */
    for (uint32_t i = 0; i < mesh->face_count; i++) {
        bvh->face_indices[i] = i;
    }

    bvh->node_count = 0;
    bvh->node_capacity = estimated_nodes;
    bvh->avg_normal = VEC3(0, 0, 0);
    bvh->normal_coherence = 0.0f;
    bvh->arena = arena;

    /* Build context */
    BuildContext ctx = {
        .arena = arena,
        .mesh = mesh,
        .face_indices = bvh->face_indices,
        .nodes = bvh->nodes,
        .node_count = 0,
        .node_capacity = estimated_nodes
    };

    /* Build tree */
    build_recursive(&ctx, 0, mesh->face_count);

    /* Update BVH from context */
    bvh->nodes = ctx.nodes;
    bvh->node_count = ctx.node_count;
    bvh->node_capacity = ctx.node_capacity;

    /* Compute area-weighted average normal and normal coherence.
     * For terrain meshes: avg_normal points "up", coherence ≈ 1.0.
     * For closed meshes: avg_normal ≈ 0, coherence ≈ 0.0. */
    Vec3 avg_n = VEC3(0, 0, 0);
    float total_area = 0.0f;
    for (uint32_t f = 0; f < mesh->face_count; f++) {
        Vec3 fn = mesh_face_normal(mesh, f);
        avg_n = vec3_add(avg_n, fn);
        total_area += vec3_length(fn);
    }
    bvh->avg_normal = avg_n;
    bvh->normal_coherence = (total_area > 1e-6f)
        ? vec3_length(avg_n) / total_area : 0.0f;

    return bvh;
}

/* ============================================================================
 * Ray-Triangle Intersection (Moller-Trumbore)
 * ============================================================================ */

static bool ray_triangle_intersect(Vec3 origin, Vec3 dir,
                                   Vec3 v0, Vec3 v1, Vec3 v2,
                                   float* t, float* u, float* v) {
    const float EPSILON = 1e-8f;

    Vec3 e1 = vec3_sub(v1, v0);
    Vec3 e2 = vec3_sub(v2, v0);

    Vec3 h = vec3_cross(dir, e2);
    float a = vec3_dot(e1, h);

    if (a > -EPSILON && a < EPSILON) {
        return false; /* Ray parallel to triangle */
    }

    float f = 1.0f / a;
    Vec3 s = vec3_sub(origin, v0);
    *u = f * vec3_dot(s, h);

    if (*u < 0.0f || *u > 1.0f) {
        return false;
    }

    Vec3 q = vec3_cross(s, e1);
    *v = f * vec3_dot(dir, q);

    if (*v < 0.0f || *u + *v > 1.0f) {
        return false;
    }

    *t = f * vec3_dot(e2, q);

    return *t > EPSILON;
}

/* ============================================================================
 * Ray-AABB Intersection
 * ============================================================================ */

static bool ray_aabb_intersect(Vec3 origin, Vec3 inv_dir, Vec3 bbox_min, Vec3 bbox_max,
                               float max_t, float* t_near) {
    float t1 = (bbox_min.x - origin.x) * inv_dir.x;
    float t2 = (bbox_max.x - origin.x) * inv_dir.x;
    float t3 = (bbox_min.y - origin.y) * inv_dir.y;
    float t4 = (bbox_max.y - origin.y) * inv_dir.y;
    float t5 = (bbox_min.z - origin.z) * inv_dir.z;
    float t6 = (bbox_max.z - origin.z) * inv_dir.z;

    float tmin = fmaxf(fmaxf(fminf(t1, t2), fminf(t3, t4)), fminf(t5, t6));
    float tmax = fminf(fminf(fmaxf(t1, t2), fmaxf(t3, t4)), fmaxf(t5, t6));

    if (tmax < 0 || tmin > tmax || tmin > max_t) {
        return false;
    }

    *t_near = tmin;
    return true;
}

/* ============================================================================
 * BVH Queries
 * ============================================================================ */

bool bvh_ray_intersect(const MeshBVH* bvh, const TriangleMesh* mesh,
                       Vec3 origin, Vec3 direction, float max_t,
                       float* hit_t, uint32_t* hit_face) {
    if (!bvh || !mesh || bvh->node_count == 0) {
        return false;
    }

    Vec3 inv_dir = VEC3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);

    /* Stack-based traversal */
    uint32_t stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0; /* Root node */

    bool hit = false;
    *hit_t = max_t;
    *hit_face = UINT32_MAX;

    while (stack_ptr > 0) {
        uint32_t node_idx = stack[--stack_ptr];
        const BVHNode* node = &bvh->nodes[node_idx];

        /* Check AABB */
        float t_near;
        if (!ray_aabb_intersect(origin, inv_dir, node->bbox_min, node->bbox_max, *hit_t, &t_near)) {
            continue;
        }

        /* Leaf node? */
        if (node->left == node->right) {
            /* Test triangles */
            for (uint32_t i = 0; i < node->face_count; i++) {
                uint32_t f = bvh->face_indices[node->face_start + i];

                Vec3 v0, v1, v2;
                mesh_get_triangle(mesh, f, &v0, &v1, &v2);

                float t, u, v;
                if (ray_triangle_intersect(origin, direction, v0, v1, v2, &t, &u, &v)) {
                    if (t < *hit_t && t > 0) {
                        *hit_t = t;
                        *hit_face = f;
                        hit = true;
                    }
                }
            }
        } else {
            /* Push children (far child first for better culling) */
            if (stack_ptr < 62) {
                stack[stack_ptr++] = node->right;
                stack[stack_ptr++] = node->left;
            }
        }
    }

    return hit;
}

/* ============================================================================
 * Closest Point on Triangle
 * ============================================================================ */

static Vec3 closest_point_on_triangle(Vec3 p, Vec3 a, Vec3 b, Vec3 c) {
    Vec3 ab = vec3_sub(b, a);
    Vec3 ac = vec3_sub(c, a);
    Vec3 ap = vec3_sub(p, a);

    float d1 = vec3_dot(ab, ap);
    float d2 = vec3_dot(ac, ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return a;

    Vec3 bp = vec3_sub(p, b);
    float d3 = vec3_dot(ab, bp);
    float d4 = vec3_dot(ac, bp);
    if (d3 >= 0.0f && d4 <= d3) return b;

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        float v = d1 / (d1 - d3);
        return vec3_add(a, vec3_scale(ab, v));
    }

    Vec3 cp = vec3_sub(p, c);
    float d5 = vec3_dot(ab, cp);
    float d6 = vec3_dot(ac, cp);
    if (d6 >= 0.0f && d5 <= d6) return c;

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        float w = d2 / (d2 - d6);
        return vec3_add(a, vec3_scale(ac, w));
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return vec3_add(b, vec3_scale(vec3_sub(c, b), w));
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    return vec3_add(a, vec3_add(vec3_scale(ab, v), vec3_scale(ac, w)));
}

float bvh_closest_point(const MeshBVH* bvh, const TriangleMesh* mesh,
                        Vec3 point, Vec3* closest, uint32_t* face_idx) {
    if (!bvh || !mesh || bvh->node_count == 0) {
        if (closest) *closest = point;
        if (face_idx) *face_idx = UINT32_MAX;
        return FLT_MAX;
    }

    float best_dist_sq = FLT_MAX;
    Vec3 best_point = point;
    uint32_t best_face = UINT32_MAX;

    /* Stack-based traversal */
    uint32_t stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        uint32_t node_idx = stack[--stack_ptr];
        const BVHNode* node = &bvh->nodes[node_idx];

        /* Check if AABB could contain closer point */
        Vec3 clamped = vec3_clamp_vec(point, node->bbox_min, node->bbox_max);
        float dist_sq = vec3_length_sq(vec3_sub(clamped, point));

        if (dist_sq >= best_dist_sq) {
            continue; /* Skip this subtree */
        }

        if (node->left == node->right) {
            /* Leaf - test triangles */
            for (uint32_t i = 0; i < node->face_count; i++) {
                uint32_t f = bvh->face_indices[node->face_start + i];

                Vec3 v0, v1, v2;
                mesh_get_triangle(mesh, f, &v0, &v1, &v2);

                Vec3 cp = closest_point_on_triangle(point, v0, v1, v2);
                float d2 = vec3_length_sq(vec3_sub(cp, point));

                if (d2 < best_dist_sq) {
                    best_dist_sq = d2;
                    best_point = cp;
                    best_face = f;
                }
            }
        } else {
            /* Push children */
            if (stack_ptr < 62) {
                stack[stack_ptr++] = node->right;
                stack[stack_ptr++] = node->left;
            }
        }
    }

    if (closest) *closest = best_point;
    if (face_idx) *face_idx = best_face;

    return sqrtf(best_dist_sq);
}

/* ============================================================================
 * AABB-Mesh Intersection
 * ============================================================================ */

/**
 * Test if AABB intersects triangle (separating axis theorem)
 */
static bool aabb_triangle_intersect(Vec3 box_min, Vec3 box_max, Vec3 v0, Vec3 v1, Vec3 v2) {
    /* Translate so box center is at origin */
    Vec3 box_center = vec3_scale(vec3_add(box_min, box_max), 0.5f);
    Vec3 box_half = vec3_scale(vec3_sub(box_max, box_min), 0.5f);

    Vec3 tv0 = vec3_sub(v0, box_center);
    Vec3 tv1 = vec3_sub(v1, box_center);
    Vec3 tv2 = vec3_sub(v2, box_center);

    /* Test box axes */
    float min_x = fminf(fminf(tv0.x, tv1.x), tv2.x);
    float max_x = fmaxf(fmaxf(tv0.x, tv1.x), tv2.x);
    if (min_x > box_half.x || max_x < -box_half.x) return false;

    float min_y = fminf(fminf(tv0.y, tv1.y), tv2.y);
    float max_y = fmaxf(fmaxf(tv0.y, tv1.y), tv2.y);
    if (min_y > box_half.y || max_y < -box_half.y) return false;

    float min_z = fminf(fminf(tv0.z, tv1.z), tv2.z);
    float max_z = fmaxf(fmaxf(tv0.z, tv1.z), tv2.z);
    if (min_z > box_half.z || max_z < -box_half.z) return false;

    /* Test triangle normal */
    Vec3 e0 = vec3_sub(tv1, tv0);
    Vec3 e1 = vec3_sub(tv2, tv1);
    Vec3 e2 = vec3_sub(tv0, tv2);
    Vec3 n = vec3_cross(e0, e1);

    float r = box_half.x * fabsf(n.x) + box_half.y * fabsf(n.y) + box_half.z * fabsf(n.z);
    float s = vec3_dot(n, tv0);
    if (fabsf(s) > r) return false;

    return true; /* Simplified - could add edge axis tests for robustness */
}

bool bvh_aabb_intersect(const MeshBVH* bvh, const TriangleMesh* mesh,
                        Vec3 box_min, Vec3 box_max) {
    if (!bvh || !mesh || bvh->node_count == 0) {
        return false;
    }

    uint32_t stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    while (stack_ptr > 0) {
        uint32_t node_idx = stack[--stack_ptr];
        const BVHNode* node = &bvh->nodes[node_idx];

        /* Check AABB-AABB */
        if (box_max.x < node->bbox_min.x || box_min.x > node->bbox_max.x ||
            box_max.y < node->bbox_min.y || box_min.y > node->bbox_max.y ||
            box_max.z < node->bbox_min.z || box_min.z > node->bbox_max.z) {
            continue;
        }

        if (node->left == node->right) {
            /* Leaf - test triangles */
            for (uint32_t i = 0; i < node->face_count; i++) {
                uint32_t f = bvh->face_indices[node->face_start + i];

                Vec3 v0, v1, v2;
                mesh_get_triangle(mesh, f, &v0, &v1, &v2);

                if (aabb_triangle_intersect(box_min, box_max, v0, v1, v2)) {
                    return true;
                }
            }
        } else {
            if (stack_ptr < 62) {
                stack[stack_ptr++] = node->right;
                stack[stack_ptr++] = node->left;
            }
        }
    }

    return false;
}

/* ============================================================================
 * Inside/Outside Determination (Ray-Casting Parity)
 * ============================================================================ */

/**
 * Count ray-mesh intersections for parity-based inside/outside test.
 *
 * Traverses the BVH and counts ALL ray-triangle intersections (not just the
 * closest). Uses Moller-Trumbore with t > EPSILON to avoid self-intersection.
 */
static uint32_t count_ray_crossings(const MeshBVH* bvh, const TriangleMesh* mesh,
                                     Vec3 origin, Vec3 direction) {
    Vec3 inv_dir = VEC3(1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z);

    uint32_t stack[64];
    int stack_ptr = 0;
    stack[stack_ptr++] = 0;

    uint32_t count = 0;

    while (stack_ptr > 0) {
        uint32_t node_idx = stack[--stack_ptr];
        const BVHNode* node = &bvh->nodes[node_idx];

        float t_near;
        if (!ray_aabb_intersect(origin, inv_dir, node->bbox_min, node->bbox_max,
                                FLT_MAX, &t_near)) {
            continue;
        }

        if (node->left == node->right) {
            /* Leaf — test all triangles */
            for (uint32_t i = 0; i < node->face_count; i++) {
                uint32_t f = bvh->face_indices[node->face_start + i];
                Vec3 v0, v1, v2;
                mesh_get_triangle(mesh, f, &v0, &v1, &v2);

                float t, u, v;
                if (ray_triangle_intersect(origin, direction, v0, v1, v2, &t, &u, &v)) {
                    count++;
                }
            }
        } else {
            if (stack_ptr < 62) {
                stack[stack_ptr++] = node->right;
                stack[stack_ptr++] = node->left;
            }
        }
    }

    return count;
}

/**
 * Determine if point is inside or outside the mesh using ray-casting parity.
 *
 * Casts a ray aligned with the mesh's dominant face normal direction and
 * counts mesh crossings. Odd crossings = inside (-1), even = outside (+1).
 *
 * The ray direction is derived from the area-weighted average normal computed
 * during BVH construction. This ensures the ray goes "upward" through terrain
 * (perpendicular to the dominant surface), giving correct results for both:
 * - Open terrain: above surface → 0 crossings → outside
 *                 below surface → 1 crossing  → inside
 * - Closed meshes: standard parity (any direction works)
 *
 * Small off-axis perturbation avoids degenerate edge/vertex intersections.
 */
float bvh_inside_outside(const MeshBVH* bvh, const TriangleMesh* mesh, Vec3 point) {
    if (!bvh || !mesh || bvh->node_count == 0) {
        return 1.0f; /* Outside */
    }

    /* Select primary ray direction aligned with dominant normal.
     * For terrain: this is the "up" direction (perpendicular to surface).
     * For closed meshes: avg_normal ≈ 0, any axis works. */
    Vec3 avg = bvh->avg_normal;
    float ax = fabsf(avg.x), ay = fabsf(avg.y), az = fabsf(avg.z);

    Vec3 dir;
    if (ay >= ax && ay >= az) {
        /* Y-dominant (e.g., Y-up terrain) */
        dir = VEC3(0.0013f, (avg.y >= 0 ? 1.0f : -1.0f), 0.0027f);
    } else if (ax >= ay && ax >= az) {
        /* X-dominant */
        dir = VEC3((avg.x >= 0 ? 1.0f : -1.0f), 0.0013f, 0.0027f);
    } else {
        /* Z-dominant (e.g., Z-up terrain) */
        dir = VEC3(0.0013f, 0.0027f, (avg.z >= 0 ? 1.0f : -1.0f));
    }

    uint32_t crossings = count_ray_crossings(bvh, mesh, point, dir);
    return (crossings & 1) ? -1.0f : 1.0f;
}

/**
 * Robust inside/outside determination using 3-ray majority voting.
 *
 * For complex surfaces (gyroid, thin shells), a single ray may graze edges
 * or pass through narrow folds, producing ambiguous crossing counts.
 * Three non-coplanar diagonal rays avoid axis-aligned degeneracies.
 * Majority vote (2-of-3) determines the result.
 */
float bvh_inside_outside_robust(const MeshBVH* bvh, const TriangleMesh* mesh, Vec3 point) {
    if (!bvh || !mesh || bvh->node_count == 0) {
        return 1.0f; /* Outside */
    }

    /* Three non-coplanar directions with irrational-looking components.
     * Avoids axis-alignment AND edge-grazing on axis-aligned geometry.
     * Each direction has 3 significantly different components. */
    static const Vec3 ROBUST_DIRS[3] = {
        { 0.8017f,  0.2673f,  0.5345f, 0.0f},  /* ~(3,1,2)/sqrt(14) */
        { 0.2357f,  0.9428f, -0.2357f, 0.0f},  /* ~(1,4,-1)/sqrt(18) */
        {-0.3015f, -0.3015f,  0.9045f, 0.0f}   /* ~(-1,-1,3)/sqrt(11) */
    };

    int inside_votes = 0;
    for (int r = 0; r < 3; r++) {
        uint32_t crossings = count_ray_crossings(bvh, mesh, point, ROBUST_DIRS[r]);
        if (crossings & 1) inside_votes++;
    }

    /* Majority vote: 2 or 3 inside -> inside */
    return (inside_votes >= 2) ? -1.0f : 1.0f;
}
