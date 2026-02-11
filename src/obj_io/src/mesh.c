/**
 * Triangle Mesh Implementation
 *
 * SoA (Structure-of-Arrays) mesh representation with:
 * - Separate x/y/z arrays for SIMD-friendly access
 * - Dynamic growth via arena reallocation
 * - Area-weighted normal computation
 * - Bounding box tracking
 */

#include "../include/obj_io.h"
#include <string.h>
#include <math.h>
#include <stdint.h>

/* ============================================================================
 * Mesh Creation and Memory Management
 * ============================================================================ */

TriangleMesh* mesh_create(Arena* arena, uint32_t vertex_capacity, uint32_t face_capacity) {
    if (!arena || vertex_capacity == 0 || face_capacity == 0) {
        return NULL;
    }

    TriangleMesh* mesh = arena_alloc_type(arena, TriangleMesh);
    if (!mesh) {
        return NULL;
    }

    /* Allocate vertex arrays (SoA) */
    mesh->vx = arena_alloc_array(arena, float, vertex_capacity);
    mesh->vy = arena_alloc_array(arena, float, vertex_capacity);
    mesh->vz = arena_alloc_array(arena, float, vertex_capacity);

    if (!mesh->vx || !mesh->vy || !mesh->vz) {
        return NULL;
    }

    mesh->vertex_count = 0;
    mesh->vertex_capacity = vertex_capacity;

    /* Normal arrays allocated lazily */
    mesh->nx = NULL;
    mesh->ny = NULL;
    mesh->nz = NULL;
    mesh->has_normals = false;

    /* Allocate face arrays */
    mesh->face_v = arena_alloc_array(arena, uint32_t, face_capacity * 3);
    mesh->face_mat = arena_alloc_array(arena, uint8_t, face_capacity);

    if (!mesh->face_v || !mesh->face_mat) {
        return NULL;
    }

    mesh->face_count = 0;
    mesh->face_capacity = face_capacity;

    /* Material names (populated during OBJ parsing) */
    mesh->material_names = NULL;
    mesh->material_name_count = 0;

    /* Initialize bbox to invalid state */
    mesh->bbox_min = VEC3(1e30f, 1e30f, 1e30f);
    mesh->bbox_max = VEC3(-1e30f, -1e30f, -1e30f);

    mesh->arena = arena;

    return mesh;
}

/**
 * Internal: Grow vertex capacity
 */
static bool mesh_grow_vertices(TriangleMesh* mesh) {
    uint32_t new_capacity = mesh->vertex_capacity * 2;

    float* new_vx = arena_alloc_array(mesh->arena, float, new_capacity);
    float* new_vy = arena_alloc_array(mesh->arena, float, new_capacity);
    float* new_vz = arena_alloc_array(mesh->arena, float, new_capacity);

    if (!new_vx || !new_vy || !new_vz) {
        return false;
    }

    /* Copy existing data */
    memcpy(new_vx, mesh->vx, mesh->vertex_count * sizeof(float));
    memcpy(new_vy, mesh->vy, mesh->vertex_count * sizeof(float));
    memcpy(new_vz, mesh->vz, mesh->vertex_count * sizeof(float));

    mesh->vx = new_vx;
    mesh->vy = new_vy;
    mesh->vz = new_vz;
    mesh->vertex_capacity = new_capacity;

    /* Grow normals if they exist */
    if (mesh->has_normals) {
        float* new_nx = arena_alloc_array(mesh->arena, float, new_capacity);
        float* new_ny = arena_alloc_array(mesh->arena, float, new_capacity);
        float* new_nz = arena_alloc_array(mesh->arena, float, new_capacity);

        if (!new_nx || !new_ny || !new_nz) {
            return false;
        }

        memcpy(new_nx, mesh->nx, mesh->vertex_count * sizeof(float));
        memcpy(new_ny, mesh->ny, mesh->vertex_count * sizeof(float));
        memcpy(new_nz, mesh->nz, mesh->vertex_count * sizeof(float));

        mesh->nx = new_nx;
        mesh->ny = new_ny;
        mesh->nz = new_nz;
    }

    return true;
}

/**
 * Internal: Grow face capacity
 */
static bool mesh_grow_faces(TriangleMesh* mesh) {
    uint32_t new_capacity = mesh->face_capacity * 2;

    uint32_t* new_face_v = arena_alloc_array(mesh->arena, uint32_t, new_capacity * 3);
    uint8_t* new_face_mat = arena_alloc_array(mesh->arena, uint8_t, new_capacity);

    if (!new_face_v || !new_face_mat) {
        return false;
    }

    memcpy(new_face_v, mesh->face_v, mesh->face_count * 3 * sizeof(uint32_t));
    memcpy(new_face_mat, mesh->face_mat, mesh->face_count * sizeof(uint8_t));

    mesh->face_v = new_face_v;
    mesh->face_mat = new_face_mat;
    mesh->face_capacity = new_capacity;

    return true;
}

uint32_t mesh_add_vertex(TriangleMesh* mesh, float x, float y, float z) {
    if (!mesh) {
        return UINT32_MAX;
    }

    /* Grow if needed */
    if (mesh->vertex_count >= mesh->vertex_capacity) {
        if (!mesh_grow_vertices(mesh)) {
            return UINT32_MAX;
        }
    }

    uint32_t idx = mesh->vertex_count++;
    mesh->vx[idx] = x;
    mesh->vy[idx] = y;
    mesh->vz[idx] = z;

    return idx;
}

/* ============================================================================
 * Vertex Welding (Spatial Hash Deduplication)
 * ============================================================================ */

WeldContext* weld_context_create(Arena* arena, TriangleMesh* mesh,
                                  float tolerance, uint32_t estimated_verts) {
    if (!arena || !mesh) return NULL;

    WeldContext* ctx = arena_alloc_type(arena, WeldContext);
    if (!ctx) return NULL;

    /* Power-of-2 hash capacity: ~2x estimated unique verts */
    uint32_t cap = 1;
    while (cap < estimated_verts * 2) cap <<= 1;

    ctx->hash_table = arena_alloc_array(arena, uint32_t, cap);
    ctx->hash_next = arena_alloc_array(arena, uint32_t, estimated_verts);
    if (!ctx->hash_table || !ctx->hash_next) return NULL;

    memset(ctx->hash_table, 0xFF, cap * sizeof(uint32_t)); /* UINT32_MAX = empty */
    ctx->hash_capacity = cap;
    ctx->hash_next_capacity = estimated_verts;
    ctx->inv_cell_size = 1.0f / tolerance;
    ctx->tolerance_sq = tolerance * tolerance;
    ctx->mesh = mesh;

    return ctx;
}

static bool weld_grow_next(WeldContext* ctx) {
    uint32_t new_cap = ctx->hash_next_capacity * 2;
    uint32_t* new_next = arena_alloc_array(ctx->mesh->arena, uint32_t, new_cap);
    if (!new_next) return false;
    memcpy(new_next, ctx->hash_next, ctx->mesh->vertex_count * sizeof(uint32_t));
    ctx->hash_next = new_next;
    ctx->hash_next_capacity = new_cap;
    return true;
}

uint32_t mesh_add_vertex_welded(WeldContext* ctx, float x, float y, float z) {
    if (!ctx || !ctx->mesh) return UINT32_MAX;

    /* Spatial hash: quantize position to grid cell */
    int32_t ix = (int32_t)floorf(x * ctx->inv_cell_size);
    int32_t iy = (int32_t)floorf(y * ctx->inv_cell_size);
    int32_t iz = (int32_t)floorf(z * ctx->inv_cell_size);

    /* Search all 27 neighboring cells (3x3x3) for matches */
    for (int32_t dz = -1; dz <= 1; dz++) {
        for (int32_t dy = -1; dy <= 1; dy++) {
            for (int32_t dx = -1; dx <= 1; dx++) {
                uint32_t hash = ((uint32_t)(ix + dx) * 73856093u ^
                                 (uint32_t)(iy + dy) * 19349669u ^
                                 (uint32_t)(iz + dz) * 83492791u) & (ctx->hash_capacity - 1);

                uint32_t vi = ctx->hash_table[hash];
                while (vi != UINT32_MAX) {
                    float ex = ctx->mesh->vx[vi] - x;
                    float ey = ctx->mesh->vy[vi] - y;
                    float ez = ctx->mesh->vz[vi] - z;
                    if (ex * ex + ey * ey + ez * ez <= ctx->tolerance_sq) {
                        return vi; /* Found existing vertex within tolerance */
                    }
                    vi = ctx->hash_next[vi];
                }
            }
        }
    }

    /* No match found — add new vertex */
    uint32_t idx = mesh_add_vertex(ctx->mesh, x, y, z);
    if (idx == UINT32_MAX) return UINT32_MAX;

    /* Grow hash_next if needed */
    if (idx >= ctx->hash_next_capacity) {
        if (!weld_grow_next(ctx)) return idx; /* Still valid, just won't be welded */
    }

    /* Insert into hash table (current cell only) */
    uint32_t hash = ((uint32_t)ix * 73856093u ^
                     (uint32_t)iy * 19349669u ^
                     (uint32_t)iz * 83492791u) & (ctx->hash_capacity - 1);

    ctx->hash_next[idx] = ctx->hash_table[hash];
    ctx->hash_table[hash] = idx;

    return idx;
}

/* ============================================================================
 * Watertight Detection (Edge Boundary Counting)
 * ============================================================================ */

float mesh_boundary_edge_ratio(const TriangleMesh* mesh) {
    if (!mesh || mesh->face_count == 0) return 0.0f;

    /* Scratch arena for edge hash table — freed when done */
    size_t arena_size = (size_t)mesh->face_count * 32 + 1024 * 1024;
    Arena* scratch = arena_create(arena_size);
    if (!scratch) return 0.0f;

    /* Hash table capacity: next power of 2 above 4x face count (load < 0.75) */
    uint32_t cap = 1;
    while (cap < mesh->face_count * 4) cap <<= 1;

    uint64_t* keys = arena_alloc_array(scratch, uint64_t, cap);
    uint8_t* counts = arena_alloc_array(scratch, uint8_t, cap);
    if (!keys || !counts) {
        arena_destroy(scratch);
        return 0.0f;
    }

    for (uint32_t i = 0; i < cap; i++) keys[i] = UINT64_MAX;
    memset(counts, 0, cap);

    uint32_t unique_edges = 0;

    for (uint32_t f = 0; f < mesh->face_count; f++) {
        uint32_t base = f * 3;
        for (int e = 0; e < 3; e++) {
            uint32_t a = mesh->face_v[base + e];
            uint32_t b = mesh->face_v[base + ((e + 1) % 3)];
            if (a > b) { uint32_t tmp = a; a = b; b = tmp; }

            uint64_t key = ((uint64_t)a << 32) | (uint64_t)b;
            uint32_t hash = (a * 73856093u ^ b * 19349669u) & (cap - 1);

            /* Linear probing */
            while (1) {
                if (keys[hash] == UINT64_MAX) {
                    keys[hash] = key;
                    counts[hash] = 1;
                    unique_edges++;
                    break;
                } else if (keys[hash] == key) {
                    if (counts[hash] < 255) counts[hash]++;
                    break;
                }
                hash = (hash + 1) & (cap - 1);
            }
        }
    }

    /* Count boundary edges (shared by exactly 1 face) */
    uint32_t boundary = 0;
    for (uint32_t i = 0; i < cap; i++) {
        if (counts[i] == 1) boundary++;
    }

    arena_destroy(scratch);

    if (unique_edges < 10) return 0.0f;
    return (float)boundary / (float)unique_edges;
}

bool mesh_is_watertight(const TriangleMesh* mesh) {
    return mesh_boundary_edge_ratio(mesh) < 0.005f;
}

uint32_t mesh_add_face(TriangleMesh* mesh, uint32_t v0, uint32_t v1, uint32_t v2, uint8_t material) {
    if (!mesh) {
        return UINT32_MAX;
    }

    /* Validate indices */
    if (v0 >= mesh->vertex_count || v1 >= mesh->vertex_count || v2 >= mesh->vertex_count) {
        return UINT32_MAX;
    }

    /* Grow if needed */
    if (mesh->face_count >= mesh->face_capacity) {
        if (!mesh_grow_faces(mesh)) {
            return UINT32_MAX;
        }
    }

    uint32_t idx = mesh->face_count++;
    uint32_t base = idx * 3;

    mesh->face_v[base + 0] = v0;
    mesh->face_v[base + 1] = v1;
    mesh->face_v[base + 2] = v2;
    mesh->face_mat[idx] = material;

    return idx;
}

/* ============================================================================
 * Geometry Queries
 * ============================================================================ */

void mesh_get_triangle(const TriangleMesh* mesh, uint32_t face_idx,
                       Vec3* v0, Vec3* v1, Vec3* v2) {
    if (!mesh || face_idx >= mesh->face_count) {
        *v0 = *v1 = *v2 = VEC3_ZERO;
        return;
    }

    uint32_t base = face_idx * 3;
    uint32_t i0 = mesh->face_v[base + 0];
    uint32_t i1 = mesh->face_v[base + 1];
    uint32_t i2 = mesh->face_v[base + 2];

    *v0 = VEC3(mesh->vx[i0], mesh->vy[i0], mesh->vz[i0]);
    *v1 = VEC3(mesh->vx[i1], mesh->vy[i1], mesh->vz[i1]);
    *v2 = VEC3(mesh->vx[i2], mesh->vy[i2], mesh->vz[i2]);
}

Vec3 mesh_face_normal(const TriangleMesh* mesh, uint32_t face_idx) {
    Vec3 v0, v1, v2;
    mesh_get_triangle(mesh, face_idx, &v0, &v1, &v2);

    Vec3 e1 = vec3_sub(v1, v0);
    Vec3 e2 = vec3_sub(v2, v0);

    return vec3_cross(e1, e2);
}

float mesh_face_area(const TriangleMesh* mesh, uint32_t face_idx) {
    Vec3 n = mesh_face_normal(mesh, face_idx);
    return vec3_length(n) * 0.5f;
}

void mesh_compute_bbox(TriangleMesh* mesh) {
    if (!mesh || mesh->vertex_count == 0) {
        return;
    }

    mesh->bbox_min = VEC3(mesh->vx[0], mesh->vy[0], mesh->vz[0]);
    mesh->bbox_max = mesh->bbox_min;

    for (uint32_t i = 1; i < mesh->vertex_count; i++) {
        float x = mesh->vx[i];
        float y = mesh->vy[i];
        float z = mesh->vz[i];

        if (x < mesh->bbox_min.x) mesh->bbox_min.x = x;
        if (y < mesh->bbox_min.y) mesh->bbox_min.y = y;
        if (z < mesh->bbox_min.z) mesh->bbox_min.z = z;

        if (x > mesh->bbox_max.x) mesh->bbox_max.x = x;
        if (y > mesh->bbox_max.y) mesh->bbox_max.y = y;
        if (z > mesh->bbox_max.z) mesh->bbox_max.z = z;
    }
}

/* ============================================================================
 * Normal Computation
 * ============================================================================ */

void mesh_compute_normals(TriangleMesh* mesh) {
    if (!mesh || mesh->vertex_count == 0 || mesh->face_count == 0) {
        return;
    }

    /* Allocate normal arrays if not present */
    if (!mesh->nx) {
        mesh->nx = arena_alloc_array(mesh->arena, float, mesh->vertex_capacity);
        mesh->ny = arena_alloc_array(mesh->arena, float, mesh->vertex_capacity);
        mesh->nz = arena_alloc_array(mesh->arena, float, mesh->vertex_capacity);

        if (!mesh->nx || !mesh->ny || !mesh->nz) {
            return;
        }
    }

    /* Zero out normals */
    memset(mesh->nx, 0, mesh->vertex_count * sizeof(float));
    memset(mesh->ny, 0, mesh->vertex_count * sizeof(float));
    memset(mesh->nz, 0, mesh->vertex_count * sizeof(float));

    /* Accumulate area-weighted face normals at vertices */
    for (uint32_t f = 0; f < mesh->face_count; f++) {
        uint32_t base = f * 3;
        uint32_t i0 = mesh->face_v[base + 0];
        uint32_t i1 = mesh->face_v[base + 1];
        uint32_t i2 = mesh->face_v[base + 2];

        /* Get face normal (unnormalized, length = 2 * area) */
        Vec3 n = mesh_face_normal(mesh, f);

        /* Add to each vertex (area-weighted by virtue of unnormalized cross product) */
        mesh->nx[i0] += n.x;
        mesh->ny[i0] += n.y;
        mesh->nz[i0] += n.z;

        mesh->nx[i1] += n.x;
        mesh->ny[i1] += n.y;
        mesh->nz[i1] += n.z;

        mesh->nx[i2] += n.x;
        mesh->ny[i2] += n.y;
        mesh->nz[i2] += n.z;
    }

    /* Normalize vertex normals */
    for (uint32_t i = 0; i < mesh->vertex_count; i++) {
        float nx = mesh->nx[i];
        float ny = mesh->ny[i];
        float nz = mesh->nz[i];

        float len = sqrtf(nx * nx + ny * ny + nz * nz);
        if (len > 1e-6f) {
            float inv_len = 1.0f / len;
            mesh->nx[i] = nx * inv_len;
            mesh->ny[i] = ny * inv_len;
            mesh->nz[i] = nz * inv_len;
        } else {
            /* Degenerate - use up vector */
            mesh->nx[i] = 0.0f;
            mesh->ny[i] = 1.0f;
            mesh->nz[i] = 0.0f;
        }
    }

    mesh->has_normals = true;
}
