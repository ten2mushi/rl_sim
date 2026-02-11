/**
 * Voxelization Implementation
 *
 * Three-phase sparse voxelization for efficient OBJ -> SDF conversion:
 *
 * Phase 1: Coarse Classification
 *   - Test each brick AABB against mesh BVH
 *   - Skip bricks with no intersection (90%+ in urban environments)
 *
 * Phase 2: Fine Classification
 *   - For candidate bricks, cast rays from center
 *   - Classify as INSIDE, OUTSIDE, or SURFACE
 *   - INSIDE/OUTSIDE use sentinel indices (no atlas memory)
 *
 * Phase 3: Surface Voxelization
 *   - Only SURFACE bricks get per-voxel SDF computation
 *   - Transfer face material to voxels via closest point query
 *   - Quantize SDF to int8 for storage
 */

#include "../include/obj_io.h"
#include <string.h>
#include <math.h>
#include <float.h>

/* ============================================================================
 * Constants and Default Options
 * ============================================================================ */

const VoxelizeOptions VOXELIZE_DEFAULTS = {
    .voxel_size = 0.1f,
    .padding = 0.0f,        /* Will be set to 1 brick if 0 */
    .preserve_materials = true,
    .max_bricks = 0,        /* Auto-calculate */
    .shell_mode = false,
    .shell_thickness = 0.0f /* Auto: 2 * voxel_size when shell_mode enabled */
};

/* RAY_DIRS[6] removed — multi-ray voting now in bvh_inside_outside_robust() */

/* ============================================================================
 * Phase 1: Coarse Brick Classification
 * ============================================================================ */

BrickClassification* classify_bricks_coarse(Arena* arena, const MeshBVH* bvh,
                                            const TriangleMesh* mesh,
                                            const WorldBrickMap* world) {
    if (!arena || !bvh || !mesh || !world) {
        return NULL;
    }

    /* Allocate classification structure */
    BrickClassification* cls = arena_alloc_type(arena, BrickClassification);
    if (!cls) return NULL;

    cls->grid_x = world->grid_x;
    cls->grid_y = world->grid_y;
    cls->grid_z = world->grid_z;
    uint32_t grid_total = cls->grid_x * cls->grid_y * cls->grid_z;

    cls->classes = arena_alloc_array(arena, BrickClass, grid_total);
    if (!cls->classes) return NULL;

    cls->outside_count = 0;
    cls->inside_count = 0;
    cls->surface_count = 0;
    cls->arena = arena;

    /* Initialize all as UNKNOWN */
    memset(cls->classes, BRICK_CLASS_UNKNOWN, grid_total * sizeof(BrickClass));

    float brick_size = world->brick_size_world;

    /* Test each brick AABB against mesh BVH */
    for (uint32_t bz = 0; bz < cls->grid_z; bz++) {
        for (uint32_t by = 0; by < cls->grid_y; by++) {
            for (uint32_t bx = 0; bx < cls->grid_x; bx++) {
                uint32_t idx = bx + by * cls->grid_x + bz * cls->grid_x * cls->grid_y;

                /* Compute brick AABB */
                Vec3 brick_min = VEC3(
                    world->world_min.x + (float)bx * brick_size,
                    world->world_min.y + (float)by * brick_size,
                    world->world_min.z + (float)bz * brick_size
                );
                Vec3 brick_max = vec3_add(brick_min, VEC3(brick_size, brick_size, brick_size));

                /* Expand slightly for safety margin (SDF falloff) */
                float margin = world->sdf_scale;
                Vec3 test_min = vec3_sub(brick_min, VEC3(margin, margin, margin));
                Vec3 test_max = vec3_add(brick_max, VEC3(margin, margin, margin));

                /* Test intersection */
                if (bvh_aabb_intersect(bvh, mesh, test_min, test_max)) {
                    /* Needs further classification */
                    cls->classes[idx] = BRICK_CLASS_SURFACE;
                    cls->surface_count++;
                } else {
                    /* No intersection - initially mark as OUTSIDE */
                    cls->classes[idx] = BRICK_CLASS_OUTSIDE;
                    cls->outside_count++;
                }
            }
        }
    }

    return cls;
}

/* ============================================================================
 * Phase 2: Fine Brick Classification
 * ============================================================================ */

void classify_bricks_fine(BrickClassification* classes, const MeshBVH* bvh,
                          const TriangleMesh* mesh, const WorldBrickMap* world,
                          const VoxelizeOptions* options) {
    if (!classes || !bvh || !mesh || !world) {
        return;
    }

    bool shell_mode = options && options->shell_mode;
    float half_thickness = 0.0f;
    if (shell_mode) {
        float thickness = (options->shell_thickness > 0.0f)
            ? options->shell_thickness : 2.0f * world->voxel_size;
        half_thickness = thickness * 0.5f;
    }

    float brick_size = world->brick_size_world;

    /* Process SURFACE bricks from Phase 1 to refine classification */
    for (uint32_t bz = 0; bz < classes->grid_z; bz++) {
        for (uint32_t by = 0; by < classes->grid_y; by++) {
            for (uint32_t bx = 0; bx < classes->grid_x; bx++) {
                uint32_t idx = bx + by * classes->grid_x + bz * classes->grid_x * classes->grid_y;

                if (classes->classes[idx] != BRICK_CLASS_SURFACE) {
                    continue; /* Already classified */
                }

                /* Compute brick center */
                Vec3 brick_center = VEC3(
                    world->world_min.x + ((float)bx + 0.5f) * brick_size,
                    world->world_min.y + ((float)by + 0.5f) * brick_size,
                    world->world_min.z + ((float)bz + 0.5f) * brick_size
                );

                /* Find closest point on mesh */
                Vec3 closest;
                float dist = bvh_closest_point(bvh, mesh, brick_center, &closest, NULL);

                /* Half-diagonal of brick */
                float brick_half_diag = brick_size * 0.866f; /* sqrt(3)/2 */

                /* If closest point is much farther than brick half-diagonal,
                   the entire brick is either all inside or all outside */
                if (dist > brick_half_diag + world->sdf_scale) {
                    if (shell_mode) {
                        /* In shell mode: classify based on distance to surface.
                           If the entire brick is beyond the shell, it's OUTSIDE.
                           No bricks are ever classified as INSIDE. */
                        if (dist - brick_half_diag > half_thickness) {
                            classes->classes[idx] = BRICK_CLASS_OUTSIDE;
                            classes->surface_count--;
                            classes->outside_count++;
                        }
                        /* else: brick overlaps the shell band, remains SURFACE */
                    } else {
                        /* Normal mode: check inside/outside with normal test */
                        float sign = bvh_inside_outside(bvh, mesh, brick_center);

                        if (sign > 0) {
                            /* Entire brick is outside */
                            classes->classes[idx] = BRICK_CLASS_OUTSIDE;
                            classes->surface_count--;
                            classes->outside_count++;
                        } else if (bvh->normal_coherence > 0.3f) {
                            /* Open mesh (terrain): no well-defined interior.
                             * Force to OUTSIDE to prevent solid slab underneath. */
                            classes->classes[idx] = BRICK_CLASS_OUTSIDE;
                            classes->surface_count--;
                            classes->outside_count++;
                        } else {
                            /* Closed mesh: entire brick is inside */
                            classes->classes[idx] = BRICK_CLASS_INSIDE;
                            classes->surface_count--;
                            classes->inside_count++;
                        }
                    }
                }
                /* else: remains SURFACE - needs per-voxel voxelization */
            }
        }
    }
}

/* ============================================================================
 * Phase 3: Surface Brick Voxelization
 * ============================================================================ */

/**
 * Voxelize a single brick
 */
static void voxelize_brick(WorldBrickMap* world, uint32_t bx, uint32_t by, uint32_t bz,
                           const MeshBVH* bvh, const TriangleMesh* mesh,
                           bool preserve_materials, bool shell_mode,
                           float shell_half_thickness) {
    /* Allocate brick in atlas */
    int32_t atlas_idx = world_alloc_brick(world, (int32_t)bx, (int32_t)by, (int32_t)bz);
    if (atlas_idx == BRICK_EMPTY_INDEX) {
        return; /* Atlas full */
    }

    /* Get SDF and material arrays */
    int8_t* sdf = world_brick_sdf(world, atlas_idx);
    uint8_t* material = world_brick_material(world, atlas_idx);

    if (!sdf || !material) {
        return;
    }

    float brick_size = world->brick_size_world;
    float voxel_size = world->voxel_size;
    float inv_sdf_scale = world->inv_sdf_scale;

    /* Brick origin in world space */
    Vec3 brick_origin = VEC3(
        world->world_min.x + (float)bx * brick_size,
        world->world_min.y + (float)by * brick_size,
        world->world_min.z + (float)bz * brick_size
    );

    /* Threshold for considering a voxel "close" to surface - within a few voxels */
    float close_threshold = voxel_size * 3.0f;

    /* Process each voxel in brick */
    for (int32_t vz = 0; vz < BRICK_SIZE; vz++) {
        for (int32_t vy = 0; vy < BRICK_SIZE; vy++) {
            for (int32_t vx = 0; vx < BRICK_SIZE; vx++) {
                uint32_t voxel_idx = (uint32_t)vx + ((uint32_t)vy << BRICK_SHIFT) +
                                     ((uint32_t)vz << (BRICK_SHIFT * 2));

                /* Voxel center in world space */
                Vec3 voxel_center = VEC3(
                    brick_origin.x + ((float)vx + 0.5f) * voxel_size,
                    brick_origin.y + ((float)vy + 0.5f) * voxel_size,
                    brick_origin.z + ((float)vz + 0.5f) * voxel_size
                );

                /* Find closest point on mesh */
                Vec3 closest;
                uint32_t closest_face;
                float distance = bvh_closest_point(bvh, mesh, voxel_center, &closest, &closest_face);

                /* Compute signed distance */
                float signed_dist;
                if (shell_mode) {
                    /* Shell mode: SDF = distance - half_thickness
                     * Negative within the shell band, positive outside.
                     * Both sides of the surface are treated symmetrically. */
                    signed_dist = distance - shell_half_thickness;
                } else {
                    /* Normal mode: determine sign via inside/outside test.
                     * Near the surface (within 2 voxels), use 3-ray majority
                     * voting for robustness against complex/symmetric geometry.
                     * Far voxels use single-ray (faster, sign errors don't
                     * create zero-crossings due to large |SDF|). */
                    float sign;
                    if (distance < voxel_size * 1.5f) {
                        sign = bvh_inside_outside_robust(bvh, mesh, voxel_center);
                    } else {
                        sign = bvh_inside_outside(bvh, mesh, voxel_center);
                    }

                    /* For CLOSED meshes: force voxels far from surface and outside
                     * the mesh bbox to be outside, preventing spurious surfaces at
                     * brick boundaries in padding regions.
                     * For OPEN meshes (terrain): skip this — forcing positive at the
                     * bbox boundary creates a floor artifact (zero-crossing between
                     * forced-positive padding voxels and ray-cast-negative interior). */
                    if (bvh->normal_coherence <= 0.3f && distance > close_threshold) {
                        bool outside_bbox =
                            voxel_center.x < mesh->bbox_min.x || voxel_center.x > mesh->bbox_max.x ||
                            voxel_center.y < mesh->bbox_min.y || voxel_center.y > mesh->bbox_max.y ||
                            voxel_center.z < mesh->bbox_min.z || voxel_center.z > mesh->bbox_max.z;

                        if (outside_bbox) {
                            sign = 1.0f; /* Force outside */
                        }
                    }

                    signed_dist = sign * distance;
                }

                /* Quantize and store */
                sdf[voxel_idx] = sdf_quantize(signed_dist, inv_sdf_scale);

                /* Transfer material from closest face */
                if (preserve_materials && closest_face != UINT32_MAX && signed_dist < 0) {
                    material[voxel_idx] = mesh->face_mat[closest_face];
                } else {
                    material[voxel_idx] = 0; /* Default material */
                }
            }
        }
    }
}

void voxelize_surface_bricks(WorldBrickMap* world, const BrickClassification* classes,
                             const MeshBVH* bvh, const TriangleMesh* mesh,
                             const VoxelizeOptions* options) {
    if (!world || !classes || !bvh || !mesh) {
        return;
    }

    bool shell_mode = options && options->shell_mode;
    float shell_half_thickness = 0.0f;
    if (shell_mode) {
        float thickness = (options->shell_thickness > 0.0f)
            ? options->shell_thickness : 2.0f * world->voxel_size;
        shell_half_thickness = thickness * 0.5f;
    }
    bool preserve_materials = options ? options->preserve_materials : true;

    /* Process each brick according to classification */
    for (uint32_t bz = 0; bz < classes->grid_z; bz++) {
        for (uint32_t by = 0; by < classes->grid_y; by++) {
            for (uint32_t bx = 0; bx < classes->grid_x; bx++) {
                uint32_t idx = bx + by * classes->grid_x + bz * classes->grid_x * classes->grid_y;
                BrickClass bc = classes->classes[idx];

                if (bc == BRICK_CLASS_SURFACE) {
                    /* Full voxelization needed */
                    voxelize_brick(world, bx, by, bz, bvh, mesh,
                                   preserve_materials, shell_mode, shell_half_thickness);
                }
                else if (bc == BRICK_CLASS_INSIDE) {
                    /* Mark as uniform inside (no atlas allocation) */
                    world_mark_brick_uniform_inside(world, (int32_t)bx, (int32_t)by, (int32_t)bz);
                }
                /* OUTSIDE bricks remain as BRICK_EMPTY_INDEX (default) */
            }
        }
    }
}

/* ============================================================================
 * Auto-Detection: Shell vs Solid Voxelization Mode
 * ============================================================================ */

/* Ratio of max interior distance to bounding sphere radius below which
 * a watertight mesh is considered a thin surface (not a solid volume).
 * Validated against: gyroid (0.02), sphere (0.58), torus (0.20),
 * thin-walled sphere (0.01), elongated cylinder (0.10). */
#define THIN_SURFACE_RATIO 0.05f

/* Number of quasi-random interior samples for thickness estimation. */
#define THICKNESS_SAMPLE_COUNT 128

/* Minimum interior samples required; fewer → assume thin surface. */
#define THICKNESS_MIN_INSIDE 4

/* Boundary edge ratio above which a non-watertight mesh is considered
 * disconnected geometry (building walls) rather than a continuous sheet
 * (terrain). Disconnected surfaces need shell mode because inside/outside
 * classification is undefined for them.
 * Validated: buildings (0.80), open cube (0.25), terrain (0.01). */
#define DISCONNECTED_BOUNDARY_RATIO 0.2f

void voxelize_options_auto_detect(VoxelizeOptions* opts, const MeshBVH* bvh,
                                  const TriangleMesh* mesh) {
    if (!opts || !bvh || !mesh) return;
    if (opts->shell_mode) return;           /* Already set by caller */

    float boundary_ratio = mesh_boundary_edge_ratio(mesh);
    bool watertight = boundary_ratio < 0.005f;

    if (!watertight) {
        /* Non-watertight mesh. Three sub-cases:
         *
         * 1. Disconnected surfaces (buildings, separate wall planes):
         *    High boundary ratio. Inside/outside undefined → shell mode.
         *
         * 2. Low-coherence open mesh (trimmed gyroid, single sheet):
         *    Inside/outside unreliable → shell mode.
         *
         * 3. Continuous terrain (high coherence, low boundary ratio):
         *    Inside/outside works (normals define consistent half-space).
         *    Leave unchanged — terrain-specific handling in Phase 2/3. */
        if (boundary_ratio > DISCONNECTED_BOUNDARY_RATIO ||
            bvh->normal_coherence <= 0.3f) {
            opts->shell_mode = true;
            if (opts->shell_thickness <= 0.0f)
                opts->shell_thickness = 2.0f * opts->voxel_size;
        }
        return;
    }

    /* Watertight + high coherence: solid or closed terrain → leave as-is */
    if (bvh->normal_coherence > 0.3f) return;

    /* Watertight + low coherence: could be a solid (sphere, cube) or a
     * thin closed surface (gyroid, Klein bottle). Distinguish by measuring
     * geometric thickness — sample quasi-random interior points and compare
     * the maximum distance-to-surface against the bounding sphere radius.
     * Thin surfaces have tiny max_inside_dist relative to the mesh size. */
    Vec3 extent = vec3_sub(mesh->bbox_max, mesh->bbox_min);
    float half_diag = 0.5f * vec3_length(extent);
    if (half_diag < 1e-6f) return;

    float max_inside_dist = 0.0f;
    int inside_count = 0;

    for (int i = 1; i <= THICKNESS_SAMPLE_COUNT; i++) {
        /* Quasi-random Halton-like sequence using irrational multipliers */
        Vec3 point = VEC3(
            mesh->bbox_min.x + fmodf((float)i * 0.6180339887f, 1.0f) * extent.x,
            mesh->bbox_min.y + fmodf((float)i * 0.3247179572f, 1.0f) * extent.y,
            mesh->bbox_min.z + fmodf((float)i * 0.2207440846f, 1.0f) * extent.z
        );

        if (bvh_inside_outside(bvh, mesh, point) < 0) {
            Vec3 closest;
            float dist = bvh_closest_point(bvh, mesh, point, &closest, NULL);
            if (dist > max_inside_dist) max_inside_dist = dist;
            inside_count++;
        }
    }

    /* Too few interior samples → degenerate or nearly zero-volume interior */
    if (inside_count < THICKNESS_MIN_INSIDE ||
        max_inside_dist / half_diag < THIN_SURFACE_RATIO) {
        opts->shell_mode = true;
        if (opts->shell_thickness <= 0.0f)
            opts->shell_thickness = 2.0f * opts->voxel_size;
    }
}

/* ============================================================================
 * High-Level mesh_to_sdf
 * ============================================================================ */

ObjIOResult mesh_to_sdf(Arena* arena, const TriangleMesh* mesh,
                        const VoxelizeOptions* options,
                        WorldBrickMap** out_world, char* error) {
    if (!arena || !mesh || !out_world) {
        if (error) snprintf(error, 256, "Invalid parameters");
        return OBJ_IO_ERROR_INVALID_PARAMETER;
    }

    if (!options) {
        options = &VOXELIZE_DEFAULTS;
    }

    *out_world = NULL;

    /* Validate mesh */
    if (mesh->vertex_count == 0 || mesh->face_count == 0) {
        if (error) snprintf(error, 256, "Empty mesh");
        return OBJ_IO_ERROR_EMPTY_MESH;
    }

    /* Compute world bounds with padding */
    float voxel_size = options->voxel_size;
    float brick_size = voxel_size * BRICK_SIZE;
    float padding = options->padding > 0 ? options->padding : brick_size;

    Vec3 world_min = vec3_sub(mesh->bbox_min, VEC3(padding, padding, padding));
    Vec3 world_max = vec3_add(mesh->bbox_max, VEC3(padding, padding, padding));

    /* Expand to user-specified bounds if larger */
    if (options->world_min.x != 0.0f || options->world_min.y != 0.0f || options->world_min.z != 0.0f ||
        options->world_max.x != 0.0f || options->world_max.y != 0.0f || options->world_max.z != 0.0f) {
        world_min = vec3_min(world_min, options->world_min);
        world_max = vec3_max(world_max, options->world_max);
    }

    /* Calculate grid dimensions */
    Vec3 world_size = vec3_sub(world_max, world_min);
    uint32_t grid_x = (uint32_t)ceilf(world_size.x / brick_size);
    uint32_t grid_y = (uint32_t)ceilf(world_size.y / brick_size);
    uint32_t grid_z = (uint32_t)ceilf(world_size.z / brick_size);
    uint32_t grid_total = grid_x * grid_y * grid_z;

    /* Determine max bricks */
    uint32_t max_bricks = options->max_bricks;
    if (max_bricks == 0) {
        max_bricks = grid_total;
        if (max_bricks < 1024) max_bricks = 1024;
    }

    /* Build BVH */
    MeshBVH* bvh = bvh_build(arena, mesh);
    if (!bvh) {
        if (error) snprintf(error, 256, "Failed to build BVH");
        return OBJ_IO_ERROR_BVH_BUILD_FAILED;
    }

    /* Auto-detect shell mode for thin surfaces (open or closed) */
    VoxelizeOptions effective_opts = *options;
    voxelize_options_auto_detect(&effective_opts, bvh, mesh);

    /* Create world brick map */
    WorldBrickMap* world = world_create(arena, world_min, world_max, voxel_size, max_bricks, 256);
    if (!world) {
        if (error) snprintf(error, 256, "Failed to create world brick map");
        return OBJ_IO_ERROR_OUT_OF_MEMORY;
    }

    /* Register materials from mesh (in the order they appear in OBJ's usemtl directives)
     * This ensures face_mat IDs match world material IDs. */
    if (mesh->material_names && mesh->material_name_count > 0) {
        for (uint32_t i = 0; i < mesh->material_name_count; i++) {
            if (mesh->material_names[i]) {
                /* Register with white color by default - actual colors come from MTL */
                world_register_material(world, mesh->material_names[i], VEC3(1.0f, 1.0f, 1.0f));
            }
        }
    }

    /* Phase 1: Coarse classification */
    BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world);
    if (!classes) {
        if (error) snprintf(error, 256, "Failed to classify bricks (phase 1)");
        return OBJ_IO_ERROR_VOXELIZE_FAILED;
    }

    /* Phase 2: Fine classification */
    classify_bricks_fine(classes, bvh, mesh, world, &effective_opts);

    /* Phase 3: Voxelize surface bricks */
    voxelize_surface_bricks(world, classes, bvh, mesh, &effective_opts);

    *out_world = world;
    return OBJ_IO_SUCCESS;
}

/* ============================================================================
 * High-Level obj_to_world
 * ============================================================================ */

ObjIOResult obj_to_world(Arena* arena, const char* path,
                         const VoxelizeOptions* options,
                         WorldBrickMap** out_world, char* error) {
    if (!arena || !path || !out_world || !options) {
        if (error) snprintf(error, 256, "Invalid parameters");
        return OBJ_IO_ERROR_INVALID_PARAMETER;
    }

    *out_world = NULL;

    /* Parse OBJ file */
    TriangleMesh* mesh = NULL;
    MtlLibrary* mtl = NULL;

    ObjIOResult result = obj_parse_file(arena, path, &OBJ_PARSE_DEFAULTS, &mesh, &mtl, error);
    if (result != OBJ_IO_SUCCESS) {
        return result;
    }

    /* Voxelize mesh */
    WorldBrickMap* world = NULL;
    result = mesh_to_sdf(arena, mesh, options, &world, error);
    if (result != OBJ_IO_SUCCESS) {
        return result;
    }

    /* Register materials from MTL */
    if (mtl) {
        mtl_register_materials(world, mtl);
    }

    *out_world = world;
    return OBJ_IO_SUCCESS;
}
