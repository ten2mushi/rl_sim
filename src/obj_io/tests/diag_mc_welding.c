/**
 * Diagnostic: Vertex duplication in Marching Cubes output
 *
 * Creates a sphere SDF (known-good, no sign issues), runs sdf_to_mesh()
 * with and without vertex welding, and reports:
 * - Total vertices vs unique positions
 * - Duplication ratio
 * - Vertex reduction from welding
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <math.h>

static WorldBrickMap* create_sphere_sdf(Arena* arena, float radius, float voxel_size) {
    float extent = radius * 1.5f;
    WorldBrickMap* world = world_create(arena,
        VEC3(-extent, -extent, -extent),
        VEC3(extent, extent, extent),
        voxel_size, 4096, 128);
    if (!world) return NULL;

    float brick_size = world->brick_size_world;

    for (float bz = world->world_min.z; bz < world->world_max.z; bz += brick_size) {
        for (float by = world->world_min.y; by < world->world_max.y; by += brick_size) {
            for (float bx = world->world_min.x; bx < world->world_max.x; bx += brick_size) {
                Vec3 brick_center = VEC3(bx + brick_size/2, by + brick_size/2, bz + brick_size/2);
                float dist_to_surface = fabsf(vec3_length(brick_center) - radius);

                if (dist_to_surface < brick_size * 2) {
                    int32_t bxi, byi, bzi;
                    world_pos_to_brick(world, brick_center, &bxi, &byi, &bzi);
                    int32_t atlas_idx = world_alloc_brick(world, bxi, byi, bzi);
                    if (atlas_idx < 0) continue;

                    int8_t* sdf = world_brick_sdf(world, atlas_idx);
                    for (int vz = 0; vz < 8; vz++) {
                        for (int vy = 0; vy < 8; vy++) {
                            for (int vx = 0; vx < 8; vx++) {
                                Vec3 voxel_pos = VEC3(
                                    bx + (vx + 0.5f) * world->voxel_size,
                                    by + (vy + 0.5f) * world->voxel_size,
                                    bz + (vz + 0.5f) * world->voxel_size
                                );
                                float d = vec3_length(voxel_pos) - radius;
                                int idx = vx + (vy << 3) + (vz << 6);
                                sdf[idx] = sdf_quantize(d, world->inv_sdf_scale);
                            }
                        }
                    }
                }
            }
        }
    }

    return world;
}

/**
 * Count unique vertex positions using brute-force comparison.
 * O(n^2) but fine for diagnostic purposes.
 */
static uint32_t count_unique_positions(const TriangleMesh* mesh, float tolerance) {
    float tol_sq = tolerance * tolerance;
    uint32_t unique = 0;

    for (uint32_t i = 0; i < mesh->vertex_count; i++) {
        bool is_dup = false;
        for (uint32_t j = 0; j < i; j++) {
            float dx = mesh->vx[i] - mesh->vx[j];
            float dy = mesh->vy[i] - mesh->vy[j];
            float dz = mesh->vz[i] - mesh->vz[j];
            if (dx*dx + dy*dy + dz*dz <= tol_sq) {
                is_dup = true;
                break;
            }
        }
        if (!is_dup) unique++;
    }

    return unique;
}

int main(void) {
    Arena* arena = arena_create(200 * 1024 * 1024);

    printf("=== Marching Cubes Vertex Welding Diagnostic ===\n\n");

    float radii[] = {0.5f, 1.0f};
    float voxel_sizes[] = {0.1f, 0.05f};

    for (int ri = 0; ri < 2; ri++) {
        for (int vi = 0; vi < 2; vi++) {
            float radius = radii[ri];
            float voxel_size = voxel_sizes[vi];

            printf("--- Sphere r=%.2f  voxel=%.3f ---\n", radius, voxel_size);

            WorldBrickMap* world = create_sphere_sdf(arena, radius, voxel_size);
            if (!world) {
                printf("  Failed to create SDF world\n\n");
                continue;
            }

            /* Extract WITHOUT welding */
            TriangleMesh* mesh_noweld = NULL;
            char error[256];
            MarchingCubesOptions opts_noweld = MARCHING_CUBES_DEFAULTS;
            opts_noweld.weld_vertices = false;
            ObjIOResult result = sdf_to_mesh(arena, world, &opts_noweld, &mesh_noweld, error);
            if (result != OBJ_IO_SUCCESS) {
                printf("  sdf_to_mesh (no weld) failed: %s\n\n", error);
                continue;
            }

            /* Extract WITH welding */
            TriangleMesh* mesh_weld = NULL;
            MarchingCubesOptions opts_weld = MARCHING_CUBES_DEFAULTS;
            opts_weld.weld_vertices = true;
            opts_weld.weld_tolerance = 0.0001f;
            result = sdf_to_mesh(arena, world, &opts_weld, &mesh_weld, error);
            if (result != OBJ_IO_SUCCESS) {
                printf("  sdf_to_mesh (weld) failed: %s\n\n", error);
                continue;
            }

            printf("  Without welding: %u vertices, %u faces\n",
                   mesh_noweld->vertex_count, mesh_noweld->face_count);
            printf("  With welding:    %u vertices, %u faces\n",
                   mesh_weld->vertex_count, mesh_weld->face_count);

            if (mesh_noweld->vertex_count > 0) {
                float reduction = 100.0f * (1.0f - (float)mesh_weld->vertex_count /
                                                     (float)mesh_noweld->vertex_count);
                printf("  Vertex reduction: %.1f%%\n", reduction);
                printf("  Face count preserved: %s\n",
                       mesh_noweld->face_count == mesh_weld->face_count ? "YES" : "NO");
            }

            /* Verify bbox is preserved */
            mesh_compute_bbox(mesh_noweld);
            mesh_compute_bbox(mesh_weld);
            float bbox_diff_x = fabsf(mesh_weld->bbox_max.x - mesh_noweld->bbox_max.x);
            float bbox_diff_y = fabsf(mesh_weld->bbox_max.y - mesh_noweld->bbox_max.y);
            float bbox_diff_z = fabsf(mesh_weld->bbox_max.z - mesh_noweld->bbox_max.z);
            printf("  BBox max diff: (%.6f, %.6f, %.6f)\n",
                   bbox_diff_x, bbox_diff_y, bbox_diff_z);

            /* Check normal quality */
            if (mesh_weld->has_normals && mesh_weld->vertex_count > 0) {
                float avg_dot = 0;
                uint32_t tested = 0;
                for (uint32_t i = 0; i < mesh_weld->vertex_count && tested < 100; i++) {
                    Vec3 pos = VEC3(mesh_weld->vx[i], mesh_weld->vy[i], mesh_weld->vz[i]);
                    float len = vec3_length(pos);
                    if (len > 0.001f) {
                        Vec3 expected = vec3_scale(pos, 1.0f / len);
                        Vec3 normal = VEC3(mesh_weld->nx[i], mesh_weld->ny[i], mesh_weld->nz[i]);
                        avg_dot += vec3_dot(normal, expected);
                        tested++;
                    }
                }
                if (tested > 0) avg_dot /= tested;
                printf("  Avg normal alignment (welded): %.4f (1.0 = perfect)\n", avg_dot);
            }

            printf("\n");
        }
    }

    printf("=== Diagnostic Complete ===\n");
    arena_destroy(arena);
    return 0;
}
