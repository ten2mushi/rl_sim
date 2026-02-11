/**
 * Diagnostic: Ridge artifact investigation
 *
 * Key insight: bvh_inside_outside has a bbox early-out that returns "outside"
 * for points beyond the mesh bbox. On a real mountain, the bbox extends to the
 * tallest peak. Points above a shorter ridge are INSIDE the bbox, so the
 * flawed normal test runs. We reproduce this by adding a tall feature that
 * extends the bbox well above our test ridge.
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <math.h>

/* Create a terrain with a sharp ridge and a taller feature elsewhere.
 *
 * Geometry:
 * - Flat ground plane at z=0 covering [-5,-5] to [5,5]
 * - Sharp ridge along X-axis from x=-4 to x=4, peaking at z=ridge_height
 * - Tall pillar at corner (4,4) going up to z=pillar_height
 *
 * This ensures bbox_max.z = pillar_height >> ridge_height,
 * so points above the ridge are INSIDE the bbox.
 */
static TriangleMesh* create_terrain_with_ridge(Arena* arena,
                                                float ridge_height,
                                                float ridge_width,
                                                float pillar_height) {
    /* Estimate: ground ~8 tris, ridge ~4 tris, pillar ~20 tris = ~32 faces */
    TriangleMesh* mesh = mesh_create(arena, 64, 64);
    if (!mesh) return NULL;

    float hw = ridge_width * 0.5f;

    /* Ground plane corners (z=0) */
    uint32_t g0 = mesh->vertex_count; mesh_add_vertex(mesh, -5, -5, 0);
    uint32_t g1 = mesh->vertex_count; mesh_add_vertex(mesh,  5, -5, 0);
    uint32_t g2 = mesh->vertex_count; mesh_add_vertex(mesh,  5,  5, 0);
    uint32_t g3 = mesh->vertex_count; mesh_add_vertex(mesh, -5,  5, 0);

    /* Ridge vertices (along X, centered on Y=0) */
    uint32_t r0 = mesh->vertex_count; mesh_add_vertex(mesh, -4, -hw, 0);
    uint32_t r1 = mesh->vertex_count; mesh_add_vertex(mesh,  4, -hw, 0);
    uint32_t r2 = mesh->vertex_count; mesh_add_vertex(mesh,  4,  hw, 0);
    uint32_t r3 = mesh->vertex_count; mesh_add_vertex(mesh, -4,  hw, 0);
    uint32_t rp0 = mesh->vertex_count; mesh_add_vertex(mesh, -4, 0, ridge_height);
    uint32_t rp1 = mesh->vertex_count; mesh_add_vertex(mesh,  4, 0, ridge_height);

    /* Ground triangles (normals face up +Z) */
    /* Split ground into sections around the ridge */
    mesh_add_face(mesh, g0, g1, r1, 0); /* south */
    mesh_add_face(mesh, g0, r1, r0, 0);
    mesh_add_face(mesh, r3, r2, g2, 0); /* north */
    mesh_add_face(mesh, r3, g2, g3, 0);
    mesh_add_face(mesh, g0, r0, r3, 0); /* west */
    mesh_add_face(mesh, g0, r3, g3, 0);
    mesh_add_face(mesh, r1, g1, g2, 0); /* east */
    mesh_add_face(mesh, r1, g2, r2, 0);

    /* Ridge left slope (normal points -Y, +Z) */
    /* Vertices: r0(-4,-hw,0), r1(4,-hw,0), rp1(4,0,h), rp0(-4,0,h) */
    mesh_add_face(mesh, r0, r1, rp1, 0);
    mesh_add_face(mesh, r0, rp1, rp0, 0);

    /* Ridge right slope (normal points +Y, +Z) */
    /* Vertices: r3(-4,hw,0), rp0(-4,0,h), rp1(4,0,h), r2(4,hw,0) */
    mesh_add_face(mesh, r3, rp0, rp1, 0);
    mesh_add_face(mesh, r3, rp1, r2, 0);

    /* Tall pillar at corner (3,3) to extend bbox upward */
    float ps = 0.3f; /* pillar half-size */
    float px = 3.0f, py = 3.0f;
    uint32_t pb0 = mesh->vertex_count; mesh_add_vertex(mesh, px-ps, py-ps, 0);
    uint32_t pb1 = mesh->vertex_count; mesh_add_vertex(mesh, px+ps, py-ps, 0);
    uint32_t pb2 = mesh->vertex_count; mesh_add_vertex(mesh, px+ps, py+ps, 0);
    uint32_t pb3 = mesh->vertex_count; mesh_add_vertex(mesh, px-ps, py+ps, 0);
    uint32_t pt0 = mesh->vertex_count; mesh_add_vertex(mesh, px-ps, py-ps, pillar_height);
    uint32_t pt1 = mesh->vertex_count; mesh_add_vertex(mesh, px+ps, py-ps, pillar_height);
    uint32_t pt2 = mesh->vertex_count; mesh_add_vertex(mesh, px+ps, py+ps, pillar_height);
    uint32_t pt3 = mesh->vertex_count; mesh_add_vertex(mesh, px-ps, py+ps, pillar_height);

    /* Pillar sides */
    mesh_add_face(mesh, pb0, pb1, pt1, 0); mesh_add_face(mesh, pb0, pt1, pt0, 0);
    mesh_add_face(mesh, pb1, pb2, pt2, 0); mesh_add_face(mesh, pb1, pt2, pt1, 0);
    mesh_add_face(mesh, pb2, pb3, pt3, 0); mesh_add_face(mesh, pb2, pt3, pt2, 0);
    mesh_add_face(mesh, pb3, pb0, pt0, 0); mesh_add_face(mesh, pb3, pt0, pt3, 0);
    /* Pillar top */
    mesh_add_face(mesh, pt0, pt1, pt2, 0); mesh_add_face(mesh, pt0, pt2, pt3, 0);

    mesh_compute_bbox(mesh);
    return mesh;
}

static Vec3 get_face_normal(const TriangleMesh* mesh, uint32_t face) {
    Vec3 n = mesh_face_normal(mesh, face);
    float len = vec3_length(n);
    if (len > 1e-8f) n = vec3_scale(n, 1.0f / len);
    return n;
}

static void test_point(const MeshBVH* bvh, const TriangleMesh* mesh,
                       Vec3 point, const char* label, float expected_sign) {
    Vec3 closest;
    uint32_t closest_face;
    float dist = bvh_closest_point(bvh, mesh, point, &closest, &closest_face);
    float sign = bvh_inside_outside(bvh, mesh, point);

    Vec3 normal = (closest_face != UINT32_MAX) ? get_face_normal(mesh, closest_face) : VEC3(0,0,0);
    Vec3 to_point = vec3_sub(point, closest);
    float dot = vec3_dot(to_point, normal);

    const char* result = (sign > 0) ? "OUTSIDE" : "INSIDE";
    bool correct = ((expected_sign > 0) == (sign > 0));
    const char* status = correct ? "OK" : "** WRONG **";

    printf("  %-40s  dist=%.3f  face=%2u  n=(%.2f,%.2f,%.2f)  "
           "dot=%+.4f  -> %-7s  %s\n",
           label, dist, closest_face, normal.x, normal.y, normal.z, dot, result, status);
}

int main(void) {
    Arena* arena = arena_create(100 * 1024 * 1024);

    printf("=== Ridge Artifact Diagnostic (bbox-aware) ===\n\n");

    /* ---------------------------------------------------------------
     * Test A: Sharp ridge with tall pillar extending bbox
     * Ridge: width=1.0, height=5, Pillar: height=20
     * Points above ridge (z=5-10) are within bbox (z_max=20)
     * --------------------------------------------------------------- */
    printf("--- Test A: Sharp ridge (w=1.0, h=5) + pillar (h=20) ---\n");
    {
        TriangleMesh* mesh = create_terrain_with_ridge(arena, 5.0f, 1.0f, 20.0f);
        MeshBVH* bvh = bvh_build(arena, mesh);

        printf("  Mesh: %u verts, %u faces, bbox z=[%.1f, %.1f]\n",
               mesh->vertex_count, mesh->face_count, mesh->bbox_min.z, mesh->bbox_max.z);

        printf("\n  Points ABOVE ridge (all should be OUTSIDE):\n");
        /* Directly above ridge peak */
        for (float dz = 0.5f; dz <= 5.0f; dz += 0.5f) {
            char label[80];
            snprintf(label, sizeof(label), "above peak (0,0,%.1f)", 5.0f + dz);
            test_point(bvh, mesh, VEC3(0, 0, 5.0f + dz), label, 1.0f);
        }

        printf("\n  Points ABOVE ridge, offset from centerline:\n");
        for (float dy = 0.1f; dy <= 2.0f; dy += 0.1f) {
            char label[80];
            snprintf(label, sizeof(label), "above+right (0,%.1f,6.0)", dy);
            test_point(bvh, mesh, VEC3(0, dy, 6.0f), label, 1.0f);
        }

        printf("\n  Points ABOVE ridge, various offsets:\n");
        for (float dy = -2.0f; dy <= 2.0f; dy += 0.2f) {
            for (float dz = 0.5f; dz <= 3.0f; dz += 1.0f) {
                char label[80];
                snprintf(label, sizeof(label), "(0,%.1f,%.1f)", dy, 5.0f + dz);
                Vec3 p = VEC3(0, dy, 5.0f + dz);
                Vec3 closest;
                uint32_t closest_face;
                float dist = bvh_closest_point(bvh, mesh, p, &closest, &closest_face);
                float sign = bvh_inside_outside(bvh, mesh, p);
                if (sign < 0) {
                    /* Only print misclassified points */
                    Vec3 normal = get_face_normal(mesh, closest_face);
                    Vec3 to_point = vec3_sub(p, closest);
                    float dot = vec3_dot(to_point, normal);
                    printf("  ** WRONG: %-30s  dist=%.3f face=%u n=(%.2f,%.2f,%.2f) dot=%+.4f -> INSIDE\n",
                           label, dist, closest_face, normal.x, normal.y, normal.z, dot);
                }
            }
        }
        printf("\n");
    }

    /* ---------------------------------------------------------------
     * Test B: Very sharp ridge (narrow, tall)
     * --------------------------------------------------------------- */
    printf("--- Test B: Very sharp ridge (w=0.2, h=8) + pillar (h=20) ---\n");
    {
        TriangleMesh* mesh = create_terrain_with_ridge(arena, 8.0f, 0.2f, 20.0f);
        MeshBVH* bvh = bvh_build(arena, mesh);

        printf("  Mesh: %u verts, %u faces, bbox z=[%.1f, %.1f]\n",
               mesh->vertex_count, mesh->face_count, mesh->bbox_min.z, mesh->bbox_max.z);

        printf("\n  Scanning grid above ridge for misclassified points...\n");
        int wrong_count = 0;
        for (float x = -3.0f; x <= 3.0f; x += 1.0f) {
            for (float dy = -2.0f; dy <= 2.0f; dy += 0.1f) {
                for (float dz = 0.2f; dz <= 5.0f; dz += 0.2f) {
                    Vec3 p = VEC3(x, dy, 8.0f + dz);
                    float sign = bvh_inside_outside(bvh, mesh, p);
                    if (sign < 0) {
                        Vec3 closest;
                        uint32_t face;
                        float dist = bvh_closest_point(bvh, mesh, p, &closest, &face);
                        Vec3 normal = get_face_normal(mesh, face);
                        Vec3 to_point = vec3_sub(p, closest);
                        float dot = vec3_dot(to_point, normal);
                        if (wrong_count < 30) {
                            printf("  ** WRONG: (%.1f,%.1f,%.1f) dist=%.3f face=%u n=(%.2f,%.2f,%.2f) dot=%+.4f\n",
                                   p.x, p.y, p.z, dist, face, normal.x, normal.y, normal.z, dot);
                        }
                        wrong_count++;
                    }
                }
            }
        }
        printf("  Total misclassified: %d / ~%d points above ridge\n\n", wrong_count,
               7 * 40 * 25);
    }

    /* ---------------------------------------------------------------
     * Test C: Brick classification on sharp ridge terrain
     * --------------------------------------------------------------- */
    printf("--- Test C: Brick classification of sharp ridge terrain ---\n");
    {
        TriangleMesh* mesh = create_terrain_with_ridge(arena, 5.0f, 0.4f, 15.0f);
        MeshBVH* bvh = bvh_build(arena, mesh);

        float voxel_size = 0.5f;
        float brick_size = voxel_size * BRICK_SIZE;
        float padding = brick_size;

        WorldBrickMap* world = world_create(arena,
            vec3_sub(mesh->bbox_min, VEC3(padding, padding, padding)),
            vec3_add(mesh->bbox_max, VEC3(padding, padding, padding)),
            voxel_size, 4096, 64);

        BrickClassification* cls = classify_bricks_coarse(arena, bvh, mesh, world);
        VoxelizeOptions opts = VOXELIZE_DEFAULTS;
        opts.voxel_size = voxel_size;
        classify_bricks_fine(cls, bvh, mesh, world, &opts);

        printf("  Classification: outside=%u, inside=%u, surface=%u\n",
               cls->outside_count, cls->inside_count, cls->surface_count);

        /* Find INSIDE bricks above the ridge peak */
        float peak_z = 5.0f;
        uint32_t bad = 0;
        for (uint32_t bz = 0; bz < cls->grid_z; bz++) {
            for (uint32_t by = 0; by < cls->grid_y; by++) {
                for (uint32_t bx = 0; bx < cls->grid_x; bx++) {
                    uint32_t idx = bx + by * cls->grid_x + bz * cls->grid_x * cls->grid_y;
                    if (cls->classes[idx] == BRICK_CLASS_INSIDE) {
                        float brick_z_min = world->world_min.z + (float)bz * brick_size;
                        Vec3 bc = VEC3(
                            world->world_min.x + ((float)bx + 0.5f) * brick_size,
                            world->world_min.y + ((float)by + 0.5f) * brick_size,
                            world->world_min.z + ((float)bz + 0.5f) * brick_size
                        );
                        if (brick_z_min > peak_z) {
                            if (bad < 20) {
                                Vec3 closest;
                                uint32_t face;
                                float dist = bvh_closest_point(bvh, mesh, bc, &closest, &face);
                                printf("  ** INSIDE brick above peak: (%u,%u,%u) center=(%.1f,%.1f,%.1f) dist=%.2f face=%u\n",
                                       bx, by, bz, bc.x, bc.y, bc.z, dist, face);
                            }
                            bad++;
                        }
                    }
                }
            }
        }
        printf("  INSIDE bricks above ridge peak: %u %s\n", bad,
               bad > 0 ? "** ARTIFACT SOURCE! **" : "(good)");
    }

    printf("\n=== Diagnostic Complete ===\n");
    arena_destroy(arena);
    return 0;
}
