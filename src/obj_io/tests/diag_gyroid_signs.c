/**
 * Diagnostic: Sign-flip detection for inside/outside classification
 *
 * Tests bvh_inside_outside vs bvh_inside_outside_robust on a synthetic
 * wavy surface where single-ray classification is known to fail.
 * Reports misclassification rates for each method.
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <math.h>

/**
 * Create a synthetic wavy surface mesh (multiple overlapping folds).
 * This approximates gyroid-like complexity where normals cancel out
 * and single-ray inside/outside is unreliable.
 *
 * Geometry: sinusoidal sheet with multiple periods in a box.
 * z = A * sin(freq * x) * cos(freq * y) centered in [-2, 2]^3
 */
static TriangleMesh* create_wavy_surface(Arena* arena, int grid_n, float amplitude, float freq) {
    uint32_t n = (uint32_t)grid_n;
    uint32_t vert_count = (n + 1) * (n + 1) * 2; /* top + bottom sheets */
    uint32_t face_count = n * n * 2 * 2 + n * 4 * 2; /* 2 sheets + side walls */

    TriangleMesh* mesh = mesh_create(arena, vert_count + 256, face_count + 256);
    if (!mesh) return NULL;

    float extent = 2.0f;
    float step = 2.0f * extent / (float)n;

    /* Create top sheet: z = amplitude * sin(freq*x) * cos(freq*y) + offset */
    float offset = 0.3f; /* half-thickness offset */
    for (uint32_t iy = 0; iy <= n; iy++) {
        for (uint32_t ix = 0; ix <= n; ix++) {
            float x = -extent + (float)ix * step;
            float y = -extent + (float)iy * step;
            float z = amplitude * sinf(freq * x) * cosf(freq * y) + offset;
            mesh_add_vertex(mesh, x, y, z);
        }
    }

    /* Create bottom sheet: z = amplitude * sin(freq*x) * cos(freq*y) - offset */
    uint32_t bot_base = (n + 1) * (n + 1);
    for (uint32_t iy = 0; iy <= n; iy++) {
        for (uint32_t ix = 0; ix <= n; ix++) {
            float x = -extent + (float)ix * step;
            float y = -extent + (float)iy * step;
            float z = amplitude * sinf(freq * x) * cosf(freq * y) - offset;
            mesh_add_vertex(mesh, x, y, z);
        }
    }

    /* Top sheet faces (normal up) */
    for (uint32_t iy = 0; iy < n; iy++) {
        for (uint32_t ix = 0; ix < n; ix++) {
            uint32_t i00 = iy * (n + 1) + ix;
            uint32_t i10 = i00 + 1;
            uint32_t i01 = i00 + (n + 1);
            uint32_t i11 = i01 + 1;
            mesh_add_face(mesh, i00, i10, i11, 0);
            mesh_add_face(mesh, i00, i11, i01, 0);
        }
    }

    /* Bottom sheet faces (normal down) */
    for (uint32_t iy = 0; iy < n; iy++) {
        for (uint32_t ix = 0; ix < n; ix++) {
            uint32_t i00 = bot_base + iy * (n + 1) + ix;
            uint32_t i10 = i00 + 1;
            uint32_t i01 = i00 + (n + 1);
            uint32_t i11 = i01 + 1;
            mesh_add_face(mesh, i00, i11, i10, 0);
            mesh_add_face(mesh, i00, i01, i11, 0);
        }
    }

    /* Side walls to close the mesh */
    for (uint32_t ix = 0; ix < n; ix++) {
        /* y=-extent edge */
        uint32_t t0 = ix, t1 = ix + 1;
        uint32_t b0 = bot_base + ix, b1 = bot_base + ix + 1;
        mesh_add_face(mesh, t0, b0, b1, 0);
        mesh_add_face(mesh, t0, b1, t1, 0);

        /* y=+extent edge */
        t0 = n * (n + 1) + ix;
        t1 = t0 + 1;
        b0 = bot_base + n * (n + 1) + ix;
        b1 = b0 + 1;
        mesh_add_face(mesh, t0, t1, b1, 0);
        mesh_add_face(mesh, t0, b1, b0, 0);
    }
    for (uint32_t iy = 0; iy < n; iy++) {
        /* x=-extent edge */
        uint32_t t0 = iy * (n + 1), t1 = (iy + 1) * (n + 1);
        uint32_t b0 = bot_base + iy * (n + 1), b1 = bot_base + (iy + 1) * (n + 1);
        mesh_add_face(mesh, t0, t1, b1, 0);
        mesh_add_face(mesh, t0, b1, b0, 0);

        /* x=+extent edge */
        t0 = iy * (n + 1) + n;
        t1 = (iy + 1) * (n + 1) + n;
        b0 = bot_base + iy * (n + 1) + n;
        b1 = bot_base + (iy + 1) * (n + 1) + n;
        mesh_add_face(mesh, t0, b0, b1, 0);
        mesh_add_face(mesh, t0, b1, t1, 0);
    }

    mesh_compute_bbox(mesh);
    return mesh;
}

int main(void) {
    Arena* arena = arena_create(200 * 1024 * 1024);

    printf("=== Gyroid Sign-Flip Diagnostic ===\n\n");

    /* Test with increasing frequency (more folds = more ambiguity) */
    float freqs[] = {1.0f, 2.0f, 3.14159f, 5.0f};
    int num_freqs = 4;

    for (int fi = 0; fi < num_freqs; fi++) {
        float freq = freqs[fi];
        float amplitude = 1.0f;

        TriangleMesh* mesh = create_wavy_surface(arena, 40, amplitude, freq);
        if (!mesh) {
            printf("  Failed to create mesh for freq=%.2f\n", freq);
            continue;
        }

        MeshBVH* bvh = bvh_build(arena, mesh);
        if (!bvh) {
            printf("  Failed to build BVH for freq=%.2f\n", freq);
            continue;
        }

        printf("--- freq=%.2f  amplitude=%.1f  coherence=%.3f  avg_normal=(%.3f,%.3f,%.3f) ---\n",
               freq, amplitude, bvh->normal_coherence,
               bvh->avg_normal.x, bvh->avg_normal.y, bvh->avg_normal.z);
        printf("  Mesh: %u verts, %u faces\n", mesh->vertex_count, mesh->face_count);

        /* Test grid of points near the surface */
        int single_errors = 0, robust_errors = 0, total_tests = 0;
        float step = 0.15f;

        for (float x = -1.8f; x <= 1.8f; x += step) {
            for (float y = -1.8f; y <= 1.8f; y += step) {
                /* Point just above the surface */
                float surf_z = amplitude * sinf(freq * x) * cosf(freq * y);
                for (float dz = 0.05f; dz <= 0.25f; dz += 0.05f) {
                    Vec3 above = VEC3(x, y, surf_z + dz + 0.3f);
                    Vec3 below = VEC3(x, y, surf_z - dz - 0.3f);

                    /* Both should be outside (points are above/below the shell) */
                    float sign_single_above = bvh_inside_outside(bvh, mesh, above);
                    float sign_robust_above = bvh_inside_outside_robust(bvh, mesh, above);
                    float sign_single_below = bvh_inside_outside(bvh, mesh, below);
                    float sign_robust_below = bvh_inside_outside_robust(bvh, mesh, below);

                    /* Check: points outside the shell should be classified outside */
                    Vec3 closest;
                    float dist_above = bvh_closest_point(bvh, mesh, above, &closest, NULL);
                    float dist_below = bvh_closest_point(bvh, mesh, below, &closest, NULL);

                    /* Only check points that are clearly outside (dist > shell offset) */
                    if (dist_above > 0.05f) {
                        total_tests++;
                        /* For a closed shell, points far from surface should be outside */
                        if (sign_single_above < 0) single_errors++;
                        if (sign_robust_above < 0) robust_errors++;
                    }
                    if (dist_below > 0.05f) {
                        total_tests++;
                        if (sign_single_below < 0) single_errors++;
                        if (sign_robust_below < 0) robust_errors++;
                    }
                }
            }
        }

        float single_rate = (total_tests > 0) ? 100.0f * single_errors / total_tests : 0.0f;
        float robust_rate = (total_tests > 0) ? 100.0f * robust_errors / total_tests : 0.0f;

        printf("  Test points: %d\n", total_tests);
        printf("  Single-ray errors: %d (%.2f%%)\n", single_errors, single_rate);
        printf("  Robust 3-ray errors: %d (%.2f%%)\n", robust_errors, robust_rate);
        printf("  Improvement: %.1fx fewer errors\n\n",
               (robust_errors > 0) ? (float)single_errors / robust_errors : (single_errors > 0 ? 999.0f : 1.0f));
    }

    printf("=== Diagnostic Complete ===\n");
    arena_destroy(arena);
    return 0;
}
