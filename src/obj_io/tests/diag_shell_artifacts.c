/**
 * Diagnostic: Shell Mode Artifact Hypothesis Testing
 *
 * Tests potential root causes of wall artifacts in gyroid shell-mode roundtrip.
 * Loads an OBJ mesh, voxelizes with shell mode, and probes the resulting SDF
 * field to produce quantitative evidence for/against each hypothesis.
 *
 * Usage: diag_shell_artifacts <input.obj> [voxel_size]
 *
 * Hypotheses tested:
 *   H1: Shell overlap — nearby gyroid surfaces within 2*half_thickness merge
 *   H2: Mesh boundary caps — flat faces at bbox create spurious shell walls
 *   H3: Phase 2 brick misclassification discards shell-containing bricks
 *   H4: Directional SDF asymmetry — channels open in Z but blocked in X/Y
 *   H5: Channel width vs shell thickness — channels narrower than 2*half_thickness
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * H2: Mesh Boundary Cap Detection
 *
 * Counts faces whose normal is axis-aligned AND whose center lies at the
 * mesh bounding box boundary. These would create spurious shell walls.
 * ============================================================================ */
static void test_h2_boundary_caps(const TriangleMesh* mesh) {
    printf("\n=== H2: Mesh Boundary Cap Detection ===\n");

    float epsilon = 0.01f; /* tolerance for bbox boundary check */
    uint32_t cap_count[6] = {0}; /* -X, +X, -Y, +Y, -Z, +Z */
    const char* cap_names[6] = {"-X", "+X", "-Y", "+Y", "-Z", "+Z"};

    Vec3 bmin = mesh->bbox_min;
    Vec3 bmax = mesh->bbox_max;

    /* Also track normals histogram for overall mesh character */
    uint32_t normal_bins[6] = {0}; /* faces with dominant normal in each axis dir */

    for (uint32_t f = 0; f < mesh->face_count; f++) {
        uint32_t i0 = mesh->face_v[f * 3 + 0];
        uint32_t i1 = mesh->face_v[f * 3 + 1];
        uint32_t i2 = mesh->face_v[f * 3 + 2];

        /* Face vertices */
        Vec3 v0 = VEC3(mesh->vx[i0], mesh->vy[i0], mesh->vz[i0]);
        Vec3 v1 = VEC3(mesh->vx[i1], mesh->vy[i1], mesh->vz[i1]);
        Vec3 v2 = VEC3(mesh->vx[i2], mesh->vy[i2], mesh->vz[i2]);

        /* Face normal (unnormalized for area-weighting, then normalize) */
        Vec3 e1 = vec3_sub(v1, v0);
        Vec3 e2 = vec3_sub(v2, v0);
        Vec3 normal = vec3_cross(e1, e2);
        float len = vec3_length(normal);
        if (len < 1e-10f) continue;
        normal = vec3_scale(normal, 1.0f / len);

        /* Face center */
        Vec3 center = vec3_scale(vec3_add(vec3_add(v0, v1), v2), 1.0f / 3.0f);

        /* Classify dominant normal direction */
        float ax = fabsf(normal.x), ay = fabsf(normal.y), az = fabsf(normal.z);
        if (ax > ay && ax > az) {
            normal_bins[normal.x < 0 ? 0 : 1]++;
        } else if (ay > az) {
            normal_bins[normal.y < 0 ? 2 : 3]++;
        } else {
            normal_bins[normal.z < 0 ? 4 : 5]++;
        }

        /* Check if face is axis-aligned AND at bbox boundary */
        float threshold = 0.9f; /* normal alignment threshold */

        /* Check each axis direction */
        if (ax > threshold) {
            /* All 3 vertices should be near the same bbox face */
            bool all_at_min = fabsf(v0.x - bmin.x) < epsilon &&
                              fabsf(v1.x - bmin.x) < epsilon &&
                              fabsf(v2.x - bmin.x) < epsilon;
            bool all_at_max = fabsf(v0.x - bmax.x) < epsilon &&
                              fabsf(v1.x - bmax.x) < epsilon &&
                              fabsf(v2.x - bmax.x) < epsilon;
            if (all_at_min) cap_count[0]++;
            if (all_at_max) cap_count[1]++;
        }
        if (ay > threshold) {
            bool all_at_min = fabsf(v0.y - bmin.y) < epsilon &&
                              fabsf(v1.y - bmin.y) < epsilon &&
                              fabsf(v2.y - bmin.y) < epsilon;
            bool all_at_max = fabsf(v0.y - bmax.y) < epsilon &&
                              fabsf(v1.y - bmax.y) < epsilon &&
                              fabsf(v2.y - bmax.y) < epsilon;
            if (all_at_min) cap_count[2]++;
            if (all_at_max) cap_count[3]++;
        }
        if (az > threshold) {
            bool all_at_min = fabsf(v0.z - bmin.z) < epsilon &&
                              fabsf(v1.z - bmin.z) < epsilon &&
                              fabsf(v2.z - bmin.z) < epsilon;
            bool all_at_max = fabsf(v0.z - bmax.z) < epsilon &&
                              fabsf(v1.z - bmax.z) < epsilon &&
                              fabsf(v2.z - bmax.z) < epsilon;
            if (all_at_min) cap_count[4]++;
            if (all_at_max) cap_count[5]++;
        }
    }

    printf("  Face normal distribution:\n");
    printf("    -X: %u  +X: %u  -Y: %u  +Y: %u  -Z: %u  +Z: %u\n",
           normal_bins[0], normal_bins[1], normal_bins[2],
           normal_bins[3], normal_bins[4], normal_bins[5]);

    uint32_t total_caps = 0;
    printf("  Boundary cap faces (axis-aligned at bbox):\n");
    for (int i = 0; i < 6; i++) {
        printf("    %s: %u faces\n", cap_names[i], cap_count[i]);
        total_caps += cap_count[i];
    }

    if (total_caps > 0) {
        printf("  VERDICT: %u cap faces detected — these create shell walls at boundaries\n",
               total_caps);
    } else {
        printf("  VERDICT: No boundary caps found — artifact cause is elsewhere\n");
    }
}

/* ============================================================================
 * H5/H1: Channel Width vs Shell Thickness
 *
 * Samples closest-point distance along axis-aligned lines through the mesh
 * center. Local maxima of distance represent channel midpoints (farthest from
 * any surface). If a channel midpoint distance < half_thickness, the shell
 * from both sides overlaps and the channel is closed.
 * ============================================================================ */
static void test_h5_channel_width(const MeshBVH* bvh, const TriangleMesh* mesh,
                                   float voxel_size) {
    printf("\n=== H5: Channel Width vs Shell Thickness ===\n");

    float half_thickness = voxel_size; /* default: shell_thickness = 2 * voxel_size */
    printf("  Shell half_thickness: %.4f\n", half_thickness);

    Vec3 center = vec3_scale(vec3_add(mesh->bbox_min, mesh->bbox_max), 0.5f);
    Vec3 extent = vec3_sub(mesh->bbox_max, mesh->bbox_min);
    float step = voxel_size * 0.25f; /* Fine sampling */

    const char* axis_names[3] = {"X", "Y", "Z"};
    float axis_extent[3] = {extent.x, extent.y, extent.z};

    for (int axis = 0; axis < 3; axis++) {
        float half_len = axis_extent[axis] * 0.5f;
        int num_samples = (int)(2.0f * half_len / step) + 1;

        /* Track local maxima of distance (channel midpoints) */
        float prev_dist = 0.0f, curr_dist = 0.0f;
        float min_channel_width = 1e6f;
        int channel_count = 0;
        int closed_channel_count = 0;
        bool in_channel = false; /* currently in a region where dist > half_thickness */

        printf("  Axis %s (%.1f to %.1f, %d samples):\n",
               axis_names[axis],
               -half_len, half_len, num_samples);

        for (int i = 0; i < num_samples; i++) {
            float t = -half_len + (float)i * step;
            Vec3 probe;
            if (axis == 0) probe = VEC3(center.x + t, center.y, center.z);
            else if (axis == 1) probe = VEC3(center.x, center.y + t, center.z);
            else probe = VEC3(center.x, center.y, center.z + t);

            Vec3 closest;
            float dist = bvh_closest_point(bvh, mesh, probe, &closest, NULL);

            /* Detect transitions */
            if (dist > half_thickness && !in_channel) {
                in_channel = true;
                channel_count++;
            } else if (dist <= half_thickness && in_channel) {
                in_channel = false;
            }

            /* Track local maxima (channel midpoint widths) */
            if (i > 0 && i < num_samples - 1) {
                /* We check if curr_dist was a local max next iteration */
                float next_dist = dist;
                if (curr_dist > prev_dist && curr_dist > next_dist && curr_dist > 0.0f) {
                    /* Local maximum at previous sample */
                    if (curr_dist < min_channel_width) {
                        min_channel_width = curr_dist;
                    }
                    if (curr_dist < half_thickness) {
                        closed_channel_count++;
                    }
                }
            }

            prev_dist = curr_dist;
            curr_dist = dist;
        }

        printf("    Channels detected: %d\n", channel_count);
        if (min_channel_width < 1e5f) {
            printf("    Narrowest channel midpoint distance: %.4f\n", min_channel_width);
            printf("    Ratio to half_thickness: %.2f\n", min_channel_width / half_thickness);
        }
        printf("    Channels with midpoint < half_thickness (CLOSED): %d\n",
               closed_channel_count);

        if (closed_channel_count > 0) {
            printf("    VERDICT: Shell overlap closes %d channels on %s axis\n",
                   closed_channel_count, axis_names[axis]);
        } else if (channel_count == 0) {
            printf("    VERDICT: No channels found on %s axis line — possible wall\n",
                   axis_names[axis]);
        } else {
            printf("    VERDICT: All channels wider than shell — OK\n");
        }
    }
}

/* ============================================================================
 * H3: Phase 2 Brick Misclassification Check
 *
 * After Phase 2 classification, checks if any OUTSIDE bricks have corners
 * within reach of the shell. If so, they were incorrectly eliminated.
 * ============================================================================ */
static void test_h3_brick_classification(const BrickClassification* classes,
                                          const MeshBVH* bvh,
                                          const TriangleMesh* mesh,
                                          const WorldBrickMap* world,
                                          float half_thickness) {
    printf("\n=== H3: Phase 2 Brick Classification ===\n");
    printf("  Grid: %ux%ux%u = %u bricks\n",
           classes->grid_x, classes->grid_y, classes->grid_z,
           classes->grid_x * classes->grid_y * classes->grid_z);
    printf("  Surface: %u  Inside: %u  Outside: %u\n",
           classes->surface_count, classes->inside_count, classes->outside_count);

    if (classes->inside_count > 0) {
        printf("  WARNING: %u INSIDE bricks in shell mode — should be 0\n",
               classes->inside_count);
    }

    float brick_size = world->brick_size_world;
    uint32_t misclassified = 0;
    uint32_t misclassified_by_axis[3] = {0}; /* near which bbox face */

    for (uint32_t bz = 0; bz < classes->grid_z; bz++) {
        for (uint32_t by = 0; by < classes->grid_y; by++) {
            for (uint32_t bx = 0; bx < classes->grid_x; bx++) {
                uint32_t idx = bx + by * classes->grid_x +
                               bz * classes->grid_x * classes->grid_y;
                if (classes->classes[idx] != BRICK_CLASS_OUTSIDE) continue;

                /* Check all 8 corners of the brick */
                Vec3 brick_min = VEC3(
                    world->world_min.x + (float)bx * brick_size,
                    world->world_min.y + (float)by * brick_size,
                    world->world_min.z + (float)bz * brick_size
                );

                bool any_corner_in_shell = false;
                for (int cz = 0; cz <= 1 && !any_corner_in_shell; cz++) {
                    for (int cy = 0; cy <= 1 && !any_corner_in_shell; cy++) {
                        for (int cx = 0; cx <= 1 && !any_corner_in_shell; cx++) {
                            Vec3 corner = VEC3(
                                brick_min.x + (float)cx * brick_size,
                                brick_min.y + (float)cy * brick_size,
                                brick_min.z + (float)cz * brick_size
                            );
                            Vec3 closest;
                            float dist = bvh_closest_point(bvh, mesh, corner,
                                                            &closest, NULL);
                            if (dist < half_thickness) {
                                any_corner_in_shell = true;
                            }
                        }
                    }
                }

                if (any_corner_in_shell) {
                    misclassified++;
                    /* Which axis is this brick near? */
                    Vec3 brick_center = VEC3(
                        brick_min.x + 0.5f * brick_size,
                        brick_min.y + 0.5f * brick_size,
                        brick_min.z + 0.5f * brick_size
                    );
                    Vec3 mesh_center = vec3_scale(
                        vec3_add(mesh->bbox_min, mesh->bbox_max), 0.5f);
                    Vec3 diff = vec3_sub(brick_center, mesh_center);
                    float ax = fabsf(diff.x), ay = fabsf(diff.y), az = fabsf(diff.z);
                    if (ax > ay && ax > az) misclassified_by_axis[0]++;
                    else if (ay > az) misclassified_by_axis[1]++;
                    else misclassified_by_axis[2]++;
                }
            }
        }
    }

    printf("  OUTSIDE bricks with corners in shell range: %u\n", misclassified);
    if (misclassified > 0) {
        printf("    By dominant axis: X=%u  Y=%u  Z=%u\n",
               misclassified_by_axis[0], misclassified_by_axis[1],
               misclassified_by_axis[2]);
        printf("  VERDICT: Phase 2 misclassifies some bricks — shell is clipped\n");
    } else {
        printf("  VERDICT: Phase 2 classification looks correct\n");
    }
}

/* ============================================================================
 * H4: Directional SDF Asymmetry
 *
 * After voxelization, samples the SDF along axis-aligned lines through the
 * world center. Counts zero-crossings (negative->positive transitions) which
 * correspond to shell boundaries. A correctly voxelized gyroid should have
 * similar zero-crossing counts in all 3 directions.
 * ============================================================================ */
static void test_h4_sdf_asymmetry(const WorldBrickMap* world) {
    printf("\n=== H4: Directional SDF Asymmetry (post-voxelization) ===\n");

    Vec3 center = vec3_scale(vec3_add(world->world_min, world->world_max), 0.5f);
    Vec3 extent = vec3_sub(world->world_max, world->world_min);
    float step = world->voxel_size;

    const char* axis_names[3] = {"X", "Y", "Z"};
    float axis_extent[3] = {extent.x, extent.y, extent.z};

    for (int axis = 0; axis < 3; axis++) {
        float half_len = axis_extent[axis] * 0.5f;
        int num_samples = (int)(2.0f * half_len / step) + 1;

        int zero_crossings = 0;
        int negative_count = 0;
        int positive_count = 0;
        float prev_sdf = 0.0f;
        bool first = true;

        /* Also track longest continuous negative run (= wall thickness) */
        int max_neg_run = 0;
        int curr_neg_run = 0;

        for (int i = 0; i < num_samples; i++) {
            float t = -half_len + (float)i * step;
            Vec3 probe;
            if (axis == 0) probe = VEC3(center.x + t, center.y, center.z);
            else if (axis == 1) probe = VEC3(center.x, center.y + t, center.z);
            else probe = VEC3(center.x, center.y, center.z + t);

            float sdf = world_sdf_query(world, probe);

            if (sdf < 0) {
                negative_count++;
                curr_neg_run++;
                if (curr_neg_run > max_neg_run) max_neg_run = curr_neg_run;
            } else {
                positive_count++;
                curr_neg_run = 0;
            }

            if (!first) {
                if ((prev_sdf < 0 && sdf >= 0) || (prev_sdf >= 0 && sdf < 0)) {
                    zero_crossings++;
                }
            }
            prev_sdf = sdf;
            first = false;
        }

        printf("  Axis %s: %d samples, %d negative, %d positive, %d zero-crossings\n",
               axis_names[axis], num_samples, negative_count, positive_count,
               zero_crossings);
        printf("    Max continuous negative run: %d voxels (%.3f units)\n",
               max_neg_run, (float)max_neg_run * step);
        printf("    Negative fraction: %.1f%%\n",
               100.0f * (float)negative_count / (float)num_samples);
    }

    printf("  NOTE: Similar zero-crossings across axes = symmetric.\n");
    printf("        Large negative runs = thick walls or merged shells.\n");
}

/* ============================================================================
 * H4b: Multi-Line SDF Probes
 *
 * Samples SDF along a GRID of parallel lines per axis (not just the center
 * line). This reveals whether walls are localized or span the full cross-section.
 * ============================================================================ */
static void test_h4b_multiline_probes(const WorldBrickMap* world) {
    printf("\n=== H4b: Multi-Line SDF Probes (wall extent) ===\n");

    Vec3 center = vec3_scale(vec3_add(world->world_min, world->world_max), 0.5f);
    Vec3 extent = vec3_sub(world->world_max, world->world_min);
    float step = world->voxel_size;
    int probe_grid = 5; /* 5x5 grid of parallel lines per axis */

    const char* axis_names[3] = {"X", "Y", "Z"};
    float axis_extent[3] = {extent.x, extent.y, extent.z};

    for (int axis = 0; axis < 3; axis++) {
        float half_len = axis_extent[axis] * 0.5f;
        int num_samples = (int)(2.0f * half_len / step) + 1;

        /* Perpendicular axes */
        int perp1 = (axis + 1) % 3;
        int perp2 = (axis + 2) % 3;
        float perp1_extent = axis_extent[perp1];
        float perp2_extent = axis_extent[perp2];

        int lines_with_long_negative = 0;
        int total_lines = 0;

        printf("  Axis %s (%dx%d probe grid):\n", axis_names[axis],
               probe_grid, probe_grid);

        for (int g1 = 0; g1 < probe_grid; g1++) {
            for (int g2 = 0; g2 < probe_grid; g2++) {
                /* Offset from center in perpendicular directions */
                float off1 = (perp1_extent * 0.3f) *
                    ((float)g1 / (float)(probe_grid - 1) - 0.5f);
                float off2 = (perp2_extent * 0.3f) *
                    ((float)g2 / (float)(probe_grid - 1) - 0.5f);

                int max_neg_run = 0;
                int curr_neg_run = 0;

                for (int i = 0; i < num_samples; i++) {
                    float t = -half_len + (float)i * step;
                    float coords[3];
                    coords[axis] = ((float[]){center.x, center.y, center.z})[axis] + t;
                    coords[perp1] = ((float[]){center.x, center.y, center.z})[perp1] + off1;
                    coords[perp2] = ((float[]){center.x, center.y, center.z})[perp2] + off2;

                    Vec3 probe = VEC3(coords[0], coords[1], coords[2]);
                    float sdf = world_sdf_query(world, probe);

                    if (sdf < 0) {
                        curr_neg_run++;
                        if (curr_neg_run > max_neg_run) max_neg_run = curr_neg_run;
                    } else {
                        curr_neg_run = 0;
                    }
                }

                total_lines++;
                /* A "long" negative run is more than 3 voxels = likely a wall, not shell */
                if (max_neg_run > 3) {
                    lines_with_long_negative++;
                }
            }
        }

        printf("    Lines with long negative runs (>3 voxels): %d/%d\n",
               lines_with_long_negative, total_lines);
        if (lines_with_long_negative > total_lines / 2) {
            printf("    VERDICT: Widespread walls blocking %s-axis channels\n",
                   axis_names[axis]);
        } else if (lines_with_long_negative > 0) {
            printf("    VERDICT: Partial walls on some %s-axis lines\n",
                   axis_names[axis]);
        } else {
            printf("    VERDICT: No walls blocking %s-axis channels\n",
                   axis_names[axis]);
        }
    }
}

/* ============================================================================
 * H_boundary: Boundary Edge Shell Artifacts
 *
 * At the mesh bbox boundary, trimmed gyroid surface creates open edges.
 * The shell wraps around these edges, creating tube-like structures.
 * This test measures how far the shell extends beyond the mesh bbox.
 * ============================================================================ */
static void test_boundary_edge_shells(const MeshBVH* bvh, const TriangleMesh* mesh,
                                       const WorldBrickMap* world) {
    printf("\n=== H_boundary: Boundary Edge Shell Extension ===\n");

    float voxel_size = world->voxel_size;
    float half_thickness = voxel_size; /* default shell */

    Vec3 bmin = mesh->bbox_min;
    Vec3 bmax = mesh->bbox_max;
    Vec3 center = vec3_scale(vec3_add(bmin, bmax), 0.5f);

    const char* axis_names[3] = {"X", "Y", "Z"};
    float step = voxel_size * 0.5f;

    /* For each axis, probe outward from the bbox face to see how far
     * the shell extends beyond the mesh boundary */
    for (int axis = 0; axis < 3; axis++) {
        float face_min = ((float[]){bmin.x, bmin.y, bmin.z})[axis];
        float face_max = ((float[]){bmax.x, bmax.y, bmax.z})[axis];

        /* Sample along the axis through the center, starting from beyond bbox */
        int outside_negative_min = 0;
        int outside_negative_max = 0;

        /* Check 20 points beyond each face */
        for (int i = 1; i <= 20; i++) {
            float coords[3];
            coords[0] = center.x;
            coords[1] = center.y;
            coords[2] = center.z;

            /* Beyond min face */
            coords[axis] = face_min - (float)i * step;
            Vec3 probe_min = VEC3(coords[0], coords[1], coords[2]);
            float sdf_min = world_sdf_query(world, probe_min);
            if (sdf_min < 0) outside_negative_min++;

            /* Beyond max face */
            coords[axis] = face_max + (float)i * step;
            Vec3 probe_max = VEC3(coords[0], coords[1], coords[2]);
            float sdf_max = world_sdf_query(world, probe_max);
            if (sdf_max < 0) outside_negative_max++;
        }

        printf("  %s axis: negative SDF beyond bbox: -%s=%d  +%s=%d (of 20 probes)\n",
               axis_names[axis], axis_names[axis], outside_negative_min,
               axis_names[axis], outside_negative_max);
    }

    /* Also check: are there shell surfaces AT the bbox faces?
     * Sample a grid on each bbox face and count negative SDF */
    printf("\n  Shell presence at mesh bbox faces:\n");
    int grid_res = 10;

    for (int axis = 0; axis < 3; axis++) {
        int perp1 = (axis + 1) % 3;
        int perp2 = (axis + 2) % 3;

        float face_val[2] = {
            ((float[]){bmin.x, bmin.y, bmin.z})[axis],
            ((float[]){bmax.x, bmax.y, bmax.z})[axis]
        };
        float p1_min = ((float[]){bmin.x, bmin.y, bmin.z})[perp1];
        float p1_max = ((float[]){bmax.x, bmax.y, bmax.z})[perp1];
        float p2_min = ((float[]){bmin.x, bmin.y, bmin.z})[perp2];
        float p2_max = ((float[]){bmax.x, bmax.y, bmax.z})[perp2];

        for (int side = 0; side < 2; side++) {
            int negative_at_face = 0;
            int total = 0;

            for (int g1 = 0; g1 < grid_res; g1++) {
                for (int g2 = 0; g2 < grid_res; g2++) {
                    float t1 = p1_min + (p1_max - p1_min) * ((float)g1 + 0.5f) / (float)grid_res;
                    float t2 = p2_min + (p2_max - p2_min) * ((float)g2 + 0.5f) / (float)grid_res;

                    float coords[3];
                    coords[axis] = face_val[side];
                    coords[perp1] = t1;
                    coords[perp2] = t2;

                    Vec3 probe = VEC3(coords[0], coords[1], coords[2]);
                    float sdf = world_sdf_query(world, probe);
                    if (sdf < 0) negative_at_face++;
                    total++;
                }
            }

            printf("    %s%s face: %d/%d negative (%.0f%%)\n",
                   side == 0 ? "-" : "+", axis_names[axis],
                   negative_at_face, total,
                   100.0f * (float)negative_at_face / (float)total);
        }
    }
}

/* ============================================================================
 * H_sdf_dump: Detailed SDF slice through center
 *
 * Prints a 2D ASCII heatmap of the SDF on a slice through the world center,
 * for each axis pair. Shows the actual shell structure visually.
 * ============================================================================ */
static void test_sdf_slice(const WorldBrickMap* world) {
    printf("\n=== SDF Slices Through Center ===\n");

    Vec3 center = vec3_scale(vec3_add(world->world_min, world->world_max), 0.5f);
    Vec3 extent = vec3_sub(world->world_max, world->world_min);
    float step = world->voxel_size;

    /* XY slice (looking down Z) */
    printf("\n  XY slice at Z=%.2f (. = outside, # = inside shell, @ = deep inside):\n",
           center.z);
    {
        int nx = (int)(extent.x * 0.6f / step);
        int ny = (int)(extent.y * 0.6f / step);
        /* Limit resolution for readability */
        int skip = 1;
        if (nx > 60) skip = nx / 60 + 1;

        for (int iy = ny; iy >= -ny; iy -= skip) {
            printf("    ");
            for (int ix = -nx; ix <= nx; ix += skip) {
                Vec3 probe = VEC3(
                    center.x + (float)ix * step,
                    center.y + (float)iy * step,
                    center.z
                );
                float sdf = world_sdf_query(world, probe);
                if (sdf > 0.01f) printf(".");
                else if (sdf > -0.01f) printf(":");
                else if (sdf > -0.05f) printf("#");
                else printf("@");
            }
            printf("\n");
        }
    }

    /* XZ slice (looking from Y) */
    printf("\n  XZ slice at Y=%.2f (. = outside, # = inside shell, @ = deep inside):\n",
           center.y);
    {
        int nx = (int)(extent.x * 0.6f / step);
        int nz = (int)(extent.z * 0.6f / step);
        int skip = 1;
        if (nx > 60) skip = nx / 60 + 1;

        for (int iz = nz; iz >= -nz; iz -= skip) {
            printf("    ");
            for (int ix = -nx; ix <= nx; ix += skip) {
                Vec3 probe = VEC3(
                    center.x + (float)ix * step,
                    center.y,
                    center.z + (float)iz * step
                );
                float sdf = world_sdf_query(world, probe);
                if (sdf > 0.01f) printf(".");
                else if (sdf > -0.01f) printf(":");
                else if (sdf > -0.05f) printf("#");
                else printf("@");
            }
            printf("\n");
        }
    }

    /* YZ slice (looking from X) */
    printf("\n  YZ slice at X=%.2f (. = outside, # = inside shell, @ = deep inside):\n",
           center.x);
    {
        int ny = (int)(extent.y * 0.6f / step);
        int nz = (int)(extent.z * 0.6f / step);
        int skip = 1;
        if (ny > 60) skip = ny / 60 + 1;

        for (int iz = nz; iz >= -nz; iz -= skip) {
            printf("    ");
            for (int iy = -ny; iy <= ny; iy += skip) {
                Vec3 probe = VEC3(
                    center.x,
                    center.y + (float)iy * step,
                    center.z + (float)iz * step
                );
                float sdf = world_sdf_query(world, probe);
                if (sdf > 0.01f) printf(".");
                else if (sdf > -0.01f) printf(":");
                else if (sdf > -0.05f) printf("#");
                else printf("@");
            }
            printf("\n");
        }
    }
}

/* ============================================================================
 * Main
 * ============================================================================ */
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr,
                "Usage: %s <input.obj> [voxel_size]\n"
                "\n"
                "Tests hypotheses for shell mode artifacts in gyroid roundtrip.\n",
                argv[0]);
        return 1;
    }

    const char* input_path = argv[1];
    float voxel_size = (argc > 2) ? (float)atof(argv[2]) : 0.1f;
    char error[256] = {0};

    Arena* arena = arena_create(1024ULL * 1024 * 1024 * 2); /* 2 GB */
    if (!arena) {
        fprintf(stderr, "Failed to create arena\n");
        return 1;
    }

    printf("=== Shell Mode Artifact Diagnostic ===\n");
    printf("  Input: %s\n", input_path);
    printf("  Voxel size: %.4f\n", voxel_size);
    printf("  Shell thickness: %.4f (half: %.4f)\n",
           2.0f * voxel_size, voxel_size);

    /* Parse mesh */
    TriangleMesh* mesh = NULL;
    MtlLibrary* mtl = NULL;
    ObjIOResult result = obj_parse_file(arena, input_path, &OBJ_PARSE_DEFAULTS,
                                         &mesh, &mtl, error);
    if (result != OBJ_IO_SUCCESS) {
        fprintf(stderr, "Parse failed: %s\n", error);
        arena_destroy(arena);
        return 1;
    }

    printf("  Mesh: %u verts, %u faces\n", mesh->vertex_count, mesh->face_count);
    printf("  BBox: (%.3f, %.3f, %.3f) to (%.3f, %.3f, %.3f)\n",
           mesh->bbox_min.x, mesh->bbox_min.y, mesh->bbox_min.z,
           mesh->bbox_max.x, mesh->bbox_max.y, mesh->bbox_max.z);
    printf("  Watertight: %s\n", mesh_is_watertight(mesh) ? "yes" : "no");

    /* Build BVH */
    MeshBVH* bvh = bvh_build(arena, mesh);
    if (!bvh) {
        fprintf(stderr, "Failed to build BVH\n");
        arena_destroy(arena);
        return 1;
    }
    printf("  Normal coherence: %.4f\n", bvh->normal_coherence);

    /* === Run pre-voxelization tests === */
    test_h2_boundary_caps(mesh);
    test_h5_channel_width(bvh, mesh, voxel_size);

    /* === Voxelize with shell mode === */
    printf("\n=== Voxelization (shell mode) ===\n");

    VoxelizeOptions opts = VOXELIZE_DEFAULTS;
    opts.voxel_size = voxel_size;
    opts.shell_mode = true;
    opts.shell_thickness = 2.0f * voxel_size;

    /* Manual phases for intermediate inspection */
    float brick_size = voxel_size * BRICK_SIZE;
    float padding = brick_size;
    Vec3 world_min = vec3_sub(mesh->bbox_min, VEC3(padding, padding, padding));
    Vec3 world_max = vec3_add(mesh->bbox_max, VEC3(padding, padding, padding));

    uint32_t grid_x = (uint32_t)ceilf((world_max.x - world_min.x) / brick_size);
    uint32_t grid_y = (uint32_t)ceilf((world_max.y - world_min.y) / brick_size);
    uint32_t grid_z = (uint32_t)ceilf((world_max.z - world_min.z) / brick_size);
    uint32_t grid_total = grid_x * grid_y * grid_z;
    uint32_t max_bricks = grid_total < 1024 ? 1024 : grid_total;

    WorldBrickMap* world = world_create(arena, world_min, world_max,
                                         voxel_size, max_bricks, 256);
    if (!world) {
        fprintf(stderr, "Failed to create world\n");
        arena_destroy(arena);
        return 1;
    }

    /* Phase 1 */
    BrickClassification* classes = classify_bricks_coarse(arena, bvh, mesh, world);
    if (!classes) {
        fprintf(stderr, "Phase 1 failed\n");
        arena_destroy(arena);
        return 1;
    }
    printf("  Phase 1: %u surface, %u outside\n",
           classes->surface_count, classes->outside_count);

    /* Phase 2 */
    classify_bricks_fine(classes, bvh, mesh, world, &opts);
    printf("  Phase 2: %u surface, %u inside, %u outside\n",
           classes->surface_count, classes->inside_count, classes->outside_count);

    /* Test Phase 2 classification */
    test_h3_brick_classification(classes, bvh, mesh, world, voxel_size);

    /* Phase 3 */
    voxelize_surface_bricks(world, classes, bvh, mesh, &opts);

    WorldStats stats = world_get_stats(world);
    printf("  Phase 3: %u active bricks\n", stats.active_bricks);

    /* === Run post-voxelization tests === */
    test_h4_sdf_asymmetry(world);
    test_h4b_multiline_probes(world);
    test_boundary_edge_shells(bvh, mesh, world);
    test_sdf_slice(world);

    printf("\n=== Summary ===\n");
    printf("  Review the results above to identify the root cause.\n");
    printf("  Key indicators:\n");
    printf("    - H2 cap faces > 0 → mesh has boundary walls (fix: remove caps)\n");
    printf("    - H5 channel < half_thickness → shell overlap (fix: reduce thickness)\n");
    printf("    - H3 misclassified > 0 → Phase 2 too aggressive (fix: margin)\n");
    printf("    - H4 asymmetric crossings → directional bias in voxelization\n");
    printf("    - SDF slices show walls → visual confirmation of artifact location\n");

    arena_destroy(arena);
    return 0;
}
