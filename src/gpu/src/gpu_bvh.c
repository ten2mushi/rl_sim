/**
 * GPU BVH and Triangle Data Upload
 *
 * Converts CPU MeshBVH (AoS BVHNode) and TriangleMesh (SoA vertices)
 * to GPU buffers for use by the voxelization compute shader.
 */

#include "gpu_hal.h"
#include "sdf_types.h"
#include "obj_io.h"

#include <string.h>
#include <stdlib.h>

/* ============================================================================
 * GPU BVH Creation
 * ============================================================================ */

GpuLinearBVH gpu_linear_bvh_create(GpuDevice* device, const MeshBVH* bvh) {
    GpuLinearBVH result = {0};

    if (!device || !bvh || bvh->node_count == 0) {
        return result;
    }

    /* Convert CPU BVHNode array to GpuBVHNode array */
    size_t node_bytes = bvh->node_count * sizeof(GpuBVHNode);
    result.nodes = gpu_buffer_create(device, node_bytes, GPU_MEMORY_SHARED);
    if (!result.nodes) return result;

    GpuBVHNode* gpu_nodes = (GpuBVHNode*)gpu_buffer_map(result.nodes);
    if (!gpu_nodes) {
        gpu_buffer_destroy(result.nodes);
        result.nodes = NULL;
        return result;
    }

    for (uint32_t i = 0; i < bvh->node_count; i++) {
        const BVHNode* src = &bvh->nodes[i];
        GpuBVHNode* dst = &gpu_nodes[i];

        dst->bbox_min_x = src->bbox_min.x;
        dst->bbox_min_y = src->bbox_min.y;
        dst->bbox_min_z = src->bbox_min.z;
        dst->bbox_max_x = src->bbox_max.x;
        dst->bbox_max_y = src->bbox_max.y;
        dst->bbox_max_z = src->bbox_max.z;
        dst->left = src->left;
        dst->right = src->right;
        dst->face_start = src->face_start;
        dst->face_count = src->face_count;
        dst->_pad0 = 0;
        dst->_pad1 = 0;
    }

    /* Upload face indices */
    /* Count total faces referenced by leaves */
    uint32_t max_face_idx = 0;
    for (uint32_t i = 0; i < bvh->node_count; i++) {
        if (bvh->nodes[i].left == bvh->nodes[i].right) {
            uint32_t end = bvh->nodes[i].face_start + bvh->nodes[i].face_count;
            if (end > max_face_idx) max_face_idx = end;
        }
    }

    size_t face_bytes = max_face_idx * sizeof(uint32_t);
    result.face_indices = gpu_buffer_create(device, face_bytes, GPU_MEMORY_SHARED);
    if (!result.face_indices) {
        gpu_buffer_destroy(result.nodes);
        result.nodes = NULL;
        return result;
    }

    gpu_buffer_upload(result.face_indices, bvh->face_indices, face_bytes, 0);

    result.node_count = bvh->node_count;
    result.face_count = max_face_idx;
    result.avg_normal_x = bvh->avg_normal.x;
    result.avg_normal_y = bvh->avg_normal.y;
    result.avg_normal_z = bvh->avg_normal.z;
    result.normal_coherence = bvh->normal_coherence;

    return result;
}

void gpu_linear_bvh_destroy(GpuLinearBVH* bvh) {
    if (!bvh) return;
    if (bvh->nodes) gpu_buffer_destroy(bvh->nodes);
    if (bvh->face_indices) gpu_buffer_destroy(bvh->face_indices);
    *bvh = (GpuLinearBVH){0};
}

/* ============================================================================
 * GPU Triangle Data Creation
 * ============================================================================ */

GpuTriangleData gpu_triangle_data_create(GpuDevice* device,
                                          const TriangleMesh* mesh) {
    GpuTriangleData result = {0};

    if (!device || !mesh || mesh->vertex_count == 0 || mesh->face_count == 0) {
        return result;
    }

    size_t vert_bytes = mesh->vertex_count * sizeof(float);
    size_t face_bytes = mesh->face_count * 3 * sizeof(uint32_t);
    size_t mat_bytes = mesh->face_count * sizeof(uint8_t);

    /* Upload vertex positions (SoA) */
    result.vertices_x = gpu_buffer_create(device, vert_bytes, GPU_MEMORY_SHARED);
    result.vertices_y = gpu_buffer_create(device, vert_bytes, GPU_MEMORY_SHARED);
    result.vertices_z = gpu_buffer_create(device, vert_bytes, GPU_MEMORY_SHARED);

    if (!result.vertices_x || !result.vertices_y || !result.vertices_z) {
        gpu_triangle_data_destroy(&result);
        return result;
    }

    gpu_buffer_upload(result.vertices_x, mesh->vx, vert_bytes, 0);
    gpu_buffer_upload(result.vertices_y, mesh->vy, vert_bytes, 0);
    gpu_buffer_upload(result.vertices_z, mesh->vz, vert_bytes, 0);

    /* Upload face vertex indices */
    result.face_v = gpu_buffer_create(device, face_bytes, GPU_MEMORY_SHARED);
    if (!result.face_v) {
        gpu_triangle_data_destroy(&result);
        return result;
    }
    gpu_buffer_upload(result.face_v, mesh->face_v, face_bytes, 0);

    /* Upload face materials */
    result.face_mat = gpu_buffer_create(device, mat_bytes, GPU_MEMORY_SHARED);
    if (!result.face_mat) {
        gpu_triangle_data_destroy(&result);
        return result;
    }
    gpu_buffer_upload(result.face_mat, mesh->face_mat, mat_bytes, 0);

    result.vertex_count = mesh->vertex_count;
    result.face_count = mesh->face_count;
    result.bbox_min_x = mesh->bbox_min.x;
    result.bbox_min_y = mesh->bbox_min.y;
    result.bbox_min_z = mesh->bbox_min.z;
    result.bbox_max_x = mesh->bbox_max.x;
    result.bbox_max_y = mesh->bbox_max.y;
    result.bbox_max_z = mesh->bbox_max.z;

    return result;
}

void gpu_triangle_data_destroy(GpuTriangleData* data) {
    if (!data) return;
    if (data->vertices_x) gpu_buffer_destroy(data->vertices_x);
    if (data->vertices_y) gpu_buffer_destroy(data->vertices_y);
    if (data->vertices_z) gpu_buffer_destroy(data->vertices_z);
    if (data->face_v) gpu_buffer_destroy(data->face_v);
    if (data->face_mat) gpu_buffer_destroy(data->face_mat);
    *data = (GpuTriangleData){0};
}
