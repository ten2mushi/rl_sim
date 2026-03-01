/**
 * OBJ I/O Module - Bidirectional conversion between OBJ mesh files and WorldBrickMap SDF
 *
 * Provides:
 * 1. Streaming OBJ/MTL parser for large mesh files
 * 2. BVH-accelerated mesh queries
 * 3. Three-phase sparse voxelization (OBJ -> SDF)
 * 4. Marching Cubes extraction (SDF -> OBJ)
 * 5. Material/color preservation through roundtrip
 *
 * Dependencies:
 * - foundation (Arena, Vec3, SIMD utilities)
 * - world_brick_map (WorldBrickMap, MaterialMetadata)
 */

#ifndef OBJ_IO_H
#define OBJ_IO_H

#include "foundation.h"
#include "world_brick_map.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Result Types and Error Handling
 * ============================================================================ */

/**
 * Result codes for OBJ I/O operations
 */
typedef enum ObjIOResult {
    OBJ_IO_SUCCESS = 0,
    OBJ_IO_ERROR_FILE_NOT_FOUND,
    OBJ_IO_ERROR_FILE_READ,
    OBJ_IO_ERROR_FILE_WRITE,
    OBJ_IO_ERROR_OUT_OF_MEMORY,
    OBJ_IO_ERROR_INVALID_FORMAT,
    OBJ_IO_ERROR_EMPTY_MESH,
    OBJ_IO_ERROR_BVH_BUILD_FAILED,
    OBJ_IO_ERROR_VOXELIZE_FAILED,
    OBJ_IO_ERROR_MARCHING_CUBES_FAILED,
    OBJ_IO_ERROR_INVALID_PARAMETER
} ObjIOResult;

/**
 * Get human-readable error message for result code
 */
const char* obj_io_result_string(ObjIOResult result);

/* ============================================================================
 * Section 2: TriangleMesh - SoA Mesh Representation
 * ============================================================================ */

/**
 * Triangle mesh with Structure-of-Arrays layout for SIMD efficiency
 *
 * Vertices stored as separate x/y/z arrays for vectorized operations.
 * Faces reference vertex indices (3 per triangle).
 * Optional normals and per-face material IDs.
 */
typedef struct TriangleMesh {
    /* Vertex positions (SoA layout) */
    float* vx;              /* X coordinates [vertex_count] */
    float* vy;              /* Y coordinates [vertex_count] */
    float* vz;              /* Z coordinates [vertex_count] */
    uint32_t vertex_count;
    uint32_t vertex_capacity;

    /* Vertex normals (SoA layout, optional) */
    float* nx;              /* Normal X [vertex_count] */
    float* ny;              /* Normal Y [vertex_count] */
    float* nz;              /* Normal Z [vertex_count] */
    bool has_normals;

    /* Face vertex indices (3 per face) */
    uint32_t* face_v;       /* Vertex indices [face_count * 3] */
    uint8_t* face_mat;      /* Material ID per face [face_count] */
    uint32_t face_count;
    uint32_t face_capacity;

    /* Material names (order matches face_mat IDs) */
    char** material_names;     /* Material names [material_name_count] */
    uint32_t material_name_count;

    /* Bounding box */
    Vec3 bbox_min;
    Vec3 bbox_max;

    /* Memory arena */
    Arena* arena;
} TriangleMesh;

/**
 * Create an empty triangle mesh
 *
 * @param arena           Memory arena for allocation
 * @param vertex_capacity Initial vertex capacity
 * @param face_capacity   Initial face capacity
 * @return New mesh or NULL on failure
 */
TriangleMesh* mesh_create(Arena* arena, uint32_t vertex_capacity, uint32_t face_capacity);

/**
 * Add a vertex to the mesh
 *
 * @param mesh Mesh to modify
 * @param x    X coordinate
 * @param y    Y coordinate
 * @param z    Z coordinate
 * @return Vertex index or UINT32_MAX on failure
 */
uint32_t mesh_add_vertex(TriangleMesh* mesh, float x, float y, float z);

/* ============================================================================
 * Section 2a: WeldContext - Spatial Hash Vertex Deduplication
 * ============================================================================ */

/**
 * Context for vertex welding via spatial hashing.
 * Deduplicates vertices within a distance tolerance using a hash table
 * keyed by quantized spatial coordinates.
 */
typedef struct WeldContext {
    uint32_t* hash_table;       /* hash -> first vertex index (UINT32_MAX = empty) */
    uint32_t* hash_next;        /* per-vertex collision chain */
    uint32_t hash_capacity;     /* power-of-2 */
    uint32_t hash_next_capacity;/* allocated size of hash_next */
    float inv_cell_size;        /* 1.0 / weld_tolerance */
    float tolerance_sq;         /* weld_tolerance^2 for distance check */
    TriangleMesh* mesh;
} WeldContext;

/**
 * Create a weld context for vertex deduplication
 *
 * @param arena           Memory arena
 * @param mesh            Target mesh
 * @param tolerance       Distance threshold for considering vertices identical
 * @param estimated_verts Estimated number of unique vertices
 * @return WeldContext or NULL on failure
 */
WeldContext* weld_context_create(Arena* arena, TriangleMesh* mesh,
                                  float tolerance, uint32_t estimated_verts);

/**
 * Add a vertex with deduplication — returns existing index if within tolerance
 *
 * @param ctx Weld context
 * @param x   X coordinate
 * @param y   Y coordinate
 * @param z   Z coordinate
 * @return Vertex index (existing or new), or UINT32_MAX on failure
 */
uint32_t mesh_add_vertex_welded(WeldContext* ctx, float x, float y, float z);

/**
 * Compute the fraction of edges that are boundary edges (shared by only 1 face).
 * Uses a scratch arena internally (freed after computation).
 *
 * Returns 0.0 for watertight meshes (no boundary edges),
 * ~1.0 for fully disconnected surfaces (most edges are boundary).
 * Terrain meshes typically have low ratios (<0.05), while disconnected
 * building geometry (separate wall planes) has high ratios (>0.5).
 *
 * @param mesh Mesh to analyze
 * @return Boundary edge ratio in [0, 1]
 */
float mesh_boundary_edge_ratio(const TriangleMesh* mesh);

/**
 * Check if mesh is watertight (no boundary edges).
 * Equivalent to mesh_boundary_edge_ratio(mesh) < 0.005f.
 *
 * @param mesh Mesh to check
 * @return True if mesh has no boundary edges (or negligible count)
 */
bool mesh_is_watertight(const TriangleMesh* mesh);

/**
 * Add a face (triangle) to the mesh
 *
 * @param mesh     Mesh to modify
 * @param v0       First vertex index
 * @param v1       Second vertex index
 * @param v2       Third vertex index
 * @param material Material ID for this face
 * @return Face index or UINT32_MAX on failure
 */
uint32_t mesh_add_face(TriangleMesh* mesh, uint32_t v0, uint32_t v1, uint32_t v2, uint8_t material);

/**
 * Compute per-vertex normals from face geometry
 * Area-weighted average of incident face normals.
 *
 * @param mesh Mesh to compute normals for
 */
void mesh_compute_normals(TriangleMesh* mesh);

/**
 * Recompute bounding box from vertices
 *
 * @param mesh Mesh to update
 */
void mesh_compute_bbox(TriangleMesh* mesh);

/**
 * Get triangle vertices
 *
 * @param mesh      Mesh to query
 * @param face_idx  Face index
 * @param v0        Output: first vertex position
 * @param v1        Output: second vertex position
 * @param v2        Output: third vertex position
 */
void mesh_get_triangle(const TriangleMesh* mesh, uint32_t face_idx,
                       Vec3* v0, Vec3* v1, Vec3* v2);

/**
 * Compute face normal (unnormalized)
 *
 * @param mesh     Mesh to query
 * @param face_idx Face index
 * @return Face normal vector
 */
Vec3 mesh_face_normal(const TriangleMesh* mesh, uint32_t face_idx);

/**
 * Compute face area
 *
 * @param mesh     Mesh to query
 * @param face_idx Face index
 * @return Face area
 */
float mesh_face_area(const TriangleMesh* mesh, uint32_t face_idx);

/* ============================================================================
 * Section 3: MeshBVH - Bounding Volume Hierarchy
 * ============================================================================ */

/**
 * BVH node for acceleration structure
 * Leaf nodes have left == right == node index, face_start/count valid
 * Internal nodes have left != right, children valid
 */
typedef struct BVHNode {
    Vec3 bbox_min;          /* AABB minimum */
    Vec3 bbox_max;          /* AABB maximum */
    uint32_t left;          /* Left child or self for leaf */
    uint32_t right;         /* Right child or self for leaf */
    uint32_t face_start;    /* First face index (leaf only) */
    uint32_t face_count;    /* Number of faces (leaf only) */
} BVHNode;

/**
 * BVH acceleration structure for mesh queries
 */
typedef struct MeshBVH {
    BVHNode* nodes;         /* Node array */
    uint32_t* face_indices; /* Reordered face indices */
    uint32_t node_count;
    uint32_t node_capacity;
    Vec3 avg_normal;        /* Area-weighted average face normal (for ray direction) */
    float normal_coherence; /* 0=closed mesh (normals cancel), 1=open/terrain (normals aligned) */
    Arena* arena;
} MeshBVH;

/**
 * Build BVH from triangle mesh using SAH (Surface Area Heuristic)
 *
 * @param arena Memory arena
 * @param mesh  Source mesh
 * @return BVH or NULL on failure
 */
MeshBVH* bvh_build(Arena* arena, const TriangleMesh* mesh);

/**
 * Ray-BVH intersection test
 *
 * @param bvh        BVH to query
 * @param mesh       Source mesh
 * @param origin     Ray origin
 * @param direction  Ray direction (normalized)
 * @param max_t      Maximum ray parameter
 * @param hit_t      Output: hit distance
 * @param hit_face   Output: hit face index
 * @return True if ray hits mesh
 */
bool bvh_ray_intersect(const MeshBVH* bvh, const TriangleMesh* mesh,
                       Vec3 origin, Vec3 direction, float max_t,
                       float* hit_t, uint32_t* hit_face);

/**
 * Find closest point on mesh to query point
 *
 * @param bvh         BVH to query
 * @param mesh        Source mesh
 * @param point       Query point
 * @param closest     Output: closest point on mesh
 * @param face_idx    Output: face containing closest point
 * @return Unsigned distance to closest point
 */
float bvh_closest_point(const MeshBVH* bvh, const TriangleMesh* mesh,
                        Vec3 point, Vec3* closest, uint32_t* face_idx);

/**
 * Test if AABB intersects any triangle in BVH
 *
 * @param bvh      BVH to query
 * @param mesh     Source mesh
 * @param box_min  AABB minimum
 * @param box_max  AABB maximum
 * @return True if AABB intersects mesh
 */
bool bvh_aabb_intersect(const MeshBVH* bvh, const TriangleMesh* mesh,
                        Vec3 box_min, Vec3 box_max);

/**
 * Determine if point is inside mesh (using ray casting)
 *
 * @param bvh   BVH to query
 * @param mesh  Source mesh
 * @param point Query point
 * @return 1.0f if outside, -1.0f if inside
 */
float bvh_inside_outside(const MeshBVH* bvh, const TriangleMesh* mesh, Vec3 point);

/**
 * Robust inside/outside using 3-ray majority voting.
 * More reliable than single-ray for complex/symmetric surfaces (gyroid).
 * ~3x slower — use near surface where accuracy matters most.
 *
 * @param bvh   BVH to query
 * @param mesh  Source mesh
 * @param point Query point
 * @return 1.0f if outside, -1.0f if inside
 */
float bvh_inside_outside_robust(const MeshBVH* bvh, const TriangleMesh* mesh, Vec3 point);

/* ============================================================================
 * Section 4: MTL Material Library
 * ============================================================================ */

/**
 * Single material definition from MTL file
 */
typedef struct MtlMaterial {
    char name[64];          /* Material name */
    Vec3 Kd;                /* Diffuse color (Kd r g b) */
    Vec3 Ks;                /* Specular color (optional) */
    float Ns;               /* Specular exponent (optional) */
    char map_Kd[256];       /* Diffuse texture path (optional, future) */
    bool has_Kd;            /* True if Kd was explicitly set */
} MtlMaterial;

/**
 * Material library parsed from MTL file
 */
typedef struct MtlLibrary {
    MtlMaterial* materials; /* Array of materials */
    uint32_t count;         /* Number of materials */
    uint32_t capacity;      /* Allocated capacity */
    Arena* arena;
} MtlLibrary;

/**
 * Parse MTL file
 *
 * @param arena Memory arena
 * @param path  Path to MTL file
 * @return Material library or NULL on failure (missing file returns empty library)
 */
MtlLibrary* mtl_parse_file(Arena* arena, const char* path);

/**
 * Find material by name in library
 *
 * @param mtl  Material library
 * @param name Material name to find
 * @return Material pointer or NULL if not found
 */
const MtlMaterial* mtl_find_material(const MtlLibrary* mtl, const char* name);

/**
 * Register all MTL materials in WorldBrickMap
 *
 * @param world World brick map
 * @param mtl   Material library
 */
void mtl_register_materials(WorldBrickMap* world, const MtlLibrary* mtl);

/* ============================================================================
 * Section 5: OBJ Parser
 * ============================================================================ */

/**
 * Parse options for OBJ loading
 */
typedef struct ObjParseOptions {
    bool compute_normals;   /* Compute normals if not in file */
    bool load_materials;    /* Load MTL file if referenced */
    const char* mtl_dir;    /* Directory for MTL file lookup (NULL = same as OBJ) */
} ObjParseOptions;

/**
 * Default parse options
 */
extern const ObjParseOptions OBJ_PARSE_DEFAULTS;

/**
 * Parse OBJ file into triangle mesh
 *
 * @param arena   Memory arena
 * @param path    Path to OBJ file
 * @param options Parse options
 * @param mesh    Output: triangle mesh
 * @param mtl     Output: material library (optional, can be NULL)
 * @param error   Output: error message buffer (at least 256 bytes)
 * @return Result code
 */
ObjIOResult obj_parse_file(Arena* arena, const char* path,
                           const ObjParseOptions* options,
                           TriangleMesh** mesh, MtlLibrary** mtl,
                           char* error);

/* ============================================================================
 * Section 6: Brick Classification for Sparse Voxelization
 * ============================================================================ */

/**
 * Brick classification for sparse voxelization
 */
typedef enum BrickClass {
    BRICK_CLASS_UNKNOWN = 0,    /* Not yet classified */
    BRICK_CLASS_OUTSIDE,        /* No mesh intersection, far from surface */
    BRICK_CLASS_INSIDE,         /* Fully enclosed by mesh */
    BRICK_CLASS_SURFACE         /* Contains surface, needs voxelization */
} BrickClass;

/**
 * Brick classification result for sparse voxelization
 */
typedef struct BrickClassification {
    BrickClass* classes;    /* Classification per brick [grid_total] */
    uint32_t outside_count; /* Bricks classified OUTSIDE */
    uint32_t inside_count;  /* Bricks classified INSIDE */
    uint32_t surface_count; /* Bricks classified SURFACE */
    uint32_t grid_x, grid_y, grid_z;
    Arena* arena;
} BrickClassification;

/* ============================================================================
 * Section 7: Voxelization (Mesh to SDF)
 * ============================================================================ */

/**
 * Voxelization options
 */
typedef struct VoxelizeOptions {
    float voxel_size;       /* Size of single voxel in world units */
    float padding;          /* Padding around mesh bbox (default: 1 brick) */
    bool preserve_materials;/* Transfer face materials to voxels */
    uint32_t max_bricks;    /* Maximum bricks for WorldBrickMap (0 = auto) */
    bool shell_mode;        /* Treat mesh as thin shell surface (e.g., gyroid).
                               Uses unsigned distance with shell thickness offset
                               instead of inside/outside classification. */
    float shell_thickness;  /* Shell thickness in world units (0 = auto: 2 * voxel_size) */
    Vec3 world_min;         /* Override world min (requires use_custom_bounds) */
    Vec3 world_max;         /* Override world max (requires use_custom_bounds) */
    bool use_custom_bounds; /* Explicit flag: apply world_min/world_max overrides */
} VoxelizeOptions;

/**
 * Default voxelization options
 */
extern const VoxelizeOptions VOXELIZE_DEFAULTS;

/**
 * Phase 1: Coarse brick classification via BVH
 * Fast culling of bricks that don't intersect the mesh.
 *
 * @param arena  Memory arena
 * @param bvh    Mesh BVH
 * @param world  World brick map (provides grid dimensions)
 * @return Classification or NULL on failure
 */
BrickClassification* classify_bricks_coarse(Arena* arena, const MeshBVH* bvh,
                                            const TriangleMesh* mesh,
                                            const WorldBrickMap* world);

/**
 * Phase 2: Refine classification with ray tests
 * Distinguishes INSIDE from OUTSIDE for candidate bricks.
 * In shell mode, bricks far from surface are classified as OUTSIDE (never INSIDE).
 *
 * @param classes Classification to refine
 * @param bvh     Mesh BVH
 * @param mesh    Triangle mesh
 * @param world   World brick map
 * @param options Voxelization options (shell_mode, shell_thickness)
 */
void classify_bricks_fine(BrickClassification* classes, const MeshBVH* bvh,
                          const TriangleMesh* mesh, const WorldBrickMap* world,
                          const VoxelizeOptions* options);

/**
 * Phase 3: Voxelize only SURFACE bricks
 * In shell mode, uses unsigned distance with shell thickness offset.
 *
 * @param world   World brick map to populate
 * @param classes Brick classification
 * @param bvh     Mesh BVH
 * @param mesh    Triangle mesh
 * @param options Voxelization options (shell_mode, shell_thickness)
 */
void voxelize_surface_bricks(WorldBrickMap* world, const BrickClassification* classes,
                             const MeshBVH* bvh, const TriangleMesh* mesh,
                             const VoxelizeOptions* options);

/**
 * Eliminate phantom int8=0 voxels from quantization dead zone.
 *
 * Int8 quantization via C truncation maps SDF values in
 * (-sdf_scale/127, +sdf_scale/127) to int8=0 (dequant 0.0). The raymarcher
 * treats 0.0 < RAYMARCH_HIT_DIST as a surface hit, creating phantom surfaces.
 * This sweep detects isolated zeros (no negative face-neighbor) and promotes
 * them to int8=+1.
 *
 * Must be called after voxelize_surface_bricks() or GPU voxelization.
 *
 * @param world   World brick map with populated SDF data
 * @param classes Brick classification (only SURFACE bricks are scanned)
 */
void cleanup_phantom_zeros(WorldBrickMap* world, const BrickClassification* classes);

/**
 * Auto-detect optimal voxelization mode for a mesh.
 *
 * Analyzes mesh topology and geometry to determine if shell mode should
 * be enabled. Modifies opts in-place. No-op if shell_mode is already set.
 *
 * Detection criteria:
 * - High normal coherence (>0.3): terrain/heightmap — left unchanged
 * - Non-watertight + low coherence: thin open surface — shell mode
 * - Watertight + low coherence + thin interior: thin closed surface — shell mode
 * - Watertight + low coherence + thick interior: solid — left unchanged
 *
 * "Thin interior" is determined by sampling random interior points and
 * comparing max distance-to-surface against the mesh bounding sphere.
 *
 * @param opts  Voxelization options to modify
 * @param bvh   Mesh BVH (for inside/outside and closest-point queries)
 * @param mesh  Triangle mesh (for watertight check and bounding box)
 */
void voxelize_options_auto_detect(VoxelizeOptions* opts, const MeshBVH* bvh,
                                  const TriangleMesh* mesh);

/**
 * Full mesh-to-SDF conversion
 *
 * @param arena      Memory arena
 * @param mesh       Source mesh
 * @param options    Voxelization options
 * @param out_world  Output: populated world brick map
 * @param error      Output: error message buffer
 * @return Result code
 */
ObjIOResult mesh_to_sdf(Arena* arena, const TriangleMesh* mesh,
                        const VoxelizeOptions* options,
                        WorldBrickMap** out_world, char* error);

/* ============================================================================
 * Section 8: Marching Cubes (SDF to Mesh)
 * ============================================================================ */

/**
 * Marching Cubes options
 */
typedef struct MarchingCubesOptions {
    float iso_value;        /* Isosurface value (default: 0) */
    bool compute_normals;   /* Compute vertex normals from SDF gradient */
    bool weld_vertices;     /* Weld duplicate vertices */
    float weld_tolerance;   /* Distance threshold for vertex welding */
} MarchingCubesOptions;

/**
 * Default Marching Cubes options
 */
extern const MarchingCubesOptions MARCHING_CUBES_DEFAULTS;

/**
 * Extract mesh from SDF using Marching Cubes
 *
 * @param arena    Memory arena
 * @param world    Source world brick map
 * @param options  Marching Cubes options
 * @param out_mesh Output: extracted mesh
 * @param error    Output: error message buffer
 * @return Result code
 */
ObjIOResult sdf_to_mesh(Arena* arena, const WorldBrickMap* world,
                        const MarchingCubesOptions* options,
                        TriangleMesh** out_mesh, char* error);

/* ============================================================================
 * Section 9: OBJ Export
 * ============================================================================ */

/**
 * Export options for OBJ writing
 */
typedef struct ObjExportOptions {
    bool write_normals;     /* Include vertex normals */
    bool write_mtl;         /* Write MTL file for materials */
    const char* mtl_name;   /* MTL filename (NULL = auto from obj name) */
} ObjExportOptions;

/**
 * Default export options
 */
extern const ObjExportOptions OBJ_EXPORT_DEFAULTS;

/**
 * Export triangle mesh to OBJ file
 *
 * @param path    Output path
 * @param mesh    Mesh to export
 * @param world   World with material registry (optional, for MTL)
 * @param options Export options
 * @param error   Output: error message buffer
 * @return Result code
 */
ObjIOResult obj_export_file(const char* path, const TriangleMesh* mesh,
                            const WorldBrickMap* world,
                            const ObjExportOptions* options, char* error);

/* ============================================================================
 * Section 10: High-Level Convenience API
 * ============================================================================ */

/**
 * Load OBJ file directly into WorldBrickMap
 *
 * @param arena      Memory arena
 * @param path       Path to OBJ file
 * @param voxel_size Voxel size in world units
 * @param out_world  Output: populated world
 * @param error      Output: error message buffer
 * @return Result code
 */
ObjIOResult obj_to_world(Arena* arena, const char* path,
                         const VoxelizeOptions* options,
                         WorldBrickMap** out_world, char* error);

/**
 * Export WorldBrickMap to OBJ file
 *
 * @param world World to export
 * @param path  Output path
 * @param error Output: error message buffer
 * @return Result code
 */
ObjIOResult world_to_obj(const WorldBrickMap* world, const char* path, char* error);

/* ============================================================================
 * Section 11: Validation and Comparison
 * ============================================================================ */

/**
 * Roundtrip comparison metrics
 */
typedef struct MeshCompareResult {
    float hausdorff_distance;   /* Maximum distance between surfaces */
    float mean_distance;        /* Average distance between surfaces */
    float rms_distance;         /* RMS distance */
    uint32_t sample_count;      /* Number of samples used */
    bool passed;                /* True if within tolerance */
} MeshCompareResult;

/**
 * Compare two meshes for roundtrip validation
 *
 * @param arena     Scratch arena for computation
 * @param mesh_a    First mesh
 * @param mesh_b    Second mesh
 * @param tolerance Maximum acceptable Hausdorff distance
 * @return Comparison result
 */
MeshCompareResult mesh_compare(Arena* arena, const TriangleMesh* mesh_a,
                               const TriangleMesh* mesh_b, float tolerance);

#ifdef __cplusplus
}
#endif

#endif /* OBJ_IO_H */
