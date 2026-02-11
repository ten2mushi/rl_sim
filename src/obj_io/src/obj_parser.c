/**
 * OBJ Parser Implementation
 *
 * Streaming parser for Wavefront OBJ files with:
 * - 64KB read buffer for large files
 * - Fast float parsing without locale overhead
 * - Support for both f v//vn and f v/vt/vn formats
 * - Material tracking via usemtl directive
 * - Auto-compute bounding box during parse
 */

#include "../include/obj_io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define READ_BUFFER_SIZE (64 * 1024)    /* 64KB read buffer */
#define LINE_BUFFER_SIZE 4096           /* Max line length */
#define INITIAL_VERTEX_CAPACITY 65536
#define INITIAL_FACE_CAPACITY 131072
#define MAX_MATERIALS 256

/* ============================================================================
 * Fast Float Parsing
 * ============================================================================ */

/**
 * Parse float without locale overhead
 * Returns pointer to character after parsed number
 */
static const char* parse_float_fast(const char* str, float* out) {
    const char* p = str;
    float sign = 1.0f;
    float value = 0.0f;
    float fraction = 0.0f;
    float divisor = 10.0f;
    bool has_decimal = false;
    bool has_digits = false;

    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') p++;

    /* Sign */
    if (*p == '-') {
        sign = -1.0f;
        p++;
    } else if (*p == '+') {
        p++;
    }

    /* Integer part */
    while (*p >= '0' && *p <= '9') {
        value = value * 10.0f + (float)(*p - '0');
        has_digits = true;
        p++;
    }

    /* Decimal part */
    if (*p == '.') {
        has_decimal = true;
        p++;
        while (*p >= '0' && *p <= '9') {
            fraction += (float)(*p - '0') / divisor;
            divisor *= 10.0f;
            has_digits = true;
            p++;
        }
    }

    /* Exponent part (scientific notation) */
    if ((*p == 'e' || *p == 'E') && has_digits) {
        p++;
        int exp_sign = 1;
        int exp_value = 0;

        if (*p == '-') {
            exp_sign = -1;
            p++;
        } else if (*p == '+') {
            p++;
        }

        while (*p >= '0' && *p <= '9') {
            exp_value = exp_value * 10 + (*p - '0');
            p++;
        }

        float multiplier = 1.0f;
        for (int i = 0; i < exp_value; i++) {
            if (exp_sign > 0) {
                multiplier *= 10.0f;
            } else {
                multiplier *= 0.1f;
            }
        }
        value = (value + fraction) * multiplier;
    } else {
        value = value + fraction;
    }

    *out = sign * value;
    return has_digits ? p : str;
}

/**
 * Parse integer (vertex index)
 */
static const char* parse_int_fast(const char* str, int32_t* out) {
    const char* p = str;
    int32_t sign = 1;
    int32_t value = 0;
    bool has_digits = false;

    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') p++;

    /* Sign */
    if (*p == '-') {
        sign = -1;
        p++;
    } else if (*p == '+') {
        p++;
    }

    /* Digits */
    while (*p >= '0' && *p <= '9') {
        value = value * 10 + (*p - '0');
        has_digits = true;
        p++;
    }

    *out = sign * value;
    return has_digits ? p : str;
}

/* ============================================================================
 * Material Tracking
 * ============================================================================ */

typedef struct MaterialEntry {
    char name[64];
    uint8_t id;
} MaterialEntry;

typedef struct MaterialTracker {
    MaterialEntry entries[MAX_MATERIALS];
    uint32_t count;
    uint8_t current_id;
} MaterialTracker;

static void material_tracker_init(MaterialTracker* tracker) {
    tracker->count = 0;
    tracker->current_id = 0;
}

static uint8_t material_tracker_get_or_add(MaterialTracker* tracker, const char* name) {
    /* Search existing */
    for (uint32_t i = 0; i < tracker->count; i++) {
        if (strncmp(tracker->entries[i].name, name, 63) == 0) {
            return tracker->entries[i].id;
        }
    }

    /* Add new */
    if (tracker->count < MAX_MATERIALS) {
        uint8_t id = (uint8_t)tracker->count;
        strncpy(tracker->entries[tracker->count].name, name, 63);
        tracker->entries[tracker->count].name[63] = '\0';
        tracker->entries[tracker->count].id = id;
        tracker->count++;
        return id;
    }

    return 0; /* Default material if full */
}

/* ============================================================================
 * Line Parsing
 * ============================================================================ */

/**
 * Parse vertex line: v x y z [w]
 */
static bool parse_vertex_line(const char* line, float* x, float* y, float* z) {
    const char* p = line + 1; /* Skip 'v' */

    p = parse_float_fast(p, x);
    if (p == line + 1) return false;

    p = parse_float_fast(p, y);
    p = parse_float_fast(p, z);

    return true;
}

/**
 * Parse vertex normal line: vn x y z
 */
static bool parse_normal_line(const char* line, float* x, float* y, float* z) {
    const char* p = line + 2; /* Skip 'vn' */

    p = parse_float_fast(p, x);
    if (p == line + 2) return false;

    p = parse_float_fast(p, y);
    p = parse_float_fast(p, z);

    return true;
}

/**
 * Parse face index: v, v/vt, v//vn, or v/vt/vn
 * Returns pointer after parsed index, or NULL on error
 */
static const char* parse_face_index(const char* str, int32_t* v, int32_t* vt, int32_t* vn) {
    const char* p = str;
    *v = 0;
    *vt = 0;
    *vn = 0;

    /* Skip whitespace */
    while (*p == ' ' || *p == '\t') p++;

    /* Parse vertex index (required) */
    const char* start = p;
    p = parse_int_fast(p, v);
    if (p == start) return NULL;

    /* Check for texture/normal indices */
    if (*p == '/') {
        p++;
        if (*p == '/') {
            /* v//vn format */
            p++;
            p = parse_int_fast(p, vn);
        } else if (*p >= '0' && *p <= '9') {
            /* v/vt or v/vt/vn format */
            p = parse_int_fast(p, vt);
            if (*p == '/') {
                p++;
                p = parse_int_fast(p, vn);
            }
        }
    }

    return p;
}

/**
 * Parse face line: f v1 v2 v3 [v4 ...]
 * Handles triangles and quads (triangulates quads)
 */
static int parse_face_line(const char* line, int32_t* indices, int max_indices,
                           uint32_t vertex_count) {
    const char* p = line + 1; /* Skip 'f' */
    int count = 0;

    while (*p && count < max_indices) {
        int32_t v, vt, vn;
        const char* next = parse_face_index(p, &v, &vt, &vn);
        if (next == NULL || next == p) break;

        /* Handle negative indices (relative to end) */
        if (v < 0) {
            v = (int32_t)vertex_count + v + 1;
        }

        /* Convert to 0-based indexing */
        if (v > 0) {
            indices[count++] = v - 1;
        }

        p = next;
    }

    return count;
}

/* ============================================================================
 * Main Parser
 * ============================================================================ */

const ObjParseOptions OBJ_PARSE_DEFAULTS = {
    .compute_normals = true,
    .load_materials = true,
    .mtl_dir = NULL
};

ObjIOResult obj_parse_file(Arena* arena, const char* path,
                           const ObjParseOptions* options,
                           TriangleMesh** out_mesh, MtlLibrary** out_mtl,
                           char* error) {
    if (!arena || !path || !out_mesh) {
        if (error) snprintf(error, 256, "Invalid parameters");
        return OBJ_IO_ERROR_INVALID_PARAMETER;
    }

    if (!options) {
        options = &OBJ_PARSE_DEFAULTS;
    }

    *out_mesh = NULL;
    if (out_mtl) *out_mtl = NULL;

    /* Open file */
    FILE* file = fopen(path, "r");
    if (!file) {
        if (error) snprintf(error, 256, "Cannot open file: %s", path);
        return OBJ_IO_ERROR_FILE_NOT_FOUND;
    }

    /* Allocate read buffer */
    char* line_buffer = arena_alloc(arena, LINE_BUFFER_SIZE);
    if (!line_buffer) {
        fclose(file);
        if (error) snprintf(error, 256, "Out of memory for line buffer");
        return OBJ_IO_ERROR_OUT_OF_MEMORY;
    }

    /* Create mesh */
    TriangleMesh* mesh = mesh_create(arena, INITIAL_VERTEX_CAPACITY, INITIAL_FACE_CAPACITY);
    if (!mesh) {
        fclose(file);
        if (error) snprintf(error, 256, "Out of memory for mesh");
        return OBJ_IO_ERROR_OUT_OF_MEMORY;
    }

    /* Material tracking */
    MaterialTracker mat_tracker;
    material_tracker_init(&mat_tracker);
    char mtl_filename[256] = {0};

    /* Initialize bbox */
    mesh->bbox_min = VEC3(1e30f, 1e30f, 1e30f);
    mesh->bbox_max = VEC3(-1e30f, -1e30f, -1e30f);

    /* Parse line by line */
    int32_t face_indices[16]; /* Support up to 16-gon */
    uint32_t line_number = 0;

    while (fgets(line_buffer, LINE_BUFFER_SIZE, file)) {
        line_number++;
        char* line = line_buffer;

        /* Skip leading whitespace */
        while (*line == ' ' || *line == '\t') line++;

        /* Skip empty lines and comments */
        if (*line == '\0' || *line == '\n' || *line == '#') continue;

        /* Vertex: v x y z */
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (parse_vertex_line(line, &x, &y, &z)) {
                uint32_t idx = mesh_add_vertex(mesh, x, y, z);
                if (idx == UINT32_MAX) {
                    fclose(file);
                    if (error) snprintf(error, 256, "Out of memory at line %u", line_number);
                    return OBJ_IO_ERROR_OUT_OF_MEMORY;
                }

                /* Update bbox */
                if (x < mesh->bbox_min.x) mesh->bbox_min.x = x;
                if (y < mesh->bbox_min.y) mesh->bbox_min.y = y;
                if (z < mesh->bbox_min.z) mesh->bbox_min.z = z;
                if (x > mesh->bbox_max.x) mesh->bbox_max.x = x;
                if (y > mesh->bbox_max.y) mesh->bbox_max.y = y;
                if (z > mesh->bbox_max.z) mesh->bbox_max.z = z;
            }
        }
        /* Vertex normal: vn x y z */
        else if (line[0] == 'v' && line[1] == 'n' && line[2] == ' ') {
            /* Normals stored but we'll recompute if requested */
            float nx, ny, nz;
            if (parse_normal_line(line, &nx, &ny, &nz)) {
                /* Could store file normals here if needed */
            }
        }
        /* Face: f v1 v2 v3 ... */
        else if (line[0] == 'f' && line[1] == ' ') {
            int vertex_count = parse_face_line(line, face_indices, 16, mesh->vertex_count);

            if (vertex_count >= 3) {
                /* Triangulate (fan triangulation for convex polygons) */
                for (int i = 1; i < vertex_count - 1; i++) {
                    uint32_t v0 = (uint32_t)face_indices[0];
                    uint32_t v1 = (uint32_t)face_indices[i];
                    uint32_t v2 = (uint32_t)face_indices[i + 1];

                    /* Validate indices */
                    if (v0 < mesh->vertex_count &&
                        v1 < mesh->vertex_count &&
                        v2 < mesh->vertex_count) {
                        uint32_t face_idx = mesh_add_face(mesh, v0, v1, v2, mat_tracker.current_id);
                        if (face_idx == UINT32_MAX) {
                            fclose(file);
                            if (error) snprintf(error, 256, "Out of memory at line %u", line_number);
                            return OBJ_IO_ERROR_OUT_OF_MEMORY;
                        }
                    }
                }
            }
        }
        /* Material library: mtllib filename.mtl */
        else if (strncmp(line, "mtllib ", 7) == 0) {
            char* name_start = line + 7;
            while (*name_start == ' ') name_start++;

            /* Copy filename, strip newline */
            strncpy(mtl_filename, name_start, 255);
            mtl_filename[255] = '\0';
            char* newline = strchr(mtl_filename, '\n');
            if (newline) *newline = '\0';
            char* cr = strchr(mtl_filename, '\r');
            if (cr) *cr = '\0';
        }
        /* Use material: usemtl name */
        else if (strncmp(line, "usemtl ", 7) == 0) {
            char* name_start = line + 7;
            while (*name_start == ' ') name_start++;

            /* Extract material name */
            char mat_name[64];
            int i = 0;
            while (name_start[i] && name_start[i] != '\n' &&
                   name_start[i] != '\r' && i < 63) {
                mat_name[i] = name_start[i];
                i++;
            }
            mat_name[i] = '\0';

            mat_tracker.current_id = material_tracker_get_or_add(&mat_tracker, mat_name);
        }
    }

    fclose(file);

    /* Validate mesh */
    if (mesh->vertex_count == 0 || mesh->face_count == 0) {
        if (error) snprintf(error, 256, "Empty mesh: %u vertices, %u faces",
                           mesh->vertex_count, mesh->face_count);
        return OBJ_IO_ERROR_EMPTY_MESH;
    }

    /* Compute normals if requested */
    if (options->compute_normals) {
        mesh_compute_normals(mesh);
    }

    /* Load MTL file if requested and referenced */
    if (options->load_materials && mtl_filename[0] && out_mtl) {
        /* Build full MTL path */
        char mtl_path[512];
        if (options->mtl_dir) {
            snprintf(mtl_path, sizeof(mtl_path), "%s/%s", options->mtl_dir, mtl_filename);
        } else {
            /* Use directory of OBJ file */
            const char* last_slash = strrchr(path, '/');
            const char* last_backslash = strrchr(path, '\\');
            const char* dir_end = last_slash > last_backslash ? last_slash : last_backslash;

            if (dir_end) {
                size_t dir_len = (size_t)(dir_end - path + 1);
                if (dir_len < sizeof(mtl_path) - 1) {
                    memcpy(mtl_path, path, dir_len);
                    mtl_path[dir_len] = '\0';
                    strncat(mtl_path, mtl_filename, sizeof(mtl_path) - dir_len - 1);
                } else {
                    strncpy(mtl_path, mtl_filename, sizeof(mtl_path) - 1);
                }
            } else {
                strncpy(mtl_path, mtl_filename, sizeof(mtl_path) - 1);
            }
            mtl_path[sizeof(mtl_path) - 1] = '\0';
        }

        *out_mtl = mtl_parse_file(arena, mtl_path);
        /* MTL parse failure is non-fatal - continue with default materials */
    }

    /* Copy material names from tracker to mesh (for proper world registration) */
    if (mat_tracker.count > 0) {
        mesh->material_names = arena_alloc_array(arena, char*, mat_tracker.count);
        if (mesh->material_names) {
            for (uint32_t i = 0; i < mat_tracker.count; i++) {
                mesh->material_names[i] = arena_alloc_array(arena, char, 64);
                if (mesh->material_names[i]) {
                    strncpy(mesh->material_names[i], mat_tracker.entries[i].name, 63);
                    mesh->material_names[i][63] = '\0';
                }
            }
            mesh->material_name_count = mat_tracker.count;
        }
    }

    *out_mesh = mesh;
    return OBJ_IO_SUCCESS;
}

/* ============================================================================
 * Result String
 * ============================================================================ */

const char* obj_io_result_string(ObjIOResult result) {
    switch (result) {
        case OBJ_IO_SUCCESS:                    return "Success";
        case OBJ_IO_ERROR_FILE_NOT_FOUND:       return "File not found";
        case OBJ_IO_ERROR_FILE_READ:            return "File read error";
        case OBJ_IO_ERROR_FILE_WRITE:           return "File write error";
        case OBJ_IO_ERROR_OUT_OF_MEMORY:        return "Out of memory";
        case OBJ_IO_ERROR_INVALID_FORMAT:       return "Invalid format";
        case OBJ_IO_ERROR_EMPTY_MESH:           return "Empty mesh";
        case OBJ_IO_ERROR_BVH_BUILD_FAILED:     return "BVH build failed";
        case OBJ_IO_ERROR_VOXELIZE_FAILED:      return "Voxelization failed";
        case OBJ_IO_ERROR_MARCHING_CUBES_FAILED: return "Marching cubes failed";
        case OBJ_IO_ERROR_INVALID_PARAMETER:    return "Invalid parameter";
        default:                                 return "Unknown error";
    }
}
