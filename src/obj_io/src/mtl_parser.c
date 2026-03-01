/**
 * MTL Parser Implementation
 *
 * Parses Wavefront MTL material files:
 * - Material names (newmtl)
 * - Diffuse colors (Kd r g b)
 * - Specular colors (Ks r g b) - optional
 * - Specular exponent (Ns) - optional
 * - Texture paths (map_Kd) - stored but not loaded
 *
 * Missing MTL files are handled gracefully (empty library returned).
 */

#include "../include/obj_io.h"
#include "parse_utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

#define MTL_LINE_BUFFER_SIZE 1024
#define MTL_INITIAL_CAPACITY 32

/* ============================================================================
 * MTL Parser
 * ============================================================================ */

MtlLibrary* mtl_parse_file(Arena* arena, const char* path) {
    if (!arena || !path) {
        return NULL;
    }

    /* Allocate library */
    MtlLibrary* lib = arena_alloc_type(arena, MtlLibrary);
    if (!lib) {
        return NULL;
    }

    lib->materials = arena_alloc_array(arena, MtlMaterial, MTL_INITIAL_CAPACITY);
    if (!lib->materials) {
        return NULL;
    }

    lib->count = 0;
    lib->capacity = MTL_INITIAL_CAPACITY;
    lib->arena = arena;

    /* Open file */
    FILE* file = fopen(path, "r");
    if (!file) {
        /* Missing MTL is not an error - return empty library */
        return lib;
    }

    /* Line buffer */
    char line_buffer[MTL_LINE_BUFFER_SIZE];
    MtlMaterial* current = NULL;

    while (fgets(line_buffer, MTL_LINE_BUFFER_SIZE, file)) {
        char* line = line_buffer;

        /* Skip leading whitespace */
        while (*line == ' ' || *line == '\t') line++;

        /* Skip empty lines and comments */
        if (*line == '\0' || *line == '\n' || *line == '#') continue;

        /* New material: newmtl name */
        if (strncmp(line, "newmtl ", 7) == 0) {
            /* Check capacity */
            if (lib->count >= lib->capacity) {
                /* Expand capacity */
                uint32_t new_capacity = lib->capacity * 2;
                MtlMaterial* new_materials = arena_alloc_array(arena, MtlMaterial, new_capacity);
                if (!new_materials) {
                    fclose(file);
                    return lib; /* Return what we have */
                }
                memcpy(new_materials, lib->materials, lib->count * sizeof(MtlMaterial));
                lib->materials = new_materials;
                lib->capacity = new_capacity;
            }

            /* Start new material */
            current = &lib->materials[lib->count++];
            memset(current, 0, sizeof(MtlMaterial));

            /* Extract name */
            char* name_start = line + 7;
            while (*name_start == ' ' || *name_start == '\t') name_start++;

            int i = 0;
            while (name_start[i] && name_start[i] != '\n' &&
                   name_start[i] != '\r' && i < 63) {
                current->name[i] = name_start[i];
                i++;
            }
            current->name[i] = '\0';

            /* Default values */
            current->Kd = VEC3(1.0f, 1.0f, 1.0f); /* White */
            current->Ks = VEC3(0.0f, 0.0f, 0.0f);
            current->Ns = 0.0f;
            current->has_Kd = false;
        }
        /* Diffuse color: Kd r g b */
        else if (current && line[0] == 'K' && line[1] == 'd' && line[2] == ' ') {
            const char* p = line + 3;
            float r, g, b;

            p = parse_float_fast(p, &r);
            p = parse_float_fast(p, &g);
            p = parse_float_fast(p, &b);

            current->Kd = VEC3(r, g, b);
            current->has_Kd = true;
        }
        /* Specular color: Ks r g b */
        else if (current && line[0] == 'K' && line[1] == 's' && line[2] == ' ') {
            const char* p = line + 3;
            float r, g, b;

            p = parse_float_fast(p, &r);
            p = parse_float_fast(p, &g);
            p = parse_float_fast(p, &b);

            current->Ks = VEC3(r, g, b);
        }
        /* Specular exponent: Ns value */
        else if (current && line[0] == 'N' && line[1] == 's' && line[2] == ' ') {
            const char* p = line + 3;
            parse_float_fast(p, &current->Ns);
        }
        /* Diffuse texture: map_Kd path */
        else if (current && strncmp(line, "map_Kd ", 7) == 0) {
            char* path_start = line + 7;
            while (*path_start == ' ' || *path_start == '\t') path_start++;

            int i = 0;
            while (path_start[i] && path_start[i] != '\n' &&
                   path_start[i] != '\r' && i < 255) {
                current->map_Kd[i] = path_start[i];
                i++;
            }
            current->map_Kd[i] = '\0';
        }
    }

    fclose(file);
    return lib;
}

const MtlMaterial* mtl_find_material(const MtlLibrary* mtl, const char* name) {
    if (!mtl || !name) {
        return NULL;
    }

    for (uint32_t i = 0; i < mtl->count; i++) {
        if (strcmp(mtl->materials[i].name, name) == 0) {
            return &mtl->materials[i];
        }
    }

    return NULL;
}

void mtl_register_materials(WorldBrickMap* world, const MtlLibrary* mtl) {
    if (!world || !mtl) {
        return;
    }

    for (uint32_t i = 0; i < mtl->count; i++) {
        const MtlMaterial* mat = &mtl->materials[i];

        /* Use Kd color if available, otherwise default white */
        Vec3 color = mat->has_Kd ? mat->Kd : VEC3(1.0f, 1.0f, 1.0f);

        world_register_material(world, mat->name, color);
    }
}
