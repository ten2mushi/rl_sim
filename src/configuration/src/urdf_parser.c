/**
 * @file urdf_parser.c
 * @brief URDF Parser Implementation
 *
 * Uses yxml minimal XML parser to extract drone properties from URDF files.
 */

#include "urdf_parser.h"
#include "configuration.h"
#include "platform_quadcopter.h"
#include "yxml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <inttypes.h>

/* ============================================================================
 * Internal Constants
 * ============================================================================ */

#define YXML_STACK_SIZE 4096
#define ATTR_VALUE_MAX 256

/* Parser context tracking current element path */
typedef struct {
    yxml_t xml;
    char xml_stack[YXML_STACK_SIZE];

    /* Element path tracking */
    int depth;
    char current_elem[64];
    char parent_elem[64];
    char grandparent_elem[64];

    /* Attribute accumulation */
    char attr_name[64];
    char attr_value[ATTR_VALUE_MAX];
    int attr_value_len;

    /* Output properties */
    URDFProperties* props;

    /* Current context flags */
    bool in_inertial;
    bool in_collision;
    bool in_geometry;

    /* First-occurrence tracking - set when we exit a section with data */
    bool found_first_inertial;
    bool found_first_collision;
    bool found_first_properties;

    /* Error tracking */
    char* error_msg;
    int error_code;
} ParseContext;

/* ============================================================================
 * String Utilities
 * ============================================================================ */

static void str_copy(char* dst, const char* src, size_t max_len) {
    if (!dst || !src || max_len == 0) return;
    size_t len = strlen(src);
    if (len >= max_len) len = max_len - 1;
    memcpy(dst, src, len);
    dst[len] = '\0';
}

static float parse_float(const char* str) {
    if (!str || !*str) return 0.0f;
    return (float)atof(str);
}

/* ============================================================================
 * Context Initialization
 * ============================================================================ */

static void context_init(ParseContext* ctx, URDFProperties* props, char* error_msg) {
    memset(ctx, 0, sizeof(ParseContext));
    yxml_init(&ctx->xml, ctx->xml_stack, YXML_STACK_SIZE);
    ctx->props = props;
    ctx->error_msg = error_msg;
    ctx->depth = 0;
    ctx->in_inertial = false;
    ctx->in_collision = false;
    ctx->in_geometry = false;
    ctx->found_first_inertial = false;
    ctx->found_first_collision = false;
    ctx->found_first_properties = false;
}

/* ============================================================================
 * Element Path Tracking
 * ============================================================================ */

static void push_element(ParseContext* ctx, const char* elem) {
    str_copy(ctx->grandparent_elem, ctx->parent_elem, sizeof(ctx->grandparent_elem));
    str_copy(ctx->parent_elem, ctx->current_elem, sizeof(ctx->parent_elem));
    str_copy(ctx->current_elem, elem, sizeof(ctx->current_elem));
    ctx->depth++;

    /* Track context */
    if (strcmp(elem, "inertial") == 0) {
        ctx->in_inertial = true;
    } else if (strcmp(elem, "collision") == 0) {
        ctx->in_collision = true;
    } else if (strcmp(elem, "geometry") == 0) {
        ctx->in_geometry = true;
    }
}

static void pop_element(ParseContext* ctx) {
    /* Track context exit - mark first-occurrence when leaving sections with data */
    if (strcmp(ctx->current_elem, "inertial") == 0) {
        if (ctx->props->has_inertial) {
            ctx->found_first_inertial = true;
        }
        ctx->in_inertial = false;
    } else if (strcmp(ctx->current_elem, "collision") == 0) {
        if (ctx->props->has_collision) {
            ctx->found_first_collision = true;
        }
        ctx->in_collision = false;
    } else if (strcmp(ctx->current_elem, "geometry") == 0) {
        ctx->in_geometry = false;
    } else if (strcmp(ctx->current_elem, "properties") == 0) {
        if (ctx->props->has_properties) {
            ctx->found_first_properties = true;
        }
    }

    str_copy(ctx->current_elem, ctx->parent_elem, sizeof(ctx->current_elem));
    str_copy(ctx->parent_elem, ctx->grandparent_elem, sizeof(ctx->parent_elem));
    ctx->grandparent_elem[0] = '\0';
    if (ctx->depth > 0) ctx->depth--;
}

/* ============================================================================
 * Attribute Processing
 * ============================================================================ */

static void process_attribute(ParseContext* ctx) {
    URDFProperties* p = ctx->props;
    const char* elem = ctx->current_elem;
    const char* attr = ctx->attr_name;
    const char* val = ctx->attr_value;

    /* <robot name="..."> */
    if (strcmp(elem, "robot") == 0 && strcmp(attr, "name") == 0) {
        str_copy(p->robot_name, val, URDF_NAME_MAX);
        return;
    }

    /* <mass value="..."/> inside <inertial> - only use first inertial section */
    if (ctx->in_inertial && !ctx->found_first_inertial &&
        strcmp(elem, "mass") == 0 && strcmp(attr, "value") == 0) {
        p->mass = parse_float(val);
        p->has_inertial = true;
        return;
    }

    /* <inertia ixx="..." iyy="..." izz="..." .../> inside <inertial>
     * Only use first inertial section */
    if (ctx->in_inertial && !ctx->found_first_inertial && strcmp(elem, "inertia") == 0) {
        if (strcmp(attr, "ixx") == 0) { p->ixx = parse_float(val); p->has_inertial = true; }
        else if (strcmp(attr, "iyy") == 0) { p->iyy = parse_float(val); }
        else if (strcmp(attr, "izz") == 0) { p->izz = parse_float(val); }
        else if (strcmp(attr, "ixy") == 0) { p->ixy = parse_float(val); }
        else if (strcmp(attr, "ixz") == 0) { p->ixz = parse_float(val); }
        else if (strcmp(attr, "iyz") == 0) { p->iyz = parse_float(val); }
        return;
    }

    /* Collision geometry - only use first collision section */
    if (ctx->in_collision && ctx->in_geometry && !ctx->found_first_collision) {
        /* <cylinder radius="..." length="..."/> */
        if (strcmp(elem, "cylinder") == 0) {
            if (strcmp(attr, "radius") == 0) {
                p->collision_radius = parse_float(val);
                p->has_collision = true;
            } else if (strcmp(attr, "length") == 0) {
                p->collision_length = parse_float(val);
            }
            return;
        }

        /* <sphere radius="..."/> */
        if (strcmp(elem, "sphere") == 0 && strcmp(attr, "radius") == 0) {
            p->collision_radius = parse_float(val);
            p->has_collision = true;
            return;
        }

        /* <box size="x y z"/> - use min dimension as collision radius */
        if (strcmp(elem, "box") == 0 && strcmp(attr, "size") == 0) {
            float x = 0, y = 0, z = 0;
            sscanf(val, "%f %f %f", &x, &y, &z);
            float min_dim = x < y ? x : y;
            min_dim = min_dim < z ? min_dim : z;
            p->collision_radius = min_dim / 2.0f;
            p->has_collision = true;
            return;
        }
    }

    /* <properties arm="..." kf="..." km="..." .../> custom drone extension
     * Only use first properties section */
    if (strcmp(elem, "properties") == 0 && !ctx->found_first_properties) {
        p->has_properties = true;
        if (strcmp(attr, "arm") == 0 || strcmp(attr, "arm_length") == 0) {
            p->arm_length = parse_float(val);
        } else if (strcmp(attr, "kf") == 0 || strcmp(attr, "k_thrust") == 0) {
            p->k_thrust = parse_float(val);
        } else if (strcmp(attr, "km") == 0 || strcmp(attr, "k_torque") == 0) {
            p->k_torque = parse_float(val);
        } else if (strcmp(attr, "motor_tau") == 0 || strcmp(attr, "tau") == 0) {
            p->motor_tau = parse_float(val);
        } else if (strcmp(attr, "max_rpm") == 0) {
            p->max_rpm = parse_float(val);
        }
        return;
    }
}

/* ============================================================================
 * Core Parse Function
 * ============================================================================ */

static int parse_urdf_content(const char* content, size_t len,
                              URDFProperties* props, char* error_msg) {
    ParseContext ctx;
    context_init(&ctx, props, error_msg);

    for (size_t i = 0; i < len; i++) {
        yxml_ret_t r = yxml_parse(&ctx.xml, content[i]);

        switch (r) {
            case YXML_ELEMSTART:
                push_element(&ctx, ctx.xml.elem);
                ctx.attr_name[0] = '\0';
                ctx.attr_value[0] = '\0';
                ctx.attr_value_len = 0;
                break;

            case YXML_ELEMEND:
                pop_element(&ctx);
                break;

            case YXML_ATTRSTART:
                str_copy(ctx.attr_name, ctx.xml.attr, sizeof(ctx.attr_name));
                ctx.attr_value[0] = '\0';
                ctx.attr_value_len = 0;
                break;

            case YXML_ATTRVAL:
                /* Accumulate attribute value */
                if (ctx.xml.data[0] && ctx.attr_value_len < ATTR_VALUE_MAX - 1) {
                    ctx.attr_value[ctx.attr_value_len++] = ctx.xml.data[0];
                    ctx.attr_value[ctx.attr_value_len] = '\0';
                }
                break;

            case YXML_ATTREND:
                process_attribute(&ctx);
                ctx.attr_name[0] = '\0';
                break;

            case YXML_OK:
            case YXML_CONTENT:
            case YXML_PISTART:
            case YXML_PICONTENT:
            case YXML_PIEND:
                /* Ignore */
                break;

            default:
                if (r < 0) {
                    /* Parse error */
                    if (error_msg) {
                        snprintf(error_msg, URDF_ERROR_MAX,
                                "XML parse error at byte %" PRIu64 ", line %" PRIu32,
                                ctx.xml.byte, ctx.xml.line);
                    }
                    return -4;
                }
                break;
        }
    }

    /* Check for EOF */
    yxml_ret_t eof = yxml_eof(&ctx.xml);
    if (eof < 0) {
        if (error_msg) {
            snprintf(error_msg, URDF_ERROR_MAX,
                    "XML incomplete: unexpected end of document");
        }
        return -4;
    }

    return 0;
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

void urdf_properties_init(URDFProperties* props) {
    if (!props) return;
    memset(props, 0, sizeof(URDFProperties));
    /* All fields default to 0.0f via memset */
}

int urdf_parse_file(const char* path, URDFProperties* props, char* error_msg) {
    if (!path || !props) {
        if (error_msg) str_copy(error_msg, "NULL argument", URDF_ERROR_MAX);
        return -1;
    }

    urdf_properties_init(props);

    /* Open file */
    FILE* f = fopen(path, "rb");
    if (!f) {
        if (error_msg) {
            snprintf(error_msg, URDF_ERROR_MAX, "Cannot open file: %s", path);
        }
        return -1;
    }

    /* Get file size */
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size <= 0) {
        fclose(f);
        if (error_msg) str_copy(error_msg, "Empty file", URDF_ERROR_MAX);
        return -2;
    }

    if (file_size > URDF_MAX_FILE_SIZE) {
        fclose(f);
        if (error_msg) {
            snprintf(error_msg, URDF_ERROR_MAX,
                    "File too large: %ld bytes (max %d)", file_size, URDF_MAX_FILE_SIZE);
        }
        return -3;
    }

    /* Allocate and read content */
    char* content = (char*)malloc((size_t)file_size + 1);
    if (!content) {
        fclose(f);
        if (error_msg) str_copy(error_msg, "Memory allocation failed", URDF_ERROR_MAX);
        return -5;
    }

    size_t bytes_read = fread(content, 1, (size_t)file_size, f);
    fclose(f);

    if (bytes_read != (size_t)file_size) {
        free(content);
        if (error_msg) str_copy(error_msg, "File read error", URDF_ERROR_MAX);
        return -2;
    }

    content[bytes_read] = '\0';

    /* Parse */
    int result = parse_urdf_content(content, bytes_read, props, error_msg);
    free(content);

    return result;
}

int urdf_parse_string(const char* urdf_str, URDFProperties* props, char* error_msg) {
    if (!urdf_str || !props) {
        if (error_msg) str_copy(error_msg, "NULL argument", URDF_ERROR_MAX);
        return -1;
    }

    urdf_properties_init(props);
    return parse_urdf_content(urdf_str, strlen(urdf_str), props, error_msg);
}

void urdf_apply_to_platform_config(const URDFProperties* urdf, struct PlatformConfig* config) {
    if (!urdf || !config) return;

    /* Copy robot name */
    if (urdf->robot_name[0]) {
        str_copy(config->name, urdf->robot_name, CONFIG_NAME_MAX);
    }

    /* Apply inertial properties if found */
    if (urdf->has_inertial) {
        if (urdf->mass > 0.0f) config->mass = urdf->mass;
        if (urdf->ixx > 0.0f) config->ixx = urdf->ixx;
        if (urdf->iyy > 0.0f) config->iyy = urdf->iyy;
        if (urdf->izz > 0.0f) config->izz = urdf->izz;
    }

    /* Apply collision properties if found */
    if (urdf->has_collision && urdf->collision_radius > 0.0f) {
        config->collision_radius = urdf->collision_radius;
    }

    /* Apply custom drone properties if found (quadcopter-specific via platform_specific) */
    if (urdf->has_properties && config->platform_specific) {
        QuadcopterConfig* quad = (QuadcopterConfig*)config->platform_specific;
        if (urdf->arm_length > 0.0f) quad->arm_length = urdf->arm_length;
        if (urdf->k_thrust > 0.0f) quad->k_thrust = urdf->k_thrust;
        if (urdf->k_torque > 0.0f) quad->k_torque = urdf->k_torque;
        if (urdf->motor_tau > 0.0f) quad->motor_tau = urdf->motor_tau;
        if (urdf->max_rpm > 0.0f) quad->max_rpm = urdf->max_rpm;
    }
}

int config_load_urdf_with_overlay(const char* urdf_path,
                                  const char* toml_path,
                                  Config* config,
                                  char* error_msg) {
    if (!urdf_path || !config) {
        if (error_msg) str_copy(error_msg, "NULL argument", URDF_ERROR_MAX);
        return -1;
    }

    /* Set defaults first */
    config_set_defaults(config);

    /* Parse URDF for physical properties */
    URDFProperties urdf;
    int urdf_result = urdf_parse_file(urdf_path, &urdf, error_msg);
    if (urdf_result != 0) {
        return urdf_result;
    }

    /* Apply URDF properties to platform config */
    urdf_apply_to_platform_config(&urdf, &config->platform);

    /* Load TOML overlay if provided */
    if (toml_path) {
        int toml_result = config_load(toml_path, config, error_msg);
        if (toml_result != 0) {
            return toml_result;
        }
    }

    return 0;
}

void urdf_properties_print(const URDFProperties* props) {
    if (!props) return;

    printf("URDFProperties:\n");
    printf("  robot_name: %s\n", props->robot_name);
    printf("  has_inertial: %s\n", props->has_inertial ? "true" : "false");
    if (props->has_inertial) {
        printf("    mass: %.6f kg\n", props->mass);
        printf("    ixx:  %.6e kg*m^2\n", props->ixx);
        printf("    iyy:  %.6e kg*m^2\n", props->iyy);
        printf("    izz:  %.6e kg*m^2\n", props->izz);
    }
    printf("  has_collision: %s\n", props->has_collision ? "true" : "false");
    if (props->has_collision) {
        printf("    collision_radius: %.4f m\n", props->collision_radius);
        printf("    collision_length: %.4f m\n", props->collision_length);
    }
    printf("  has_properties: %s\n", props->has_properties ? "true" : "false");
    if (props->has_properties) {
        printf("    arm_length: %.4f m\n", props->arm_length);
        printf("    k_thrust:   %.6e N/(rad/s)^2\n", props->k_thrust);
        printf("    k_torque:   %.6e N*m/(rad/s)^2\n", props->k_torque);
        printf("    motor_tau:  %.4f s\n", props->motor_tau);
        printf("    max_rpm:    %.0f\n", props->max_rpm);
    }
}

int urdf_properties_validate(const URDFProperties* props, char* error_msg) {
    if (!props) {
        if (error_msg) str_copy(error_msg, "NULL properties", URDF_ERROR_MAX);
        return -1;
    }

    if (props->has_inertial) {
        if (props->mass <= 0.0f) {
            if (error_msg) str_copy(error_msg, "mass must be positive", URDF_ERROR_MAX);
            return -1;
        }
        if (props->ixx <= 0.0f || props->iyy <= 0.0f || props->izz <= 0.0f) {
            if (error_msg) str_copy(error_msg, "inertia diagonal must be positive", URDF_ERROR_MAX);
            return -1;
        }
    }

    if (props->has_collision && props->collision_radius <= 0.0f) {
        if (error_msg) str_copy(error_msg, "collision_radius must be positive", URDF_ERROR_MAX);
        return -1;
    }

    if (props->has_properties && props->arm_length <= 0.0f) {
        if (error_msg) str_copy(error_msg, "arm_length must be positive", URDF_ERROR_MAX);
        return -1;
    }

    return 0;
}
