/**
 * @file urdf_parser.h
 * @brief URDF (Universal Robot Description Format) Parser
 *
 * Parses standard URDF files to extract platform physical properties.
 * Supports extracting mass, inertia tensor, collision geometry, and
 * custom drone properties from <properties> extension tags.
 *
 * This parser uses yxml, a minimal XML parser, for lightweight parsing
 * with minimal dependencies.
 *
 * Usage:
 *   URDFProperties props;
 *   char error[256];
 *   if (urdf_parse_file("crazyflie.urdf", &props, error) == 0) {
 *       urdf_apply_to_platform_config(&props, &platform_config);
 *   }
 *
 * URDF to PlatformConfig Field Mapping:
 *   <mass value="..."/>           -> mass
 *   <inertia ixx="..." .../>      -> ixx, iyy, izz
 *   <properties arm="..."/>       -> QuadcopterConfig.arm_length (via platform_specific)
 *   <properties kf="..."/>        -> QuadcopterConfig.k_thrust
 *   <properties km="..."/>        -> QuadcopterConfig.k_torque
 *   <cylinder radius="..."/>      -> collision_radius (from <collision>)
 *   <sphere radius="..."/>        -> collision_radius (from <collision>)
 */

#ifndef URDF_PARSER_H
#define URDF_PARSER_H

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration */
struct PlatformConfig;

/* ============================================================================
 * Constants
 * ============================================================================ */

/** Maximum robot name length */
#define URDF_NAME_MAX 64

/** Maximum error message length */
#define URDF_ERROR_MAX 256

/** Maximum file size supported (4MB) */
#define URDF_MAX_FILE_SIZE (4 * 1024 * 1024)

/* ============================================================================
 * URDFProperties Structure
 * ============================================================================ */

/**
 * Properties extracted from URDF file.
 *
 * Contains physical properties parsed from standard URDF tags:
 * - <robot name="...">
 * - <inertial>/<mass>, <inertial>/<inertia>
 * - <collision>/<geometry>/<cylinder> or <sphere>
 *
 * And custom drone extension tags:
 * - <properties arm="..." kf="..." km="..."/>
 */
typedef struct URDFProperties {
    /* Robot identity */
    char robot_name[URDF_NAME_MAX];      /**< From <robot name="..."> */

    /* Inertial properties (from <inertial>) */
    float mass;                           /**< kg, from <mass value="..."/> */
    float ixx;                            /**< kg*m^2, from <inertia ixx="..."/> */
    float iyy;                            /**< kg*m^2, from <inertia iyy="..."/> */
    float izz;                            /**< kg*m^2, from <inertia izz="..."/> */
    float ixy;                            /**< kg*m^2, off-diagonal (usually 0) */
    float ixz;                            /**< kg*m^2, off-diagonal (usually 0) */
    float iyz;                            /**< kg*m^2, off-diagonal (usually 0) */

    /* Collision geometry (from <collision>/<geometry>) */
    float collision_radius;               /**< m, from cylinder/sphere radius */
    float collision_length;               /**< m, from cylinder length */

    /* Custom drone properties (from <properties> extension tag) */
    float arm_length;                     /**< m, motor to center distance */
    float k_thrust;                       /**< N/(rad/s)^2, thrust coefficient */
    float k_torque;                       /**< N*m/(rad/s)^2, torque coefficient */
    float motor_tau;                      /**< s, motor time constant */
    float max_rpm;                        /**< Maximum motor RPM */

    /* Flags indicating which sections were found */
    bool has_inertial;                    /**< True if <inertial> found */
    bool has_collision;                   /**< True if <collision> found */
    bool has_properties;                  /**< True if <properties> found */
} URDFProperties;

/* ============================================================================
 * Parsing API
 * ============================================================================ */

/**
 * Parse URDF file and extract drone properties.
 *
 * @param path Path to URDF file
 * @param props Output properties structure (caller allocated)
 * @param error_msg Buffer for error message (at least URDF_ERROR_MAX bytes)
 * @return 0 on success, negative error code on failure:
 *         -1: File open error
 *         -2: File read error
 *         -3: File too large
 *         -4: XML parse error
 *         -5: Memory allocation error
 */
int urdf_parse_file(const char* path, URDFProperties* props, char* error_msg);

/**
 * Parse URDF from string buffer.
 *
 * @param urdf_str URDF content as null-terminated string
 * @param props Output properties structure (caller allocated)
 * @param error_msg Buffer for error message (at least URDF_ERROR_MAX bytes)
 * @return 0 on success, negative error code on failure
 */
int urdf_parse_string(const char* urdf_str, URDFProperties* props, char* error_msg);

/* ============================================================================
 * Conversion API
 * ============================================================================ */

/**
 * Apply URDF properties to PlatformConfig.
 *
 * Copies extracted URDF values to corresponding PlatformConfig fields.
 * Only overwrites fields that were actually found in the URDF (has_* flags).
 * Quadcopter-specific fields (arm_length, k_thrust, etc.) are applied to
 * the QuadcopterConfig pointed to by config->platform_specific.
 *
 * @param urdf Source URDF properties
 * @param config Target platform configuration
 */
void urdf_apply_to_platform_config(const URDFProperties* urdf, struct PlatformConfig* config);

/**
 * Load URDF file with optional TOML overlay.
 *
 * Parses URDF for physical properties, then applies TOML config on top
 * for environment, physics, reward, and training settings.
 *
 * @param urdf_path Path to URDF file
 * @param toml_path Path to TOML config (NULL for no overlay)
 * @param config Output complete config structure
 * @param error_msg Error message buffer
 * @return 0 on success, negative on error
 */
struct Config;
int config_load_urdf_with_overlay(const char* urdf_path,
                                  const char* toml_path,
                                  struct Config* config,
                                  char* error_msg);

/* ============================================================================
 * Utility API
 * ============================================================================ */

/**
 * Initialize URDFProperties with default values.
 *
 * @param props Properties to initialize
 */
void urdf_properties_init(URDFProperties* props);

/**
 * Print URDFProperties to stdout for debugging.
 *
 * @param props Properties to print
 */
void urdf_properties_print(const URDFProperties* props);

/**
 * Validate URDFProperties for physical plausibility.
 *
 * Checks:
 * - mass > 0
 * - inertia > 0 (diagonal elements)
 * - collision_radius > 0 if has_collision
 * - arm_length > 0 if has_properties
 *
 * @param props Properties to validate
 * @param error_msg Buffer for error description
 * @return 0 if valid, -1 if invalid
 */
int urdf_properties_validate(const URDFProperties* props, char* error_msg);

#ifdef __cplusplus
}
#endif

#endif /* URDF_PARSER_H */
