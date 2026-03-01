/**
 * Diff-Drive Platform - Two-wheel differential drive ground robot
 *
 * State extensions:  [0]=wheel_vel_l, [1]=wheel_vel_r
 * Params extensions: [0]=wheel_radius, [1]=axle_length, [2]=max_wheel_vel
 *
 * Action space: 2 actions (left wheel, right wheel), range [-1, 1]
 * Maps to target wheel angular velocities: action * max_wheel_vel
 *
 * Physics Model:
 *   Kinematic diff-drive with spring-based velocity tracking.
 *   The robot is constrained to the z=0 ground plane with yaw-only rotation.
 *   Forward velocity = (wl + wr) * R / 2
 *   Yaw rate = (wr - wl) * R / L
 *
 * Default parameters modeled after TurtleBot3 Burger:
 *   wheel_radius = 0.033 m
 *   axle_length  = 0.16 m (wheel center-to-center)
 *   max_wheel_vel = 6.67 rad/s (~200 RPM)
 */

#ifndef PLATFORM_DIFF_DRIVE_H
#define PLATFORM_DIFF_DRIVE_H

#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Extension Index Constants
 * ============================================================================ */

/* State extension indices */
#define DD_EXT_WHEEL_VEL_L     0
#define DD_EXT_WHEEL_VEL_R     1
#define DD_STATE_EXT_COUNT     2

/* Params extension indices */
#define DD_PEXT_WHEEL_RADIUS   0
#define DD_PEXT_AXLE_LENGTH    1
#define DD_PEXT_MAX_WHEEL_VEL  2
#define DD_PARAMS_EXT_COUNT    3

/* ============================================================================
 * Section 2: DiffDriveConfig
 * ============================================================================ */

/**
 * Diff-drive-specific configuration.
 *
 * Populated from [platform.diff_drive] TOML section or programmatically.
 */
typedef struct DiffDriveConfig {
    float wheel_radius;     /* m (default: 0.033 like TurtleBot3) */
    float axle_length;      /* m (center-to-center wheel distance, default: 0.16) */
    float max_wheel_vel;    /* rad/s (max wheel angular velocity, default: 6.67 ~= 200 RPM) */
} DiffDriveConfig;

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_DIFF_DRIVE_H */
