/**
 * Quadcopter Platform - X-configuration quadcopter with 4 motors
 *
 * State extensions:  [0]=rpm_0, [1]=rpm_1, [2]=rpm_2, [3]=rpm_3
 * Params extensions: [0]=arm_length, [1]=k_thrust, [2]=k_torque,
 *                    [3]=motor_tau, [4]=max_rpm, [5]=k_drag, [6]=k_ang_damp
 *
 * Motor Convention (X-configuration):
 *   M0 = Front-Right (CW)
 *   M1 = Rear-Left (CW)
 *   M2 = Front-Left (CCW)
 *   M3 = Rear-Right (CCW)
 */

#ifndef PLATFORM_QUADCOPTER_H
#define PLATFORM_QUADCOPTER_H

#include "platform.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Extension Index Constants
 * ============================================================================ */

/* State extension indices */
#define QUAD_EXT_RPM_0      0
#define QUAD_EXT_RPM_1      1
#define QUAD_EXT_RPM_2      2
#define QUAD_EXT_RPM_3      3
#define QUAD_STATE_EXT_COUNT 4

/* Params extension indices */
#define QUAD_PEXT_ARM_LENGTH  0
#define QUAD_PEXT_K_THRUST    1
#define QUAD_PEXT_K_TORQUE    2
#define QUAD_PEXT_MOTOR_TAU   3
#define QUAD_PEXT_MAX_RPM     4
#define QUAD_PEXT_K_DRAG      5
#define QUAD_PEXT_K_ANG_DAMP  6
#define QUAD_PARAMS_EXT_COUNT 7

/* ============================================================================
 * Section 2: QuadcopterConfig
 * ============================================================================ */

/**
 * Quadcopter-specific configuration.
 *
 * Populated from [platform.quadcopter] TOML section or programmatically.
 */
typedef struct QuadcopterConfig {
    /* Geometry */
    float arm_length;       /* m (motor arm length from center, default: 0.1) */

    /* Thrust and torque coefficients */
    float k_thrust;         /* N/(rad/s)^2 (default: 3.16e-10) */
    float k_torque;         /* N*m/(rad/s)^2 (default: 7.94e-12) */

    /* Motor dynamics */
    float motor_tau;        /* s (motor time constant, default: 0.02) */
    float max_rpm;          /* rad/s (maximum motor angular velocity, default: 2500.0) */

    /* Damping */
    float k_drag;           /* N/(m/s) (linear drag, default: 0.1) */
    float k_ang_damp;       /* N*m/(rad/s) (angular damping, default: 0.01) */
} QuadcopterConfig;

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_QUADCOPTER_H */
