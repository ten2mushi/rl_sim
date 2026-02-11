/**
 * Physics Engine Module - RK4-integrated quadcopter physics with SIMD batch processing
 *
 * Provides high-performance physics simulation for 1024+ parallel drones using:
 * - RK4 integration for numerical accuracy
 * - SIMD vectorization (AVX2/NEON) for batch processing
 * - Structure-of-Arrays layout for cache efficiency
 * - Configurable physics features (drag, ground effect, motor dynamics)
 *
 * Performance Targets:
 * - Full RK4 step: <5ms for 1024 drones
 * - Single derivative computation: <1ms
 * - Total physics (4 substeps): <20ms per frame
 *
 * Physics Model:
 * - X-configuration quadcopter with 4 motors
 * - First-order motor dynamics with time constant
 * - Thrust/torque from RPM^2 relationship
 * - Euler equations for angular dynamics
 * - Quaternion-based orientation (no gimbal lock)
 * - Optional: drag, ground effect, gyroscopic precession
 */

#ifndef PHYSICS_H
#define PHYSICS_H

#include "foundation.h"
#include "drone_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Physics Configuration
 * ============================================================================ */

/**
 * Physics simulation configuration
 *
 * Contains all tunable parameters for the physics engine.
 * Use physics_config_default() to get sensible defaults.
 */
typedef struct PhysicsConfig {
    /* Timestep configuration */
    float dt;                    /* Base timestep in seconds (default: 0.02s = 50Hz) */
    float dt_variance;           /* Domain randomization variance [0, 0.1] (default: 0) */
    uint32_t substeps;           /* Physics substeps per frame (default: 4) */

    /* Physical constants */
    float gravity;               /* Gravitational acceleration m/s^2 (default: 9.81) */
    float air_density;           /* Air density kg/m^3 (default: 1.225) */

    /* Feature flags */
    bool enable_drag;            /* Enable linear/quadratic drag (default: true) */
    bool enable_ground_effect;   /* Enable ground effect thrust boost (default: true) */
    bool enable_motor_dynamics;  /* Enable first-order motor lag (default: true) */
    bool enable_gyroscopic;      /* Enable gyroscopic precession (default: false) */

    /* Ground effect parameters */
    float ground_effect_height;  /* Height for full effect in meters (default: 0.5) */
    float ground_effect_coeff;   /* Maximum thrust multiplier (default: 1.15) */

    /* Numerical stability limits */
    float max_linear_accel;      /* Clamp linear acceleration m/s^2 (default: 100) */
    float max_angular_accel;     /* Clamp angular acceleration rad/s^2 (default: 200) */
} PhysicsConfig;

/* ============================================================================
 * Section 2: Physics System
 * ============================================================================ */

/**
 * Physics simulation system
 *
 * Contains all state needed for physics simulation including:
 * - Configuration parameters
 * - Scratch memory for RK4 integration
 * - Intermediate force/torque buffers
 * - Performance statistics
 */
typedef struct PhysicsSystem {
    PhysicsConfig config;

    /* Memory management */
    Arena* scratch_arena;        /* Scratch arena for RK4 temporaries (reset each step) */

    /* RK4 scratch storage (reuse DroneStateSOA layout for derivatives) */
    DroneStateSOA* k1;           /* First derivative evaluation */
    DroneStateSOA* k2;           /* Second derivative evaluation */
    DroneStateSOA* k3;           /* Third derivative evaluation */
    DroneStateSOA* k4;           /* Fourth derivative evaluation */
    DroneStateSOA* temp_state;   /* Temporary state for intermediate RK4 steps */

    /* Intermediate force/torque buffers */
    float* forces_x;             /* [capacity] World-frame force X component */
    float* forces_y;             /* [capacity] World-frame force Y component */
    float* forces_z;             /* [capacity] World-frame force Z component */
    float* torques_x;            /* [capacity] Body-frame torque X component */
    float* torques_y;            /* [capacity] Body-frame torque Y component */
    float* torques_z;            /* [capacity] Body-frame torque Z component */

    /* SDF distances from collision system (set externally before physics_step) */
    const float* sdf_distances;  /* [capacity] SDF distance per drone (NULL = no ground effect) */

    /* Statistics */
    uint64_t step_count;         /* Total physics steps executed */
    double total_integration_time; /* Cumulative integration time in seconds */
    uint32_t max_drones;         /* Maximum drone capacity */
} PhysicsSystem;

/* ============================================================================
 * Section 3: Lifecycle Functions
 * ============================================================================ */

/**
 * Get default physics configuration with sensible values.
 *
 * Default values:
 * - dt: 0.02s (50 Hz)
 * - substeps: 4
 * - gravity: 9.81 m/s^2
 * - All features enabled except gyroscopic precession
 *
 * @return PhysicsConfig with default values
 */
PhysicsConfig physics_config_default(void);

/**
 * Create a new physics system.
 *
 * Allocates all necessary memory from the arenas:
 * - RK4 scratch buffers (5 DroneStateSOA structures)
 * - Force/torque intermediate buffers (6 float arrays)
 *
 * @param persistent_arena Arena for long-lived allocations (PhysicsSystem struct)
 * @param scratch_arena Arena for temporary allocations (reset each step)
 * @param config Physics configuration (NULL for defaults)
 * @param max_drones Maximum number of drones to simulate
 * @return Pointer to PhysicsSystem, or NULL on failure
 */
PhysicsSystem* physics_create(Arena* persistent_arena, Arena* scratch_arena,
                              const PhysicsConfig* config, uint32_t max_drones);

/**
 * Destroy a physics system and release resources.
 *
 * Note: Memory allocated from arenas cannot be individually freed,
 * but this function resets statistics and nullifies pointers.
 *
 * @param physics Physics system to destroy (NULL safe)
 */
void physics_destroy(PhysicsSystem* physics);

/* ============================================================================
 * Section 4: Main Physics Step Functions
 * ============================================================================ */

/**
 * Advance physics simulation by one frame.
 *
 * Performs substeps × RK4 integration with:
 * 1. Motor dynamics update
 * 2. RK4 integration of state
 * 3. Quaternion normalization
 * 4. Velocity clamping
 * 5. State sanitization
 *
 * @param physics Physics system
 * @param states Drone states to update (modified in place)
 * @param params Drone physical parameters
 * @param actions Motor commands [count × 4] in range [0, 1]
 * @param count Number of drones to simulate
 */
void physics_step(PhysicsSystem* physics, DroneStateSOA* states,
                  const DroneParamsSOA* params, const float* actions, uint32_t count);

/**
 * Advance physics simulation with custom timestep.
 *
 * Same as physics_step but allows overriding the configured dt.
 * Useful for variable timestep simulation or testing.
 *
 * @param physics Physics system
 * @param states Drone states to update (modified in place)
 * @param params Drone physical parameters
 * @param actions Motor commands [count × 4] in range [0, 1]
 * @param count Number of drones to simulate
 * @param dt Custom timestep in seconds
 */
void physics_step_dt(PhysicsSystem* physics, DroneStateSOA* states,
                     const DroneParamsSOA* params, const float* actions,
                     uint32_t count, float dt);

/* ============================================================================
 * Section 5: Component Physics Functions
 * ============================================================================ */

/**
 * Compute state derivatives for RK4 integration.
 *
 * Computes:
 * - Position derivative = velocity
 * - Velocity derivative = (forces / mass) including gravity, thrust, drag
 * - Quaternion derivative = 0.5 * q * omega
 * - Angular velocity derivative = I^-1 * (torques - omega x (I * omega))
 *
 * @param states Current drone states
 * @param params Drone physical parameters
 * @param actions Motor commands [count × 4]
 * @param derivatives Output state derivatives (same layout as DroneStateSOA)
 * @param count Number of drones
 * @param config Physics configuration
 * @param sdf_distances SDF distance per drone [count] (NULL = no ground effect)
 */
void physics_compute_derivatives(const DroneStateSOA* states, const DroneParamsSOA* params,
                                 const float* actions, DroneStateSOA* derivatives,
                                 uint32_t count, const PhysicsConfig* config,
                                 const float* sdf_distances);

/**
 * Apply first-order motor dynamics.
 *
 * Updates actual RPMs towards commanded RPMs with time constant:
 * rpm_new = rpm_old + (rpm_target - rpm_old) * min(dt / tau, 1.0)
 *
 * @param rpm_commands Target RPM commands [count × 4]
 * @param actual_rpms Current RPMs to update [count × 4]
 * @param params Drone parameters (contains motor_tau, max_rpm)
 * @param dt Timestep in seconds
 * @param count Number of drones
 */
void physics_motor_dynamics(const float* rpm_commands, float* actual_rpms,
                            const DroneParamsSOA* params, float dt, uint32_t count);

/**
 * Compute thrust forces and torques from motor RPMs.
 *
 * Uses X-configuration motor layout:
 * - Total thrust = sum of k_thrust * rpm^2 for all motors
 * - Roll torque from differential thrust (left vs right)
 * - Pitch torque from differential thrust (front vs back)
 * - Yaw torque from differential reaction torque (CW vs CCW motors)
 *
 * @param states Drone states (contains RPMs)
 * @param params Drone parameters (contains k_thrust, k_torque, arm_length)
 * @param forces_x Output world-frame force X [count]
 * @param forces_y Output world-frame force Y [count]
 * @param forces_z Output world-frame force Z [count]
 * @param torques_x Output body-frame torque X (roll) [count]
 * @param torques_y Output body-frame torque Y (pitch) [count]
 * @param torques_z Output body-frame torque Z (yaw) [count]
 * @param count Number of drones
 */
void physics_compute_forces_torques(const DroneStateSOA* states, const DroneParamsSOA* params,
                                    float* forces_x, float* forces_y, float* forces_z,
                                    float* torques_x, float* torques_y, float* torques_z,
                                    uint32_t count);

/**
 * Apply aerodynamic drag to forces.
 *
 * Adds drag force opposing velocity:
 * F_drag = -k_drag * |v| * v (linear drag model)
 *
 * @param states Drone states (contains velocities)
 * @param params Drone parameters (contains k_drag)
 * @param forces_x Force X to modify [count]
 * @param forces_y Force Y to modify [count]
 * @param forces_z Force Z to modify [count]
 * @param count Number of drones
 * @param air_density Air density kg/m^3
 */
void physics_apply_drag(const DroneStateSOA* states, const DroneParamsSOA* params,
                        float* forces_x, float* forces_y, float* forces_z,
                        uint32_t count, float air_density);

/**
 * Apply ground effect to vertical thrust using SDF proximity.
 *
 * Increases thrust when close to any surface:
 * k_ge = 1 + (k_max - 1) * exp(-sdf / h_ref)
 *
 * @param states Drone states
 * @param params Drone parameters
 * @param forces_z Vertical force to modify [count]
 * @param sdf_distances SDF distance per drone [count] (NULL = no effect)
 * @param count Number of drones
 * @param ground_height SDF distance for full effect (meters)
 * @param effect_coeff Maximum thrust multiplier
 */
void physics_apply_ground_effect(const DroneStateSOA* states, const DroneParamsSOA* params,
                                 float* forces_z, const float* sdf_distances,
                                 uint32_t count,
                                 float ground_height, float effect_coeff);

/* ============================================================================
 * Section 6: RK4 Integration Functions
 * ============================================================================ */

/**
 * Perform full RK4 integration step.
 *
 * Implements classic 4th-order Runge-Kutta:
 * k1 = f(t, y)
 * k2 = f(t + dt/2, y + k1*dt/2)
 * k3 = f(t + dt/2, y + k2*dt/2)
 * k4 = f(t + dt, y + k3*dt)
 * y_new = y + (k1 + 2*k2 + 2*k3 + k4) * dt/6
 *
 * @param physics Physics system (contains scratch buffers)
 * @param states Drone states to integrate (modified in place)
 * @param params Drone physical parameters
 * @param actions Motor commands [count × 4]
 * @param dt Timestep in seconds
 * @param count Number of drones
 */
void physics_rk4_integrate(PhysicsSystem* physics, DroneStateSOA* states,
                           const DroneParamsSOA* params, const float* actions,
                           float dt, uint32_t count);

/**
 * Perform RK4 substep: output = current + derivative * dt_scale.
 *
 * Used to compute intermediate states for k2, k3, k4 evaluation.
 *
 * @param current Current state
 * @param derivative State derivative (from physics_compute_derivatives)
 * @param output Output state = current + derivative * dt_scale
 * @param dt_scale Timestep scale factor (0.5 for k2/k3, 1.0 for k4)
 * @param count Number of drones
 */
void physics_rk4_substep(const DroneStateSOA* current, const DroneStateSOA* derivative,
                         DroneStateSOA* output, float dt_scale, uint32_t count);

/**
 * Combine RK4 derivatives to produce final state update.
 *
 * Applies weighted combination: (k1 + 2*k2 + 2*k3 + k4) / 6
 *
 * @param states State to update (modified in place)
 * @param k1 First derivative
 * @param k2 Second derivative
 * @param k3 Third derivative
 * @param k4 Fourth derivative
 * @param dt Timestep in seconds
 * @param count Number of drones
 */
void physics_rk4_combine(DroneStateSOA* states, const DroneStateSOA* k1,
                         const DroneStateSOA* k2, const DroneStateSOA* k3,
                         const DroneStateSOA* k4, float dt, uint32_t count);

/* ============================================================================
 * Section 7: Numerical Stability Functions
 * ============================================================================ */

/**
 * Normalize quaternions to unit length.
 *
 * Renormalizes quaternions that have drifted from unit norm during integration.
 * Uses Newton-Raphson refinement on rsqrt for better precision.
 *
 * @param states Drone states to normalize (modified in place)
 * @param count Number of drones
 */
void physics_normalize_quaternions(DroneStateSOA* states, uint32_t count);

/**
 * Clamp velocities to maximum allowed values.
 *
 * Clamps both linear and angular velocities to their per-drone limits
 * specified in the parameters.
 *
 * @param states Drone states to clamp (modified in place)
 * @param params Drone parameters (contains max_vel, max_omega)
 * @param count Number of drones
 */
void physics_clamp_velocities(DroneStateSOA* states, const DroneParamsSOA* params, uint32_t count);

/**
 * Clamp accelerations to prevent numerical instability.
 *
 * @param accel_x Acceleration X to clamp [count]
 * @param accel_y Acceleration Y to clamp [count]
 * @param accel_z Acceleration Z to clamp [count]
 * @param max_accel Maximum allowed acceleration magnitude
 * @param count Number of drones
 */
void physics_clamp_accelerations(float* accel_x, float* accel_y, float* accel_z,
                                 float max_accel, uint32_t count);

/**
 * Sanitize drone states by detecting and resetting invalid values.
 *
 * Detects NaN and Inf values in state arrays and resets affected drones
 * to safe default values (origin position, identity quaternion, zero velocities).
 *
 * @param states Drone states to sanitize (modified in place)
 * @param count Number of drones
 * @return Number of drones that were reset due to invalid values
 */
uint32_t physics_sanitize_state(DroneStateSOA* states, uint32_t count);

/* ============================================================================
 * Section 8: Inline Utility Functions
 * ============================================================================ */

/**
 * Convert normalized action [0, 1] to RPM.
 */
FOUNDATION_INLINE float action_to_rpm(float action, float max_rpm) {
    return clampf(action, 0.0f, 1.0f) * max_rpm;
}

/**
 * Convert RPM to thrust using quadratic relationship.
 * T = k_thrust * rpm^2
 */
FOUNDATION_INLINE float rpm_to_thrust(float rpm, float k_thrust) {
    return k_thrust * rpm * rpm;
}

/**
 * Convert RPM to reaction torque using quadratic relationship.
 * tau = k_torque * rpm^2
 */
FOUNDATION_INLINE float rpm_to_torque(float rpm, float k_torque) {
    return k_torque * rpm * rpm;
}

/**
 * Rotate a body-frame vector to world frame using quaternion.
 *
 * For F_body = [0, 0, fz] (thrust along body Z), the optimized formula is:
 * fx_world = 2 * (qx*qz + qw*qy) * fz
 * fy_world = 2 * (qy*qz - qw*qx) * fz
 * fz_world = (qw^2 - qx^2 - qy^2 + qz^2) * fz
 */
FOUNDATION_INLINE Vec3 quat_rotate_body_z_to_world(Quat q, float fz_body) {
    float fx = 2.0f * (q.x * q.z + q.w * q.y) * fz_body;
    float fy = 2.0f * (q.y * q.z - q.w * q.x) * fz_body;
    float fz = (q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z) * fz_body;
    return VEC3(fx, fy, fz);
}

/**
 * Compute quaternion derivative from angular velocity.
 *
 * q_dot = 0.5 * q * [0, omega_x, omega_y, omega_z]
 */
FOUNDATION_INLINE Quat quat_derivative(Quat q, Vec3 omega) {
    float qw_dot = 0.5f * (-q.x * omega.x - q.y * omega.y - q.z * omega.z);
    float qx_dot = 0.5f * ( q.w * omega.x + q.y * omega.z - q.z * omega.y);
    float qy_dot = 0.5f * ( q.w * omega.y + q.z * omega.x - q.x * omega.z);
    float qz_dot = 0.5f * ( q.w * omega.z + q.x * omega.y - q.y * omega.x);
    return QUAT(qw_dot, qx_dot, qy_dot, qz_dot);
}

/* ============================================================================
 * Section 9: Memory Size Calculation
 * ============================================================================ */

/**
 * Calculate memory required for physics system.
 *
 * Includes:
 * - PhysicsSystem struct
 * - 5 DroneStateSOA structures for RK4
 * - 6 float arrays for force/torque buffers
 *
 * @param max_drones Maximum drone capacity
 * @return Total bytes required
 */
size_t physics_memory_size(uint32_t max_drones);

#ifdef __cplusplus
}
#endif

#endif /* PHYSICS_H */
