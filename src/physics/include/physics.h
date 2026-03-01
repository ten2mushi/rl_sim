/**
 * Physics Engine Module - RK4-integrated rigid body physics with SIMD batch processing
 *
 * Provides high-performance physics simulation for 1024+ parallel agents using:
 * - RK4 integration for numerical accuracy
 * - SIMD vectorization (AVX2/NEON) for batch processing
 * - Structure-of-Arrays layout for cache efficiency
 * - Platform-agnostic via VTable dispatch for forces/torques
 *
 * Performance Targets:
 * - Full RK4 step: <5ms for 1024 agents
 * - Single derivative computation: <1ms
 * - Total physics (4 substeps): <20ms per frame
 *
 * Physics Model:
 * - Generic rigid body with platform-specific forces via VTable
 * - Euler equations for angular dynamics
 * - Quaternion-based orientation (no gimbal lock)
 * - Platform-dispatched: actuator dynamics, force/torque, effects
 */

#ifndef PHYSICS_H
#define PHYSICS_H

#include "foundation.h"
#include "drone_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declare PlatformVTable to avoid circular dependency.
 * physics.h needs PlatformVTable* in PhysicsSystem and physics_step_dt().
 * platform.h includes physics.h for PhysicsConfig. */
typedef struct PlatformVTable PlatformVTable;

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
 * - Platform VTable for dispatching forces/torques
 * - Scratch memory for RK4 integration
 * - Intermediate force/torque buffers
 * - Performance statistics
 */
typedef struct PhysicsSystem {
    PhysicsConfig config;

    /* Platform dispatch */
    const PlatformVTable* vtable;    /* Platform vtable for forces/torques */

    /* Memory management */
    Arena* scratch_arena;        /* Scratch arena for RK4 temporaries (reset each step) */

    /* RK4 scratch storage (rigid body only - 13 arrays) */
    RigidBodyStateSOA* k1;           /* First derivative evaluation */
    RigidBodyStateSOA* k2;           /* Second derivative evaluation */
    RigidBodyStateSOA* k3;           /* Third derivative evaluation */
    RigidBodyStateSOA* k4;           /* Fourth derivative evaluation */
    RigidBodyStateSOA* temp_state;   /* Temporary state for intermediate RK4 steps */

    /* Intermediate force/torque buffers */
    float* forces_x;             /* [capacity] World-frame force X component */
    float* forces_y;             /* [capacity] World-frame force Y component */
    float* forces_z;             /* [capacity] World-frame force Z component */
    float* torques_x;            /* [capacity] Body-frame torque X component */
    float* torques_y;            /* [capacity] Body-frame torque Y component */
    float* torques_z;            /* [capacity] Body-frame torque Z component */

    /* SDF distances from collision system (set externally before physics_step) */
    const float* sdf_distances;  /* [capacity] SDF distance per agent (NULL = no ground effect) */

    /* Statistics */
    uint64_t step_count;         /* Total physics steps executed */
    float total_integration_time;  /* Cumulative integration time in seconds */
    uint32_t max_agents;         /* Maximum agent capacity */
} PhysicsSystem;

/* ============================================================================
 * Section 3: Lifecycle Functions
 * ============================================================================ */

/**
 * Get default physics configuration with sensible values.
 *
 * @return PhysicsConfig with default values
 */
PhysicsConfig physics_config_default(void);

/**
 * Create a new physics system.
 *
 * Allocates all necessary memory from the arenas:
 * - RK4 scratch buffers (5 RigidBodyStateSOA structures)
 * - Force/torque intermediate buffers (6 float arrays)
 *
 * @param persistent_arena Arena for long-lived allocations (PhysicsSystem struct)
 * @param scratch_arena Arena for temporary allocations (reset each step)
 * @param config Physics configuration (NULL for defaults)
 * @param max_agents Maximum number of agents to simulate
 * @param vtable Platform vtable for forces/torques dispatch
 * @return Pointer to PhysicsSystem, or NULL on failure
 */
PhysicsSystem* physics_create(Arena* persistent_arena, Arena* scratch_arena,
                              const PhysicsConfig* config, uint32_t max_agents,
                              const PlatformVTable* vtable);

/**
 * Destroy a physics system and release resources.
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
 * Performs substeps x RK4 integration with platform-dispatched:
 * 1. Action mapping (vtable->map_actions)
 * 2. Actuator dynamics (vtable->actuator_dynamics)
 * 3. RK4 integration of rigid body state
 * 4. Quaternion normalization
 * 5. Velocity clamping
 * 6. Platform effects (vtable->apply_platform_effects)
 * 7. State sanitization
 *
 * @param physics Physics system
 * @param states Platform states to update (modified in place)
 * @param params Platform physical parameters
 * @param actions Raw actions [count x action_dim] in range [0, 1]
 * @param count Number of agents to simulate
 */
void physics_step(PhysicsSystem* physics, PlatformStateSOA* states,
                  const PlatformParamsSOA* params, const float* actions, uint32_t count);

/**
 * Advance physics simulation with custom timestep.
 *
 * @param physics Physics system
 * @param states Platform states to update
 * @param params Platform physical parameters
 * @param actions Raw actions [count x action_dim]
 * @param count Number of agents to simulate
 * @param dt Custom timestep in seconds
 */
void physics_step_dt(PhysicsSystem* physics, PlatformStateSOA* states,
                     const PlatformParamsSOA* params, const float* actions,
                     uint32_t count, float dt);

/* ============================================================================
 * Section 5: Component Physics Functions
 * ============================================================================ */

/**
 * Compute state derivatives for RK4 integration.
 *
 * Computes:
 * - Position derivative = velocity
 * - Velocity derivative = (forces / mass) including gravity
 * - Quaternion derivative = 0.5 * q * omega
 * - Angular velocity derivative = I^-1 * (torques - omega x (I * omega))
 *
 * Forces and torques are computed via the platform vtable.
 *
 * @param physics Physics system (contains vtable and force/torque buffers)
 * @param states Current platform states
 * @param params Platform physical parameters
 * @param derivatives Output state derivatives (RigidBodyStateSOA layout)
 * @param count Number of agents
 */
void physics_compute_derivatives(PhysicsSystem* physics,
                                 const PlatformStateSOA* states,
                                 const PlatformParamsSOA* params,
                                 RigidBodyStateSOA* derivatives,
                                 uint32_t count);

/* ============================================================================
 * Section 6: RK4 Integration Functions
 * ============================================================================ */

/**
 * Perform full RK4 integration step.
 *
 * @param physics Physics system (contains scratch buffers)
 * @param states Platform states to integrate (modified in place)
 * @param params Platform physical parameters
 * @param dt Timestep in seconds
 * @param count Number of agents
 */
void physics_rk4_integrate(PhysicsSystem* physics, PlatformStateSOA* states,
                           const PlatformParamsSOA* params,
                           float dt, uint32_t count);

/**
 * Perform RK4 substep: output = current + derivative * dt_scale.
 *
 * @param current Current rigid body state
 * @param derivative State derivative
 * @param output Output state = current + derivative * dt_scale
 * @param dt_scale Timestep scale factor
 * @param count Number of agents
 */
void physics_rk4_substep(const RigidBodyStateSOA* current, const RigidBodyStateSOA* derivative,
                         RigidBodyStateSOA* output, float dt_scale, uint32_t count);

/**
 * Combine RK4 derivatives to produce final state update.
 *
 * @param states Rigid body state to update (modified in place)
 * @param k1 First derivative
 * @param k2 Second derivative
 * @param k3 Third derivative
 * @param k4 Fourth derivative
 * @param dt Timestep in seconds
 * @param count Number of agents
 */
void physics_rk4_combine(RigidBodyStateSOA* states, const RigidBodyStateSOA* k1,
                         const RigidBodyStateSOA* k2, const RigidBodyStateSOA* k3,
                         const RigidBodyStateSOA* k4, float dt, uint32_t count);

/* ============================================================================
 * Section 7: Numerical Stability Functions
 * ============================================================================ */

/**
 * Normalize quaternions to unit length.
 *
 * @param states Platform states to normalize (modified in place)
 * @param count Number of agents
 */
void physics_normalize_quaternions(PlatformStateSOA* states, uint32_t count);

/**
 * Clamp velocities to maximum allowed values.
 *
 * @param states Platform states to clamp (modified in place)
 * @param params Platform parameters (contains max_vel, max_omega)
 * @param count Number of agents
 */
void physics_clamp_velocities(PlatformStateSOA* states, const PlatformParamsSOA* params, uint32_t count);

/**
 * Clamp accelerations to prevent numerical instability.
 *
 * @param accel_x Acceleration X to clamp [count]
 * @param accel_y Acceleration Y to clamp [count]
 * @param accel_z Acceleration Z to clamp [count]
 * @param max_accel Maximum allowed acceleration magnitude
 * @param count Number of agents
 */
void physics_clamp_accelerations(float* accel_x, float* accel_y, float* accel_z,
                                 float max_accel, uint32_t count);

/**
 * Sanitize states by detecting and resetting invalid values.
 *
 * @param states Platform states to sanitize (modified in place)
 * @param count Number of agents
 * @return Number of agents that were reset due to invalid values
 */
uint32_t physics_sanitize_state(PlatformStateSOA* states, uint32_t count);

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
 */
FOUNDATION_INLINE Vec3 quat_rotate_body_z_to_world(Quat q, float fz_body) {
    float fx = 2.0f * (q.x * q.z + q.w * q.y) * fz_body;
    float fy = 2.0f * (q.y * q.z - q.w * q.x) * fz_body;
    float fz = (q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z) * fz_body;
    return VEC3(fx, fy, fz);
}

/**
 * Compute quaternion derivative from angular velocity.
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
 * @param max_agents Maximum agent capacity
 * @return Total bytes required
 */
size_t physics_memory_size(uint32_t max_agents);

#ifdef __cplusplus
}
#endif

#endif /* PHYSICS_H */
