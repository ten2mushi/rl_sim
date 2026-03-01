/**
 * Platform Abstraction Module - VTable-based robot platform polymorphism
 *
 * Defines the PlatformVTable that allows the engine to simulate any robotic
 * platform (quadcopter, diff-drive, etc.) with zero overhead via static
 * const vtables set once at engine creation time.
 *
 * Each platform implements the vtable functions for:
 * - Action mapping (normalized actions -> platform-specific commands)
 * - Actuator dynamics (first-order lag, motor models, etc.)
 * - Force/torque computation (thrust, wheel forces, etc.)
 * - Platform-specific effects (drag, ground constraint, etc.)
 * - State lifecycle (init, reset, validate, sanitize)
 * - Configuration (defaults, TOML parsing, param population)
 */

#ifndef PLATFORM_H
#define PLATFORM_H

#include "foundation.h"
#include "rigid_body_state.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Section 1: Forward Declarations
 * ============================================================================ */

typedef struct PlatformVTable PlatformVTable;
typedef struct PlatformRegistry PlatformRegistry;

/* Forward declare PhysicsConfig to avoid circular dependency */
struct PhysicsConfig;

/* ============================================================================
 * Section 2: PlatformVTable
 * ============================================================================ */

/**
 * Platform vtable - static const, set once at engine creation.
 *
 * All function pointers operate on the platform-agnostic types
 * (RigidBodyStateSOA + extension arrays). Platform-specific state
 * lives in extensions indexed by platform-defined constants.
 */
struct PlatformVTable {
    /* --- Dimensions --- */
    const char* name;                   /* e.g. "quadcopter", "diff_drive" */
    uint32_t action_dim;                /* Action space dimensionality */
    uint32_t state_extension_count;     /* Number of state extension arrays */
    uint32_t params_extension_count;    /* Number of params extension arrays */

    /* --- Hot Path Functions --- */

    /**
     * Map normalized actions [0,1] to platform-specific commands.
     *
     * @param actions   Raw actions [count * action_dim], range [0,1]
     * @param commands  Output command buffer [count * action_dim] (scratch-allocated)
     * @param params_ext Platform params extension arrays
     * @param params_ext_count Number of params extension arrays
     * @param count     Number of agents
     */
    void (*map_actions)(const float* actions, float* commands,
                        float* const* params_ext, uint32_t params_ext_count,
                        uint32_t count);

    /**
     * Apply actuator dynamics (e.g., motor lag).
     *
     * @param commands  Target commands [count * action_dim]
     * @param state_ext State extension arrays (e.g., current RPMs)
     * @param state_ext_count Number of state extension arrays
     * @param params_ext Platform params extension arrays
     * @param params_ext_count Number of params extension arrays
     * @param dt        Substep timestep
     * @param count     Number of agents
     */
    void (*actuator_dynamics)(const float* commands,
                              float** state_ext, uint32_t state_ext_count,
                              float* const* params_ext, uint32_t params_ext_count,
                              float dt, uint32_t count);

    /**
     * Compute platform-specific forces and torques.
     *
     * Reads actuator state from extensions, writes to force/torque buffers.
     *
     * @param rb        Rigid body state (for quaternion rotation etc.)
     * @param state_ext State extension arrays
     * @param state_ext_count Number of state extension arrays
     * @param params_ext Platform params extension arrays
     * @param params_ext_count Number of params extension arrays
     * @param rb_params Rigid body parameters
     * @param forces_x  Output world-frame force X [count]
     * @param forces_y  Output world-frame force Y [count]
     * @param forces_z  Output world-frame force Z [count]
     * @param torques_x Output body-frame torque X [count]
     * @param torques_y Output body-frame torque Y [count]
     * @param torques_z Output body-frame torque Z [count]
     * @param count     Number of agents
     */
    void (*compute_forces_torques)(const RigidBodyStateSOA* rb,
                                   float* const* state_ext, uint32_t state_ext_count,
                                   float* const* params_ext, uint32_t params_ext_count,
                                   const RigidBodyParamsSOA* rb_params,
                                   float* forces_x, float* forces_y, float* forces_z,
                                   float* torques_x, float* torques_y, float* torques_z,
                                   uint32_t count);

    /**
     * Apply platform-specific effects (drag, ground constraint, etc.).
     *
     * Called after integration. May modify rigid body state directly.
     * NULL if no platform effects needed.
     *
     * @param rb          Rigid body state (may be modified)
     * @param state_ext   State extension arrays
     * @param state_ext_count Number of state extension arrays
     * @param rb_params   Rigid body parameters
     * @param params_ext  Platform params extension arrays
     * @param params_ext_count Number of params extension arrays
     * @param forces_x    Force X buffer (for drag accumulation)
     * @param forces_y    Force Y buffer
     * @param forces_z    Force Z buffer
     * @param sdf_distances SDF distance per agent (NULL = no ground effect)
     * @param physics_config Physics configuration
     * @param count       Number of agents
     */
    void (*apply_platform_effects)(RigidBodyStateSOA* rb,
                                   float** state_ext, uint32_t state_ext_count,
                                   const RigidBodyParamsSOA* rb_params,
                                   float* const* params_ext, uint32_t params_ext_count,
                                   float* forces_x, float* forces_y, float* forces_z,
                                   const float* sdf_distances,
                                   const struct PhysicsConfig* physics_config,
                                   uint32_t count);

    /* --- Lifecycle Functions --- */

    /**
     * Initialize platform-specific state extensions for a single agent.
     */
    void (*init_state)(float** state_ext, uint32_t ext_count, uint32_t index);

    /**
     * Reset platform-specific state extensions for a single agent.
     */
    void (*reset_state)(float** state_ext, uint32_t ext_count, uint32_t index);

    /**
     * Initialize platform-specific params extensions for a single agent.
     */
    void (*init_params)(float** params_ext, uint32_t ext_count, uint32_t index);

    /* --- Configuration Functions --- */

    /**
     * Get the size of the platform-specific config struct.
     */
    size_t (*config_size)(void);

    /**
     * Set default values in a platform-specific config struct.
     */
    void (*config_set_defaults)(void* platform_config);

    /**
     * Populate params extensions from platform-specific config.
     *
     * Called once during engine creation for each agent.
     *
     * @param platform_config Platform-specific config struct
     * @param params_ext      Params extension arrays
     * @param ext_count       Number of extension arrays
     * @param rb_params       Rigid body params (may also be set)
     * @param index           Agent index
     */
    void (*config_to_params)(const void* platform_config,
                             float** params_ext, uint32_t ext_count,
                             RigidBodyParamsSOA* rb_params, uint32_t index);
};

/* ============================================================================
 * Section 3: PlatformRegistry
 * ============================================================================ */

#define PLATFORM_REGISTRY_MAX_SLOTS 16

/**
 * Registry of available platform vtables.
 *
 * Allows lookup by name (e.g., from TOML config "type = quadcopter").
 */
struct PlatformRegistry {
    const PlatformVTable* vtables[PLATFORM_REGISTRY_MAX_SLOTS];
    uint32_t count;
};

/**
 * Initialize a platform registry and register built-in platforms.
 *
 * @param registry Registry to initialize
 */
void platform_registry_init(PlatformRegistry* registry);

/**
 * Register a platform vtable.
 *
 * @param registry Registry to add to
 * @param vtable   Platform vtable to register
 * @return 0 on success, -1 if registry full
 */
int platform_registry_register(PlatformRegistry* registry, const PlatformVTable* vtable);

/**
 * Find a platform vtable by name.
 *
 * @param registry Registry to search
 * @param name     Platform name (e.g., "quadcopter")
 * @return Platform vtable, or NULL if not found
 */
const PlatformVTable* platform_registry_find(const PlatformRegistry* registry, const char* name);

/* ============================================================================
 * Section 4: Built-in Platform Externs
 * ============================================================================ */

extern const PlatformVTable PLATFORM_QUADCOPTER;
extern const PlatformVTable PLATFORM_DIFF_DRIVE;

#ifdef __cplusplus
}
#endif

#endif /* PLATFORM_H */
