"""
DroneEnv - High-Performance Vectorized Drone Swarm Environment

PufferLib-compatible environment wrapper for the C simulation engine.
Provides zero-copy observation/action buffers for maximum performance.

Features:
- Vectorized environments with configurable drone counts
- 10 sensor types including 3D LiDAR and depth cameras
- Box action space for 4 motor commands per drone
- Configurable via TOML files or programmatic API

Usage:
    from rl_engine import DroneEnv

    # Create environment
    env = DroneEnv(num_envs=64, drones_per_env=16)

    # Reset to get initial observations
    obs, info = env.reset()

    # Training loop
    for _ in range(1000):
        actions = env.action_space.sample()
        obs, rewards, dones, truncs, info = env.step(actions)
        if info:
            print(f"Episode return: {info[0]['episode_return']:.2f}")

    env.close()
"""

import numpy as np
import gymnasium
import pufferlib

try:
    from rl_engine import binding
except ImportError:
    # Fallback: load the .so directly to avoid circular import when
    # drone.py is imported as a standalone module (e.g. from scripts/)
    import importlib.util
    import pathlib

    _so = list(pathlib.Path(__file__).parent.glob("binding.cpython-*"))
    if not _so:
        raise ImportError(
            "rl_engine.binding not found. Please build the C extension:\n"
            "  cd rl_engine && pip install -e ."
        )
    _spec = importlib.util.spec_from_file_location("binding", str(_so[0]))
    binding = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(binding)


class DroneEnv(pufferlib.PufferEnv):
    """High-performance vectorized drone swarm RL environment.

    This environment wraps a C simulation engine providing batch physics
    simulation of quadcopter drones with various sensor configurations.

    Args:
        num_envs: Number of parallel environments (default: 64)
        drones_per_env: Drones per environment (default: 16)
        config_path: Path to TOML configuration file (optional)
        render_mode: Render mode ("human" or None, default: None)
        report_interval: Steps between log aggregation (default: 1024)
        buf: Pre-allocated buffer dict (for async training, default: None)
        seed: Random seed for reproducibility (default: 0)
        obs_dim: Observation dimension override (default: auto from config)
        action_dim: Action dimension (default: 4 for quadcopter)

    Attributes:
        num_agents: Total number of agents (num_envs * drones_per_env)
        single_observation_space: Gymnasium Box space for single agent
        single_action_space: Gymnasium Box space for single agent
        observations: Shared observation buffer [num_agents, obs_dim]
        actions: Shared action buffer [num_agents, action_dim]
        rewards: Shared reward buffer [num_agents]
        terminals: Shared terminal flags [num_agents]
        truncations: Shared truncation flags [num_agents]

    Example:
        >>> env = DroneEnv(num_envs=4, drones_per_env=4)
        >>> obs, _ = env.reset()
        >>> print(f"Observation shape: {obs.shape}")
        Observation shape: (16, 26)
    """

    def __init__(
        self,
        num_envs: int = 64,
        drones_per_env: int = 16,
        config_path: str = None,
        render_mode: str = None,
        report_interval: int = 1024,
        buf: dict = None,
        seed: int = 0,
        obs_dim: int = None,
        action_dim: int = 4,
        obj_path: str = None,
        spawn_min: tuple = None,
        spawn_max: tuple = None,
        termination_min: tuple = None,
        termination_max: tuple = None,
        drone_radius: float = None,
        air_density: float = None,
        enable_ground_effect: bool = None,
        enable_drag: bool = None,
        enable_motor_dynamics: bool = None,
        ground_effect_height: float = None,
        ground_effect_coeff: float = None,
        collision_cell_size: float = None,
        camera_width: int = None,
        camera_height: int = None,
        camera_fov: float = None,
        camera_far: float = None,
        add_position_sensor: bool = False,
        add_velocity_sensor: bool = False,
        add_imu_sensor: bool = False,
        voxel_size: float = None,
        world_min: tuple = None,
        world_max: tuple = None,
        max_bricks: int = None,
        use_gpu_voxelization: bool = None,
    ):
        self.num_envs = num_envs
        self.drones_per_env = drones_per_env
        self.config_path = config_path
        self.render_mode = render_mode
        self.report_interval = report_interval
        self._seed = seed
        self.tick = 0

        # World/physics kwargs to pass through to binding
        self._engine_kwargs = {}
        if obj_path is not None:
            self._engine_kwargs["obj_path"] = obj_path
        if spawn_min is not None and spawn_max is not None:
            self._engine_kwargs["spawn_min"] = spawn_min
            self._engine_kwargs["spawn_max"] = spawn_max
        if termination_min is not None and termination_max is not None:
            self._engine_kwargs["termination_min"] = termination_min
            self._engine_kwargs["termination_max"] = termination_max
        if drone_radius is not None:
            self._engine_kwargs["drone_radius"] = drone_radius
        if air_density is not None:
            self._engine_kwargs["air_density"] = air_density
        if enable_ground_effect is not None:
            self._engine_kwargs["enable_ground_effect"] = enable_ground_effect
        if enable_drag is not None:
            self._engine_kwargs["enable_drag"] = enable_drag
        if enable_motor_dynamics is not None:
            self._engine_kwargs["enable_motor_dynamics"] = enable_motor_dynamics
        if ground_effect_height is not None:
            self._engine_kwargs["ground_effect_height"] = ground_effect_height
        if ground_effect_coeff is not None:
            self._engine_kwargs["ground_effect_coeff"] = ground_effect_coeff
        if collision_cell_size is not None:
            self._engine_kwargs["collision_cell_size"] = collision_cell_size
        if voxel_size is not None:
            self._engine_kwargs["voxel_size"] = voxel_size
        if world_min is not None:
            self._engine_kwargs["world_min"] = world_min
        if world_max is not None:
            self._engine_kwargs["world_max"] = world_max
        if max_bricks is not None:
            self._engine_kwargs["max_bricks"] = max_bricks
        if use_gpu_voxelization is not None:
            self._engine_kwargs["use_gpu_voxelization"] = use_gpu_voxelization

        # Sensor kwargs
        if camera_width is not None and camera_height is not None:
            self._engine_kwargs["camera_width"] = camera_width
            self._engine_kwargs["camera_height"] = camera_height
            if camera_fov is not None:
                self._engine_kwargs["camera_fov"] = camera_fov
            if camera_far is not None:
                self._engine_kwargs["camera_far"] = camera_far
        if add_position_sensor:
            self._engine_kwargs["add_position_sensor"] = True
        if add_velocity_sensor:
            self._engine_kwargs["add_velocity_sensor"] = True
        if add_imu_sensor:
            self._engine_kwargs["add_imu_sensor"] = True

        # Total number of agents across all environments
        self.num_agents = num_envs * drones_per_env

        # Determine observation dimension
        if obs_dim is None:
            # Auto-compute from sensor params if cameras are configured
            dim = 0
            if camera_width and camera_height:
                dim += camera_width * camera_height * 3  # RGB
                dim += camera_width * camera_height       # Depth
            if add_position_sensor:
                dim += 3
            if add_velocity_sensor:
                dim += 6
            if add_imu_sensor:
                dim += 6  # accel(3) + gyro(3)
            if dim > 0:
                obs_dim = dim
            else:
                obs_dim = 15  # Default: IMU(6) + Position(3) + Velocity(6)

        self._obs_dim = obs_dim
        self._action_dim = action_dim

        # Define observation space for single agent
        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Define action space for single agent
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        # Call parent constructor (allocates buffers if buf is None)
        super().__init__(buf)

        # Ensure actions are float32 (PufferLib may default to float64)
        self.actions = self.actions.astype(np.float32)

        # Initialize C environments
        self._init_c_envs()

    def _init_c_envs(self):
        """Initialize C environment handles."""
        c_envs = []

        for i in range(self.num_envs):
            start_idx = i * self.drones_per_env
            end_idx = (i + 1) * self.drones_per_env

            # Get slices of shared buffers for this environment
            obs_slice = self.observations[start_idx:end_idx]
            actions_slice = self.actions[start_idx:end_idx]
            rewards_slice = self.rewards[start_idx:end_idx]
            terminals_slice = self.terminals[start_idx:end_idx]
            truncations_slice = self.truncations[start_idx:end_idx]

            # Initialize C environment with buffer slices
            kwargs = {
                "num_envs": 1,  # Each C env handles 1 sub-environment
                "drones_per_env": self.drones_per_env,
                "seed": self._seed + i,
            }

            if self.config_path:
                kwargs["config_path"] = self.config_path

            # Pass through engine kwargs (obj_path, spawn, physics tunables, etc.)
            kwargs.update(self._engine_kwargs)

            c_env = binding.env_init(
                obs_slice,
                actions_slice,
                rewards_slice,
                terminals_slice,
                truncations_slice,
                self._seed + i,
                **kwargs,
            )
            c_envs.append(c_env)

        # Store individual C env handles before vectorizing
        self._c_env_handles = list(c_envs)

        # Validate obs_dim matches engine's actual dimension
        engine_obs_dim = binding.get_obs_dim(c_envs[0])
        if engine_obs_dim != self._obs_dim:
            raise ValueError(
                f"obs_dim mismatch: Python expected {self._obs_dim} but engine "
                f"reports {engine_obs_dim}. Pass obs_dim={engine_obs_dim} or "
                f"configure sensors to match."
            )

        # Vectorize all environments
        self.c_envs = binding.vectorize(*c_envs)

    def reset(self, seed: int = None):
        """Reset all environments to initial state.

        Args:
            seed: Optional seed for reproducibility

        Returns:
            observations: Initial observations [num_agents, obs_dim]
            info: Empty list (no initial info)
        """
        self.tick = 0

        if seed is not None:
            self._seed = seed

        binding.vec_reset(self.c_envs, self._seed)

        return self.observations, []

    def step(self, actions):
        """Execute one timestep across all environments.

        Args:
            actions: Action array [num_agents, action_dim]

        Returns:
            observations: Updated observations [num_agents, obs_dim]
            rewards: Step rewards [num_agents]
            terminals: Episode termination flags [num_agents]
            truncations: Episode truncation flags [num_agents]
            info: List of log dicts (if report_interval reached)
        """
        # Copy actions to shared buffer (for policies that don't write directly)
        np.copyto(self.actions, actions)

        # Step all environments
        binding.vec_step(self.c_envs)

        self.tick += 1

        # Collect aggregated logs periodically
        info = []
        if self.tick % self.report_interval == 0:
            log_data = binding.vec_log(self.c_envs)
            if log_data:  # Non-empty if episodes completed
                info.append(log_data)

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        """Render the first environment (if render_mode is set)."""
        if self.render_mode == "human":
            binding.vec_render(self.c_envs, 0)

    def close(self):
        """Close all environments and free C resources."""
        if hasattr(self, "c_envs") and self.c_envs is not None:
            binding.vec_close(self.c_envs)
            self.c_envs = None

    # ---- New methods for state teleportation and sensor-only stepping ----

    def set_drone_state(self, drone_idx, position, orientation):
        """Teleport a drone to a given position and orientation.

        Args:
            drone_idx: Global drone index (across all envs)
            position: (x, y, z) tuple
            orientation: (w, x, y, z) quaternion tuple
        """
        env_i = drone_idx // self.drones_per_env
        local_idx = drone_idx % self.drones_per_env
        handle = self._c_env_handles[env_i]
        px, py, pz = position
        qw, qx, qy, qz = orientation
        binding.set_drone_state(handle, local_idx, px, py, pz, qw, qx, qy, qz)

    def get_drone_state(self, drone_idx):
        """Get the state of a drone.

        Args:
            drone_idx: Global drone index

        Returns:
            dict with 'position', 'orientation', 'velocity', 'angular_velocity'
        """
        env_i = drone_idx // self.drones_per_env
        local_idx = drone_idx % self.drones_per_env
        handle = self._c_env_handles[env_i]
        return binding.get_drone_state(handle, local_idx)

    def step_sensors(self):
        """Run sensor-only step across all environments.

        Updates self.observations in-place from engine sensor system.
        """
        for handle in self._c_env_handles:
            binding.step_sensors(handle)

    def get_obs_dim(self):
        """Get the actual observation dimension from the C engine.

        Returns:
            int: Number of floats per drone observation
        """
        return binding.get_obs_dim(self._c_env_handles[0])

    def set_gpu_enabled(self, enabled):
        """Enable or disable GPU sensor acceleration.

        When disabled, the engine falls back to CPU-only sensor processing.
        Re-enabling restores GPU acceleration.

        Args:
            enabled: True to enable GPU, False to force CPU-only
        """
        for handle in self._c_env_handles:
            binding.set_gpu_enabled(handle, int(enabled))

    def is_gpu_enabled(self):
        """Check if GPU sensor acceleration is currently active.

        Returns:
            bool: True if GPU sensors are enabled
        """
        return binding.is_gpu_enabled(self._c_env_handles[0])


def test_performance(timeout: float = 10.0, atn_cache: int = 1024):
    """Benchmark environment throughput.

    Args:
        timeout: Benchmark duration in seconds
        atn_cache: Number of action samples to pre-generate

    Returns:
        steps_per_second: Achieved throughput
    """
    import time

    env = DroneEnv(num_envs=100, drones_per_env=16)
    env.reset()
    tick = 0

    # Pre-generate actions to avoid sampling overhead
    actions = [env.action_space.sample() for _ in range(atn_cache)]

    start = time.time()
    while time.time() - start < timeout:
        atn = actions[tick % atn_cache]
        env.step(atn)
        tick += 1

    elapsed = time.time() - start
    sps = env.num_agents * tick / elapsed

    print(f"Steps per second: {sps:,.0f}")
    print(f"Total steps: {tick:,}")
    print(f"Agents: {env.num_agents}")
    print(f"Elapsed: {elapsed:.2f}s")

    env.close()
    return sps


if __name__ == "__main__":
    test_performance()
