"""
RL Engine - High-Performance Drone Swarm Simulation for Reinforcement Learning

This package provides a vectorized drone swarm RL environment with:
- 10 sensor types (IMU, ToF, LiDAR 2D/3D, Camera RGB/Depth/Seg, Position, Velocity, Neighbor)
- Zero-copy observation buffers for PufferLib integration
- Batch physics simulation with RK4 integration
- Sparse SDF world representation with CSG operations
- 1M+ steps/second performance target

Usage:
    from rl_engine import DroneEnv

    env = DroneEnv(num_envs=64, drones_per_env=16)
    obs, info = env.reset()

    for _ in range(1000):
        actions = env.action_space.sample()
        obs, rewards, dones, truncs, info = env.step(actions)
"""

from rl_engine.drone import DroneEnv

__version__ = "1.0.0"
__all__ = ["DroneEnv"]
