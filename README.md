# RL Engine

High-performance drone swarm simulation for reinforcement learning.

Vectorized C engine with Python/PufferLib integration delivering 1M+ steps/second. Features 10 sensor types (IMU, ToF, LiDAR 2D/3D, Camera RGB/Depth/Segmentation, Position, Velocity, Neighbor), batch RK4 physics, sparse SDF worlds with CSG operations, and optional Metal GPU compute on macOS.

## Prerequisites

- CMake 3.16+
- C11 compiler (clang or gcc)
- Python 3.10+
- [Poetry](https://python-poetry.org/docs/#installation)
- macOS for GPU acceleration (Metal compute shaders)

## Quick Start

### Build the C library

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Run tests

```bash
cd build
ctest --output-on-failure
```

### Install the Python environment

```bash
poetry install
```

### Generate a gyroid mesh and run the demo

```bash
poetry run python utils/generate_gyroid_cube.py --size 20 --channel-diameter 1.0
PYTHONPATH=.. poetry run python scripts/demo_gyroid_orbit.py --resolution 16
```

## Project Structure

```
rl_engine/
├── CMakeLists.txt          # Top-level build configuration
├── setup.py                # Python package build (CMake + setuptools)
├── pyproject.toml           # Poetry/PEP 517 metadata
├── __init__.py             # Package entry point (exports DroneEnv)
├── drone.py                # DroneEnv — Gymnasium/PufferLib environment wrapper
├── binding.c               # CPython C extension (zero-copy obs buffers)
├── include/
│   └── drone_rl.h          # Public C API header
├── configs/
│   ├── orbit_demo.toml     # Gyroid orbit demo configuration
│   └── lidar_3d.toml       # 3D LiDAR sensor preset
├── src/                    # C engine modules (13 modules)
│   ├── foundation/         # Arena allocator, math, hash maps, SIMD
│   ├── drone_state/        # Drone state representation and lifecycle
│   ├── physics/            # RK4 integrator, aerodynamics, motor model
│   ├── world_brick_map/    # Sparse SDF voxel world, mesh loading, CSG
│   ├── collision_system/   # SDF-based collision detection and response
│   ├── sensor_system/      # Sensor registry and observation buffer management
│   ├── sensor_implementations/ # 10 sensor types (IMU, LiDAR, camera, etc.)
│   ├── reward_system/      # Configurable reward/penalty functions
│   ├── threading/          # Thread pool for parallel env stepping
│   ├── configuration/      # TOML config loading, noise pipeline
│   ├── environment_manager/ # Engine lifecycle, reset, step orchestration
│   ├── gpu/                # Metal compute shaders (macOS)
│   └── obj_io/             # OBJ/MTL mesh parser, marching cubes, voxelizer
├── scripts/
│   └── demo_gyroid_orbit.py # Orbit camera demo
├── utils/
│   └── generate_gyroid_cube.py # Gyroid mesh generator
├── benchmarks/             # C micro-benchmarks for each subsystem
├── input/environments/     # Small test meshes (1.obj, 2.obj, 3.obj)
└── NOTES.md                # Development notes
```

## Architecture

The engine is organized in dependency layers (lower layers have no upward dependencies):

| Layer | Modules |
|-------|---------|
| **Foundation** | `foundation` — arena allocator, SIMD math, hash maps |
| **State** | `drone_state` — per-drone state (pose, velocity, motors) |
| **World** | `world_brick_map`, `obj_io` — sparse SDF grid, mesh I/O |
| **Physics** | `physics` — RK4 integration, aerodynamics, motor model |
| **Collision** | `collision_system` — SDF ray marching, contact response |
| **Sensors** | `sensor_system`, `sensor_implementations` — 10 sensor types |
| **Rewards** | `reward_system` — configurable reward functions |
| **Config** | `configuration` — TOML loading, noise pipeline |
| **Threading** | `threading` — thread pool for parallel stepping |
| **GPU** | `gpu` — Metal compute shaders (sensors, collision) |
| **Orchestration** | `environment_manager` — engine lifecycle, step/reset |

The Python layer (`drone.py` + `binding.c`) wraps the C engine as a Gymnasium-compatible environment with zero-copy NumPy observation buffers for PufferLib vectorized training.

## Usage

```python
from rl_engine import DroneEnv

env = DroneEnv(num_envs=64, drones_per_env=16)
obs, info = env.reset()

for _ in range(1000):
    actions = env.action_space.sample()
    obs, rewards, dones, truncs, info = env.step(actions)
```
