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

## Benchmarks

Measured on Apple M3 Max. Build with `-DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON`.

### Full Pipeline (GPU, 1024 drones)

| Profile | Sensors | ms/step | drone-steps/s | FPS @1024 |
|---------|---------|--------:|--------------:|----------:|
| MINIMAL | IMU + position | 0.61 | 1,677,000 | 1,639 |
| LIGHT | IMU + position + velocity | 0.61 | 1,677,000 | 1,639 |
| NAVIGATION | LiDAR 2D-64 + IMU + position | 1.04 | 983,000 | 961 |
| VISION | Depth cam 32 + IMU + position | 1.46 | 701,000 | 685 |
| FULL | Depth cam 32 + LiDAR 2D + IMU + pos + vel | 1.63 | 628,000 | 614 |
| STRESS | Depth cam 64 + LiDAR 3D + all sensors | 7.45 | 137,000 | 134 |

Target: 1024 drones at 50 FPS (20 ms/step) — met with 13x headroom on VISION.

### GPU vs CPU Speedup (256 drones)

| Sensor Config | CPU (ms) | GPU (ms) | Speedup |
|---------------|--------:|---------:|--------:|
| Depth cam 32 + IMU | 69.0 | 1.3 | 55x |
| Depth cam 64 + IMU | 273.1 | 1.1 | 241x |
| RGB cam 32 + IMU | 69.1 | 1.0 | 72x |
| LiDAR 3D 16x64 + IMU | 62.7 | 0.8 | 75x |
| LiDAR 2D-64 + IMU | 2.0 | 0.8 | 2.5x |

### GPU Scaling (VISION profile)

| Drones | GPU (ms) | Speedup vs CPU | us/drone |
|-------:|---------:|---------------:|---------:|
| 64 | 0.81 | 23x | 3.21 |
| 256 | 0.88 | 84x | 1.96 |
| 512 | 1.08 | 139x | 1.96 |
| 1024 | 1.48 | 201x | 1.44 |
| 2048 | 2.80 | — | 1.37 |
| 4096 | 5.48 | — | 1.34 |

### CPU Subsystem Breakdown (1024 drones)

| Subsystem | ms/step |
|-----------|--------:|
| Physics (RK4) | 0.15 |
| Collision (spatial hash + SDF) | 0.045 |
| Rewards (hover task) | 0.004 |
| LiDAR 2D-64 | 4.66 |
| Depth cam 32 | 129.1 |
| Neighbor K=5 | 0.16 |

### Physics Scaling

| Drones | ms/step |
|-------:|--------:|
| 256 | 0.031 |
| 512 | 0.070 |
| 1024 | 0.149 |
| 2048 | 0.373 |
| 4096 | 1.004 |

## Usage

```python
from rl_engine import DroneEnv

env = DroneEnv(num_envs=64, drones_per_env=16)
obs, info = env.reset()

for _ in range(1000):
    actions = env.action_space.sample()
    obs, rewards, dones, truncs, info = env.step(actions)
```
