#!/usr/bin/env python3
# NOTE: Run all scripts via the poetry environment:
#   cd rl_engine && PYTHONPATH=.. poetry run python scripts/demo_gyroid_orbit.py
# Rebuild the C binding first if needed:
#   cd rl_engine && poetry run pip install -e .
"""
Gyroid Orbit Demo: Drone cameras orbit around a gyroid_cube.obj world.

Runs N parallel environments (each with 1 drone) through an identical circular
orbit, collecting RGB + depth + position + velocity observations.  All envs
are stepped together in one engine call, so N envs cost roughly the same wall
time as 1.

Supports CPU-only, GPU-only, or side-by-side benchmarking.

Run from rl_engine/:
    PYTHONPATH=.. poetry run python scripts/demo_gyroid_orbit.py --resolution 16
    PYTHONPATH=.. poetry run python scripts/demo_gyroid_orbit.py --resolution 64 --num-envs 2
    PYTHONPATH=.. poetry run python scripts/demo_gyroid_orbit.py --resolution 32 --mode both
"""

import argparse
import math
import os
import sys
import time

import numpy as np

# Resolve rl_engine/ root (parent of scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
rl_engine_dir = os.path.dirname(script_dir)
if rl_engine_dir not in sys.path:
    sys.path.insert(0, rl_engine_dir)

from drone import DroneEnv


def look_at_origin_quat(theta):
    """Compute quaternion (w,x,y,z) pointing body +X toward the origin.

    The drone is at angle `theta` on the orbit circle.
    To look at the origin, rotate by (theta + pi) about Z.
    """
    alpha = theta + math.pi
    half = alpha / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def quat_rotate_vec(qw, qx, qy, qz, vx, vy, vz):
    """Rotate vector (vx,vy,vz) by quaternion (qw,qx,qy,qz)."""
    # q * v * q_conj using Hamilton product
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


def render_env_dashboards(env_dir, env_idx, resolution, orbit_radius,
                          positions, orientations, rgb_frames, depth_frames,
                          depth_stats, orbit_frames):
    """Generate per-environment dashboard PNGs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    positions_arr = np.array(positions)
    key_frames = list(range(0, orbit_frames, max(1, orbit_frames // 8)))

    for fi in key_frames:
        fig = plt.figure(figsize=(16, 5))

        # Left: 3D trajectory
        ax3d = fig.add_subplot(131, projection="3d")
        ax3d.plot(positions_arr[:, 0], positions_arr[:, 1], positions_arr[:, 2],
                  "b-", alpha=0.3, linewidth=0.8)
        ax3d.scatter(*positions_arr[fi], color="red", s=60, zorder=5)

        # Draw body axes at current position
        qw, qx, qy, qz = orientations[fi]
        px, py_, pz = positions[fi]
        axis_len = 2.0
        for body_axis, color in [((1, 0, 0), "red"), ((0, 1, 0), "green"), ((0, 0, 1), "blue")]:
            wx, wy, wz = quat_rotate_vec(qw, qx, qy, qz, *body_axis)
            ax3d.quiver(px, py_, pz, wx * axis_len, wy * axis_len, wz * axis_len,
                        color=color, arrow_length_ratio=0.15, linewidth=1.5)

        # Draw gyroid bounding box approximation
        half = 10.0
        for z in [-half, half]:
            xs = [-half, half, half, -half, -half]
            ys = [-half, -half, half, half, -half]
            ax3d.plot(xs, ys, [z] * 5, "gray", alpha=0.2, linewidth=0.5)

        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Z")
        ax3d.set_title(f"Trajectory (frame {fi})")
        lim = orbit_radius * 1.3
        ax3d.set_xlim(-lim, lim)
        ax3d.set_ylim(-lim, lim)
        ax3d.set_zlim(-lim, lim)

        # Center: RGB camera
        ax_rgb = fig.add_subplot(132)
        ax_rgb.imshow(rgb_frames[fi])
        ax_rgb.set_title(f"RGB Camera ({resolution}x{resolution})")
        ax_rgb.axis("off")

        # Right: Depth camera
        ax_depth = fig.add_subplot(133)
        im = ax_depth.imshow(depth_frames[fi], cmap="viridis", vmin=0, vmax=1)
        ax_depth.set_title(f"Depth (range [{depth_stats[fi][0]:.2f}, {depth_stats[fi][1]:.2f}])")
        ax_depth.axis("off")
        plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)

        fig.suptitle(f"Env {env_idx} - Frame {fi}/{orbit_frames}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(env_dir, f"frame_{fi:04d}.png"), dpi=120)
        plt.close(fig)

    # Overview: depth statistics over time
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stats_arr = np.array(depth_stats)
    axes[0].plot(stats_arr[:, 0], label="min", alpha=0.8)
    axes[0].plot(stats_arr[:, 1], label="max", alpha=0.8)
    axes[0].plot(stats_arr[:, 2], label="mean", alpha=0.8)
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Depth value")
    axes[0].set_title("Depth Statistics Over Orbit")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Top-down trajectory
    axes[1].plot(positions_arr[:, 0], positions_arr[:, 1], "b-", alpha=0.6)
    axes[1].scatter(positions_arr[0, 0], positions_arr[0, 1], color="green",
                    s=80, zorder=5, label="Start")
    axes[1].scatter(0, 0, color="orange", s=100, zorder=5, marker="*",
                    label="Origin (gyroid)")
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].set_title("Orbit Trajectory (top-down)")
    axes[1].set_aspect("equal")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Env {env_idx} - Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(env_dir, "overview.png"), dpi=120)
    plt.close(fig)


def create_env(num_envs, resolution, orbit_radius, orbit_height,
               camera_fov, camera_far, voxel_size, obs_dim,
               use_gpu_voxelization=True, config_path=None,
               add_imu=False):
    """Create a DroneEnv configured for the gyroid orbit demo."""
    obj_path = os.path.join(rl_engine_dir, "input", "from_utils", "gyroid_cube.obj")
    if not os.path.exists(obj_path):
        print(f"ERROR: Gyroid OBJ not found at {obj_path}")
        sys.exit(1)

    wb = orbit_radius + 7.0  # 7m margin beyond orbit

    kwargs = dict(
        num_envs=num_envs,
        drones_per_env=1,
        obj_path=obj_path,
        camera_width=resolution,
        camera_height=resolution,
        camera_fov=camera_fov,
        camera_far=camera_far,
        add_position_sensor=True,
        add_velocity_sensor=True,
        add_imu_sensor=add_imu,
        voxel_size=voxel_size,
        obs_dim=obs_dim,
        spawn_min=(orbit_radius - 1.0, -1.0, orbit_height - 1.0),
        spawn_max=(orbit_radius + 1.0,  1.0, orbit_height + 1.0),
        termination_min=(-wb, -wb, -wb),
        termination_max=( wb,  wb,  wb),
        world_min=(-wb, -wb, -wb),
        world_max=( wb,  wb,  wb),
        use_gpu_voxelization=use_gpu_voxelization,
    )

    if config_path is not None:
        kwargs["config_path"] = config_path

    env = DroneEnv(**kwargs)
    return env


def run_orbit_loop(env, num_envs, resolution, orbit_frames, orbit_radius,
                   orbit_height, collect_frames=True):
    """Run the orbit loop, returning timing and optionally frame data.

    Returns:
        (elapsed_ms, positions, orientations, all_rgb, all_depth, all_depth_stats)
        If collect_frames=False, frame lists are empty.
    """
    rgb_dim = resolution * resolution * 3
    depth_dim = resolution * resolution

    positions = []
    orientations = []
    all_rgb = [[] for _ in range(num_envs)]
    all_depth = [[] for _ in range(num_envs)]
    all_depth_stats = [[] for _ in range(num_envs)]

    t_start = time.perf_counter()

    for step in range(orbit_frames):
        theta = 2.0 * math.pi * step / orbit_frames
        px = orbit_radius * math.cos(theta)
        py = orbit_radius * math.sin(theta)
        pz = orbit_height

        qw, qx, qy, qz = look_at_origin_quat(theta)

        for e in range(num_envs):
            env.set_drone_state(e, (px, py, pz), (qw, qx, qy, qz))

        env.step_sensors()

        if collect_frames:
            positions.append((px, py, pz))
            orientations.append((qw, qx, qy, qz))

            for e in range(num_envs):
                obs = env.observations[e]
                rgb_flat = obs[:rgb_dim]
                depth_flat = obs[rgb_dim:rgb_dim + depth_dim]

                rgb_img = np.clip(rgb_flat.reshape(resolution, resolution, 3), 0, 1)
                depth_img = depth_flat.reshape(resolution, resolution).copy()
                all_rgb[e].append(rgb_img)
                all_depth[e].append(depth_img)

                d_min = float(np.min(depth_img))
                d_max = float(np.max(depth_img))
                d_mean = float(np.mean(depth_img))
                all_depth_stats[e].append((d_min, d_max, d_mean))

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0
    return elapsed_ms, positions, orientations, all_rgb, all_depth, all_depth_stats


def run_orbit(num_envs=1, resolution=16, orbit_frames=120, orbit_radius=15.0,
              orbit_height=0.0, camera_fov=1.5708, camera_far=20.0,
              voxel_size=0.25, output_dir=None, mode="auto", warmup=10,
              no_viz=False, gpu_voxelization=True, config_path=None,
              add_imu=False):
    """Run the orbit demo across N parallel environments.

    Gyroid cube spans +/-10m.  A circular orbit at radius r puts the drone
    at (r/sqrt2, r/sqrt2) at 45 deg -- to stay fully outside the cube at all
    angles we need r > 10*sqrt2 ~ 14.14m.  Default radius 15m gives ~0.9m
    clearance at the worst-case diagonal.

    mode: "cpu", "gpu", "auto" (use whatever engine decides), or "both" (benchmark)
    gpu_voxelization: Use GPU for OBJ voxelization Phase 3 (default: True)
    """

    if output_dir is None:
        output_dir = os.path.join(rl_engine_dir, "orbit_output")
    os.makedirs(output_dir, exist_ok=True)

    rgb_dim = resolution * resolution * 3
    depth_dim = resolution * resolution
    pos_dim = 3
    vel_dim = 6
    imu_dim = 6 if add_imu else 0
    obs_dim = rgb_dim + depth_dim + pos_dim + vel_dim + imu_dim

    print(f"Environments: {num_envs}")
    print(f"Resolution:   {resolution}x{resolution}")
    print(f"Orbit:        radius={orbit_radius}m, height={orbit_height}m, frames={orbit_frames}")
    print(f"Voxel size:   {voxel_size}m")
    print(f"Mode:         {mode}")
    if config_path:
        print(f"Config:       {config_path}")
    if add_imu:
        print(f"IMU sensor:   enabled (6 floats: accel + gyro)")
    print(f"OBJ voxelization: {'GPU (Phase 3)' if gpu_voxelization else 'CPU only'}")
    print(f"Obs dim:      {obs_dim} (RGB:{rgb_dim} + Depth:{depth_dim} + Pos:{pos_dim} + Vel:{vel_dim}" +
          (f" + IMU:{imu_dim}" if imu_dim else "") + ")")

    t_load = time.perf_counter()
    env = create_env(num_envs, resolution, orbit_radius, orbit_height,
                     camera_fov, camera_far, voxel_size, obs_dim,
                     use_gpu_voxelization=gpu_voxelization,
                     config_path=config_path, add_imu=add_imu)
    t_load = time.perf_counter() - t_load
    print(f"OBJ loaded:   {t_load:.2f}s")
    env.reset()

    gpu_available = env.is_gpu_enabled()
    print(f"GPU sensors:  {'available' if gpu_available else 'not available'}")

    engine_obs_dim = env.get_obs_dim()
    if engine_obs_dim != obs_dim:
        print(f"WARNING: obs_dim mismatch! Engine={engine_obs_dim}, expected={obs_dim}")
        obs_dim = engine_obs_dim

    # ---- Mode dispatch ----

    if mode == "both":
        # Benchmark both paths using the same engine
        if not gpu_available:
            print("\nGPU not available -- running CPU-only benchmark")
            mode = "cpu"
        else:
            print(f"\n--- Warmup ({warmup} frames) ---")
            run_orbit_loop(env, num_envs, resolution, warmup,
                           orbit_radius, orbit_height, collect_frames=False)

            # GPU run
            env.set_gpu_enabled(True)
            print(f"\n--- GPU benchmark ({orbit_frames} frames) ---")
            gpu_ms, *_ = run_orbit_loop(
                env, num_envs, resolution, orbit_frames,
                orbit_radius, orbit_height, collect_frames=False)

            # CPU run
            env.set_gpu_enabled(False)
            print(f"--- CPU benchmark ({orbit_frames} frames) ---")
            cpu_ms, *_ = run_orbit_loop(
                env, num_envs, resolution, orbit_frames,
                orbit_radius, orbit_height, collect_frames=False)

            # Restore GPU for viz pass
            env.set_gpu_enabled(True)

            gpu_per_frame = gpu_ms / orbit_frames
            cpu_per_frame = cpu_ms / orbit_frames
            speedup = cpu_ms / gpu_ms if gpu_ms > 0.001 else float('inf')

            print()
            print("=" * 60)
            print(f"  {'':20s} {'Total':>10s} {'Per Frame':>12s}")
            print(f"  {'CPU':20s} {cpu_ms:>9.1f}ms {cpu_per_frame:>10.3f}ms")
            print(f"  {'GPU':20s} {gpu_ms:>9.1f}ms {gpu_per_frame:>10.3f}ms")
            print(f"  {'Speedup':20s} {speedup:>9.1f}x")
            print("=" * 60)

            if not no_viz:
                # Collect frames with GPU for visualization
                print(f"\nCollecting frames for visualization (GPU)...")
                _, positions, orientations, all_rgb, all_depth, all_depth_stats = \
                    run_orbit_loop(env, num_envs, resolution, orbit_frames,
                                   orbit_radius, orbit_height, collect_frames=True)
                env.close()
                _generate_viz(output_dir, num_envs, resolution, orbit_radius,
                              positions, orientations, all_rgb, all_depth,
                              all_depth_stats, orbit_frames)
            else:
                env.close()
            return

    # Single-mode run (cpu, gpu, or auto)
    if mode == "cpu":
        env.set_gpu_enabled(False)
    elif mode == "gpu":
        if not gpu_available:
            print("WARNING: GPU not available, falling back to CPU")
        else:
            env.set_gpu_enabled(True)
    # mode == "auto": leave as-is

    active_mode = "GPU" if env.is_gpu_enabled() else "CPU"
    print(f"\nRunning with: {active_mode}")

    # Warmup
    if warmup > 0:
        run_orbit_loop(env, num_envs, resolution, warmup,
                       orbit_radius, orbit_height, collect_frames=False)

    # Timed run with frame collection
    elapsed_ms, positions, orientations, all_rgb, all_depth, all_depth_stats = \
        run_orbit_loop(env, num_envs, resolution, orbit_frames,
                       orbit_radius, orbit_height, collect_frames=not no_viz)

    per_frame = elapsed_ms / orbit_frames
    print(f"\n{active_mode} total: {elapsed_ms:.1f}ms  ({per_frame:.3f}ms/frame)")

    env.close()

    if not no_viz and positions:
        _generate_viz(output_dir, num_envs, resolution, orbit_radius,
                      positions, orientations, all_rgb, all_depth,
                      all_depth_stats, orbit_frames)


def _generate_viz(output_dir, num_envs, resolution, orbit_radius,
                  positions, orientations, all_rgb, all_depth,
                  all_depth_stats, orbit_frames):
    """Generate per-env dashboard PNGs."""
    env_dirs = []
    for e in range(num_envs):
        d = os.path.join(output_dir, f"env_{e}")
        os.makedirs(d, exist_ok=True)
        env_dirs.append(d)

    print(f"\nGenerating dashboards to {output_dir}/...")
    for e in range(num_envs):
        print(f"  env_{e}/")
        render_env_dashboards(
            env_dirs[e], e, resolution, orbit_radius,
            positions, orientations,
            all_rgb[e], all_depth[e], all_depth_stats[e],
            orbit_frames,
        )

    print(f"\nDone! Output in {output_dir}/")
    for e in range(num_envs):
        print(f"  env_{e}/  ({len(os.listdir(env_dirs[e]))} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gyroid Orbit Demo")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (default: 1)")
    parser.add_argument("--resolution", type=int, default=16,
                        help="Camera resolution (default: 16)")
    parser.add_argument("--frames", type=int, default=120,
                        help="Number of orbit frames (default: 120)")
    parser.add_argument("--radius", type=float, default=15.0,
                        help="Orbit radius in meters (default: 15.0, min safe ~ 14.14)")
    parser.add_argument("--height", type=float, default=0.0,
                        help="Orbit height Z (default: 0.0)")
    parser.add_argument("--voxel-size", type=float, default=0.25,
                        help="SDF voxel size (default: 0.25)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: rl_engine/orbit_output)")
    parser.add_argument("--mode", choices=["cpu", "gpu", "auto", "both"], default="auto",
                        help="Sensor backend: cpu, gpu, auto, or both for benchmark (default: auto)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup frames before timing (default: 10)")
    parser.add_argument("--no-viz", action="store_true",
                        help="Skip visualization output (benchmark only)")
    parser.add_argument("--no-gpu-voxelization", action="store_true",
                        help="Force CPU-only OBJ voxelization (default: GPU)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to TOML config file (sensors + noise)")
    parser.add_argument("--add-imu", action="store_true",
                        help="Add IMU sensor (6 floats: accel + gyro)")
    args = parser.parse_args()

    run_orbit(
        num_envs=args.num_envs,
        resolution=args.resolution,
        orbit_frames=args.frames,
        orbit_radius=args.radius,
        orbit_height=args.height,
        voxel_size=args.voxel_size,
        output_dir=args.output,
        mode=args.mode,
        warmup=args.warmup,
        no_viz=args.no_viz,
        gpu_voxelization=not args.no_gpu_voxelization,
        config_path=args.config,
        add_imu=args.add_imu,
    )
