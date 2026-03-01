#!/usr/bin/env python3
"""PPO training loop for quadcopter hover task with GPU training and periodic rendering.

Uses PufferLib's PuffeRL trainer with the native PufferEnv backend.
Defaults to MPS (Metal GPU) for policy training. Periodically runs a
greedy eval episode and saves an animated 4-panel GIF dashboard.

Usage:
    PYTHONPATH=.. poetry run python scripts/training/train_hover.py
    PYTHONPATH=.. poetry run python scripts/training/train_hover.py --no-render
    PYTHONPATH=.. poetry run python scripts/training/train_hover.py --total-timesteps 100000 --device cpu
    PYTHONPATH=.. poetry run python scripts/training/train_hover.py --resolution 16 --render-interval 5
"""

import argparse
import math
import os
import tempfile

import numpy as np
import torch

import pufferlib
import pufferlib.vector
import pufferlib.models
import pufferlib.pufferl
import pufferlib.pytorch

from rl_engine.robot import RobotEnv


def _create_ground_plane_obj(half_extent=15.0, thickness=0.5):
    """Generate thin-box OBJ for ground plane. Returns path to temp file.

    Box spans (-half_extent, -half_extent, -thickness) to (half_extent, half_extent, 0).
    8 vertices, 12 triangle faces (6 quad faces, each split into 2 triangles).
    """
    he = half_extent
    t = thickness
    vertices = [
        (-he, -he, -t), ( he, -he, -t), ( he,  he, -t), (-he,  he, -t),  # bottom
        (-he, -he,  0), ( he, -he,  0), ( he,  he,  0), (-he,  he,  0),  # top
    ]
    # CCW winding, outward normals
    faces = [
        (5, 6, 7), (5, 7, 8),  # top    (+z)
        (1, 4, 3), (1, 3, 2),  # bottom (-z)
        (1, 2, 6), (1, 6, 5),  # front  (-y)
        (3, 4, 8), (3, 8, 7),  # back   (+y)
        (1, 5, 8), (1, 8, 4),  # left   (-x)
        (2, 3, 7), (2, 7, 6),  # right  (+x)
    ]

    fd, path = tempfile.mkstemp(suffix=".obj", prefix="ground_plane_")
    with os.fdopen(fd, "w") as f:
        f.write("# Ground plane thin box\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    return path


def make_env(num_envs=16, agents_per_env=1, seed=0, resolution=32,
             ground_obj_path=None):
    """Create a RobotEnv configured for the hover task with cameras and ground."""
    kwargs = dict(
        platform="quadcopter",
        num_envs=num_envs,
        agents_per_env=agents_per_env,
        seed=seed,
        camera_width=resolution,
        camera_height=resolution,
        camera_fov=1.5708,
        camera_far=20.0,
        add_position_sensor=True,
        add_velocity_sensor=True,
        add_imu_sensor=True,
        termination_min=(-10, -10, 0),
        termination_max=(10, 10, 20),
        spawn_min=(-2, -2, 1),
        spawn_max=(2, 2, 3),
        world_min=(-15, -15, -2),
        world_max=(15, 15, 22),
    )
    if ground_obj_path is not None:
        kwargs["obj_path"] = ground_obj_path
        kwargs["voxel_size"] = 0.5
    return RobotEnv(**kwargs)


def make_eval_env(ground_obj_path, resolution=32, seed=99):
    """Create a standalone 1-env, 1-agent RobotEnv for eval rollouts."""
    kwargs = dict(
        platform="quadcopter",
        num_envs=1,
        agents_per_env=1,
        seed=seed,
        camera_width=resolution,
        camera_height=resolution,
        camera_fov=1.5708,
        camera_far=20.0,
        add_position_sensor=True,
        add_velocity_sensor=True,
        add_imu_sensor=True,
        termination_min=(-10, -10, 0),
        termination_max=(10, 10, 20),
        spawn_min=(-2, -2, 1),
        spawn_max=(2, 2, 3),
        world_min=(-15, -15, -2),
        world_max=(15, 15, 22),
    )
    if ground_obj_path is not None:
        kwargs["obj_path"] = ground_obj_path
        kwargs["voxel_size"] = 0.5
    return RobotEnv(**kwargs)


@torch.no_grad()
def run_eval_episode(policy, eval_env, device, resolution, max_steps=500):
    """Run a greedy eval episode and collect frame data for GIF rendering.

    Returns:
        dict with keys: positions, rgb_frames, depth_frames, rewards, velocities
    """
    obs, _ = eval_env.reset()
    policy.eval()

    data = {
        "positions": [],
        "rgb_frames": [],
        "depth_frames": [],
        "rewards": [],
        "velocities": [],
    }
    rgb_dim = resolution * resolution * 3
    depth_dim = resolution * resolution

    for _ in range(max_steps):
        obs_t = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        logits, _ = policy(obs_t)
        # Greedy: use mean of Normal distribution, clamp to action range
        action = logits.mean.clamp(-1, 1).cpu().numpy()

        obs, rewards, terminals, truncations, _ = eval_env.step(action)

        # Extract camera data from observation
        o = obs[0]
        rgb = np.clip(o[:rgb_dim].reshape(resolution, resolution, 3), 0, 1)
        depth = o[rgb_dim:rgb_dim + depth_dim].reshape(resolution, resolution)
        data["rgb_frames"].append(rgb.copy())
        data["depth_frames"].append(depth.copy())
        data["rewards"].append(float(rewards[0]))

        # Extract position and velocity from agent state
        state = eval_env.get_agent_state(0)
        px, py, pz = state["position"]
        data["positions"].append((px, py, pz))
        vx, vy, vz = state["velocity"]
        data["velocities"].append(math.sqrt(vx * vx + vy * vy + vz * vz))

        if terminals[0] or truncations[0]:
            break

    policy.train()
    return data


def make_config(args):
    """Build PuffeRL config dict from CLI args."""
    num_agents = args.num_envs * args.agents_per_env
    batch_size = num_agents * args.bptt_horizon

    # PuffeRL computes total_epochs = total_timesteps // batch_size.
    # Zero epochs causes division by zero, so clamp upward.
    if args.total_timesteps < batch_size:
        args.total_timesteps = batch_size
        print(f"  (clamped total_timesteps to {batch_size} to fill at least 1 batch)")

    return dict(
        env="quadcopter_hover",
        seed=args.seed,
        torch_deterministic=True,
        cpu_offload=False,
        device=args.device,
        precision="float32",
        optimizer="adam",
        learning_rate=3e-4,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-8,
        anneal_lr=True,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        vf_coef=0.5,
        vf_clip_coef=0.2,
        ent_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=4,
        total_timesteps=args.total_timesteps,
        batch_size=batch_size,
        bptt_horizon=args.bptt_horizon,
        minibatch_size=args.minibatch_size,
        max_minibatch_size=2048,
        vtrace_rho_clip=1.0,
        vtrace_c_clip=1.0,
        prio_alpha=0.0,
        prio_beta0=1.0,
        data_dir="experiments",
        checkpoint_interval=50,
        compile=False,
        compile_mode="default",
        compile_fullgraph=False,
        use_rnn=False,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train quadcopter hover with PPO")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="mps",
                        choices=["cpu", "mps"])
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--agents-per-env", type=int, default=1)
    parser.add_argument("--bptt-horizon", type=int, default=128)
    parser.add_argument("--minibatch-size", type=int, default=512)
    parser.add_argument("--resolution", type=int, default=32,
                        help="Camera resolution (default: 32)")
    parser.add_argument("--render-interval", type=int, default=50,
                        help="Render eval GIF every N training steps (default: 50)")
    parser.add_argument("--eval-steps", type=int, default=500,
                        help="Max steps per eval episode (default: 500)")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable periodic eval rendering")
    return parser.parse_args()


def main():
    args = parse_args()

    # Generate ground plane OBJ
    ground_obj_path = _create_ground_plane_obj()
    print(f"Ground plane OBJ: {ground_obj_path}")

    # Create vectorized environment (native PufferEnv backend)
    vecenv = pufferlib.vector.make(
        make_env,
        env_kwargs=dict(
            num_envs=args.num_envs,
            agents_per_env=args.agents_per_env,
            seed=args.seed,
            resolution=args.resolution,
            ground_obj_path=ground_obj_path,
        ),
    )

    # Create policy and move to target device
    policy = pufferlib.models.Default(vecenv.driver_env,
                                      hidden_size=args.hidden_size)
    policy = policy.to(args.device)

    # Create trainer config
    config = make_config(args)

    # Create PuffeRL trainer
    trainer = pufferlib.pufferl.PuffeRL(config, vecenv, policy)

    # Create eval environment and renderer
    eval_env = None
    renderer = None
    if not args.no_render:
        try:
            from scripts.training.visualizer import TrainingRenderer
            eval_env = make_eval_env(ground_obj_path, resolution=args.resolution)
            renderer = TrainingRenderer()
            print(f"Rendering to: {renderer.run_dir}")
        except Exception as e:
            print(f"Renderer unavailable ({e}), running headless")
            renderer = None

    rgb_dim = args.resolution * args.resolution * 3
    depth_dim = args.resolution * args.resolution
    obs_dim = rgb_dim + depth_dim + 3 + 6 + 6

    print(f"Training quadcopter hover: {config['total_timesteps']} timesteps")
    print(f"  Agents: {vecenv.num_agents}, Obs: {vecenv.single_observation_space.shape}")
    print(f"  Action: {vecenv.single_action_space.shape}, Device: {args.device}")
    print(f"  Batch: {config['batch_size']}, Horizon: {config['bptt_horizon']}")
    print(f"  Camera: {args.resolution}x{args.resolution}, Obs dim: {obs_dim}")
    total_steps = config["total_timesteps"] // config["batch_size"]
    if renderer:
        if args.render_interval > total_steps:
            print(f"  WARNING: render_interval ({args.render_interval}) > total training steps ({total_steps})")
            print(f"           Only a final eval GIF will be rendered.")
        print(f"  Eval: every {args.render_interval} steps, max {args.eval_steps} steps/episode")
    print()

    def _render_eval(tag=""):
        """Run eval episode and save GIF. Returns True if successful."""
        frames = run_eval_episode(
            policy, eval_env, args.device, args.resolution,
            max_steps=args.eval_steps,
        )
        episode_ret = sum(frames["rewards"])
        path = renderer.render_episode_gif(
            frames=frames,
            global_step=trainer.global_step,
            episode_return=episode_ret,
            resolution=args.resolution,
        )
        n_frames = len(frames["positions"])
        print(f"  [render{tag}] {path} ({n_frames} frames, return={episode_ret:.2f})")
        return True

    step_count = 0
    rendered = False

    try:
        while trainer.global_step < config["total_timesteps"]:
            trainer.evaluate()
            logs = trainer.train()
            step_count += 1

            if renderer and step_count % args.render_interval == 0:
                _render_eval()
                rendered = True

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Always render a final eval episode
    if renderer and not rendered:
        print("  Rendering final eval episode...")
        _render_eval(tag=" final")

    # Save final checkpoint and clean up
    model_path = trainer.close()
    print(f"Final model saved to: {model_path}")

    if eval_env is not None:
        eval_env.close()

    if renderer:
        print(f"Episode GIFs saved to: {renderer.run_dir}")

    # Clean up temp ground plane OBJ
    try:
        os.unlink(ground_obj_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
