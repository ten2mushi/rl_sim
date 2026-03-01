"""Episode GIF renderer for quadcopter hover training.

Saves periodic animated GIF dashboards showing full eval episodes:
1. 3D trajectory with hover target and ground plane
2. RGB camera view
3. Depth camera view (inferno colormap + sky masking)
4. Per-step metrics (reward, altitude, velocity)

Output structure:
    output/training/run_<timestamp>/episode_step_NNNNNN.gif
"""

import os
from datetime import datetime

import numpy as np


class TrainingRenderer:
    """Headless renderer that saves episode GIF dashboards to disk."""

    def __init__(self, output_root="output/training"):
        import matplotlib
        matplotlib.use("Agg")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_root, f"run_{timestamp}")
        os.makedirs(self.run_dir, exist_ok=True)

    def render_episode_gif(self, frames, global_step, episode_return,
                           resolution):
        """Render a 4-panel animated GIF from eval episode data.

        Args:
            frames: Dict with keys 'positions', 'rgb_frames', 'depth_frames',
                    'rewards', 'velocities'.
            global_step: Global training step number.
            episode_return: Cumulative episode reward.
            resolution: Camera resolution (width = height).

        Returns:
            Path to the saved GIF file.
        """
        import matplotlib.pyplot as plt
        from PIL import Image

        positions = frames["positions"]
        rgb_frames = frames["rgb_frames"]
        depth_frames = frames["depth_frames"]
        rewards = frames["rewards"]
        velocities = frames["velocities"]
        num_frames = len(positions)

        if num_frames == 0:
            return None

        positions_arr = np.array(positions)
        cumulative_rewards = np.cumsum(rewards)
        altitudes = positions_arr[:, 2]

        pil_frames = []

        for fi in range(num_frames):
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))

            # --- Top-left: 3D trajectory ---
            ax3d = fig.add_subplot(2, 2, 1, projection="3d")
            axes[0, 0].remove()  # Remove the 2D axes, replace with 3D

            # Trail up to current frame
            trail = positions_arr[:fi + 1]
            ax3d.plot(trail[:, 0], trail[:, 1], trail[:, 2],
                      "b-", alpha=0.4, linewidth=1.0)
            # Current position
            px, py, pz = positions[fi]
            ax3d.scatter(px, py, pz, color="red", s=60, zorder=5)
            # Hover target at (0, 0, 1)
            ax3d.scatter(0, 0, 1, color="green", marker="*", s=200, zorder=10)
            # Ground plane wireframe at z=0
            gx = [-5, 5, 5, -5, -5]
            gy = [-5, -5, 5, 5, -5]
            gz = [0, 0, 0, 0, 0]
            ax3d.plot(gx, gy, gz, "k-", alpha=0.3, linewidth=0.8)

            ax3d.set_xlabel("X")
            ax3d.set_ylabel("Y")
            ax3d.set_zlabel("Z")
            ax3d.set_xlim(-5, 5)
            ax3d.set_ylim(-5, 5)
            ax3d.set_zlim(-0.5, 5)
            ax3d.set_title(f"({px:.1f}, {py:.1f}, {pz:.1f})", fontsize=9)

            # --- Top-right: RGB camera ---
            ax_rgb = axes[0, 1]
            ax_rgb.imshow(rgb_frames[fi])
            ax_rgb.set_title(f"RGB ({resolution}x{resolution})", fontsize=9)
            ax_rgb.axis("off")

            # --- Bottom-left: Depth camera (inferno + sky masking) ---
            ax_depth = axes[1, 0]
            depth_img = depth_frames[fi]
            sky_mask = depth_img >= 0.999
            hit_vals = depth_img[~sky_mask]
            if len(hit_vals) > 0:
                d_min = float(hit_vals.min())
                d_max = float(hit_vals.max())
            else:
                d_min, d_max = 0.0, 1.0
            cmap = plt.cm.inferno.copy()
            cmap.set_bad(color=(0.5, 0.7, 0.9))
            depth_masked = np.ma.masked_where(sky_mask, depth_img)
            ax_depth.imshow(depth_masked, cmap=cmap, vmin=d_min, vmax=d_max)
            ax_depth.set_title(f"Depth [{d_min:.2f}, {d_max:.2f}]", fontsize=9)
            ax_depth.axis("off")

            # --- Bottom-right: Metrics ---
            ax_met = axes[1, 1]
            steps_so_far = range(fi + 1)
            ax_met.plot(steps_so_far, rewards[:fi + 1],
                        "g-", alpha=0.7, linewidth=0.8, label="reward")
            ax_alt = ax_met.twinx()
            ax_alt.plot(steps_so_far, altitudes[:fi + 1],
                        "b-", alpha=0.7, linewidth=0.8, label="altitude")
            ax_alt.plot(steps_so_far, velocities[:fi + 1],
                        "r-", alpha=0.5, linewidth=0.8, label="speed")
            # Current step marker
            ax_met.axvline(x=fi, color="gray", alpha=0.4, linewidth=0.5)

            ax_met.set_xlabel("Step", fontsize=8)
            ax_met.set_ylabel("Reward", fontsize=8, color="green")
            ax_alt.set_ylabel("Alt / Speed", fontsize=8)
            ax_met.tick_params(labelsize=7)
            ax_alt.tick_params(labelsize=7)
            ax_met.set_xlim(0, max(num_frames - 1, 1))

            # Combined legend
            lines_met, labels_met = ax_met.get_legend_handles_labels()
            lines_alt, labels_alt = ax_alt.get_legend_handles_labels()
            ax_met.legend(lines_met + lines_alt, labels_met + labels_alt,
                          fontsize=6, loc="upper left")

            fig.suptitle(
                f"Step {global_step:,} | Frame {fi}/{num_frames} | "
                f"Return: {episode_return:.2f}",
                fontsize=11,
            )
            fig.tight_layout()

            # Convert matplotlib figure to PIL Image
            fig.canvas.draw()
            buf = fig.canvas.buffer_rgba()
            img_arr = np.asarray(buf)
            pil_frames.append(Image.fromarray(img_arr[:, :, :3].copy()))
            plt.close(fig)

        # Save as animated GIF
        duration = max(50, 5000 // max(1, num_frames))
        path = os.path.join(self.run_dir, f"episode_step_{global_step:06d}.gif")
        pil_frames[0].save(
            path, save_all=True, append_images=pil_frames[1:],
            duration=duration, loop=0,
        )
        return path
