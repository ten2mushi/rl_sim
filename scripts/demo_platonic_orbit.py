#!/usr/bin/env python3
# NOTE: Run all scripts via the poetry environment:
#   cd rl_engine && PYTHONPATH=.. poetry run python scripts/demo_platonic_orbit.py
# Rebuild the C binding first if needed:
#   cd rl_engine && ./install.sh
"""
Platonic Solids Orbit Demo: Drone cameras orbit around a scene of random
platonic solids (tetrahedra, cubes, octahedra, dodecahedra, icosahedra).

Generates a random scene by scattering solids with random types, positions,
scales, and rotations, writes a combined OBJ, then runs the same 3-plane
circular orbit (XY, XZ, YZ) as the gyroid demo.

Run from rl_engine/:
    PYTHONPATH=.. poetry run python scripts/demo_platonic_orbit.py --resolution 256
    PYTHONPATH=.. poetry run python scripts/demo_platonic_orbit.py --resolution 64 --num-solids 12 --seed 7
    PYTHONPATH=.. poetry run python scripts/demo_platonic_orbit.py --resolution 32 --mode both
    PYTHONPATH=.. poetry run python scripts/demo_platonic_orbit.py --resolution 256 --output orbit_output/platonic_3plane_256 --mode gpu --frames 90 --num-solids 8 --seed 42
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

from robot import RobotEnv


# ---------------------------------------------------------------------------
# Platonic solid geometry — all vertices at unit circumradius, all faces
# triangulated with outward-facing CCW winding.
# ---------------------------------------------------------------------------

def _platonic_solids():
    """Return dict mapping name -> (vertices, faces) for all 5 Platonic solids.

    Vertices are numpy arrays (N, 3) at unit circumradius.
    Faces are lists of (i, j, k) triangle index tuples.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0  # golden ratio

    solids = {}

    # --- Tetrahedron: 4 vertices, 4 triangles ---
    inv_sqrt3 = 1.0 / math.sqrt(3.0)
    solids["tetrahedron"] = (
        np.array([
            ( 1,  1,  1), ( 1, -1, -1),
            (-1,  1, -1), (-1, -1,  1),
        ], dtype=np.float64) * inv_sqrt3,
        [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)],
    )

    # --- Cube (hexahedron): 8 vertices, 12 triangles ---
    # Indices: 0=(1,1,1) 1=(1,1,-1) 2=(1,-1,1) 3=(1,-1,-1)
    #          4=(-1,1,1) 5=(-1,1,-1) 6=(-1,-1,1) 7=(-1,-1,-1)
    solids["cube"] = (
        np.array([
            ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
            (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1),
        ], dtype=np.float64) * inv_sqrt3,
        [
            (0, 2, 3), (0, 3, 1),  # +x
            (4, 5, 7), (4, 7, 6),  # -x
            (0, 1, 5), (0, 5, 4),  # +y
            (2, 6, 7), (2, 7, 3),  # -y
            (0, 4, 6), (0, 6, 2),  # +z
            (1, 3, 7), (1, 7, 5),  # -z
        ],
    )

    # --- Octahedron: 6 vertices, 8 triangles ---
    # Already unit circumradius (vertices on axes at distance 1)
    solids["octahedron"] = (
        np.array([
            ( 1, 0, 0), (-1, 0, 0),
            ( 0, 1, 0), ( 0,-1, 0),
            ( 0, 0, 1), ( 0, 0,-1),
        ], dtype=np.float64),
        [
            (0, 2, 4), (0, 4, 3), (0, 3, 5), (0, 5, 2),
            (1, 4, 2), (1, 3, 4), (1, 5, 3), (1, 2, 5),
        ],
    )

    # --- Icosahedron: 12 vertices, 20 triangles ---
    # Circumradius = sqrt(1 + phi^2)
    ico_cr = math.sqrt(1.0 + phi * phi)
    ico_inv = 1.0 / ico_cr
    solids["icosahedron"] = (
        np.array([
            ( 0,  1,  phi), ( 0,  1, -phi), ( 0, -1,  phi), ( 0, -1, -phi),
            ( 1,  phi, 0),  ( 1, -phi, 0),  (-1,  phi, 0),  (-1, -phi, 0),
            ( phi, 0,  1),  ( phi, 0, -1),  (-phi, 0,  1),  (-phi, 0, -1),
        ], dtype=np.float64) * ico_inv,
        [
            (0, 2, 8),  (0, 8, 4),   (0, 4, 6),   (0, 6, 10),  (0, 10, 2),
            (2, 10, 7), (2, 7, 5),   (2, 5, 8),   (8, 5, 9),   (8, 9, 4),
            (4, 9, 1),  (4, 1, 6),   (6, 1, 11),  (6, 11, 10), (10, 11, 7),
            (3, 5, 7),  (3, 9, 5),   (3, 1, 9),   (3, 11, 1),  (3, 7, 11),
        ],
    )

    # --- Dodecahedron: 20 vertices, 36 triangles ---
    # Vertex groups: 8 cube verts (±1,±1,±1), plus 12 golden-ratio verts
    # Circumradius = sqrt(3)
    inv_phi = 1.0 / phi
    dodec_verts = np.array([
        ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),  # 0-3
        (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1),  # 4-7
        ( 0,  inv_phi,  phi), ( 0,  inv_phi, -phi),               # 8-9
        ( 0, -inv_phi,  phi), ( 0, -inv_phi, -phi),               # 10-11
        ( inv_phi,  phi, 0), (-inv_phi,  phi, 0),                  # 12-13 (note: 13=-1/phi,phi,0)
        ( inv_phi, -phi, 0), (-inv_phi, -phi, 0),                  # 14-15 (note: 14=1/phi,-phi,0)
        ( phi, 0,  inv_phi), ( phi, 0, -inv_phi),                  # 16-17
        (-phi, 0,  inv_phi), (-phi, 0, -inv_phi),                  # 18-19
    ], dtype=np.float64) * inv_sqrt3

    # NOTE: The plan's vertex indexing used indices 12-15 as (±1/φ, ±φ, 0).
    # We match that here but must remap: plan's v12=(1/φ,φ,0) is our v12,
    # plan's v13=(1/φ,-φ,0) is our v14, plan's v14=(-1/φ,φ,0) is our v13,
    # plan's v15=(-1/φ,-φ,0) is our v15.
    # Adjacency (from distance computation, edge length² = 4/φ²):
    #   0:{8,12,16}  1:{9,12,17}  2:{10,14,16}  3:{11,14,17}
    #   4:{8,13,18}  5:{9,13,19}  6:{10,15,18}  7:{11,15,19}
    #   8:{0,4,10}   9:{1,5,11}   10:{2,6,8}    11:{3,7,9}
    #   12:{0,1,13}  13:{4,5,12}  14:{2,3,15}   15:{6,7,14}
    #   16:{0,2,17}  17:{1,3,16}  18:{4,6,19}   19:{5,7,18}
    # 12 pentagonal faces (reversed for outward CCW winding), each fan-triangulated:
    dodec_faces = [
        # Face 1: (0,12,13,4,8)
        (0, 12, 13), (0, 13, 4), (0, 4, 8),
        # Face 2: (0,8,10,2,16)
        (0, 8, 10), (0, 10, 2), (0, 2, 16),
        # Face 3: (0,16,17,1,12)
        (0, 16, 17), (0, 17, 1), (0, 1, 12),
        # Face 4: (8,4,18,6,10)
        (8, 4, 18), (8, 18, 6), (8, 6, 10),
        # Face 5: (4,13,5,19,18)
        (4, 13, 5), (4, 5, 19), (4, 19, 18),
        # Face 6: (13,12,1,9,5)
        (13, 12, 1), (13, 1, 9), (13, 9, 5),
        # Face 7: (2,14,3,17,16)
        (2, 14, 3), (2, 3, 17), (2, 17, 16),
        # Face 8: (2,10,6,15,14)
        (2, 10, 6), (2, 6, 15), (2, 15, 14),
        # Face 9: (1,17,3,11,9)
        (1, 17, 3), (1, 3, 11), (1, 11, 9),
        # Face 10: (5,9,11,7,19)
        (5, 9, 11), (5, 11, 7), (5, 7, 19),
        # Face 11: (18,19,7,15,6)
        (18, 19, 7), (18, 7, 15), (18, 15, 6),
        # Face 12: (14,15,7,11,3)
        (14, 15, 7), (14, 7, 11), (14, 11, 3),
    ]
    solids["dodecahedron"] = (dodec_verts, dodec_faces)

    return solids


PLATONIC_SOLIDS = _platonic_solids()
SOLID_NAMES = list(PLATONIC_SOLIDS.keys())


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _axis_angle_to_matrix(axis, angle):
    """Build 3x3 rotation matrix from unit axis and angle (Rodrigues)."""
    c = math.cos(angle)
    s = math.sin(angle)
    t = 1.0 - c
    x, y, z = axis
    return np.array([
        [t*x*x + c,   t*x*y - s*z, t*x*z + s*y],
        [t*x*y + s*z, t*y*y + c,   t*y*z - s*x],
        [t*x*z - s*y, t*y*z + s*x, t*z*z + c  ],
    ])


# ---------------------------------------------------------------------------
# Scene generation
# ---------------------------------------------------------------------------

def generate_platonic_scene(num_solids, scale_min, scale_max, spread, seed,
                            output_path):
    """Scatter random platonic solids and write a combined OBJ.

    Returns (obj_path, auto_radius, scene_info) where scene_info is a list
    of (solid_type, position, scale) tuples.
    """
    rng = np.random.default_rng(seed)

    scene_info = []
    all_verts = []
    all_faces = []
    vert_offset = 0

    positions = []

    for _ in range(num_solids):
        # Pick random type, position, scale, rotation
        solid_name = SOLID_NAMES[rng.integers(len(SOLID_NAMES))]
        pos = rng.uniform(-spread, spread, size=3)
        scale = rng.uniform(scale_min, scale_max)

        # Random rotation: uniform random axis + angle
        raw_axis = rng.standard_normal(3)
        axis_len = np.linalg.norm(raw_axis)
        if axis_len < 1e-8:
            raw_axis = np.array([1.0, 0.0, 0.0])
            axis_len = 1.0
        axis = raw_axis / axis_len
        angle = rng.uniform(0, 2.0 * math.pi)
        R = _axis_angle_to_matrix(axis, angle)

        base_verts, base_faces = PLATONIC_SOLIDS[solid_name]
        # Transform: rotate, scale, translate
        transformed = (base_verts @ R.T) * scale + pos

        all_verts.append(transformed)
        for f in base_faces:
            all_faces.append((f[0] + vert_offset, f[1] + vert_offset,
                              f[2] + vert_offset))
        vert_offset += len(base_verts)

        positions.append(pos)
        scene_info.append((solid_name, pos.copy(), scale))

    # Shift everything so centroid of solid centers = origin
    positions = np.array(positions)
    centroid = positions.mean(axis=0)
    combined_verts = np.vstack(all_verts) - centroid

    # Also shift scene_info positions
    for i in range(len(scene_info)):
        name, pos, sc = scene_info[i]
        scene_info[i] = (name, pos - centroid, sc)

    # Compute auto_radius: max distance from origin to any vertex + clearance
    dists = np.linalg.norm(combined_verts, axis=1)
    auto_radius = float(dists.max()) + 5.0  # 5m clearance

    # Write OBJ
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"# Platonic solids scene: {num_solids} solids, seed={seed}\n")
        for v in combined_verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in all_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")  # OBJ is 1-indexed

    return output_path, auto_radius, scene_info


# ---------------------------------------------------------------------------
# Orbit plane definitions: (name, position_func, up_hint)
# ---------------------------------------------------------------------------

ORBIT_PLANES = [
    ("XY", lambda r, t: (r * math.cos(t), r * math.sin(t), 0.0), (0, 0, 1)),
    ("XZ", lambda r, t: (r * math.cos(t), 0.0, r * math.sin(t)), (0, 1, 0)),
    ("YZ", lambda r, t: (0.0, r * math.cos(t), r * math.sin(t)), (1, 0, 0)),
]


def look_at_origin_quat_3d(position, up_hint):
    """Compute quaternion (w,x,y,z) pointing body +X toward the origin."""
    px, py, pz = position
    ux, uy, uz = up_hint

    d = math.sqrt(px * px + py * py + pz * pz)
    if d < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    fx, fy, fz = -px / d, -py / d, -pz / d

    rx = uy * fz - uz * fy
    ry = uz * fx - ux * fz
    rz = ux * fy - uy * fx
    rd = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rd < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    rx, ry, rz = rx / rd, ry / rd, rz / rd

    upx = fy * rz - fz * ry
    upy = fz * rx - fx * rz
    upz = fx * ry - fy * rx

    m00, m01, m02 = fx, rx, upx
    m10, m11, m12 = fy, ry, upy
    m20, m21, m22 = fz, rz, upz

    tr = m00 + m11 + m22
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    qd = math.sqrt(w * w + x * x + y * y + z * z)
    if qd < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / qd, x / qd, y / qd, z / qd)


def quat_rotate_vec(qw, qx, qy, qz, vx, vy, vz):
    """Rotate vector (vx,vy,vz) by quaternion (qw,qx,qy,qz)."""
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    rx = vx + qw * tx + (qy * tz - qz * ty)
    ry = vy + qw * ty + (qz * tx - qx * tz)
    rz = vz + qw * tz + (qx * ty - qy * tx)
    return rx, ry, rz


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_depth_frame(depth_img, plt):
    """Render a single depth frame with inferno colormap + sky masking."""
    sky_mask = depth_img >= 0.999
    hit_vals = depth_img[~sky_mask]
    if len(hit_vals) > 0:
        d_min_hit = float(hit_vals.min())
        d_max_hit = float(hit_vals.max())
    else:
        d_min_hit, d_max_hit = 0.0, 1.0

    cmap = plt.cm.inferno.copy()
    if d_max_hit > d_min_hit:
        norm_img = (depth_img - d_min_hit) / (d_max_hit - d_min_hit)
    else:
        norm_img = np.zeros_like(depth_img)
    rgba = cmap(norm_img)
    rgba[sky_mask] = [0.5, 0.7, 0.9, 1.0]
    return (rgba * 255).astype(np.uint8)


def render_env_gifs(env_dir, rgb_frames, depth_frames, orbit_frames, plt):
    """Generate animated GIFs for RGB and depth orbit sequences."""
    try:
        from PIL import Image
    except ImportError:
        print("    Pillow not available, skipping GIF generation")
        return

    duration = max(50, 5000 // max(1, orbit_frames))

    rgb_pil = []
    for f in rgb_frames:
        img_uint8 = (np.clip(f, 0, 1) * 255).astype(np.uint8)
        rgb_pil.append(Image.fromarray(img_uint8))
    if rgb_pil:
        rgb_path = os.path.join(env_dir, "rgb_orbit.gif")
        rgb_pil[0].save(rgb_path, save_all=True, append_images=rgb_pil[1:],
                        duration=duration, loop=0)
        print(f"    {rgb_path}")

    depth_pil = []
    for f in depth_frames:
        rgba = render_depth_frame(f, plt)
        depth_pil.append(Image.fromarray(rgba[:, :, :3]))
    if depth_pil:
        depth_path = os.path.join(env_dir, "depth_orbit.gif")
        depth_pil[0].save(depth_path, save_all=True, append_images=depth_pil[1:],
                          duration=duration, loop=0)
        print(f"    {depth_path}")


def render_env_dashboards(env_dir, env_idx, resolution, orbit_radius,
                          positions, orientations, rgb_frames, depth_frames,
                          depth_stats, orbit_frames, scene_info=None):
    """Generate per-environment dashboard PNGs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

    # Color map for solid types
    type_colors = {
        "tetrahedron": "red",
        "cube": "blue",
        "octahedron": "green",
        "dodecahedron": "purple",
        "icosahedron": "orange",
    }

    positions_arr = np.array(positions)
    key_frames = list(range(0, orbit_frames, max(1, orbit_frames // 8)))

    for fi in key_frames:
        fig = plt.figure(figsize=(16, 5))

        # Left: 3D trajectory
        ax3d = fig.add_subplot(131, projection="3d")
        ax3d.plot(positions_arr[:, 0], positions_arr[:, 1], positions_arr[:, 2],
                  "b-", alpha=0.3, linewidth=0.8)
        ax3d.scatter(*positions_arr[fi], color="red", s=60, zorder=5)

        # Draw body axes
        qw, qx, qy, qz = orientations[fi]
        px, py_, pz = positions[fi]
        axis_len = 2.0
        for body_axis, color in [((1, 0, 0), "red"), ((0, 1, 0), "green"), ((0, 0, 1), "blue")]:
            wx, wy, wz = quat_rotate_vec(qw, qx, qy, qz, *body_axis)
            ax3d.quiver(px, py_, pz, wx * axis_len, wy * axis_len, wz * axis_len,
                        color=color, arrow_length_ratio=0.15, linewidth=1.5)

        # Draw solid positions as colored markers
        if scene_info:
            for sname, spos, sscale in scene_info:
                ax3d.scatter(*spos, color=type_colors.get(sname, "gray"),
                             s=sscale * 30, alpha=0.6, marker="D", edgecolors="k",
                             linewidths=0.3)

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
        depth_display = depth_frames[fi].copy()
        sky_mask = depth_display >= 0.999
        hit_vals = depth_display[~sky_mask]
        if len(hit_vals) > 0:
            d_min_hit = float(hit_vals.min())
            d_max_hit = float(hit_vals.max())
        else:
            d_min_hit, d_max_hit = 0.0, 1.0
        cmap = plt.cm.inferno.copy()
        cmap.set_bad(color=(0.5, 0.7, 0.9))
        depth_masked = np.ma.masked_where(sky_mask, depth_display)
        im = ax_depth.imshow(depth_masked, cmap=cmap, vmin=d_min_hit, vmax=d_max_hit)
        ax_depth.set_title(f"Depth (hits [{d_min_hit:.2f}, {d_max_hit:.2f}])")
        ax_depth.axis("off")
        plt.colorbar(im, ax=ax_depth, fraction=0.046, pad=0.04)

        fig.suptitle(f"Env {env_idx} - Frame {fi}/{orbit_frames}", fontsize=14)
        fig.tight_layout()
        fig.savefig(os.path.join(env_dir, f"frame_{fi:04d}.png"), dpi=120)
        plt.close(fig)

    # Overview: depth statistics + 3D trajectory with solid markers
    fig = plt.figure(figsize=(14, 5))

    stats_arr = np.array(depth_stats)
    ax_stats = fig.add_subplot(121)
    ax_stats.plot(stats_arr[:, 0], label="min", alpha=0.8)
    ax_stats.plot(stats_arr[:, 1], label="max", alpha=0.8)
    ax_stats.plot(stats_arr[:, 2], label="mean", alpha=0.8)
    ax_stats.set_xlabel("Frame")
    ax_stats.set_ylabel("Depth value")
    ax_stats.set_title("Depth Statistics Over Orbit")
    ax_stats.legend()
    ax_stats.grid(True, alpha=0.3)

    # 3D trajectory overview with plane labels + solid markers
    ax3d = fig.add_subplot(122, projection="3d")
    n = len(positions_arr)
    seg = n // 3
    colors = ["tab:blue", "tab:orange", "tab:green"]
    labels = ["XY plane", "XZ plane", "YZ plane"]
    for i, (c, lbl) in enumerate(zip(colors, labels)):
        s = i * seg
        e = (i + 1) * seg if i < 2 else n
        ax3d.plot(positions_arr[s:e, 0], positions_arr[s:e, 1],
                  positions_arr[s:e, 2], color=c, alpha=0.7, linewidth=1.2, label=lbl)
    ax3d.scatter(*positions_arr[0], color="green", s=80, zorder=5, label="Start")
    ax3d.scatter(0, 0, 0, color="black", s=100, zorder=5, marker="+", label="Origin")

    # Solid position markers colored by type
    if scene_info:
        plotted_types = set()
        for sname, spos, sscale in scene_info:
            lbl = sname if sname not in plotted_types else None
            ax3d.scatter(*spos, color=type_colors.get(sname, "gray"),
                         s=sscale * 40, alpha=0.7, marker="D", edgecolors="k",
                         linewidths=0.5, label=lbl, zorder=4)
            plotted_types.add(sname)

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title("3-Plane Orbit + Solid Positions")
    lim = orbit_radius * 1.3
    ax3d.set_xlim(-lim, lim)
    ax3d.set_ylim(-lim, lim)
    ax3d.set_zlim(-lim, lim)
    ax3d.legend(fontsize=7, loc="upper left")

    fig.suptitle(f"Env {env_idx} - Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(env_dir, "overview.png"), dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------

def create_env(num_envs, resolution, orbit_radius, obj_path, world_bound,
               camera_fov, camera_far, voxel_size, obs_dim,
               use_gpu_voxelization=True, config_path=None, add_imu=False):
    """Create a RobotEnv configured for the platonic orbit demo."""
    wb = world_bound

    kwargs = dict(
        num_envs=num_envs,
        agents_per_env=1,
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
        spawn_min=(orbit_radius - 1.0, -1.0, -1.0),
        spawn_max=(orbit_radius + 1.0,  1.0,  1.0),
        termination_min=(-wb, -wb, -wb),
        termination_max=( wb,  wb,  wb),
        world_min=(-wb, -wb, -wb),
        world_max=( wb,  wb,  wb),
        use_gpu_voxelization=use_gpu_voxelization,
    )

    if config_path is not None:
        kwargs["config_path"] = config_path

    return RobotEnv(**kwargs)


# ---------------------------------------------------------------------------
# Orbit loop
# ---------------------------------------------------------------------------

def run_orbit_loop(env, num_envs, resolution, orbit_frames, orbit_radius,
                   collect_frames=True):
    """Run the 3-plane orbit loop, returning timing and optionally frame data."""
    rgb_dim = resolution * resolution * 3
    depth_dim = resolution * resolution

    positions = []
    orientations = []
    all_rgb = [[] for _ in range(num_envs)]
    all_depth = [[] for _ in range(num_envs)]
    all_depth_stats = [[] for _ in range(num_envs)]

    seg = orbit_frames // 3

    t_start = time.perf_counter()

    for step in range(orbit_frames):
        if step < seg:
            plane_idx = 0
            local_step = step
            local_total = seg
        elif step < 2 * seg:
            plane_idx = 1
            local_step = step - seg
            local_total = seg
        else:
            plane_idx = 2
            local_step = step - 2 * seg
            local_total = orbit_frames - 2 * seg

        _, pos_fn, up_hint = ORBIT_PLANES[plane_idx]
        theta = 2.0 * math.pi * local_step / local_total
        px, py, pz = pos_fn(orbit_radius, theta)
        qw, qx, qy, qz = look_at_origin_quat_3d((px, py, pz), up_hint)

        for e in range(num_envs):
            env.set_agent_state(e, (px, py, pz), (qw, qx, qy, qz))

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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_orbit(num_envs=1, resolution=16, orbit_frames=120, orbit_radius=0.0,
              camera_fov=1.5708, camera_far=50.0,
              voxel_size=0.1, output_dir=None, mode="auto", warmup=10,
              no_viz=False, gpu_voxelization=True, config_path=None,
              add_imu=False, num_solids=8, scale_min=1.0, scale_max=3.0,
              spread=10.0, seed=42):
    """Run the 3-plane orbit demo around a random platonic solids scene."""

    if output_dir is None:
        output_dir = os.path.join(rl_engine_dir, "orbit_output", "platonic_3plane")
    os.makedirs(output_dir, exist_ok=True)

    # Generate scene OBJ
    obj_path = os.path.join(output_dir, "platonic_scene.obj")
    print(f"Generating scene: {num_solids} platonic solids, seed={seed}")
    print(f"  scale=[{scale_min}, {scale_max}], spread={spread}")
    obj_path, auto_radius, scene_info = generate_platonic_scene(
        num_solids, scale_min, scale_max, spread, seed, obj_path)
    print(f"  OBJ written: {obj_path}")

    # Print scene summary
    print(f"\nScene contents ({num_solids} solids, centroid shifted to origin):")
    type_counts = {}
    for sname, spos, sscale in scene_info:
        type_counts[sname] = type_counts.get(sname, 0) + 1
        print(f"  {sname:14s}  pos=({spos[0]:+6.2f}, {spos[1]:+6.2f}, {spos[2]:+6.2f})  "
              f"scale={sscale:.2f}")
    print(f"  Types: {', '.join(f'{k}={v}' for k, v in sorted(type_counts.items()))}")

    # Determine orbit radius
    if orbit_radius <= 0:
        orbit_radius = auto_radius
        print(f"\nAuto orbit radius: {orbit_radius:.2f}m (scene extent + 5m)")
    else:
        print(f"\nOrbit radius: {orbit_radius:.2f}m")

    world_bound = orbit_radius + 7.0  # 7m margin beyond orbit

    rgb_dim = resolution * resolution * 3
    depth_dim = resolution * resolution
    pos_dim = 3
    vel_dim = 6
    imu_dim = 6 if add_imu else 0
    obs_dim = rgb_dim + depth_dim + pos_dim + vel_dim + imu_dim

    seg = orbit_frames // 3
    print(f"Environments: {num_envs}")
    print(f"Resolution:   {resolution}x{resolution}")
    print(f"Orbit:        radius={orbit_radius:.1f}m, frames={orbit_frames} "
          f"({seg} XY + {seg} XZ + {orbit_frames - 2*seg} YZ)")
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
    env = create_env(num_envs, resolution, orbit_radius, obj_path, world_bound,
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
        if not gpu_available:
            print("\nGPU not available -- running CPU-only benchmark")
            mode = "cpu"
        else:
            print(f"\n--- Warmup ({warmup} frames) ---")
            run_orbit_loop(env, num_envs, resolution, warmup,
                           orbit_radius, collect_frames=False)

            env.set_gpu_enabled(True)
            print(f"\n--- GPU benchmark ({orbit_frames} frames) ---")
            gpu_ms, *_ = run_orbit_loop(
                env, num_envs, resolution, orbit_frames,
                orbit_radius, collect_frames=False)

            env.set_gpu_enabled(False)
            print(f"--- CPU benchmark ({orbit_frames} frames) ---")
            cpu_ms, *_ = run_orbit_loop(
                env, num_envs, resolution, orbit_frames,
                orbit_radius, collect_frames=False)

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
                print(f"\nCollecting frames for visualization (GPU)...")
                _, positions, orientations, all_rgb, all_depth, all_depth_stats = \
                    run_orbit_loop(env, num_envs, resolution, orbit_frames,
                                   orbit_radius, collect_frames=True)
                env.close()
                _generate_viz(output_dir, num_envs, resolution, orbit_radius,
                              positions, orientations, all_rgb, all_depth,
                              all_depth_stats, orbit_frames, scene_info)
            else:
                env.close()
            return

    # Single-mode run
    if mode == "cpu":
        env.set_gpu_enabled(False)
    elif mode == "gpu":
        if not gpu_available:
            print("WARNING: GPU not available, falling back to CPU")
        else:
            env.set_gpu_enabled(True)

    active_mode = "GPU" if env.is_gpu_enabled() else "CPU"
    print(f"\nRunning with: {active_mode}")

    if warmup > 0:
        run_orbit_loop(env, num_envs, resolution, warmup,
                       orbit_radius, collect_frames=False)

    elapsed_ms, positions, orientations, all_rgb, all_depth, all_depth_stats = \
        run_orbit_loop(env, num_envs, resolution, orbit_frames,
                       orbit_radius, collect_frames=not no_viz)

    per_frame = elapsed_ms / orbit_frames
    print(f"\n{active_mode} total: {elapsed_ms:.1f}ms  ({per_frame:.3f}ms/frame)")

    env.close()

    if not no_viz and positions:
        _generate_viz(output_dir, num_envs, resolution, orbit_radius,
                      positions, orientations, all_rgb, all_depth,
                      all_depth_stats, orbit_frames, scene_info)


def _generate_viz(output_dir, num_envs, resolution, orbit_radius,
                  positions, orientations, all_rgb, all_depth,
                  all_depth_stats, orbit_frames, scene_info=None):
    """Generate per-env dashboard PNGs and animated GIFs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping visualization")
        return

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
            orbit_frames, scene_info=scene_info,
        )

    print(f"\nGenerating animated GIFs...")
    for e in range(num_envs):
        print(f"  env_{e}/")
        render_env_gifs(env_dirs[e], all_rgb[e], all_depth[e], orbit_frames, plt)

    print(f"\nDone! Output in {output_dir}/")
    for e in range(num_envs):
        print(f"  env_{e}/  ({len(os.listdir(env_dirs[e]))} files)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Platonic Solids 3-Plane Orbit Demo")
    parser.add_argument("--num-envs", type=int, default=1,
                        help="Number of parallel environments (default: 1)")
    parser.add_argument("--resolution", type=int, default=16,
                        help="Camera resolution (default: 16)")
    parser.add_argument("--frames", type=int, default=90,
                        help="Number of orbit frames (default: 90)")
    parser.add_argument("--radius", type=float, default=0.0,
                        help="Orbit radius in meters (0 = auto from scene extent)")
    parser.add_argument("--voxel-size", type=float, default=0.1,
                        help="SDF voxel size (default: 0.1)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (default: rl_engine/orbit_output/platonic_3plane)")
    parser.add_argument("--mode", choices=["cpu", "gpu", "auto", "both"], default="auto",
                        help="Sensor backend (default: auto)")
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
    # Scene generation args
    parser.add_argument("--num-solids", type=int, default=8,
                        help="Number of platonic solids to scatter (default: 8)")
    parser.add_argument("--scale-min", type=float, default=1.0,
                        help="Minimum solid scale (default: 1.0)")
    parser.add_argument("--scale-max", type=float, default=3.0,
                        help="Maximum solid scale (default: 3.0)")
    parser.add_argument("--spread", type=float, default=10.0,
                        help="Position spread in meters (default: 10.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible scenes (default: 42)")
    args = parser.parse_args()

    run_orbit(
        num_envs=args.num_envs,
        resolution=args.resolution,
        orbit_frames=args.frames,
        orbit_radius=args.radius,
        voxel_size=args.voxel_size,
        output_dir=args.output,
        mode=args.mode,
        warmup=args.warmup,
        no_viz=args.no_viz,
        gpu_voxelization=not args.no_gpu_voxelization,
        config_path=args.config,
        add_imu=args.add_imu,
        num_solids=args.num_solids,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        spread=args.spread,
        seed=args.seed,
    )
