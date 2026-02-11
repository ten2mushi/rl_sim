#!/usr/bin/env python3
"""Standalone gyroid cube OBJ generator for voxel SDF engine testing.

Generates a gyroid isosurface within a cube and exports it as a standard OBJ
file. Uses numpy for vectorized grid sampling and scikit-image marching cubes
for mesh extraction. No Blender dependency.

The gyroid surface is defined by the implicit equation:
    sin(wx)cos(wy) + sin(wy)cos(wz) + sin(wz)cos(wx) = 0

where w (omega) controls the frequency and thus the channel diameter.

Usage:
    python3 generate_gyroid_cube.py --size 20 --channel-diameter 1.0
    python3 generate_gyroid_cube.py --size 10 --channel-diameter 0.5 --resolution 200
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from skimage.measure import marching_cubes


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class MeshData(NamedTuple):
    """Extracted mesh geometry from marching cubes."""

    vertices: NDArray[np.float64]  # (V, 3)
    faces: NDArray[np.intp]        # (F, 3)


# ---------------------------------------------------------------------------
# Gyroid math
# ---------------------------------------------------------------------------

def compute_omega(channel_diameter: float) -> float:
    """Derive angular frequency from desired channel diameter.

    The gyroid channel diameter is approximately 0.35 * period, where
    period = 2*pi / omega.  Inverting: omega = 2*pi * 0.35 / d.

    Args:
        channel_diameter: Target inner channel diameter in meters.

    Returns:
        Angular frequency omega (rad/m).

    Raises:
        ValueError: If channel_diameter is not positive.
    """
    if channel_diameter <= 0.0:
        raise ValueError(
            f"channel_diameter must be positive, got {channel_diameter}"
        )
    return 2.0 * math.pi * 0.35 / channel_diameter


def auto_resolution(size: float, channel_diameter: float) -> int:
    """Calculate grid resolution to get ~8 voxels per channel diameter.

    Args:
        size: Cube edge length in meters.
        channel_diameter: Target channel diameter in meters.

    Returns:
        Number of grid points per axis.
    """
    voxels_per_channel = 8
    voxel_size = channel_diameter / voxels_per_channel
    resolution = int(math.ceil(size / voxel_size))
    # Clamp to a sane range
    return max(16, min(resolution, 1024))


# ---------------------------------------------------------------------------
# Mesh generation
# ---------------------------------------------------------------------------

def generate_gyroid(
    size: float,
    channel_diameter: float,
    resolution: int,
) -> MeshData:
    """Generate a gyroid isosurface mesh within a cube.

    Pipeline:
        1. Compute omega from channel_diameter.
        2. Create a 3D numpy grid spanning [-size/2, size/2]^3.
        3. Evaluate the gyroid function (vectorized) on the grid.
        4. Run skimage marching_cubes at level=0.
        5. Return vertices and faces.

    Args:
        size: Cube edge length in meters.
        channel_diameter: Target inner channel diameter in meters.
        resolution: Number of grid points per axis.

    Returns:
        MeshData with vertices (V, 3) and faces (F, 3).
    """
    omega = compute_omega(channel_diameter)
    period = 2.0 * math.pi / omega
    approx_diameter = 0.35 * period
    print(f"  omega          = {omega:.4f} rad/m")
    print(f"  period         = {period:.4f} m")
    print(f"  approx channel = {approx_diameter:.4f} m")

    half = size / 2.0
    spacing = size / (resolution - 1)

    # -- Phase 1: grid sampling ------------------------------------------------
    t0 = time.perf_counter()
    lin = np.linspace(-half, half, resolution)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")

    wx = omega * x
    wy = omega * y
    wz = omega * z

    volume: NDArray[np.float64] = (
        np.sin(wx) * np.cos(wy)
        + np.sin(wy) * np.cos(wz)
        + np.sin(wz) * np.cos(wx)
    )
    dt_sample = time.perf_counter() - t0
    print(f"  grid sampling  : {dt_sample:.3f}s  "
          f"({resolution}^3 = {resolution**3:,} voxels)")

    # -- Phase 2: marching cubes -----------------------------------------------
    t0 = time.perf_counter()
    verts_mc, faces_mc, _, _ = marching_cubes(
        volume,
        level=0.0,
        spacing=(spacing, spacing, spacing),
    )
    dt_mc = time.perf_counter() - t0
    print(f"  marching cubes : {dt_mc:.3f}s  "
          f"({len(verts_mc):,} verts, {len(faces_mc):,} faces)")

    # marching_cubes returns vertices in grid coordinates starting at origin.
    # Shift so the cube is centered at the origin.
    verts_mc -= half

    # -- Phase 3: vertex welding (deduplicate coincident vertices) -------------
    t0 = time.perf_counter()
    vertices, inverse = np.unique(
        np.round(verts_mc, decimals=6), axis=0, return_inverse=True
    )
    faces = inverse[faces_mc]
    dt_weld = time.perf_counter() - t0
    print(f"  vertex welding : {dt_weld:.3f}s  "
          f"({len(verts_mc):,} -> {len(vertices):,} verts)")

    return MeshData(vertices=vertices, faces=faces)


# ---------------------------------------------------------------------------
# OBJ export
# ---------------------------------------------------------------------------

def export_obj(path: str | os.PathLike[str], mesh: MeshData) -> int:
    """Write a standard Wavefront OBJ file.

    Writes vertex (``v``) and face (``f``) records. No materials, normals, or
    texture coordinates -- just geometry for the SDF engine.

    Args:
        path: Output file path.
        mesh: MeshData containing vertices and faces.

    Returns:
        File size in bytes.
    """
    t0 = time.perf_counter()
    filepath = Path(path)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)

    with open(filepath, "w") as fh:
        fh.write(f"# Gyroid cube generated by generate_gyroid_cube.py\n")
        fh.write(f"# Vertices: {n_verts}  Faces: {n_faces}\n\n")

        # Vertices
        for x, y, z in mesh.vertices:
            fh.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")

        fh.write("\n")

        # Faces (OBJ uses 1-based indexing)
        for i0, i1, i2 in mesh.faces:
            fh.write(f"f {i0 + 1} {i1 + 1} {i2 + 1}\n")

    file_size = filepath.stat().st_size
    dt_export = time.perf_counter() - t0
    print(f"  OBJ export     : {dt_export:.3f}s  ({file_size:,} bytes)")

    return file_size


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    default_output = str(
        Path(__file__).resolve().parent.parent
        / "input" / "from_utils" / "gyroid_cube.obj"
    )

    parser = argparse.ArgumentParser(
        description="Generate a gyroid cube OBJ mesh for SDF engine testing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--size",
        type=float,
        default=20.0,
        help="Cube edge length in meters.",
    )
    parser.add_argument(
        "--channel-diameter",
        type=float,
        default=1.0,
        help="Target inner channel diameter in meters.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=0,
        help=(
            "Grid points per axis.  0 = auto-calculate from size and "
            "channel-diameter (~8 voxels per channel)."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=default_output,
        help="Output OBJ file path.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the gyroid cube generator."""
    parser = build_parser()
    args = parser.parse_args(argv)

    size: float = args.size
    channel_diameter: float = args.channel_diameter
    resolution: int = args.resolution
    output: str = args.output

    if resolution <= 0:
        resolution = auto_resolution(size, channel_diameter)

    print("=" * 60)
    print("Gyroid Cube Generator")
    print("=" * 60)
    print(f"  size             = {size} m")
    print(f"  channel diameter = {channel_diameter} m")
    print(f"  resolution       = {resolution}")
    print(f"  output           = {output}")
    print("=" * 60)

    t_total = time.perf_counter()

    print("\n[1/2] Generating gyroid mesh ...")
    mesh = generate_gyroid(size, channel_diameter, resolution)

    print("\n[2/2] Exporting OBJ ...")
    file_size = export_obj(output, mesh)

    dt_total = time.perf_counter() - t_total

    print("\n" + "=" * 60)
    print("Done.")
    print(f"  vertices   : {len(mesh.vertices):,}")
    print(f"  faces      : {len(mesh.faces):,}")
    print(f"  file size  : {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")
    print(f"  total time : {dt_total:.3f}s")
    print(f"  output     : {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
