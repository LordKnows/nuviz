"""Point cloud (PLY) writing utilities."""

from __future__ import annotations

import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy_float32(data: Any) -> np.ndarray | None:
    """Convert input to numpy float32 array, handling torch tensors."""
    if data is None:
        return None

    if type(data).__module__.startswith("torch"):
        data = data.detach().cpu().numpy()

    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy array or torch tensor, got {type(data).__name__}")

    return data.astype(np.float32)


def _to_numpy_uint8(data: Any) -> Any:
    """Convert color data to uint8. Accepts float [0,1] or uint8."""
    if data is None:
        return None

    if type(data).__module__.startswith("torch"):
        data = data.detach().cpu().numpy()

    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy array or torch tensor, got {type(data).__name__}")

    if data.dtype in (np.float32, np.float64, np.float16):
        data = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)
    elif data.dtype != np.uint8:
        data = data.astype(np.uint8)

    return data


def save_pointcloud(
    tag: str,
    step: int,
    xyz: Any,
    experiment_dir: Path,
    colors: Any | None = None,
    opacities: Any | None = None,
) -> Path | None:
    """Save a point cloud as binary little-endian PLY.

    Args:
        tag: Identifier (e.g., "gaussians", "points").
        step: Current training step.
        xyz: Positions array of shape (N, 3).
        experiment_dir: Experiment directory path.
        colors: Optional RGB colors, shape (N, 3). Float [0,1] or uint8.
        opacities: Optional opacity values, shape (N,). Float.

    Returns:
        Path to saved PLY file, or None on failure.
    """
    try:
        xyz_arr = _to_numpy_float32(xyz)
        if xyz_arr is None or xyz_arr.ndim != 2 or xyz_arr.shape[1] != 3:
            raise ValueError(f"xyz must be shape (N, 3), got {getattr(xyz_arr, 'shape', None)}")

        num_points = xyz_arr.shape[0]
        if num_points == 0:
            raise ValueError("xyz array is empty")

        colors_arr = _to_numpy_uint8(colors)
        if colors_arr is not None and (colors_arr.ndim != 2 or colors_arr.shape != (num_points, 3)):
            raise ValueError(
                f"colors must be shape ({num_points}, 3), got {colors_arr.shape}"
            )

        opacities_arr = _to_numpy_float32(opacities)
        if opacities_arr is not None and opacities_arr.shape != (num_points,):
            raise ValueError(
                f"opacities must be shape ({num_points},), got {opacities_arr.shape}"
            )

        pc_dir = experiment_dir / "pointclouds"
        pc_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{tag}_step_{step:06d}.ply"
        filepath = pc_dir / filename

        _write_binary_ply(filepath, xyz_arr, colors_arr, opacities_arr)
        return filepath

    except Exception as e:
        print(f"[nuviz] Warning: failed to save pointcloud '{tag}': {e}", file=sys.stderr)
        return None


def _write_binary_ply(
    path: Path,
    xyz: np.ndarray,
    colors: np.ndarray | None,
    opacities: np.ndarray | None,
) -> None:
    """Write a binary little-endian PLY file."""
    num_points = xyz.shape[0]

    # Build header
    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
    ]

    if colors is not None:
        header_lines.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ])

    if opacities is not None:
        header_lines.append("property float opacity")

    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("ascii"))

        for i in range(num_points):
            # xyz (3 x float32)
            f.write(struct.pack("<fff", xyz[i, 0], xyz[i, 1], xyz[i, 2]))

            # colors (3 x uint8)
            if colors is not None:
                f.write(struct.pack("<BBB", colors[i, 0], colors[i, 1], colors[i, 2]))

            # opacity (float32)
            if opacities is not None:
                f.write(struct.pack("<f", opacities[i]))
