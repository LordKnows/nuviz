"""Image saving utilities."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy(data: Any) -> Any:
    """Convert input to numpy uint8 array.

    Accepts:
        - numpy array (float 0-1 or uint8)
        - torch tensor (auto-detach and convert)
    """
    # Handle torch tensors via duck typing
    if type(data).__module__.startswith("torch"):
        data = data.detach().cpu().numpy()

    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy array or torch tensor, got {type(data).__name__}")

    # Float to uint8 conversion
    if data.dtype in (np.float32, np.float64, np.float16):
        data = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)

    if data.dtype != np.uint8:
        data = data.astype(np.uint8)

    return data


def save_image(
    name: str,
    data: Any,
    step: int,
    experiment_dir: Path,
    cmap: str | None = None,
) -> Path | None:
    """Save an image to the experiment's images directory.

    Returns the saved file path, or None on failure.
    """
    try:
        arr = _to_numpy(data)

        # Apply colormap if requested (for single-channel data like depth maps)
        if cmap is not None and arr.ndim == 2:
            arr = _apply_colormap(arr, cmap)

        images_dir = experiment_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        filename = f"step_{step:06d}_{name}.png"
        filepath = images_dir / filename

        _write_png(arr, filepath)
        return filepath

    except Exception as e:
        print(f"[nuviz] Warning: failed to save image '{name}': {e}", file=sys.stderr)
        return None


def _apply_colormap(data: Any, cmap: str) -> Any:
    """Apply a simple colormap to single-channel data."""
    # Normalize to 0-255
    if data.dtype != np.uint8:
        dmin, dmax = data.min(), data.max()
        if dmax > dmin:
            data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)

    # Simple turbo-like colormap (red-yellow-green-blue)
    if cmap == "turbo":
        r = np.clip(np.where(data < 128, data * 2, 255 - (data - 128) * 2), 0, 255).astype(
            np.uint8
        )
        g = np.clip(np.where(data < 128, data * 2, (255 - data) * 2), 0, 255).astype(np.uint8)
        b = np.clip(255 - data, 0, 255).astype(np.uint8)
        return np.stack([r, g, b], axis=-1)

    # Default: grayscale to RGB
    return np.stack([data, data, data], axis=-1)


def _write_png(data: Any, path: Path) -> None:
    """Write image data as PNG using Pillow if available, else raise."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError(
            "Pillow is required for image saving. "
            "Install with: pip install nuviz[images]"
        ) from None

    if data.ndim == 2:
        img = Image.fromarray(data, mode="L")
    elif data.ndim == 3 and data.shape[2] == 3:
        img = Image.fromarray(data, mode="RGB")
    elif data.ndim == 3 and data.shape[2] == 4:
        img = Image.fromarray(data, mode="RGBA")
    else:
        raise ValueError(f"Unsupported image shape: {data.shape}")

    img.save(path, format="PNG")
