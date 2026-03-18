"""Tests for image saving."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from nuviz.image import _to_numpy, save_image


class TestToNumpy:
    def test_uint8_passthrough(self) -> None:
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = _to_numpy(arr)
        assert result.dtype == np.uint8

    def test_float_to_uint8(self) -> None:
        arr = np.ones((10, 10, 3), dtype=np.float32)
        result = _to_numpy(arr)
        assert result.dtype == np.uint8
        assert result.max() == 255

    def test_float_clipped(self) -> None:
        arr = np.array([[-1.0, 0.5, 2.0]], dtype=np.float32)
        result = _to_numpy(arr)
        assert result[0, 0] == 0
        assert result[0, 2] == 255

    def test_torch_tensor_duck_typing(self) -> None:
        """Test that torch-like objects are handled via duck typing."""
        mock_tensor = MagicMock()
        mock_tensor.__class__.__module__ = "torch"
        type(mock_tensor).__module__ = "torch"

        expected = np.zeros((10, 10, 3), dtype=np.uint8)
        mock_tensor.detach.return_value.cpu.return_value.numpy.return_value = expected

        result = _to_numpy(mock_tensor)
        np.testing.assert_array_equal(result, expected)

    def test_invalid_type_raises(self) -> None:
        with pytest.raises(TypeError, match="Expected numpy array"):
            _to_numpy([1, 2, 3])


class TestSaveImage:
    def test_saves_rgb_image(self, tmp_path: Path) -> None:
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result = save_image("render", arr, step=100, experiment_dir=tmp_path)

        assert result is not None
        assert result.exists()
        assert result.name == "step_000100_render.png"
        assert result.parent.name == "images"

    def test_saves_grayscale_image(self, tmp_path: Path) -> None:
        arr = np.random.randint(0, 255, (64, 64), dtype=np.uint8)
        result = save_image("depth", arr, step=50, experiment_dir=tmp_path)

        assert result is not None
        assert result.exists()

    def test_colormap_applied(self, tmp_path: Path) -> None:
        arr = np.random.rand(32, 32).astype(np.float32)
        result = save_image("depth", arr, step=10, experiment_dir=tmp_path, cmap="turbo")

        assert result is not None
        assert result.exists()

    def test_creates_images_directory(self, tmp_path: Path) -> None:
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        save_image("test", arr, step=0, experiment_dir=tmp_path)

        assert (tmp_path / "images").is_dir()

    def test_error_returns_none(self, tmp_path: Path) -> None:
        """Passing invalid data should return None, not raise."""
        result = save_image("test", "not_an_array", step=0, experiment_dir=tmp_path)
        assert result is None

    def test_step_zero_padding(self, tmp_path: Path) -> None:
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = save_image("test", arr, step=5, experiment_dir=tmp_path)
        assert result is not None
        assert "step_000005" in result.name
