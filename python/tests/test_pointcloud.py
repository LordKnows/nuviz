"""Tests for point cloud (PLY) saving functionality."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from nuviz.pointcloud import save_pointcloud


@pytest.fixture()
def exp_dir(tmp_path: Path) -> Path:
    return tmp_path / "test-exp"


class TestSavePointcloud:
    def test_basic_xyz(self, exp_dir: Path) -> None:
        xyz = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        result = save_pointcloud("points", 0, xyz, exp_dir)
        assert result is not None
        assert result.exists()
        assert result.name == "points_step_000000.ply"

        # Verify PLY structure
        content = result.read_bytes()
        header_end = content.index(b"end_header\n") + len(b"end_header\n")
        header = content[:header_end].decode("ascii")
        assert "format binary_little_endian 1.0" in header
        assert "element vertex 2" in header
        assert "property float x" in header

    def test_xyz_with_colors_uint8(self, exp_dir: Path) -> None:
        xyz = np.random.randn(10, 3).astype(np.float32)
        colors = np.random.randint(0, 256, (10, 3), dtype=np.uint8)
        result = save_pointcloud("colored", 5, xyz, exp_dir, colors=colors)
        assert result is not None

        header = result.read_bytes().split(b"end_header\n")[0].decode("ascii")
        assert "property uchar red" in header

    def test_xyz_with_colors_float(self, exp_dir: Path) -> None:
        xyz = np.random.randn(5, 3).astype(np.float32)
        colors = np.random.rand(5, 3).astype(np.float32)
        result = save_pointcloud("float_colors", 1, xyz, exp_dir, colors=colors)
        assert result is not None

    def test_xyz_with_opacities(self, exp_dir: Path) -> None:
        xyz = np.random.randn(5, 3).astype(np.float32)
        opacities = np.random.rand(5).astype(np.float32)
        result = save_pointcloud("with_opacity", 1, xyz, exp_dir, opacities=opacities)
        assert result is not None

        header = result.read_bytes().split(b"end_header\n")[0].decode("ascii")
        assert "property float opacity" in header

    def test_full_pointcloud(self, exp_dir: Path) -> None:
        n = 100
        xyz = np.random.randn(n, 3).astype(np.float32)
        colors = np.random.randint(0, 256, (n, 3), dtype=np.uint8)
        opacities = np.random.rand(n).astype(np.float32)

        result = save_pointcloud("gaussians", 42, xyz, exp_dir, colors=colors, opacities=opacities)
        assert result is not None

        # Verify data can be read back
        content = result.read_bytes()
        header_end = content.index(b"end_header\n") + len(b"end_header\n")
        data = content[header_end:]

        # Each vertex: 3*float + 3*uchar + 1*float = 12 + 3 + 4 = 19 bytes
        expected_size = n * 19
        assert len(data) == expected_size

        # Read first vertex
        x, y, z = struct.unpack_from("<fff", data, 0)
        assert abs(x - xyz[0, 0]) < 1e-6
        assert abs(y - xyz[0, 1]) < 1e-6

    def test_wrong_xyz_shape(self, exp_dir: Path) -> None:
        xyz = np.array([1.0, 2.0, 3.0])  # 1D, not (N, 3)
        result = save_pointcloud("bad", 0, xyz, exp_dir)
        assert result is None

    def test_wrong_colors_shape(self, exp_dir: Path) -> None:
        xyz = np.random.randn(5, 3).astype(np.float32)
        colors = np.random.randint(0, 256, (3, 3), dtype=np.uint8)  # wrong count
        result = save_pointcloud("bad", 0, xyz, exp_dir, colors=colors)
        assert result is None

    def test_wrong_opacities_shape(self, exp_dir: Path) -> None:
        xyz = np.random.randn(5, 3).astype(np.float32)
        opacities = np.random.rand(3).astype(np.float32)  # wrong count
        result = save_pointcloud("bad", 0, xyz, exp_dir, opacities=opacities)
        assert result is None

    def test_empty_array(self, exp_dir: Path) -> None:
        xyz = np.empty((0, 3), dtype=np.float32)
        result = save_pointcloud("empty", 0, xyz, exp_dir)
        assert result is None

    def test_non_array_input(self, exp_dir: Path) -> None:
        result = save_pointcloud("bad", 0, [[1, 2, 3]], exp_dir)
        assert result is None

    def test_directory_created(self, exp_dir: Path) -> None:
        xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = save_pointcloud("pts", 0, xyz, exp_dir)
        assert result is not None
        assert (exp_dir / "pointclouds").is_dir()

    def test_step_in_filename(self, exp_dir: Path) -> None:
        xyz = np.array([[0, 0, 0]], dtype=np.float32)
        result = save_pointcloud("pts", 12345, xyz, exp_dir)
        assert result is not None
        assert "step_012345" in result.name


class TestLoggerPointcloud:
    def test_logger_pointcloud_method(self, tmp_path: Path) -> None:
        from nuviz.config import NuvizConfig
        from nuviz.logger import Logger

        config = NuvizConfig(base_dir=tmp_path)
        log = Logger("pc-test", config=config, snapshot=False, alerts=False)

        xyz = np.random.randn(10, 3).astype(np.float32)
        log.step(5, loss=0.1)
        log.pointcloud("gaussians", xyz)

        pc_dir = log.experiment_dir / "pointclouds"
        assert pc_dir.is_dir()
        plys = list(pc_dir.glob("*.ply"))
        assert len(plys) == 1
        assert "step_000005" in plys[0].name

        log.finish()

    def test_logger_pointcloud_after_finish(self, tmp_path: Path) -> None:
        from nuviz.config import NuvizConfig
        from nuviz.logger import Logger

        config = NuvizConfig(base_dir=tmp_path)
        log = Logger("pc-test2", config=config, snapshot=False, alerts=False)
        log.finish()

        xyz = np.random.randn(5, 3).astype(np.float32)
        log.pointcloud("pts", xyz)  # Should be a no-op

        pc_dir = log.experiment_dir / "pointclouds"
        assert not pc_dir.exists()
