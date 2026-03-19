"""Tests for per-scene metric recording."""

from __future__ import annotations

import json
from pathlib import Path

from nuviz.config import NuvizConfig
from nuviz.logger import Logger


def _make_config(tmp_path: Path) -> NuvizConfig:
    return NuvizConfig(
        base_dir=tmp_path,
        flush_interval_seconds=0.1,
        flush_count=5,
        enable_alerts=False,
        enable_snapshot=False,
    )


class TestSceneRecording:
    def test_scene_writes_jsonl(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)

        log.scene("garden", psnr=27.41, ssim=0.945)
        log.finish()

        scenes_path = log.experiment_dir / "scenes.jsonl"
        assert scenes_path.exists()
        data = json.loads(scenes_path.read_text().strip())
        assert data["scene"] == "garden"
        assert data["metrics"]["psnr"] == 27.41
        assert data["metrics"]["ssim"] == 0.945
        assert "timestamp" in data

    def test_multiple_scenes(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)

        scenes = [
            ("bicycle", {"psnr": 25.12, "ssim": 0.912}),
            ("garden", {"psnr": 27.41, "ssim": 0.945}),
            ("stump", {"psnr": 26.88, "ssim": 0.931}),
        ]
        for name, metrics in scenes:
            log.scene(name, **metrics)
        log.finish()

        lines = (log.experiment_dir / "scenes.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

        for i, (name, metrics) in enumerate(scenes):
            data = json.loads(lines[i])
            assert data["scene"] == name
            assert data["metrics"]["psnr"] == metrics["psnr"]

    def test_scene_after_finish_ignored(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)

        log.scene("garden", psnr=27.0)
        log.finish()
        log.scene("bicycle", psnr=25.0)  # Should be ignored

        lines = (log.experiment_dir / "scenes.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_no_scenes_no_file(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)

        log.step(0, loss=1.0)
        log.finish()

        assert not (log.experiment_dir / "scenes.jsonl").exists()

    def test_scenes_with_steps(self, tmp_path: Path) -> None:
        """Scene recording works alongside step recording."""
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)

        log.step(0, loss=1.0)
        log.step(1, loss=0.5)
        log.scene("garden", psnr=27.0)
        log.finish()

        assert (log.experiment_dir / "metrics.jsonl").exists()
        assert (log.experiment_dir / "scenes.jsonl").exists()
