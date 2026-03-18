"""Integration tests — end-to-end training loop simulation."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from nuviz.config import NuvizConfig
from nuviz.logger import Logger


def _make_config(tmp_path: Path) -> NuvizConfig:
    return NuvizConfig(
        base_dir=tmp_path,
        flush_interval_seconds=0.1,
        flush_count=10,
        enable_alerts=True,
        enable_snapshot=False,
    )


class TestFullTrainingLoop:
    def test_realistic_training_simulation(self, tmp_path: Path) -> None:
        """Simulate a realistic 100-step training loop."""
        config = _make_config(tmp_path)

        with Logger("integration-test", project="test_project", config=config, snapshot=False) as log:
            for step in range(100):
                loss = 1.0 * (0.97 ** step) + 0.01 * np.random.randn()
                psnr = 20.0 + 10.0 * (1 - 0.97 ** step)

                log.step(step, loss=max(loss, 0.001), psnr=psnr)

                if step % 25 == 0:
                    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
                    log.image("render", img)

            exp_dir = log.experiment_dir

        # Verify JSONL
        jsonl_path = exp_dir / "metrics.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 100

        # Verify each line is valid JSON with correct schema
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["step"] == i
            assert "timestamp" in data
            assert "loss" in data["metrics"]
            assert "psnr" in data["metrics"]

        # Verify meta.json
        meta = json.loads((exp_dir / "meta.json").read_text())
        assert meta["status"] == "done"
        assert meta["total_steps"] == 99
        assert "loss" in meta["best_metrics"]
        assert "psnr" in meta["best_metrics"]

        # Verify images
        images = list((exp_dir / "images").glob("*.png"))
        assert len(images) == 4  # steps 0, 25, 50, 75

    def test_nan_injection(self, tmp_path: Path) -> None:
        """Verify that NaN in metrics produces alert but doesn't crash."""
        config = _make_config(tmp_path)

        with Logger("nan-test", config=config, snapshot=False) as log:
            for step in range(20):
                log.step(step, loss=0.1)

            # Inject NaN
            log.step(20, loss=float("nan"))

            # Continue training
            for step in range(21, 30):
                log.step(step, loss=0.1)

            exp_dir = log.experiment_dir

        # All 30 records should be written
        lines = (exp_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 30

    def test_concurrent_loggers(self, tmp_path: Path) -> None:
        """Two loggers writing to different experiments simultaneously."""
        config = _make_config(tmp_path)

        log1 = Logger("exp-1", config=config, snapshot=False)
        log2 = Logger("exp-2", config=config, snapshot=False)

        for step in range(50):
            log1.step(step, loss=0.1 * step)
            log2.step(step, loss=0.2 * step)

        log1.finish()
        log2.finish()

        lines1 = (log1.experiment_dir / "metrics.jsonl").read_text().strip().splitlines()
        lines2 = (log2.experiment_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines1) == 50
        assert len(lines2) == 50

    def test_large_batch_write(self, tmp_path: Path) -> None:
        """Verify correct handling of many rapid writes."""
        config = NuvizConfig(
            base_dir=tmp_path,
            flush_interval_seconds=0.1,
            flush_count=100,
            enable_alerts=False,
            enable_snapshot=False,
        )

        with Logger("large-batch", config=config, snapshot=False, alerts=False) as log:
            for step in range(1000):
                log.step(step, loss=1.0 / (step + 1))
            exp_dir = log.experiment_dir

        lines = (exp_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1000

    def test_empty_experiment(self, tmp_path: Path) -> None:
        """Logger with no steps should still produce valid meta.json."""
        config = _make_config(tmp_path)

        with Logger("empty", config=config, snapshot=False) as log:
            exp_dir = log.experiment_dir

        meta = json.loads((exp_dir / "meta.json").read_text())
        assert meta["status"] == "done"
        assert meta["total_steps"] == 0
