"""Tests for the core Logger class."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nuviz.config import NuvizConfig
from nuviz.logger import Logger


def _make_config(tmp_path: Path) -> NuvizConfig:
    return NuvizConfig(
        base_dir=tmp_path,
        flush_interval_seconds=0.1,
        flush_count=5,
        enable_alerts=True,
        enable_snapshot=False,
    )


class TestLoggerLifecycle:
    def test_basic_lifecycle(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test-exp", config=config, snapshot=False)

        log.step(0, loss=1.0)
        log.step(1, loss=0.5)
        log.finish()

        # Verify JSONL
        jsonl_path = log.experiment_dir / "metrics.jsonl"
        assert jsonl_path.exists()
        lines = jsonl_path.read_text().strip().splitlines()
        assert len(lines) == 2

        # Verify meta.json
        meta_path = log.experiment_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["status"] == "done"
        assert meta["total_steps"] == 1

    def test_context_manager(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        with Logger("ctx-test", config=config, snapshot=False) as log:
            log.step(0, loss=1.0)
            exp_dir = log.experiment_dir

        # finish() should have been called
        meta = json.loads((exp_dir / "meta.json").read_text())
        assert meta["status"] == "done"

    def test_double_finish_safe(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)
        log.step(0, loss=1.0)
        log.finish()
        log.finish()  # Should not raise or duplicate data

    def test_step_after_finish_ignored(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)
        log.step(0, loss=1.0)
        log.finish()
        log.step(1, loss=0.5)  # Should be silently ignored

        lines = (log.experiment_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_experiment_dir_created(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("dir-test", config=config, snapshot=False)
        assert log.experiment_dir.exists()
        log.finish()


class TestLoggerMetrics:
    def test_metrics_recorded_correctly(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)

        log.step(100, loss=0.05, psnr=28.4, ssim=0.93)
        log.finish()

        data = json.loads(
            (log.experiment_dir / "metrics.jsonl").read_text().strip()
        )
        assert data["step"] == 100
        assert data["metrics"]["loss"] == 0.05
        assert data["metrics"]["psnr"] == 28.4
        assert data["metrics"]["ssim"] == 0.93

    def test_best_metrics_tracked(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)

        log.step(0, loss=1.0, psnr=20.0)
        log.step(1, loss=0.5, psnr=25.0)
        log.step(2, loss=0.8, psnr=23.0)
        log.finish()

        meta = json.loads((log.experiment_dir / "meta.json").read_text())
        # loss is minimized, psnr is maximized
        assert meta["best_metrics"]["loss"] == 0.5
        assert meta["best_metrics"]["psnr"] == 25.0


class TestLoggerImages:
    def test_image_saved(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)

        log.step(100, loss=0.1)
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        log.image("render", arr)
        log.finish()

        images_dir = log.experiment_dir / "images"
        assert images_dir.exists()
        pngs = list(images_dir.glob("*.png"))
        assert len(pngs) == 1
        assert "step_000100_render" in pngs[0].name


class TestLoggerAlerts:
    def test_nan_alert(self, tmp_path: Path, capsys: object) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False, alerts=True)

        log.step(50, loss=float("nan"))
        log.finish()

    def test_alerts_disabled(self, tmp_path: Path) -> None:
        config = NuvizConfig(
            base_dir=tmp_path,
            flush_interval_seconds=0.1,
            flush_count=5,
            enable_alerts=False,
            enable_snapshot=False,
        )
        log = Logger("test", config=config, snapshot=False, alerts=False)

        # Should not crash even with NaN
        log.step(50, loss=float("nan"))
        log.finish()


class TestLoggerProject:
    def test_project_creates_nested_dir(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("exp-001", project="gaussian_splatting", config=config, snapshot=False)

        assert "gaussian_splatting" in str(log.experiment_dir)
        log.finish()


class TestMetricCoercion:
    def test_numpy_and_tensor_like_metrics(self, tmp_path: Path) -> None:
        np = __import__("pytest").importorskip("numpy")

        class FakeTensor:
            """Mimics a 0-dim torch tensor."""

            def item(self) -> float:
                return 0.25

        config = NuvizConfig(
            base_dir=tmp_path, enable_snapshot=False, enable_gpu=False, flush_interval_seconds=0.05
        )
        with Logger("coerce", config=config, snapshot=False) as log:
            log.step(0, loss=np.float32(0.5), acc=np.float64(0.9), n=np.int64(3))
            log.step(1, loss=FakeTensor(), acc=np.array(0.95), n=4)

        lines = (log.experiment_dir / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(line)["metrics"] for line in lines] == [
            {"loss": 0.5, "acc": 0.9, "n": 3},
            {"loss": 0.25, "acc": 0.95, "n": 4},
        ]
        meta = json.loads((log.experiment_dir / "meta.json").read_text())
        assert meta["best_metrics"]["loss"] == 0.25
        assert meta["best_metrics"]["acc"] == 0.95

    def test_non_numeric_metric_is_skipped_not_fatal(self, tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
        config = NuvizConfig(base_dir=tmp_path, enable_snapshot=False, enable_gpu=False)
        with Logger("bad-metric", config=config, snapshot=False) as log:
            log.step(0, loss=0.5, note="hello")  # type: ignore[arg-type]
            log.step(1, loss=0.4, note="again")  # type: ignore[arg-type]

        lines = (log.experiment_dir / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(line)["metrics"] for line in lines] == [{"loss": 0.5}, {"loss": 0.4}]
        assert capsys.readouterr().err.count("metric 'note' ignored") == 1
