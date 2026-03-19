"""Error handling hardening tests — edge cases and resilience."""

from __future__ import annotations

import json
import time
from pathlib import Path

from nuviz.anomaly import AnomalyDetector
from nuviz.config import NuvizConfig
from nuviz.logger import Logger
from nuviz.types import MetricRecord
from nuviz.writer import JsonlWriter


def _make_config(tmp_path: Path, **overrides) -> NuvizConfig:
    defaults = dict(
        base_dir=tmp_path,
        flush_interval_seconds=0.1,
        flush_count=5,
        enable_alerts=True,
        enable_snapshot=False,
    )
    defaults.update(overrides)
    return NuvizConfig(**defaults)


class TestLoggerResilience:
    def test_step_with_non_float_metric_value(self, tmp_path: Path) -> None:
        """Non-numeric values should not crash the logger."""
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)
        # Python allows passing non-float kwargs, logger should handle gracefully
        log.step(0, loss=0.1, note="hello")  # type: ignore
        log.finish()

        lines = (log.experiment_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1

    def test_step_with_very_large_values(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False, alerts=False)
        log.step(0, loss=1e308, psnr=-1e308)
        log.finish()

        data = json.loads(
            (log.experiment_dir / "metrics.jsonl").read_text().strip()
        )
        assert data["metrics"]["loss"] == 1e308

    def test_step_with_zero_metrics(self, tmp_path: Path) -> None:
        config = _make_config(tmp_path)
        log = Logger("test", config=config, snapshot=False)
        log.step(0)  # no metrics at all
        log.finish()

        data = json.loads(
            (log.experiment_dir / "metrics.jsonl").read_text().strip()
        )
        assert data["metrics"] == {}

    def test_many_rapid_steps(self, tmp_path: Path) -> None:
        """Stress test: 10k steps as fast as possible."""
        config = _make_config(tmp_path, flush_count=500)
        log = Logger("stress", config=config, snapshot=False, alerts=False)
        for i in range(10_000):
            log.step(i, loss=1.0 / (i + 1))
        log.finish()

        lines = (log.experiment_dir / "metrics.jsonl").read_text().strip().splitlines()
        assert len(lines) == 10_000

    def test_experiment_dir_deleted_mid_training(self, tmp_path: Path) -> None:
        """Logger should not crash if experiment dir is removed."""
        config = _make_config(tmp_path)
        log = Logger("fragile", config=config, snapshot=False)
        log.step(0, loss=1.0)

        # Force flush so the first step is written
        log._writer.flush()

        # Delete the experiment directory
        import shutil
        shutil.rmtree(log.experiment_dir)

        # Subsequent steps should silently fail
        log.step(1, loss=0.5)
        log.finish()  # Should not crash


class TestWriterResilience:
    def test_concurrent_writers_different_files(self, tmp_path: Path) -> None:
        """Two writers to different files should not interfere."""
        w1 = JsonlWriter(tmp_path / "a.jsonl", flush_interval=0.1, flush_count=100)
        w2 = JsonlWriter(tmp_path / "b.jsonl", flush_interval=0.1, flush_count=100)

        for i in range(20):
            w1.write(MetricRecord(step=i, timestamp=time.time(), metrics={"x": float(i)}))
            w2.write(MetricRecord(step=i, timestamp=time.time(), metrics={"y": float(i * 2)}))

        w1.close()
        w2.close()

        lines_a = (tmp_path / "a.jsonl").read_text().strip().splitlines()
        lines_b = (tmp_path / "b.jsonl").read_text().strip().splitlines()
        assert len(lines_a) == 20
        assert len(lines_b) == 20


class TestAnomalyEdgeCases:
    def test_negative_values(self) -> None:
        detector = AnomalyDetector()
        alerts = detector.check(0, {"loss": -5.0})
        assert len(alerts) == 0  # Negative is valid

    def test_integer_values(self) -> None:
        detector = AnomalyDetector()
        alerts = detector.check(0, {"epoch": 1})
        assert len(alerts) == 0

    def test_empty_metrics(self) -> None:
        detector = AnomalyDetector()
        alerts = detector.check(0, {})
        assert len(alerts) == 0

    def test_single_metric_many_steps(self) -> None:
        """Welford's algorithm should handle thousands of updates."""
        detector = AnomalyDetector(min_steps=100)
        for i in range(5000):
            alerts = detector.check(i, {"loss": 0.1 + 0.001 * (i % 5)})
            # Should not crash or produce false positives in steady state
            if i > 200:
                assert len(alerts) == 0


class TestConfigFromEnv:
    def test_env_var_override(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("NUVIZ_DIR", str(tmp_path / "custom"))
        config = NuvizConfig.from_env()
        assert config.base_dir == tmp_path / "custom"

    def test_flush_interval_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("NUVIZ_FLUSH_INTERVAL", "5.0")
        config = NuvizConfig.from_env()
        assert config.flush_interval_seconds == 5.0

    def test_alerts_disabled_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("NUVIZ_ALERTS", "0")
        config = NuvizConfig.from_env()
        assert config.enable_alerts is False

    def test_defaults_without_env(self, monkeypatch) -> None:
        monkeypatch.delenv("NUVIZ_DIR", raising=False)
        monkeypatch.delenv("NUVIZ_FLUSH_INTERVAL", raising=False)
        monkeypatch.delenv("NUVIZ_FLUSH_COUNT", raising=False)
        monkeypatch.delenv("NUVIZ_ALERTS", raising=False)
        monkeypatch.delenv("NUVIZ_SNAPSHOT", raising=False)
        config = NuvizConfig.from_env()
        assert config.flush_interval_seconds == 2.0
        assert config.flush_count == 50
        assert config.enable_alerts is True
