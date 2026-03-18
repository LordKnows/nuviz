"""Test that NaN/Inf values are serialized as null (valid JSON)."""

from __future__ import annotations

import json
import time
from pathlib import Path

from nuviz.types import MetricRecord
from nuviz.writer import JsonlWriter


class TestNanSerialization:
    def test_nan_written_as_null(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)

        record = MetricRecord(
            step=0, timestamp=time.time(),
            metrics={"loss": float("nan"), "psnr": 25.0},
        )
        writer.write(record)
        writer.close()

        data = json.loads(path.read_text().strip())
        assert data["metrics"]["loss"] is None  # NaN -> null
        assert data["metrics"]["psnr"] == 25.0

    def test_inf_written_as_null(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)

        record = MetricRecord(
            step=0, timestamp=time.time(),
            metrics={"loss": float("inf")},
        )
        writer.write(record)
        writer.close()

        data = json.loads(path.read_text().strip())
        assert data["metrics"]["loss"] is None

    def test_valid_json_parseable_by_any_parser(self, tmp_path: Path) -> None:
        """Verify output is strict JSON (no NaN/Infinity literals)."""
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)

        writer.write(MetricRecord(step=0, timestamp=1.0, metrics={"a": float("nan")}))
        writer.write(MetricRecord(step=1, timestamp=2.0, metrics={"a": float("inf")}))
        writer.write(MetricRecord(step=2, timestamp=3.0, metrics={"a": float("-inf")}))
        writer.write(MetricRecord(step=3, timestamp=4.0, metrics={"a": 0.5}))
        writer.close()

        content = path.read_text().strip()
        # Should not contain NaN or Infinity literals
        assert "NaN" not in content
        assert "Infinity" not in content

        # Every line should be valid JSON
        for line in content.splitlines():
            json.loads(line)  # Should not raise
