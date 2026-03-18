"""Tests for the background JSONL writer."""

from __future__ import annotations

import json
import time
from pathlib import Path

from nuviz.types import MetricRecord
from nuviz.writer import JsonlWriter


def _make_record(step: int, loss: float = 0.1) -> MetricRecord:
    return MetricRecord(step=step, timestamp=time.time(), metrics={"loss": loss})


class TestJsonlWriter:
    def test_write_and_close_flushes(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)

        writer.write(_make_record(1))
        writer.write(_make_record(2))
        writer.close()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["step"] == 1
        assert json.loads(lines[1])["step"] == 2

    def test_count_based_flush(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=3)

        for i in range(5):
            writer.write(_make_record(i))

        # Give writer thread time to flush
        time.sleep(0.3)

        # At least the first 3 should be flushed
        lines = path.read_text().strip().splitlines()
        assert len(lines) >= 3

        writer.close()

    def test_time_based_flush(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=0.1, flush_count=1000)

        writer.write(_make_record(1))

        # Wait for time-based flush
        time.sleep(0.3)

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

        writer.close()

    def test_explicit_flush(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=1000)

        writer.write(_make_record(1))
        writer.flush()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

        writer.close()

    def test_close_is_idempotent(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)

        writer.write(_make_record(1))
        writer.close()
        writer.close()  # Should not raise

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1

    def test_write_after_close_is_noop(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)
        writer.close()

        writer.write(_make_record(999))
        writer.flush()

        # File may not exist at all (no writes happened), or be empty
        if path.exists():
            content = path.read_text().strip()
            assert content == ""

    def test_gpu_none_omitted(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)

        writer.write(_make_record(1))
        writer.close()

        data = json.loads(path.read_text().strip())
        assert "gpu" not in data

    def test_gpu_present_when_set(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)

        record = MetricRecord(
            step=1, timestamp=time.time(), metrics={"loss": 0.1},
            gpu={"util": 87, "mem_used": 10240},
        )
        writer.write(record)
        writer.close()

        data = json.loads(path.read_text().strip())
        assert data["gpu"]["util"] == 87

    def test_jsonl_format_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)

        for i in range(10):
            writer.write(_make_record(i, loss=0.1 * i))
        writer.close()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 10
        for line in lines:
            data = json.loads(line)
            assert "step" in data
            assert "timestamp" in data
            assert "metrics" in data

    def test_error_resilience_readonly(self, tmp_path: Path) -> None:
        """Writer should not crash on I/O errors — just warn."""
        path = tmp_path / "readonly" / "metrics.jsonl"
        # Don't create parent dir — write will fail
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)
        writer.write(_make_record(1))
        writer.close()  # Should not raise
