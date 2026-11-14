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


class TestWriterRobustness:
    """Regressions for failures that used to kill or race the writer thread."""

    def test_numpy_scalars_are_written(self, tmp_path: Path) -> None:
        np = __import__("pytest").importorskip("numpy")
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=100)
        writer.write(
            MetricRecord(
                step=1,
                timestamp=1.0,
                metrics={"a": np.float32(0.5), "b": np.int64(3), "c": np.float64("nan")},
            )
        )
        writer.close()

        data = json.loads(path.read_text())
        assert data["metrics"] == {"a": 0.5, "b": 3, "c": None}

    def test_unserializable_record_is_skipped_and_thread_survives(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=1)

        writer.write(_make_record(1))
        writer.write(MetricRecord(step=2, timestamp=1.0, metrics={"bad": object()}))  # type: ignore[dict-item]
        writer.write(_make_record(3))
        assert writer.flush()
        assert writer._thread.is_alive()

        writer.write(_make_record(4))
        writer.close()

        steps = [json.loads(line)["step"] for line in path.read_text().splitlines()]
        assert steps == [1, 3, 4]

    def test_large_backlog_keeps_order_and_loses_nothing(self, tmp_path: Path) -> None:
        """close() must not write concurrently with the background thread."""
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(
            path,
            flush_interval=10.0,
            flush_count=10_000_000,  # never flush early: everything is backlog
            max_file_lines=7_000,
            close_timeout=60.0,
        )
        n = 50_000
        for i in range(n):
            writer.write(_make_record(i))
        writer.close()

        segments = sorted(
            tmp_path.glob("metrics.*.jsonl"),
            key=lambda p: -int(p.name.split(".")[1]),
        )
        steps: list[int] = []
        for file in [*segments, path]:
            steps.extend(json.loads(line)["step"] for line in file.read_text().splitlines())
        assert steps == list(range(n))

    def test_full_queue_drops_instead_of_growing(self, tmp_path: Path) -> None:
        import threading

        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=1, max_queue=5)
        writer._block_timeout = 0.05

        # Simulate a hung filesystem: the writer thread blocks inside its write.
        release = threading.Event()
        entered = threading.Event()
        real_write_batch = writer._write_batch

        def stalled(records: list[MetricRecord]) -> None:
            entered.set()
            release.wait(timeout=10.0)
            real_write_batch(records)

        writer._write_batch = stalled  # type: ignore[method-assign]

        assert writer.write(_make_record(0))
        assert entered.wait(timeout=5.0)

        accepted = [writer.write(_make_record(i)) for i in range(1, 30)]
        assert sum(accepted) == 5  # queue capacity; the rest are dropped, not buffered
        assert writer.dropped == 24

        release.set()
        writer.close()
        steps = [json.loads(line)["step"] for line in path.read_text().splitlines()]
        assert steps == [0, 1, 2, 3, 4, 5]

    def test_file_deleted_externally_is_recreated(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=1)
        writer.write(_make_record(1))
        assert writer.flush()
        path.unlink()
        writer.write(_make_record(2))
        writer.close()
        steps = [json.loads(line)["step"] for line in path.read_text().splitlines()]
        assert steps == [2]

    def test_flush_from_many_threads(self, tmp_path: Path) -> None:
        import threading

        path = tmp_path / "metrics.jsonl"
        writer = JsonlWriter(path, flush_interval=10.0, flush_count=7)

        def produce(offset: int) -> None:
            for i in range(500):
                writer.write(_make_record(offset + i))
                if i % 50 == 0:
                    writer.flush()

        threads = [threading.Thread(target=produce, args=(k * 1000,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        writer.close()

        lines = path.read_text().splitlines()
        assert len(lines) == 2000
        assert {json.loads(line)["step"] for line in lines} == {
            k * 1000 + i for k in range(4) for i in range(500)
        }
