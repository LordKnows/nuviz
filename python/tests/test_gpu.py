"""Tests for GPU metrics collection."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nuviz.config import NuvizConfig
from nuviz.gpu import GpuCollector
from nuviz.types import MetricRecord

# --- Helpers ---

NVIDIA_SMI_OUTPUT = " 42, 3072, 8192, 65\n"


def _make_successful_result(stdout: str = NVIDIA_SMI_OUTPUT) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _make_failed_result() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="error")


def _make_probe_result() -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=[], returncode=0, stdout="NVIDIA GeForce RTX 3090\n", stderr=""
    )


# --- GpuCollector unit tests ---


class TestGpuCollectorAvailable:
    """Tests when nvidia-smi is available and returns data."""

    @patch("nuviz.gpu.subprocess.run")
    def test_latest_returns_gpu_reading(self, mock_run: MagicMock) -> None:
        """After a successful poll, latest() returns a dict with GPU metrics."""
        mock_run.return_value = _make_successful_result()

        collector = GpuCollector(poll_interval=0.05)
        # Give the background thread time to probe + poll
        time.sleep(0.3)

        reading = collector.latest()
        collector.stop()

        assert reading is not None
        assert reading["gpu_util"] == 42
        assert reading["mem_used_mb"] == 3072
        assert reading["mem_total_mb"] == 8192
        assert reading["temperature_c"] == 65

    @patch("nuviz.gpu.subprocess.run")
    def test_polls_multiple_times(self, mock_run: MagicMock) -> None:
        """The collector polls nvidia-smi repeatedly at the interval."""
        mock_run.return_value = _make_successful_result()

        collector = GpuCollector(poll_interval=0.05)
        time.sleep(0.3)
        collector.stop()

        # At least probe + 1 poll
        assert mock_run.call_count >= 2


class TestGpuCollectorUnavailable:
    """Tests when nvidia-smi is not available."""

    @patch("nuviz.gpu.subprocess.run", side_effect=FileNotFoundError)
    def test_file_not_found_returns_none(self, mock_run: MagicMock) -> None:
        """When nvidia-smi is not found, latest() returns None."""
        collector = GpuCollector(poll_interval=0.05)
        time.sleep(0.2)

        assert collector.latest() is None
        collector.stop()

    @patch(
        "nuviz.gpu.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=3.0),
    )
    def test_timeout_returns_none(self, mock_run: MagicMock) -> None:
        """When nvidia-smi times out during probe, latest() returns None."""
        collector = GpuCollector(poll_interval=0.05)
        time.sleep(0.2)

        assert collector.latest() is None
        collector.stop()

    @patch("nuviz.gpu.subprocess.run")
    def test_nonzero_return_code_probe(self, mock_run: MagicMock) -> None:
        """When nvidia-smi returns non-zero on probe, thread exits and latest() is None."""
        mock_run.return_value = _make_failed_result()

        collector = GpuCollector(poll_interval=0.05)
        time.sleep(0.2)

        assert collector.latest() is None
        collector.stop()


class TestGpuCollectorPollErrors:
    """Tests for error handling during polling (after successful probe)."""

    @patch("nuviz.gpu.subprocess.run")
    def test_poll_failure_preserves_last_reading(self, mock_run: MagicMock) -> None:
        """If a poll fails after a successful one, the last good reading is preserved."""
        call_count = 0

        def side_effect(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                # Probe + first poll succeed
                return _make_successful_result()
            # Subsequent polls fail
            raise OSError("GPU fell off the bus")

        mock_run.side_effect = side_effect

        collector = GpuCollector(poll_interval=0.05)
        time.sleep(0.3)

        reading = collector.latest()
        collector.stop()

        # Should still have the reading from the first successful poll
        assert reading is not None
        assert reading["gpu_util"] == 42

    @patch("nuviz.gpu.subprocess.run")
    def test_poll_bad_output_ignored(self, mock_run: MagicMock) -> None:
        """If nvidia-smi returns garbage output, it's silently ignored."""
        call_count = 0

        def side_effect(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _make_probe_result()
            return _make_successful_result(stdout="not,enough,fields\n")

        mock_run.side_effect = side_effect

        collector = GpuCollector(poll_interval=0.05)
        time.sleep(0.2)

        # 3 fields < 4 required, so no reading stored
        assert collector.latest() is None
        collector.stop()


class TestGpuCollectorThreadSafety:
    """Tests for thread-safety guarantees."""

    def test_latest_returns_none_initially(self) -> None:
        """Before any poll completes, latest() returns None."""
        with patch("nuviz.gpu.subprocess.run", side_effect=lambda *a, **kw: time.sleep(10)):
            collector = GpuCollector(poll_interval=100)
            # Immediately check — probe hasn't completed
            assert collector.latest() is None
            collector.stop()

    @patch("nuviz.gpu.subprocess.run")
    def test_concurrent_latest_calls(self, mock_run: MagicMock) -> None:
        """Multiple threads can call latest() concurrently without errors."""
        mock_run.return_value = _make_successful_result()

        collector = GpuCollector(poll_interval=0.05)
        time.sleep(0.2)

        results: list[dict[str, int | float] | None] = []
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(50):
                    results.append(collector.latest())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        collector.stop()

        assert not errors
        assert len(results) == 250


class TestGpuCollectorStop:
    """Tests for the stop() method."""

    @patch("nuviz.gpu.subprocess.run")
    def test_stop_is_idempotent(self, mock_run: MagicMock) -> None:
        """Calling stop() multiple times does not raise."""
        mock_run.return_value = _make_successful_result()

        collector = GpuCollector(poll_interval=0.05)
        time.sleep(0.1)
        collector.stop()
        collector.stop()  # Should not raise


# --- Integration with Logger config ---


class TestGpuLoggerIntegration:
    """Tests for GPU collector integration with Logger."""

    def test_gpu_disabled_skips_collector(self, tmp_path: Path) -> None:
        """When enable_gpu=False, Logger does not create a GpuCollector."""
        config = NuvizConfig(
            base_dir=tmp_path,
            flush_interval_seconds=0.1,
            flush_count=5,
            enable_alerts=False,
            enable_snapshot=False,
            enable_gpu=False,
        )
        from nuviz.logger import Logger

        with Logger("test-no-gpu", config=config, snapshot=False) as log:
            assert log._gpu_collector is None

    @patch("nuviz.gpu.subprocess.run")
    def test_gpu_enabled_creates_collector(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """When enable_gpu=True (default), Logger creates a GpuCollector."""
        mock_run.side_effect = FileNotFoundError  # nvidia-smi not found is fine
        config = NuvizConfig(
            base_dir=tmp_path,
            flush_interval_seconds=0.1,
            flush_count=5,
            enable_alerts=False,
            enable_snapshot=False,
            enable_gpu=True,
        )
        from nuviz.logger import Logger

        with Logger("test-gpu", config=config, snapshot=False) as log:
            assert log._gpu_collector is not None

    @patch("nuviz.gpu.subprocess.run")
    def test_step_includes_gpu_data_in_record(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """When GPU data is available, it is included in the JSONL output."""
        mock_run.return_value = _make_successful_result()

        config = NuvizConfig(
            base_dir=tmp_path,
            flush_interval_seconds=0.1,
            flush_count=1,
            enable_alerts=False,
            enable_snapshot=False,
            enable_gpu=True,
        )
        from nuviz.logger import Logger

        log = Logger("test-gpu-data", config=config, snapshot=False)
        # Wait for GPU collector to get a reading
        time.sleep(0.3)
        log.step(0, loss=1.0)
        log.finish()

        jsonl_path = log.experiment_dir / "metrics.jsonl"
        assert jsonl_path.exists()
        data = json.loads(jsonl_path.read_text().strip().splitlines()[0])
        assert "gpu" in data
        assert data["gpu"]["gpu_util"] == 42
        assert data["gpu"]["mem_used_mb"] == 3072

    @patch("nuviz.gpu.subprocess.run", side_effect=FileNotFoundError)
    def test_step_omits_gpu_when_unavailable(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """When nvidia-smi is not available, gpu field is omitted from JSONL."""
        config = NuvizConfig(
            base_dir=tmp_path,
            flush_interval_seconds=0.1,
            flush_count=1,
            enable_alerts=False,
            enable_snapshot=False,
            enable_gpu=True,
        )
        from nuviz.logger import Logger

        log = Logger("test-no-nvidia", config=config, snapshot=False)
        time.sleep(0.2)
        log.step(0, loss=1.0)
        log.finish()

        jsonl_path = log.experiment_dir / "metrics.jsonl"
        data = json.loads(jsonl_path.read_text().strip().splitlines()[0])
        # gpu should be stripped by writer when None
        assert "gpu" not in data


# --- MetricRecord serialization ---


class TestMetricRecordGpuSerialization:
    """Tests that MetricRecord with gpu data serializes correctly."""

    def test_metric_record_with_gpu(self) -> None:
        """MetricRecord with gpu dict serializes to valid JSON."""
        gpu_data = {"gpu_util": 42, "mem_used_mb": 3072, "mem_total_mb": 8192, "temperature_c": 65}
        record = MetricRecord(step=0, timestamp=1.0, metrics={"loss": 0.5}, gpu=gpu_data)
        data = asdict(record)
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert parsed["gpu"]["gpu_util"] == 42
        assert parsed["metrics"]["loss"] == 0.5

    def test_metric_record_without_gpu(self) -> None:
        """MetricRecord with gpu=None serializes and gpu can be stripped."""
        record = MetricRecord(step=0, timestamp=1.0, metrics={"loss": 0.5})
        data = asdict(record)
        assert data["gpu"] is None
        # Writer strips None gpu
        del data["gpu"]
        serialized = json.dumps(data)
        parsed = json.loads(serialized)
        assert "gpu" not in parsed


# --- Config env var tests ---


class TestGpuConfigEnvVars:
    """Tests for GPU-related environment variable overrides."""

    def test_gpu_disabled_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NUVIZ_GPU=0 disables GPU collection."""
        monkeypatch.setenv("NUVIZ_GPU", "0")
        config = NuvizConfig.from_env()
        assert config.enable_gpu is False

    def test_gpu_enabled_by_default(self) -> None:
        """GPU collection is enabled by default."""
        config = NuvizConfig.from_env()
        assert config.enable_gpu is True

    def test_gpu_poll_interval_via_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """NUVIZ_GPU_POLL sets the poll interval."""
        monkeypatch.setenv("NUVIZ_GPU_POLL", "2.5")
        config = NuvizConfig.from_env()
        assert config.gpu_poll_interval == 2.5

    def test_gpu_poll_interval_default(self) -> None:
        """Default poll interval is 5.0 seconds."""
        config = NuvizConfig.from_env()
        assert config.gpu_poll_interval == 5.0
