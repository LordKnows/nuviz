"""Tests for environment snapshot capture."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from nuviz.snapshot import capture_snapshot, write_meta, _run_command


class TestRunCommand:
    def test_successful_command(self) -> None:
        result = _run_command(["echo", "hello"])
        assert result == "hello"

    def test_failed_command(self) -> None:
        result = _run_command(["false"])
        assert result is None

    def test_nonexistent_command(self) -> None:
        result = _run_command(["nonexistent_cmd_xyz"])
        assert result is None

    def test_timeout(self) -> None:
        result = _run_command(["sleep", "10"], timeout=0.1)
        assert result is None


class TestCaptureSnapshot:
    @patch("nuviz.snapshot._run_command")
    def test_captures_all_fields(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = lambda cmd, **kw: {
            ("git", "rev-parse", "HEAD"): "abc123",
            ("git", "diff", "--quiet"): "",  # clean
            ("pip", "freeze"): "numpy==1.24.0\ntorch==2.0.0",
            ("nvcc", "--version"): "Cuda compilation tools, release 12.1",
            ("nvidia-smi", "--query-gpu=name", "--format=csv,noheader"): "RTX 4090",
        }.get(tuple(cmd))

        snapshot = capture_snapshot()
        assert snapshot.git_hash == "abc123"
        assert snapshot.git_dirty is False
        assert "numpy==1.24.0" in snapshot.pip_packages
        assert snapshot.gpu_model == "RTX 4090"
        assert snapshot.python_version is not None
        assert snapshot.hostname is not None
        assert snapshot.start_time is not None

    @patch("nuviz.snapshot._run_command")
    def test_dirty_repo_detected(self, mock_run: MagicMock) -> None:
        mock_run.side_effect = lambda cmd, **kw: {
            ("git", "rev-parse", "HEAD"): "abc123",
            ("git", "diff", "--quiet"): None,  # non-zero exit = dirty
        }.get(tuple(cmd))

        snapshot = capture_snapshot()
        assert snapshot.git_dirty is True

    @patch("nuviz.snapshot._run_command", return_value=None)
    def test_all_commands_fail_gracefully(self, mock_run: MagicMock) -> None:
        snapshot = capture_snapshot()
        assert snapshot.git_hash is None
        assert snapshot.gpu_model is None
        assert snapshot.cuda_version is None
        assert snapshot.python_version is not None  # sys.version always works


class TestWriteMeta:
    def test_writes_valid_json(self, tmp_path: Path) -> None:
        from nuviz.types import EnvironmentSnapshot

        snapshot = EnvironmentSnapshot(
            git_hash="abc123",
            python_version="3.10.0",
            hostname="test-host",
        )
        meta_path = tmp_path / "meta.json"
        write_meta(snapshot, meta_path)

        data = json.loads(meta_path.read_text())
        assert data["git_hash"] == "abc123"
        assert data["hostname"] == "test-host"

    def test_write_to_invalid_path_no_crash(self, tmp_path: Path) -> None:
        from nuviz.types import EnvironmentSnapshot

        snapshot = EnvironmentSnapshot()
        write_meta(snapshot, tmp_path / "nonexistent" / "meta.json")
        # Should not raise
