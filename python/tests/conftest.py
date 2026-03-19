"""Shared fixtures for NuViz tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nuviz.config import NuvizConfig


@pytest.fixture
def tmp_base_dir(tmp_path: Path) -> Path:
    """A temporary base directory for experiments."""
    base = tmp_path / "experiments"
    base.mkdir()
    return base


@pytest.fixture
def tmp_config(tmp_base_dir: Path) -> NuvizConfig:
    """A NuvizConfig pointing to a temp directory with fast flush."""
    return NuvizConfig(
        base_dir=tmp_base_dir,
        flush_interval_seconds=0.1,
        flush_count=5,
        enable_alerts=True,
        enable_snapshot=False,  # Disable snapshot in tests by default
    )
