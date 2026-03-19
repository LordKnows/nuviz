"""Configuration for NuViz."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class NuvizConfig:
    """NuViz configuration with sensible defaults."""

    base_dir: Path = Path.home() / ".nuviz" / "experiments"
    flush_interval_seconds: float = 2.0
    flush_count: int = 50
    enable_alerts: bool = True
    enable_snapshot: bool = True
    rotate_max_size_bytes: int = 50_000_000  # 50 MB
    rotate_max_lines: int = 500_000

    @staticmethod
    def from_env() -> NuvizConfig:
        """Create config with environment variable overrides."""
        base_dir_str = os.environ.get("NUVIZ_DIR")
        base_dir = Path(base_dir_str) if base_dir_str else Path.home() / ".nuviz" / "experiments"

        return NuvizConfig(
            base_dir=base_dir,
            flush_interval_seconds=float(
                os.environ.get("NUVIZ_FLUSH_INTERVAL", "2.0")
            ),
            flush_count=int(os.environ.get("NUVIZ_FLUSH_COUNT", "50")),
            enable_alerts=os.environ.get("NUVIZ_ALERTS", "1") != "0",
            enable_snapshot=os.environ.get("NUVIZ_SNAPSHOT", "1") != "0",
            rotate_max_size_bytes=int(
                os.environ.get("NUVIZ_ROTATE_SIZE", "50000000")
            ),
            rotate_max_lines=int(
                os.environ.get("NUVIZ_ROTATE_LINES", "500000")
            ),
        )
