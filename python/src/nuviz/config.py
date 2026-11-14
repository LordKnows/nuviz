"""Configuration for NuViz."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from nuviz._util import warn


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        warn(f"ignoring invalid {name}={raw!r} (expected a number)")
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        warn(f"ignoring invalid {name}={raw!r} (expected an integer)")
        return default


def _env_flag(name: str, default: bool = True) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


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
    enable_gpu: bool = True
    gpu_poll_interval: float = 5.0
    heartbeat_interval: float = 10.0
    max_queue: int = 100_000
    close_timeout: float = 30.0

    @staticmethod
    def from_env() -> NuvizConfig:
        """Create config with environment variable overrides."""
        base_dir_str = os.environ.get("NUVIZ_DIR")
        base_dir = Path(base_dir_str) if base_dir_str else Path.home() / ".nuviz" / "experiments"

        return NuvizConfig(
            base_dir=base_dir,
            flush_interval_seconds=_env_float("NUVIZ_FLUSH_INTERVAL", 2.0),
            flush_count=_env_int("NUVIZ_FLUSH_COUNT", 50),
            enable_alerts=_env_flag("NUVIZ_ALERTS"),
            enable_snapshot=_env_flag("NUVIZ_SNAPSHOT"),
            rotate_max_size_bytes=_env_int("NUVIZ_ROTATE_SIZE", 50_000_000),
            rotate_max_lines=_env_int("NUVIZ_ROTATE_LINES", 500_000),
            enable_gpu=_env_flag("NUVIZ_GPU"),
            gpu_poll_interval=_env_float("NUVIZ_GPU_POLL", 5.0),
            heartbeat_interval=_env_float("NUVIZ_HEARTBEAT", 10.0),
            max_queue=_env_int("NUVIZ_QUEUE_SIZE", 100_000),
            close_timeout=_env_float("NUVIZ_CLOSE_TIMEOUT", 30.0),
        )
