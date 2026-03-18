"""Experiment naming and directory resolution."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path


def _sanitize_name(name: str) -> str:
    """Replace unsafe characters with underscores, collapse runs."""
    sanitized = re.sub(r"[^\w\-.]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized)
    return sanitized.strip("_")


def _generate_timestamp_name() -> str:
    """Generate a name from the current UTC timestamp."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def resolve_experiment_dir(
    name: str | None,
    project: str | None,
    base_dir: Path,
) -> Path:
    """Resolve and create the experiment directory.

    Args:
        name: User-provided experiment name, or None for timestamp fallback.
        project: Optional project grouping.
        base_dir: Root directory for all experiments.

    Returns:
        Path to the created experiment directory.
    """
    resolved_name = _sanitize_name(name) if name else _generate_timestamp_name()

    if project:
        experiment_dir = base_dir / _sanitize_name(project) / resolved_name
    else:
        experiment_dir = base_dir / resolved_name

    # Handle collision: append _1, _2, etc.
    if experiment_dir.exists():
        suffix = 1
        while True:
            candidate = experiment_dir.parent / f"{resolved_name}_{suffix}"
            if not candidate.exists():
                experiment_dir = candidate
                break
            suffix += 1

    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir
