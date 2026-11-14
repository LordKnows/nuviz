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
    resolved_name = (_sanitize_name(name) if name else "") or _generate_timestamp_name()

    if project:
        experiment_dir = base_dir / _sanitize_name(project) / resolved_name
    else:
        experiment_dir = base_dir / resolved_name

    experiment_dir.parent.mkdir(parents=True, exist_ok=True)

    # Claim the directory atomically. On collision append _1, _2, ... so two
    # runs started at the same moment with the same name can never share a
    # directory (and interleave their metrics).
    candidate = experiment_dir
    suffix = 0
    while True:
        try:
            candidate.mkdir(exist_ok=False)
            return candidate
        except FileExistsError:
            suffix += 1
            candidate = experiment_dir.parent / f"{resolved_name}_{suffix}"
