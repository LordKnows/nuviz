"""Environment snapshot capture."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from socket import gethostname

from nuviz.types import EnvironmentSnapshot


def _run_command(cmd: list[str], timeout: float = 5.0) -> str | None:
    """Run a command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def capture_snapshot() -> EnvironmentSnapshot:
    """Capture current environment. All fields are best-effort."""
    # Git info
    git_hash = _run_command(["git", "rev-parse", "HEAD"])
    git_dirty: bool | None = None
    if git_hash is not None:
        dirty_result = _run_command(["git", "diff", "--quiet"])
        git_dirty = dirty_result is None  # non-zero exit = dirty

    # Pip packages
    pip_output = _run_command(["pip", "freeze"])
    pip_packages = pip_output.splitlines() if pip_output else []

    # CUDA version
    cuda_version = _run_command(["nvcc", "--version"])
    if cuda_version:
        # Extract version number from verbose output
        for line in cuda_version.splitlines():
            if "release" in line.lower():
                cuda_version = line.strip()
                break

    # GPU model
    gpu_model = _run_command(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
    )

    # PyTorch version (dynamic import)
    pytorch_version: str | None = None
    try:
        import torch
        pytorch_version = torch.__version__
    except ImportError:
        pass

    return EnvironmentSnapshot(
        git_hash=git_hash,
        git_dirty=git_dirty,
        pip_packages=pip_packages,
        cuda_version=cuda_version,
        gpu_model=gpu_model,
        python_version=sys.version,
        pytorch_version=pytorch_version,
        hostname=gethostname(),
        start_time=datetime.now(timezone.utc).isoformat(),
    )


def write_meta(snapshot: EnvironmentSnapshot, path: Path) -> None:
    """Write snapshot to meta.json. Never raises."""
    try:
        from dataclasses import asdict
        data = asdict(snapshot)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass
