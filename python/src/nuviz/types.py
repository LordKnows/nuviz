"""Core data types for NuViz."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


@dataclass(slots=True, frozen=True)
class MetricRecord:
    """A single metrics observation at a training step."""

    step: int
    timestamp: float
    metrics: dict[str, float]
    gpu: dict[str, int | float] | None = None


class AlertType(Enum):
    """Types of anomaly alerts."""

    NAN = "nan"
    INF = "inf"
    SPIKE = "spike"


@dataclass(slots=True, frozen=True)
class AlertEvent:
    """An anomaly event detected during training."""

    step: int
    alert_type: AlertType
    metric_name: str
    message: str
    value: float | None = None


@dataclass(slots=True, frozen=True)
class EnvironmentSnapshot:
    """Captured environment at experiment start."""

    git_hash: str | None = None
    git_dirty: bool | None = None
    pip_packages: list[str] = field(default_factory=list)
    cuda_version: str | None = None
    gpu_model: str | None = None
    python_version: str | None = None
    pytorch_version: str | None = None
    hostname: str | None = None
    start_time: str | None = None


@dataclass(slots=True, frozen=True)
class SceneRecord:
    """A per-scene metrics observation (e.g., per-scene PSNR for NeRF evaluation)."""

    scene: str
    metrics: dict[str, float]
    timestamp: float


@dataclass(slots=True, frozen=True)
class PointcloudRecord:
    """A saved point cloud file reference."""

    tag: str
    step: int
    path: str
    num_points: int
    timestamp: float


@dataclass(slots=True, frozen=True)
class ExperimentMeta:
    """Metadata for an experiment, written to meta.json."""

    name: str
    project: str | None
    snapshot: EnvironmentSnapshot | None
    end_time: str | None = None
    total_steps: int | None = None
    best_metrics: dict[str, float] = field(default_factory=dict)
    status: str = "running"
    seed: int | None = None
    config_hash: str | None = None
