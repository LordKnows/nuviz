"""Core Logger class — the main public API for NuViz."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from nuviz.anomaly import AnomalyDetector
from nuviz.config import NuvizConfig
from nuviz.image import save_image
from nuviz.naming import resolve_experiment_dir
from nuviz.snapshot import capture_snapshot, write_meta
from nuviz.types import AlertEvent, ExperimentMeta, MetricRecord
from nuviz.writer import JsonlWriter


class Logger:
    """NuViz training logger.

    Records metrics, images, and environment info to structured files
    for visualization with the nuviz CLI.

    Usage::

        log = Logger("exp-001", project="my_project")
        for step in range(1000):
            log.step(step, loss=loss, psnr=psnr)
        log.finish()

    Or as a context manager::

        with Logger("exp-001") as log:
            for step in range(1000):
                log.step(step, loss=loss)
    """

    def __init__(
        self,
        name: str | None = None,
        project: str | None = None,
        snapshot: bool = True,
        alerts: bool = True,
        config: NuvizConfig | None = None,
    ) -> None:
        self._config = config or NuvizConfig.from_env()
        self._name = name
        self._project = project

        # Resolve experiment directory
        self._experiment_dir = resolve_experiment_dir(
            name=name,
            project=project,
            base_dir=self._config.base_dir,
        )

        # Initialize writer
        self._writer = JsonlWriter(
            path=self._experiment_dir / "metrics.jsonl",
            flush_interval=self._config.flush_interval_seconds,
            flush_count=self._config.flush_count,
        )

        # Initialize anomaly detector
        self._detector = AnomalyDetector() if alerts and self._config.enable_alerts else None

        # Track state
        self._current_step: int = 0
        self._best_metrics: dict[str, float] = {}
        self._finished = False
        self._alerts: list[AlertEvent] = []

        # Capture environment snapshot
        if snapshot and self._config.enable_snapshot:
            try:
                env_snapshot = capture_snapshot()
                write_meta(env_snapshot, self._experiment_dir / "meta.json")
            except Exception as e:
                print(f"[nuviz] Warning: failed to capture snapshot: {e}", file=sys.stderr)

    @property
    def experiment_dir(self) -> Path:
        """Path to the experiment directory."""
        return self._experiment_dir

    def step(self, step: int, **metrics: float) -> None:
        """Record metrics for a training step.

        Args:
            step: The current training step number.
            **metrics: Named metric values (e.g., loss=0.05, psnr=28.4).
        """
        if self._finished:
            return

        try:
            # Anomaly detection
            if self._detector:
                alerts = self._detector.check(step, metrics)
                for alert in alerts:
                    self._alerts.append(alert)
                    print(f"[nuviz] ⚠ {alert.message}", file=sys.stderr)

            # Track best metrics
            for name, value in metrics.items():
                if isinstance(value, (int, float)) and not (
                    isinstance(value, float) and (value != value or abs(value) == float("inf"))
                ):
                    if name not in self._best_metrics or self._should_update_best(
                        name, value, self._best_metrics[name]
                    ):
                        self._best_metrics[name] = value

            # Write record
            record = MetricRecord(
                step=step,
                timestamp=time.time(),
                metrics=dict(metrics),
            )
            self._writer.write(record)
            self._current_step = step

        except Exception as e:
            print(f"[nuviz] Warning: failed to record step {step}: {e}", file=sys.stderr)

    def image(self, name: str, data: Any, cmap: str | None = None) -> None:
        """Save an image for the current step.

        Args:
            name: Image identifier (e.g., "render", "gt", "depth").
            data: Image data as numpy array or torch tensor.
            cmap: Optional colormap for single-channel data.
        """
        if self._finished:
            return

        save_image(
            name=name,
            data=data,
            step=self._current_step,
            experiment_dir=self._experiment_dir,
            cmap=cmap,
        )

    def finish(self) -> None:
        """Finalize the experiment. Flushes all data and writes final metadata."""
        if self._finished:
            return
        self._finished = True

        # Flush writer
        self._writer.close()

        # Update meta.json with final info
        self._write_final_meta()

    def _write_final_meta(self) -> None:
        """Write or update meta.json with final experiment info."""
        try:
            meta_path = self._experiment_dir / "meta.json"
            existing: dict[str, Any] = {}
            if meta_path.exists():
                with open(meta_path) as f:
                    existing = json.load(f)

            from datetime import datetime, timezone

            existing["end_time"] = datetime.now(timezone.utc).isoformat()
            existing["total_steps"] = self._current_step
            existing["best_metrics"] = self._best_metrics
            existing["status"] = "done"
            existing["name"] = self._name or self._experiment_dir.name
            existing["project"] = self._project

            with open(meta_path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            print(f"[nuviz] Warning: failed to write final meta: {e}", file=sys.stderr)

    @staticmethod
    def _should_update_best(name: str, new_value: float, old_value: float) -> bool:
        """Determine if a new metric value is 'better' than the old one.

        Heuristic: metrics containing 'loss', 'lpips', or 'error' are
        minimized; all others are maximized.
        """
        lower_name = name.lower()
        minimize = any(k in lower_name for k in ("loss", "lpips", "error", "mse", "mae"))
        if minimize:
            return new_value < old_value
        return new_value > old_value

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, *args: Any) -> None:
        self.finish()
