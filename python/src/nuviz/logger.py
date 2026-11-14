"""Core Logger class — the main public API for NuViz."""

from __future__ import annotations

import atexit
import contextlib
import dataclasses
import json
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from socket import gethostname
from typing import Any

from nuviz._util import atomic_write_json, is_finite, to_number, warn
from nuviz.anomaly import AnomalyDetector
from nuviz.config import NuvizConfig
from nuviz.events import EventWriter
from nuviz.goals import Goal, default_goal, validate_goal
from nuviz.gpu import GpuCollector
from nuviz.image import save_image
from nuviz.meta import SCHEMA_VERSION, MetaStore
from nuviz.naming import _sanitize_name, resolve_experiment_dir
from nuviz.pointcloud import save_pointcloud
from nuviz.scene_writer import SceneWriter
from nuviz.snapshot import capture_snapshot
from nuviz.types import AlertEvent, MetricRecord, SceneRecord
from nuviz.writer import JsonlWriter


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_hparams(hparams: Any) -> dict[str, Any] | None:
    """Turn a dict / argparse.Namespace / dataclass into a JSON-safe dict."""
    if hparams is None:
        return None
    if dataclasses.is_dataclass(hparams) and not isinstance(hparams, type):
        raw: Any = dataclasses.asdict(hparams)
    elif isinstance(hparams, dict):
        raw = hparams
    elif hasattr(hparams, "__dict__"):
        raw = vars(hparams)
    else:
        raise TypeError(
            "hparams must be a dict, argparse.Namespace or dataclass, "
            f"got {type(hparams).__name__}"
        )
    # Round-trip through JSON so nested tuples, paths, numpy scalars etc. are
    # stored in a form the CLI can read.
    safe: dict[str, Any] = json.loads(json.dumps(raw, default=_hparam_default))
    return safe


def _hparam_default(value: Any) -> Any:
    try:
        return to_number(value)
    except TypeError:
        return str(value)


class _MetricSummary:
    """Running min / max / last for one metric."""

    __slots__ = ("count", "last", "last_step", "max", "max_step", "min", "min_step")

    def __init__(self) -> None:
        self.count = 0
        self.min = float("inf")
        self.min_step = 0
        self.max = float("-inf")
        self.max_step = 0
        self.last = 0.0
        self.last_step = 0

    def update(self, step: int, value: float) -> None:
        self.count += 1
        if value < self.min:
            self.min, self.min_step = value, step
        if value > self.max:
            self.max, self.max_step = value, step
        self.last, self.last_step = value, step

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "min": self.min,
            "min_step": self.min_step,
            "max": self.max,
            "max_step": self.max_step,
            "last": self.last,
            "last_step": self.last_step,
        }


class Logger:
    """NuViz training logger.

    Records metrics, images, and environment info to structured files
    for visualization with the nuviz CLI.

    Usage::

        log = Logger("exp-001", project="my_project", hparams={"lr": 1e-3})
        for step in range(1000):
            log.step(step, loss=loss, psnr=psnr)
        log.finish()

    Or as a context manager (marks the run ``failed`` if the block raises)::

        with Logger("exp-001") as log:
            for step in range(1000):
                log.step(step, loss=loss)

    Args:
        name: Experiment name. Defaults to a UTC timestamp. If a run with this
            name already exists a ``_1``, ``_2``, ... suffix is appended.
        project: Optional project grouping (a parent directory).
        snapshot: Capture git/package/GPU information in the background.
        alerts: Enable NaN/Inf and spike detection.
        config: Logger configuration (defaults to environment variables).
        seed: Random seed, recorded for multi-seed aggregation.
        config_hash: Groups runs that differ only by seed. Derived from
            ``hparams`` when omitted.
        hparams: Hyperparameters (dict, ``argparse.Namespace`` or dataclass),
            stored in ``meta.json`` as ``config`` for ``nuviz matrix`` and
            ``nuviz reproduce``.
        max_steps: Planned number of steps, used by ``nuviz watch`` for ETA.
        tags: Initial tags.
    """

    def __init__(
        self,
        name: str | None = None,
        project: str | None = None,
        snapshot: bool = True,
        alerts: bool = True,
        config: NuvizConfig | None = None,
        seed: int | None = None,
        config_hash: str | None = None,
        hparams: Any = None,
        max_steps: int | None = None,
        tags: list[str] | None = None,
    ) -> None:
        self._config = config or NuvizConfig.from_env()
        self._project = project
        self._seed = seed
        self._hparams = _normalize_hparams(hparams)
        self._config_hash = config_hash or self._derive_config_hash(self._hparams)

        # Resolve (and atomically claim) the experiment directory
        self._experiment_dir = resolve_experiment_dir(
            name=name,
            project=project,
            base_dir=self._config.base_dir,
        )
        # The directory name is unique within its project, so it is the name
        # the CLI can always resolve.
        self._name = self._experiment_dir.name
        if name and self._name != _sanitize_name(name):
            print(
                f"[nuviz] experiment '{_sanitize_name(name)}' already exists; "
                f"logging to '{self._name}'",
                file=sys.stderr,
            )
        self._run_id = uuid.uuid4().hex[:12]

        # Track state
        self._current_step: int = 0
        self._summaries: dict[str, _MetricSummary] = {}
        self._goals: dict[str, Goal] = {}
        self._finished = False
        self._finish_lock = threading.Lock()
        self._alerts: list[AlertEvent] = []
        self._bad_metrics: set[str] = set()
        self._warned_negative_step = False
        self._last_heartbeat = 0.0

        # meta.json exists from the first moment, so the CLI can tell a running
        # experiment from a finished or crashed one.
        self._meta = MetaStore(self._experiment_dir / "meta.json")
        initial: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "name": self._name,
            "project": project,
            "status": "running",
            "start_time": _now_iso(),
            "hostname": gethostname(),
            "pid": os.getpid(),
            "python_version": sys.version,
        }
        if seed is not None:
            initial["seed"] = seed
        if self._config_hash is not None:
            initial["config_hash"] = self._config_hash
        if self._hparams is not None:
            initial["config"] = self._hparams
        if max_steps is not None:
            initial["max_steps"] = int(max_steps)
        if tags:
            initial["tags"] = [str(t) for t in tags]
        self._meta.update(initial)

        self._events = EventWriter(self._experiment_dir / "events.jsonl")

        # Initialize writer; its thread also drives the heartbeat
        self._writer = JsonlWriter(
            path=self._experiment_dir / "metrics.jsonl",
            flush_interval=self._config.flush_interval_seconds,
            flush_count=self._config.flush_count,
            max_file_size_bytes=self._config.rotate_max_size_bytes,
            max_file_lines=self._config.rotate_max_lines,
            max_queue=self._config.max_queue,
            close_timeout=self._config.close_timeout,
            on_tick=self._tick,
        )
        self._write_heartbeat()

        # Scene writer (lazy-init on first scene() call)
        self._scene_writer: SceneWriter | None = None

        # Initialize GPU collector
        self._gpu_collector: GpuCollector | None = None
        if self._config.enable_gpu:
            self._gpu_collector = GpuCollector(
                poll_interval=self._config.gpu_poll_interval,
            )

        # Initialize anomaly detector
        self._detector = AnomalyDetector() if alerts and self._config.enable_alerts else None

        # Environment snapshot runs in the background: git, nvcc and
        # nvidia-smi can each take seconds and must not delay training.
        self._snapshot_thread: threading.Thread | None = None
        if snapshot and self._config.enable_snapshot:
            self._snapshot_thread = threading.Thread(
                target=self._capture_snapshot, daemon=True, name="nuviz-snapshot"
            )
            self._snapshot_thread.start()

        atexit.register(self._atexit)

    # ----------------------------------------------------------- properties

    @property
    def experiment_dir(self) -> Path:
        """Path to the experiment directory."""
        return self._experiment_dir

    @property
    def name(self) -> str:
        """Resolved experiment name (unique within its project)."""
        return self._name

    @property
    def run_id(self) -> str:
        """Unique identifier of this run."""
        return self._run_id

    @property
    def alerts(self) -> tuple[AlertEvent, ...]:
        """Anomaly alerts raised so far."""
        return tuple(self._alerts)

    # -------------------------------------------------------------- logging

    def define_metric(self, name: str, goal: str) -> None:
        """Declare whether lower (``"min"``) or higher (``"max"``) is better.

        Without a declaration the CLI guesses from the name (``loss``-like
        metrics are minimized, everything else maximized).
        """
        self._goals[name] = validate_goal(goal)
        self._meta.update({"metric_goals": dict(self._goals)})

    def step(self, step: int, **metrics: Any) -> None:
        """Record metrics for a training step.

        Args:
            step: The current training step number (non-negative).
            **metrics: Named scalar values (e.g., loss=0.05, psnr=28.4).
                Python numbers, numpy scalars and 0-dim tensors are accepted.
        """
        if self._finished:
            return

        try:
            step = int(step)
            if step < 0:
                if not self._warned_negative_step:
                    self._warned_negative_step = True
                    warn(f"ignoring negative step {step}; steps must be >= 0")
                return

            values = self._coerce_metrics(metrics)

            # Anomaly detection
            if self._detector:
                for alert in self._detector.check(step, values):
                    self._alerts.append(alert)
                    print(f"[nuviz] ⚠ {alert.message}", file=sys.stderr)
                    self._events.emit(
                        "alert",
                        step=step,
                        alert_type=alert.alert_type.value,
                        metric=alert.metric_name,
                        value=alert.value,
                        message=alert.message,
                    )

            # Track min / max / last per metric
            for name, value in values.items():
                if is_finite(value):
                    summary = self._summaries.get(name)
                    if summary is None:
                        summary = self._summaries[name] = _MetricSummary()
                    summary.update(step, value)

            # Collect GPU metrics
            gpu = self._gpu_collector.latest() if self._gpu_collector else None

            # Write record
            record = MetricRecord(
                step=step,
                timestamp=time.time(),
                metrics=values,
                gpu=gpu,
            )
            self._writer.write(record)
            self._current_step = step

        except Exception as e:
            warn(f"failed to record step {step}: {e}")

    def _coerce_metrics(self, metrics: dict[str, Any]) -> dict[str, float]:
        """Convert metric values to plain numbers on the caller's thread.

        Accepts Python numbers, numpy scalars and 0-dim tensors. Values that
        are not scalar numbers are skipped with a one-time warning per metric.
        """
        values: dict[str, float] = {}
        for name, value in metrics.items():
            try:
                values[name] = to_number(value)
            except TypeError as e:
                if name not in self._bad_metrics:
                    self._bad_metrics.add(name)
                    warn(f"metric '{name}' ignored: {e}")
        return values

    def image(
        self, name: str, data: Any, cmap: str | None = None, step: int | None = None
    ) -> None:
        """Save an image.

        Args:
            name: Image identifier (e.g., "render", "gt", "depth").
            data: Image data as numpy array or torch tensor.
            cmap: Optional colormap for single-channel data.
            step: Step to file the image under (defaults to the latest step).
        """
        if self._finished:
            return

        at = self._current_step if step is None else int(step)
        path = save_image(
            name=name,
            data=data,
            step=at,
            experiment_dir=self._experiment_dir,
            cmap=cmap,
        )
        if path is not None:
            self._events.emit("image", step=at, tag=name, path=self._relative(path))

    def pointcloud(
        self,
        tag: str,
        xyz: Any,
        colors: Any | None = None,
        opacities: Any | None = None,
        step: int | None = None,
    ) -> None:
        """Save a point cloud.

        Args:
            tag: Identifier (e.g., "gaussians", "points").
            xyz: Positions array of shape (N, 3).
            colors: Optional RGB colors, shape (N, 3). Float [0,1] or uint8.
            opacities: Optional opacity values, shape (N,). Float.
            step: Step to file the cloud under (defaults to the latest step).
        """
        if self._finished:
            return

        at = self._current_step if step is None else int(step)
        path = save_pointcloud(
            tag=tag,
            step=at,
            xyz=xyz,
            experiment_dir=self._experiment_dir,
            colors=colors,
            opacities=opacities,
        )
        if path is not None:
            self._events.emit("pointcloud", step=at, tag=tag, path=self._relative(path))

    def scene(self, scene_name: str, **metrics: Any) -> None:
        """Record per-scene evaluation metrics.

        Writes to scenes.jsonl for per-scene breakdowns (e.g., NeRF per-scene PSNR).

        Args:
            scene_name: Scene identifier (e.g., "bicycle", "garden").
            **metrics: Named metric values (e.g., psnr=28.4, ssim=0.92).
        """
        if self._finished:
            return

        try:
            if self._scene_writer is None:
                self._scene_writer = SceneWriter(self._experiment_dir / "scenes.jsonl")

            record = SceneRecord(
                scene=scene_name,
                metrics=self._coerce_metrics(metrics),
                timestamp=time.time(),
            )
            self._scene_writer.write(record)
        except Exception as e:
            warn(f"failed to record scene {scene_name}: {e}")

    def flush(self, timeout: float | None = 10.0) -> bool:
        """Block until all metrics logged so far are on disk."""
        return self._writer.flush(timeout=timeout)

    # ------------------------------------------------------------ lifecycle

    def finish(self, status: str = "done", error: str | None = None) -> None:
        """Finalize the experiment. Flushes all data and writes final metadata.

        Args:
            status: Final status, ``"done"`` or ``"failed"``.
            error: Optional short description stored when the run failed.
        """
        with self._finish_lock:
            if self._finished:
                return
            self._finished = True

        with contextlib.suppress(Exception):
            atexit.unregister(self._atexit)

        # Stop GPU collector
        if self._gpu_collector is not None:
            self._gpu_collector.stop()

        # Flush writer
        self._writer.close()

        # Close scene writer
        if self._scene_writer is not None:
            self._scene_writer.close()

        # Let the snapshot land so the final meta.json is complete
        if self._snapshot_thread is not None:
            self._snapshot_thread.join(timeout=10.0)

        self._write_final_meta(status, error)
        self._write_heartbeat()

    def _write_final_meta(self, status: str, error: str | None) -> None:
        """Merge final experiment info into meta.json. Never raises."""
        try:
            best: dict[str, float] = {}
            best_steps: dict[str, int] = {}
            final: dict[str, float] = {}
            for name, s in self._summaries.items():
                goal = self._goals.get(name) or default_goal(name)
                if goal == "min":
                    best[name], best_steps[name] = s.min, s.min_step
                else:
                    best[name], best_steps[name] = s.max, s.max_step
                final[name] = s.last

            fields: dict[str, Any] = {
                "end_time": _now_iso(),
                "total_steps": self._current_step,
                "best_metrics": best,
                "best_steps": best_steps,
                "final_metrics": final,
                "metric_summary": {n: s.as_dict() for n, s in self._summaries.items()},
                "status": status,
                "name": self._name,
                "project": self._project,
                "num_alerts": len(self._alerts),
            }
            if self._writer.dropped:
                fields["dropped_records"] = self._writer.dropped
            if error:
                fields["error"] = error
            if self._seed is not None:
                fields["seed"] = self._seed
            if self._config_hash is not None:
                fields["config_hash"] = self._config_hash
            self._meta.update(fields)
        except Exception as e:
            warn(f"failed to write final meta: {e}")

    # ------------------------------------------------------------ internals

    @staticmethod
    def _derive_config_hash(hparams: dict[str, Any] | None) -> str | None:
        """Group key for multi-seed runs: hash of hparams without the seed."""
        if hparams is None:
            return None
        explicit = hparams.get("_config_hash")
        if isinstance(explicit, str) and explicit:
            return explicit
        from nuviz.ablation import _config_hash

        return _config_hash(
            {k: v for k, v in hparams.items() if k != "seed" and not k.startswith("_")}
        )

    def _relative(self, path: Path) -> str:
        try:
            return path.relative_to(self._experiment_dir).as_posix()
        except ValueError:
            return str(path)

    def _capture_snapshot(self) -> None:
        try:
            snapshot = dataclasses.asdict(capture_snapshot())
            # Identity and timing fields are owned by the logger itself.
            for key in ("start_time", "hostname", "python_version"):
                snapshot.pop(key, None)
            self._meta.update(snapshot)
        except Exception as e:
            warn(f"failed to capture snapshot: {e}")

    def _tick(self) -> None:
        """Called from the writer thread after every flush cycle."""
        if time.monotonic() - self._last_heartbeat >= self._config.heartbeat_interval:
            self._write_heartbeat()

    def _write_heartbeat(self) -> None:
        """Atomically refresh heartbeat.json so the CLI can detect dead runs."""
        self._last_heartbeat = time.monotonic()
        # Best effort: a missing heartbeat only affects status display.
        with contextlib.suppress(OSError):
            atomic_write_json(
                self._experiment_dir / "heartbeat.json",
                {
                    "timestamp": time.time(),
                    "step": self._current_step,
                    # The writer thread wakes at least every flush interval, so
                    # that bounds how late a heartbeat can be.
                    "interval": max(
                        self._config.heartbeat_interval, self._config.flush_interval_seconds
                    ),
                    "pid": os.getpid(),
                },
                indent=None,
            )

    def _atexit(self) -> None:
        """Finalize a run whose script ended without calling finish()."""
        crashed = getattr(sys, "last_exc", None) or getattr(sys, "last_value", None)
        if crashed is not None and not isinstance(crashed, SystemExit):
            self.finish(status="failed", error=f"{type(crashed).__name__}: {crashed}"[:500])
        else:
            self.finish()

    @staticmethod
    def _should_update_best(name: str, new_value: float, old_value: float) -> bool:
        """Whether ``new_value`` beats ``old_value`` under the name heuristic."""
        if default_goal(name) == "min":
            return new_value < old_value
        return new_value > old_value

    def __enter__(self) -> Logger:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is None:
            self.finish()
        else:
            self.finish(status="failed", error=f"{exc_type.__name__}: {exc}"[:500])
