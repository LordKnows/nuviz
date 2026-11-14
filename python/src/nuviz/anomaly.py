"""Anomaly detection for training metrics."""

from __future__ import annotations

import math
from collections.abc import Callable

from nuviz.goals import Goal, default_goal
from nuviz.types import AlertEvent, AlertType


class _MovingStats:
    """Exponentially weighted mean/variance over roughly the last ``window`` values.

    A moving window matters: a loss that falls from 2.0 to 0.02 has a huge
    variance over its whole history, which would hide a later jump back to
    1.0. Until ``window`` values have been seen this is the plain running
    mean/variance.
    """

    __slots__ = ("alpha_min", "count", "mean", "var")

    def __init__(self, window: int) -> None:
        self.alpha_min = 2.0 / (window + 1)
        self.count = 0
        self.mean = 0.0
        self.var = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        alpha = max(1.0 / self.count, self.alpha_min)
        delta = value - self.mean
        self.mean += alpha * delta
        self.var = (1.0 - alpha) * (self.var + alpha * delta * delta)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.var)


def _loss_like_only(name: str) -> Goal | None:
    """Default policy: only watch metrics that are minimized."""
    return "min" if default_goal(name) == "min" else None


class AnomalyDetector:
    """Detects NaN/Inf values and sudden regressions in training metrics.

    NaN/Inf are reported for every metric.

    A *spike* is a value that is worse than the recent moving average by more
    than ``sigma_threshold`` standard deviations **and** by more than
    ``min_relative`` of the average. "Worse" depends on the metric's goal:
    upward for metrics that are minimized, downward for ones that are
    maximized. ``goal_for`` decides which metrics are watched; by default
    only loss-like metrics are, so schedules such as the learning rate do
    not raise alerts every time they step.

    After a spike, further spike alerts for that metric are suppressed for
    ``cooldown`` observations so one divergence does not flood the log.
    """

    def __init__(
        self,
        sigma_threshold: float = 3.0,
        min_steps: int = 10,
        window: int = 100,
        min_relative: float = 0.5,
        cooldown: int = 25,
        goal_for: Callable[[str], Goal | None] | None = None,
    ) -> None:
        self._sigma_threshold = sigma_threshold
        self._min_steps = min_steps
        self._window = window
        self._min_relative = min_relative
        self._cooldown = cooldown
        self._goal_for = goal_for or _loss_like_only
        self._states: dict[str, _MovingStats] = {}
        self._quiet_until: dict[str, int] = {}

    def check(self, step: int, metrics: dict[str, float]) -> list[AlertEvent]:
        """Check metrics for anomalies. Returns list of alerts (may be empty)."""
        alerts: list[AlertEvent] = []

        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue

            if isinstance(value, float) and math.isnan(value):
                alerts.append(
                    AlertEvent(
                        step=step,
                        alert_type=AlertType.NAN,
                        metric_name=name,
                        message=f"{name} is NaN at step {step}",
                        value=value,
                    )
                )
                continue

            if isinstance(value, float) and math.isinf(value):
                alerts.append(
                    AlertEvent(
                        step=step,
                        alert_type=AlertType.INF,
                        metric_name=name,
                        message=f"{name} is Inf at step {step}",
                        value=value,
                    )
                )
                continue

            goal = self._goal_for(name)
            if goal is None:
                continue

            state = self._states.get(name)
            if state is None:
                state = self._states[name] = _MovingStats(self._window)

            # Check against the history before folding the new value in.
            if (
                state.count >= self._min_steps
                and state.count >= self._quiet_until.get(name, 0)
                and self._is_spike(state, value, goal)
            ):
                self._quiet_until[name] = state.count + self._cooldown
                alerts.append(
                    AlertEvent(
                        step=step,
                        alert_type=AlertType.SPIKE,
                        metric_name=name,
                        message=(
                            f"{name} {'spike' if goal == 'min' else 'drop'} at step {step}: "
                            f"{value:.6g} (recent mean={state.mean:.6g}, "
                            f"std={state.stddev:.6g})"
                        ),
                        value=value,
                    )
                )

            state.update(value)

        return alerts

    def _is_spike(self, state: _MovingStats, value: float, goal: Goal) -> bool:
        std = state.stddev
        if std <= 0.0:
            return False
        worse_by = value - state.mean if goal == "min" else state.mean - value
        return (
            worse_by > self._sigma_threshold * std
            and worse_by > self._min_relative * max(abs(state.mean), 1e-12)
        )
