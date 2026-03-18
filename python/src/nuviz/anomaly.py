"""Anomaly detection for training metrics."""

from __future__ import annotations

import math

from nuviz.types import AlertEvent, AlertType


class _WelfordState:
    """Online mean/variance via Welford's algorithm."""

    __slots__ = ("count", "mean", "m2")

    def __init__(self) -> None:
        self.count: int = 0
        self.mean: float = 0.0
        self.m2: float = 0.0

    def update(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> float:
        if self.count < 2:
            return 0.0
        return self.m2 / (self.count - 1)

    @property
    def stddev(self) -> float:
        return math.sqrt(self.variance)


class AnomalyDetector:
    """Detects NaN/Inf values and loss spikes in training metrics.

    Spike detection uses Welford's online algorithm with a 3-sigma threshold.
    A minimum of `min_steps` observations is required before spike detection activates.
    """

    def __init__(self, sigma_threshold: float = 3.0, min_steps: int = 10) -> None:
        self._sigma_threshold = sigma_threshold
        self._min_steps = min_steps
        self._states: dict[str, _WelfordState] = {}

    def check(self, step: int, metrics: dict[str, float]) -> list[AlertEvent]:
        """Check metrics for anomalies. Returns list of alerts (may be empty)."""
        alerts: list[AlertEvent] = []

        for name, value in metrics.items():
            # NaN check
            if isinstance(value, float) and math.isnan(value):
                alerts.append(AlertEvent(
                    step=step,
                    alert_type=AlertType.NAN,
                    metric_name=name,
                    message=f"{name} is NaN at step {step}",
                    value=value,
                ))
                continue

            # Inf check
            if isinstance(value, float) and math.isinf(value):
                alerts.append(AlertEvent(
                    step=step,
                    alert_type=AlertType.INF,
                    metric_name=name,
                    message=f"{name} is Inf at step {step}",
                    value=value,
                ))
                continue

            # Spike detection (only for numeric values)
            if not isinstance(value, (int, float)):
                continue

            if name not in self._states:
                self._states[name] = _WelfordState()

            state = self._states[name]

            # Check for spike before updating state
            if state.count >= self._min_steps and state.stddev > 0:
                deviation = abs(value - state.mean)
                if deviation > self._sigma_threshold * state.stddev:
                    alerts.append(AlertEvent(
                        step=step,
                        alert_type=AlertType.SPIKE,
                        metric_name=name,
                        message=(
                            f"{name} spike at step {step}: "
                            f"{value:.6g} (mean={state.mean:.6g}, "
                            f"std={state.stddev:.6g})"
                        ),
                        value=value,
                    ))

            state.update(value)

        return alerts
