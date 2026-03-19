"""Tests for anomaly detection."""

from __future__ import annotations

from nuviz.anomaly import AnomalyDetector
from nuviz.types import AlertType


class TestNanInfDetection:
    def test_nan_detected(self) -> None:
        detector = AnomalyDetector()
        alerts = detector.check(100, {"loss": float("nan")})
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.NAN
        assert alerts[0].metric_name == "loss"
        assert alerts[0].step == 100

    def test_inf_detected(self) -> None:
        detector = AnomalyDetector()
        alerts = detector.check(50, {"loss": float("inf")})
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.INF

    def test_negative_inf_detected(self) -> None:
        detector = AnomalyDetector()
        alerts = detector.check(50, {"loss": float("-inf")})
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.INF

    def test_normal_value_no_alert(self) -> None:
        detector = AnomalyDetector()
        alerts = detector.check(1, {"loss": 0.5})
        assert len(alerts) == 0

    def test_multiple_metrics_checked(self) -> None:
        detector = AnomalyDetector()
        alerts = detector.check(1, {"loss": float("nan"), "psnr": float("inf")})
        assert len(alerts) == 2
        types = {a.alert_type for a in alerts}
        assert AlertType.NAN in types
        assert AlertType.INF in types


class TestSpikeDetection:
    def test_spike_detected_after_warmup(self) -> None:
        detector = AnomalyDetector(sigma_threshold=3.0, min_steps=10)

        # Stable training for 20 steps
        for i in range(20):
            alerts = detector.check(i, {"loss": 0.1 + 0.001 * (i % 3)})
            assert len(alerts) == 0

        # Inject a massive spike
        alerts = detector.check(20, {"loss": 10.0})
        assert len(alerts) == 1
        assert alerts[0].alert_type == AlertType.SPIKE

    def test_no_spike_during_warmup(self) -> None:
        detector = AnomalyDetector(min_steps=10)

        # Even a big value shouldn't trigger during warmup
        alerts = detector.check(0, {"loss": 0.1})
        assert len(alerts) == 0
        alerts = detector.check(1, {"loss": 100.0})
        assert len(alerts) == 0

    def test_no_false_positive_on_normal_descent(self) -> None:
        detector = AnomalyDetector(sigma_threshold=3.0, min_steps=10)

        # Smooth exponential decay
        for i in range(100):
            loss = 1.0 * (0.99 ** i)
            alerts = detector.check(i, {"loss": loss})
            # The gradual decrease shouldn't be flagged
            spike_alerts = [a for a in alerts if a.alert_type == AlertType.SPIKE]
            # Allow spike alerts in early steps as the distribution forms
            if i > 20:
                assert len(spike_alerts) == 0, f"False positive at step {i}, loss={loss}"

    def test_independent_metric_tracking(self) -> None:
        detector = AnomalyDetector(sigma_threshold=3.0, min_steps=10)

        # Train with two metrics, add slight noise so stddev > 0
        for i in range(20):
            detector.check(i, {"loss": 0.1 + 0.001 * (i % 3), "psnr": 25.0 + 0.01 * (i % 3)})

        # Spike only in loss
        alerts = detector.check(20, {"loss": 10.0, "psnr": 25.01})
        spike_names = [a.metric_name for a in alerts if a.alert_type == AlertType.SPIKE]
        assert "loss" in spike_names
        assert "psnr" not in spike_names

    def test_zero_variance_no_crash(self) -> None:
        detector = AnomalyDetector(min_steps=5)

        # All identical values — stddev is 0
        for i in range(10):
            alerts = detector.check(i, {"loss": 0.5})
            # Should not crash, no spike alert (stddev=0 guard)
            assert all(a.alert_type != AlertType.SPIKE for a in alerts)
