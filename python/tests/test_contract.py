"""Tests for the on-disk contract between the Python logger and the CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from nuviz import Logger
from nuviz.config import NuvizConfig
from nuviz.goals import default_goal, validate_goal
from nuviz.meta import SCHEMA_VERSION, MetaStore

FIXTURES = Path(__file__).resolve().parents[2] / "spec" / "fixtures"


def _config(tmp_path: Path, **overrides: Any) -> NuvizConfig:
    defaults: dict[str, Any] = {
        "base_dir": tmp_path,
        "flush_interval_seconds": 0.05,
        "flush_count": 5,
        "enable_snapshot": False,
        "enable_gpu": False,
    }
    defaults.update(overrides)
    return NuvizConfig(**defaults)


def _meta(log: Logger) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((log.experiment_dir / "meta.json").read_text())
    return data


def _events(log: Logger) -> list[dict[str, Any]]:
    path = log.experiment_dir / "events.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestMetaLifecycle:
    def test_meta_exists_while_running(self, tmp_path: Path) -> None:
        log = Logger("run", project="proj", config=_config(tmp_path), seed=3, max_steps=100)
        try:
            meta = _meta(log)
            assert meta["status"] == "running"
            assert meta["schema_version"] == SCHEMA_VERSION
            assert meta["name"] == "run"
            assert meta["project"] == "proj"
            assert meta["seed"] == 3
            assert meta["max_steps"] == 100
            assert len(meta["run_id"]) == 12
            assert meta["start_time"]
        finally:
            log.finish()
        assert _meta(log)["status"] == "done"
        assert _meta(log)["start_time"] == meta["start_time"]

    def test_name_is_unique_directory_name(self, tmp_path: Path) -> None:
        a = Logger("same", config=_config(tmp_path))
        b = Logger("same", config=_config(tmp_path))
        a.finish()
        b.finish()
        assert (a.name, b.name) == ("same", "same_1")
        assert _meta(a)["name"] == "same"
        assert _meta(b)["name"] == "same_1"
        assert _meta(a)["run_id"] != _meta(b)["run_id"]

    def test_concurrent_starts_never_share_a_directory(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        logs: list[Logger] = []
        lock = threading.Lock()

        def start() -> None:
            log = Logger("race", config=config)
            with lock:
                logs.append(log)

        threads = [threading.Thread(target=start) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for log in logs:
            log.finish()
        assert len({log.experiment_dir for log in logs}) == 8

    def test_exception_in_context_marks_failed(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError), Logger("boom", config=_config(tmp_path)) as log:
            log.step(0, loss=1.0)
            raise RuntimeError("diverged")
        meta = _meta(log)
        assert meta["status"] == "failed"
        assert meta["error"] == "RuntimeError: diverged"
        assert meta["total_steps"] == 0

    def test_unfinished_script_is_finalized_at_exit(self, tmp_path: Path) -> None:
        script = f"""
import pathlib, sys
from nuviz import Logger
from nuviz.config import NuvizConfig
cfg = NuvizConfig(base_dir=pathlib.Path({str(tmp_path)!r}), enable_gpu=False, enable_snapshot=False)
log = Logger(sys.argv[1], config=cfg)
for s in range(7):
    log.step(s, loss=1.0 / (s + 1))
if sys.argv[1] == "crash":
    raise ValueError("bad batch")
"""
        for name in ("clean", "crash"):
            subprocess.run([sys.executable, "-c", script, name], capture_output=True, check=False)
        clean = json.loads((tmp_path / "clean" / "meta.json").read_text())
        crash = json.loads((tmp_path / "crash" / "meta.json").read_text())
        assert clean["status"] == "done"
        assert crash["status"] == "failed"
        assert crash["error"] == "ValueError: bad batch"
        # Metrics are flushed either way
        assert len((tmp_path / "crash" / "metrics.jsonl").read_text().splitlines()) == 7

    def test_tags_written_by_cli_during_run_survive_finish(self, tmp_path: Path) -> None:
        log = Logger("tagged", config=_config(tmp_path), tags=["baseline"])
        log.step(0, loss=1.0)
        # The CLI (`nuviz tag`) patches meta.json while the run is active.
        MetaStore(log.experiment_dir / "meta.json").update({"tags": ["baseline", "best"]})
        log.finish()
        assert _meta(log)["tags"] == ["baseline", "best"]

    def test_snapshot_runs_in_background_and_lands_in_meta(self, tmp_path: Path) -> None:
        started = time.perf_counter()
        log = Logger("snap", config=_config(tmp_path, enable_snapshot=True))
        assert time.perf_counter() - started < 1.0  # does not wait for git / nvidia-smi
        log.finish()
        meta = _meta(log)
        assert "git_hash" in meta
        assert isinstance(meta["pip_packages"], list)
        assert meta["argv"]
        assert meta["status"] == "done"

    def test_final_summary_fields(self, tmp_path: Path) -> None:
        with Logger("summary", config=_config(tmp_path)) as log:
            for step, (loss, psnr) in enumerate([(1.0, 20.0), (0.2, 31.0), (0.5, 25.0)]):
                log.step(step, loss=loss, psnr=psnr)
        meta = _meta(log)
        assert meta["best_metrics"] == {"loss": 0.2, "psnr": 31.0}
        assert meta["best_steps"] == {"loss": 1, "psnr": 1}
        assert meta["final_metrics"] == {"loss": 0.5, "psnr": 25.0}
        assert meta["metric_summary"]["loss"] == {
            "count": 3,
            "min": 0.2,
            "min_step": 1,
            "max": 1.0,
            "max_step": 0,
            "last": 0.5,
            "last_step": 2,
        }


class TestHeartbeat:
    def test_heartbeat_written_at_start_and_refreshed(self, tmp_path: Path) -> None:
        log = Logger("hb", config=_config(tmp_path, heartbeat_interval=0.05))
        path = log.experiment_dir / "heartbeat.json"
        first = json.loads(path.read_text())
        assert first["step"] == 0
        assert first["interval"] > 0

        log.step(41, loss=1.0)
        deadline = time.time() + 5.0
        while time.time() < deadline:
            current = json.loads(path.read_text())
            if current["step"] == 41 and current["timestamp"] > first["timestamp"]:
                break
            time.sleep(0.02)
        else:
            pytest.fail("heartbeat was not refreshed by the writer thread")
        log.finish()


class TestHparams:
    def test_dict_hparams_stored_as_config(self, tmp_path: Path) -> None:
        hp = {"lr": 1e-3, "model": {"sh": 2, "dims": (64, 64)}, "out": Path("/tmp/x")}
        with Logger("hp", config=_config(tmp_path), hparams=hp) as log:
            pass
        assert _meta(log)["config"] == {
            "lr": 1e-3,
            "model": {"sh": 2, "dims": [64, 64]},
            "out": "/tmp/x",
        }

    def test_namespace_and_dataclass_hparams(self, tmp_path: Path) -> None:
        @dataclass
        class Cfg:
            lr: float = 0.1
            layers: int = 4

        with Logger(
            "ns", config=_config(tmp_path), hparams=argparse.Namespace(lr=0.5, bs=8)
        ) as a:
            pass
        with Logger("dc", config=_config(tmp_path), hparams=Cfg()) as b:
            pass
        assert _meta(a)["config"] == {"lr": 0.5, "bs": 8}
        assert _meta(b)["config"] == {"lr": 0.1, "layers": 4}

    def test_config_hash_groups_seeds(self, tmp_path: Path) -> None:
        hashes = []
        for seed in (0, 1):
            with Logger(
                f"run_s{seed}",
                config=_config(tmp_path),
                seed=seed,
                hparams={"lr": 1e-3, "seed": seed},
            ) as log:
                pass
            hashes.append(_meta(log)["config_hash"])
        with Logger("other", config=_config(tmp_path), hparams={"lr": 1e-4, "seed": 0}) as log:
            pass
        assert hashes[0] == hashes[1]
        assert _meta(log)["config_hash"] != hashes[0]

    def test_ablation_hash_is_reused(self, tmp_path: Path) -> None:
        from nuviz import Ablation

        cfg = Ablation("ab", base_config={"lr": 1e-3}).vary("sh", [1, 2]).generate()[0]
        with Logger("ab", config=_config(tmp_path), hparams=cfg) as log:
            pass
        assert _meta(log)["config_hash"] == cfg["_config_hash"]

    def test_invalid_hparams_type(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError):
            Logger("bad", config=_config(tmp_path), hparams=[1, 2, 3])


class TestGoals:
    def test_define_metric_overrides_heuristic(self, tmp_path: Path) -> None:
        with Logger("goal", config=_config(tmp_path)) as log:
            log.define_metric("regret", "min")
            assert _meta(log)["metric_goals"] == {"regret": "min"}
            for step, v in enumerate([5.0, 1.0, 3.0]):
                log.step(step, regret=v, score=v)
        meta = _meta(log)
        assert meta["best_metrics"] == {"regret": 1.0, "score": 5.0}
        assert meta["metric_goals"] == {"regret": "min"}

    def test_invalid_goal(self) -> None:
        with pytest.raises(ValueError):
            validate_goal("sideways")

    def test_heuristic_matches_shared_fixture(self) -> None:
        cases: dict[str, str] = json.loads((FIXTURES / "metric_goals.json").read_text())
        wrong = {n: default_goal(n) for n, want in cases.items() if default_goal(n) != want}
        assert not wrong


class TestEvents:
    def test_alerts_are_recorded_as_events(self, tmp_path: Path) -> None:
        with Logger("alerts", config=_config(tmp_path)) as log:
            log.step(0, loss=0.5)
            log.step(1, loss=float("nan"))
            log.step(2, loss=float("inf"))
        events = [e for e in _events(log) if e["type"] == "alert"]
        assert [(e["step"], e["alert_type"], e["metric"]) for e in events] == [
            (1, "nan", "loss"),
            (2, "inf", "loss"),
        ]
        assert all(e["value"] is None for e in events)  # strict JSON: no NaN literals
        assert _meta(log)["num_alerts"] == 2
        assert len(log.alerts) == 2

    def test_image_and_pointcloud_events(self, tmp_path: Path) -> None:
        np = pytest.importorskip("numpy")
        pytest.importorskip("PIL")
        with Logger("media", config=_config(tmp_path)) as log:
            log.step(5, loss=0.1)
            log.image("render", np.zeros((4, 4, 3), dtype=np.uint8))
            log.image("gt", np.zeros((4, 4, 3), dtype=np.uint8), step=9)
            log.pointcloud("pts", np.zeros((3, 3), dtype=np.float32))
        events = _events(log)
        assert [(e["type"], e["step"], e["tag"], e["path"]) for e in events] == [
            ("image", 5, "render", "images/step_000005_render.png"),
            ("image", 9, "gt", "images/step_000009_gt.png"),
            ("pointcloud", 5, "pts", "pointclouds/pts_step_000005.ply"),
        ]
        for e in events:
            assert (log.experiment_dir / e["path"]).is_file()


class TestInputValidation:
    def test_negative_step_is_ignored(self, tmp_path: Path) -> None:
        with Logger("neg", config=_config(tmp_path)) as log:
            log.step(-1, loss=1.0)
            log.step(0, loss=0.5)
        lines = (log.experiment_dir / "metrics.jsonl").read_text().splitlines()
        assert [json.loads(line)["step"] for line in lines] == [0]

    def test_flush_makes_metrics_visible(self, tmp_path: Path) -> None:
        log = Logger("flush", config=_config(tmp_path, flush_interval_seconds=60, flush_count=999))
        log.step(0, loss=1.0)
        assert log.flush()
        assert len((log.experiment_dir / "metrics.jsonl").read_text().splitlines()) == 1
        log.finish()

    def test_import_and_scalar_logging_work_without_numpy(self, tmp_path: Path) -> None:
        script = f"""
import sys
sys.modules["numpy"] = None  # simulate numpy not being installed
import pathlib
from nuviz import Logger
from nuviz.config import NuvizConfig
cfg = NuvizConfig(base_dir=pathlib.Path({str(tmp_path)!r}), enable_gpu=False, enable_snapshot=False)
with Logger("nonumpy", config=cfg) as log:
    log.step(0, loss=1.0)
    log.image("render", [[0]])  # warns, must not raise
"""
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, result.stderr
        assert "numpy is required" in result.stderr
        assert (tmp_path / "nonumpy" / "metrics.jsonl").exists()


class TestSpikeAlertsThroughLogger:
    def test_declared_max_metric_raises_drop_alert_lr_does_not(self, tmp_path: Path) -> None:
        with Logger("spikes", config=_config(tmp_path)) as log:
            log.define_metric("psnr", "max")
            for step in range(40):
                log.step(step, psnr=30.0 + 0.01 * (step % 2), lr=1e-3 * 0.1 ** (step // 10))
            log.step(40, psnr=4.0, lr=1e-7)
        alerts = [(e["metric"], e["alert_type"]) for e in _events(log) if e["type"] == "alert"]
        assert alerts == [("psnr", "spike")]
        assert "drop" in log.alerts[0].message
