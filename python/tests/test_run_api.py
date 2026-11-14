"""Tests for the wandb-style init/log/finish API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import nuviz
from nuviz.config import NuvizConfig


def _config(tmp_path: Path) -> NuvizConfig:
    return NuvizConfig(
        base_dir=tmp_path,
        flush_interval_seconds=0.05,
        flush_count=5,
        enable_snapshot=False,
        enable_gpu=False,
    )


def _records(run_dir: str) -> list[dict[str, Any]]:
    path = Path(run_dir) / "metrics.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


def _meta(run_dir: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((Path(run_dir) / "meta.json").read_text())
    return data


@pytest.fixture(autouse=True)
def _no_leftover_run() -> Any:
    yield
    nuviz.finish()


def test_module_level_flow_matches_wandb_usage(tmp_path: Path) -> None:
    run = nuviz.init(
        project="proj",
        name="exp",
        config={"lr": 1e-3},
        logger_config=_config(tmp_path),
        entity="ignored",  # wandb-only arguments are accepted
        mode="offline",
    )
    for i in range(3):
        nuviz.log({"loss": 1.0 / (i + 1), "acc": i / 3})
    run_dir = run.dir
    nuviz.finish()

    records = _records(run_dir)
    assert [r["step"] for r in records] == [0, 1, 2]  # auto-incremented
    assert records[2]["metrics"] == {"loss": 1.0 / 3, "acc": 2 / 3}
    meta = _meta(run_dir)
    assert meta["status"] == "done"
    assert meta["project"] == "proj"
    assert meta["config"] == {"lr": 1e-3}


def test_explicit_step_and_nested_dicts(tmp_path: Path) -> None:
    with nuviz.Run(name="nested", logger_config=_config(tmp_path)) as run:
        run.log({"train": {"loss": 0.5, "acc": 0.8}, "lr": 0.1}, step=10)
        run.log({"train": {"loss": 0.4}})  # continues from the explicit step
    records = _records(run.dir)
    assert [r["step"] for r in records] == [10, 11]
    assert records[0]["metrics"] == {"train/loss": 0.5, "train/acc": 0.8, "lr": 0.1}


def test_commit_false_merges_into_next_call(tmp_path: Path) -> None:
    with nuviz.Run(name="commit", logger_config=_config(tmp_path)) as run:
        run.log({"loss": 0.5}, commit=False)
        run.log({"acc": 0.9})
        run.log({"loss": 0.4})
    records = _records(run.dir)
    assert [(r["step"], r["metrics"]) for r in records] == [
        (0, {"loss": 0.5, "acc": 0.9}),
        (1, {"loss": 0.4}),
    ]


def test_uncommitted_values_are_not_lost_on_finish(tmp_path: Path) -> None:
    run = nuviz.Run(name="pending", logger_config=_config(tmp_path))
    run.log({"loss": 0.5}, commit=False)
    run.finish()
    assert _records(run.dir)[0]["metrics"] == {"loss": 0.5}


def test_arrays_become_images_and_junk_is_skipped(tmp_path: Path) -> None:
    np = pytest.importorskip("numpy")
    pytest.importorskip("PIL")
    with nuviz.Run(name="media", logger_config=_config(tmp_path)) as run:
        run.log(
            {
                "loss": np.float32(0.25),
                "val/render": np.zeros((4, 4, 3), dtype=np.uint8),
                "caption": "not a number",
            },
            step=7,
        )
    assert _records(run.dir)[0]["metrics"] == {"loss": 0.25}
    assert (Path(run.dir) / "images" / "step_000007_val_render.png").is_file()


def test_define_metric_and_failed_exit_code(tmp_path: Path) -> None:
    run = nuviz.Run(name="goal", logger_config=_config(tmp_path))
    run.define_metric("regret", summary="min")
    run.define_metric("whatever", summary="mean")  # unsupported mode: ignored
    run.log({"regret": 3.0})
    run.log({"regret": 1.0})
    run.finish(exit_code=1)
    meta = _meta(run.dir)
    assert meta["metric_goals"] == {"regret": "min"}
    assert meta["best_metrics"] == {"regret": 1.0}
    assert meta["status"] == "failed"


def test_log_before_init_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="nuviz.init"):
        nuviz.log({"loss": 1.0})


def test_second_init_finishes_the_first(tmp_path: Path) -> None:
    first = nuviz.init(name="first", logger_config=_config(tmp_path))
    nuviz.log({"loss": 1.0})
    second = nuviz.init(name="second", logger_config=_config(tmp_path))
    assert _meta(first.dir)["status"] == "done"
    assert _meta(second.dir)["status"] == "running"
    assert second.id != first.id
