"""Tests for framework adapters.

Lightning and transformers are large optional dependencies, so the adapters
are exercised against minimal stand-ins that mimic the interfaces the
adapters rely on.
"""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from nuviz.config import NuvizConfig


def _config(tmp_path: Path) -> NuvizConfig:
    return NuvizConfig(
        base_dir=tmp_path,
        flush_interval_seconds=0.05,
        flush_count=5,
        enable_snapshot=False,
        enable_gpu=False,
    )


def _records(exp_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (exp_dir / "metrics.jsonl").read_text().splitlines()]


def _meta(exp_dir: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((exp_dir / "meta.json").read_text())
    return data


@pytest.fixture
def lightning_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Install a stand-in ``lightning`` package and import the adapter."""

    class BaseLogger:
        def __init__(self) -> None:
            self.base_initialised = True

    logger_mod = types.ModuleType("lightning.pytorch.loggers.logger")
    logger_mod.Logger = BaseLogger  # type: ignore[attr-defined]
    logger_mod.rank_zero_experiment = lambda fn: fn  # type: ignore[attr-defined]
    utilities = types.ModuleType("lightning.pytorch.utilities")
    utilities.rank_zero_only = lambda fn: fn  # type: ignore[attr-defined]

    for name, module in {
        "lightning": types.ModuleType("lightning"),
        "lightning.pytorch": types.ModuleType("lightning.pytorch"),
        "lightning.pytorch.loggers": types.ModuleType("lightning.pytorch.loggers"),
        "lightning.pytorch.loggers.logger": logger_mod,
        "lightning.pytorch.utilities": utilities,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "nuviz.integrations.lightning", raising=False)
    module = importlib.import_module("nuviz.integrations.lightning")
    yield module
    sys.modules.pop("nuviz.integrations.lightning", None)


@pytest.fixture
def transformers_module(monkeypatch: pytest.MonkeyPatch) -> Any:
    stub = types.ModuleType("transformers")
    stub.TrainerCallback = type("TrainerCallback", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", stub)
    monkeypatch.delitem(sys.modules, "nuviz.integrations.transformers", raising=False)
    module = importlib.import_module("nuviz.integrations.transformers")
    yield module
    sys.modules.pop("nuviz.integrations.transformers", None)


class TestLightningLogger:
    def test_full_trainer_sequence(self, lightning_module: Any, tmp_path: Path) -> None:
        logger = lightning_module.NuvizLogger(
            "lit-run", project="lit", config=_config(tmp_path), seed=3
        )
        assert logger.base_initialised
        assert logger.name == "lit-run"
        assert logger.version == "0"  # nothing created before the first use

        # The order Lightning's Trainer calls a logger in.
        logger.log_hyperparams({"lr": 0.01, "layers": 4})
        logger.log_metrics({"train_loss": 0.9, "epoch": 0}, step=0)
        logger.log_metrics({"train_loss": 0.5, "epoch": 0}, step=50)
        logger.log_metrics({"val_loss": 0.6, "epoch": 0}, step=50)
        exp_dir = logger.experiment.experiment_dir
        assert logger.version == logger.experiment.run_id
        assert logger.save_dir == str(exp_dir.parent)
        logger.finalize("success")

        assert exp_dir == tmp_path / "lit" / "lit-run"
        records = _records(exp_dir)
        assert [r["step"] for r in records] == [0, 50, 50]
        assert records[1]["metrics"] == {"train_loss": 0.5, "epoch": 0}
        meta = _meta(exp_dir)
        assert meta["status"] == "done"
        assert meta["config"] == {"lr": 0.01, "layers": 4}
        assert meta["seed"] == 3

    def test_failure_status_and_namespace_hparams(
        self, lightning_module: Any, tmp_path: Path
    ) -> None:
        import argparse

        logger = lightning_module.NuvizLogger("lit-fail", config=_config(tmp_path))
        logger.log_metrics({"loss": 1.0}, step=1)
        # Hyperparameters that arrive after the run has started are still recorded.
        logger.log_hyperparams(argparse.Namespace(lr=0.5))
        exp_dir = logger.experiment.experiment_dir
        logger.finalize("failed")

        meta = _meta(exp_dir)
        assert meta["status"] == "failed"
        assert meta["config"] == {"lr": 0.5}

    def test_finalize_without_any_logging_creates_nothing(
        self, lightning_module: Any, tmp_path: Path
    ) -> None:
        logger = lightning_module.NuvizLogger("unused", config=_config(tmp_path))
        logger.finalize("success")
        assert not (tmp_path / "unused").exists()

    def test_missing_framework_gives_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("lightning", "pytorch_lightning"):
            monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, "nuviz.integrations.lightning", raising=False)
        with pytest.raises(ImportError, match="pip install lightning"):
            importlib.import_module("nuviz.integrations.lightning")


class TestTransformersCallback:
    @staticmethod
    def _args(**overrides: Any) -> Any:
        fields: dict[str, Any] = {"run_name": "hf-run", "output_dir": "out", "seed": 11}
        fields.update(overrides)
        args = types.SimpleNamespace(**fields)
        args.to_sanitized_dict = lambda: {"learning_rate": 5e-5, "num_train_epochs": 3}
        return args

    def test_full_trainer_sequence(self, transformers_module: Any, tmp_path: Path) -> None:
        cb = transformers_module.NuvizCallback(project="hf", config=_config(tmp_path))
        args = self._args()
        state = types.SimpleNamespace(is_world_process_zero=True, max_steps=100, global_step=0)

        cb.on_train_begin(args, state, None)
        exp_dir = cb.experiment.experiment_dir
        state.global_step = 10
        cb.on_log(args, state, None, logs={"loss": 2.5, "learning_rate": 5e-5, "epoch": 0.1})
        state.global_step = 20
        cb.on_log(args, state, None, logs={"eval_loss": 2.1, "note": "text is skipped"})
        cb.on_log(args, state, None, logs=None)
        cb.on_train_end(args, state, None)

        assert exp_dir == tmp_path / "hf" / "hf-run"
        records = _records(exp_dir)
        assert [(r["step"], sorted(r["metrics"])) for r in records] == [
            (10, ["epoch", "learning_rate", "loss"]),
            (20, ["eval_loss"]),
        ]
        meta = _meta(exp_dir)
        assert meta["status"] == "done"
        assert meta["max_steps"] == 100
        assert meta["seed"] == 11
        assert meta["config"]["learning_rate"] == 5e-5
        assert cb.experiment is None

    def test_non_main_process_does_not_log(self, transformers_module: Any, tmp_path: Path) -> None:
        cb = transformers_module.NuvizCallback(config=_config(tmp_path))
        state = types.SimpleNamespace(is_world_process_zero=False, max_steps=10, global_step=1)
        cb.on_train_begin(self._args(), state, None)
        cb.on_log(self._args(), state, None, logs={"loss": 1.0})
        cb.on_train_end(self._args(), state, None)
        assert list(tmp_path.iterdir()) == []

    def test_default_run_name_equal_to_output_dir_is_not_used(
        self, transformers_module: Any, tmp_path: Path
    ) -> None:
        cb = transformers_module.NuvizCallback(config=_config(tmp_path))
        state = types.SimpleNamespace(is_world_process_zero=True, max_steps=-1, global_step=0)
        cb.on_train_begin(self._args(run_name="out"), state, None)
        name = cb.experiment.name
        cb.on_train_end(self._args(), state, None)
        assert name != "out"  # falls back to a timestamp name
        assert "max_steps" not in _meta(tmp_path / name)
