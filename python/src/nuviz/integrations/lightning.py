"""Lightning integration.

Usage::

    from lightning.pytorch import Trainer
    from nuviz.integrations.lightning import NuvizLogger

    trainer = Trainer(logger=NuvizLogger("exp-001", project="my_project"))
"""

from __future__ import annotations

from typing import Any

try:  # Lightning >= 2.0
    from lightning.pytorch.loggers.logger import Logger as _LightningLogger
    from lightning.pytorch.loggers.logger import rank_zero_experiment
    from lightning.pytorch.utilities import rank_zero_only
except ImportError:
    try:  # the older, separately distributed package
        from pytorch_lightning.loggers.logger import (  # type: ignore[no-redef]
            Logger as _LightningLogger,
        )
        from pytorch_lightning.loggers.logger import (  # type: ignore[no-redef]
            rank_zero_experiment,
        )
        from pytorch_lightning.utilities import rank_zero_only  # type: ignore[no-redef]
    except ImportError:
        raise ImportError(
            "nuviz.integrations.lightning requires Lightning. "
            "Install it with: pip install lightning"
        ) from None

from nuviz.config import NuvizConfig
from nuviz.logger import Logger


def _to_dict(params: Any) -> dict[str, Any]:
    if params is None:
        return {}
    if isinstance(params, dict):
        return dict(params)
    if hasattr(params, "__dict__"):
        return dict(vars(params))
    return {"hparams": str(params)}


class NuvizLogger(_LightningLogger):  # type: ignore[misc]
    """Log a Lightning run to nuviz.

    The underlying :class:`nuviz.Logger` is created lazily on the rank-zero
    process only, so distributed runs produce a single experiment.

    Args:
        name: Experiment name.
        project: Optional project grouping.
        config: Logger configuration.
        **logger_kwargs: Forwarded to :class:`nuviz.Logger`
            (``seed``, ``tags``, ``max_steps``, ...).
    """

    def __init__(
        self,
        name: str | None = None,
        project: str | None = None,
        config: NuvizConfig | None = None,
        **logger_kwargs: Any,
    ) -> None:
        super().__init__()
        self._requested_name = name
        self._project = project
        self._config = config
        self._logger_kwargs = logger_kwargs
        self._hparams: dict[str, Any] = {}
        self._run: Logger | None = None

    @property
    @rank_zero_experiment
    def experiment(self) -> Logger:
        """The underlying :class:`nuviz.Logger` (created on first access)."""
        if self._run is None:
            kwargs = dict(self._logger_kwargs)
            if self._hparams and "hparams" not in kwargs:
                kwargs["hparams"] = self._hparams
            self._run = Logger(
                name=self._requested_name,
                project=self._project,
                config=self._config,
                **kwargs,
            )
        return self._run

    @property
    def name(self) -> str:
        if self._run is not None:
            return self._run.name
        return self._requested_name or "nuviz"

    @property
    def version(self) -> str:
        return self._run.run_id if self._run is not None else "0"

    @property
    def save_dir(self) -> str | None:
        return str(self._run.experiment_dir.parent) if self._run is not None else None

    @rank_zero_only
    def log_hyperparams(self, params: Any = None, *args: Any, **kwargs: Any) -> None:
        self._hparams.update(_to_dict(params))
        if self._run is not None:
            # The run already started: record the hyperparameters after the fact.
            from nuviz.logger import _normalize_hparams

            self._run._meta.update({"config": _normalize_hparams(self._hparams)})

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        run = self.experiment
        run.step(step if step is not None else run._current_step + 1, **metrics)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if self._run is None:
            return
        if status in ("success", "finished"):
            self._run.finish()
        else:
            self._run.finish(status="failed", error=f"trainer status: {status}")
