"""Hugging Face ``Trainer`` integration.

Usage::

    from transformers import Trainer
    from nuviz.integrations.transformers import NuvizCallback

    trainer = Trainer(..., callbacks=[NuvizCallback(project="my_project")])
"""

from __future__ import annotations

from typing import Any

try:
    from transformers import TrainerCallback
except ImportError:
    raise ImportError(
        "nuviz.integrations.transformers requires transformers. "
        "Install it with: pip install transformers"
    ) from None

from nuviz.config import NuvizConfig
from nuviz.logger import Logger


class NuvizCallback(TrainerCallback):  # type: ignore[misc]
    """Log a ``Trainer`` run to nuviz.

    Scalars passed to ``on_log`` are recorded at ``state.global_step``.
    Only the main process logs, so distributed training produces one
    experiment.

    Args:
        name: Experiment name. Defaults to ``TrainingArguments.run_name``.
        project: Optional project grouping.
        config: Logger configuration.
        **logger_kwargs: Forwarded to :class:`nuviz.Logger`.
    """

    def __init__(
        self,
        name: str | None = None,
        project: str | None = None,
        config: NuvizConfig | None = None,
        **logger_kwargs: Any,
    ) -> None:
        self._name = name
        self._project = project
        self._config = config
        self._logger_kwargs = logger_kwargs
        self._run: Logger | None = None

    @property
    def experiment(self) -> Logger | None:
        """The underlying :class:`nuviz.Logger`, once training has started."""
        return self._run

    def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if not getattr(state, "is_world_process_zero", True) or self._run is not None:
            return

        logger_kwargs = dict(self._logger_kwargs)
        if "hparams" not in logger_kwargs:
            to_dict = getattr(args, "to_sanitized_dict", None) or getattr(args, "to_dict", None)
            if callable(to_dict):
                logger_kwargs["hparams"] = to_dict()
        max_steps = getattr(state, "max_steps", None)
        if "max_steps" not in logger_kwargs and isinstance(max_steps, int) and max_steps > 0:
            logger_kwargs["max_steps"] = max_steps
        seed = getattr(args, "seed", None)
        if "seed" not in logger_kwargs and isinstance(seed, int):
            logger_kwargs["seed"] = seed

        run_name = self._name or getattr(args, "run_name", None)
        # Trainer defaults run_name to output_dir, which is not a usable name.
        if run_name is not None and run_name == getattr(args, "output_dir", None):
            run_name = None
        self._run = Logger(
            name=run_name, project=self._project, config=self._config, **logger_kwargs
        )

    def on_log(
        self,
        args: Any,
        state: Any,
        control: Any,
        logs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if self._run is None or not logs:
            return
        scalars = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if scalars:
            self._run.step(int(getattr(state, "global_step", 0)), **scalars)

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
