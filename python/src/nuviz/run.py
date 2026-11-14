"""A small ``init`` / ``log`` / ``finish`` API in the style of wandb.

For scripts that already look like this::

    import wandb
    wandb.init(project="p", name="exp", config=cfg)
    wandb.log({"loss": loss, "acc": acc})
    wandb.finish()

switching to nuviz is ``import nuviz as wandb`` (for the calls above).
Everything goes through :class:`nuviz.Logger`; this module only adapts the
calling convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from nuviz._util import to_number, warn
from nuviz.config import NuvizConfig
from nuviz.logger import Logger


def _flatten(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dicts into ``"outer/inner"`` keys."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(value, name))
        else:
            flat[name] = value
    return flat


def _is_image(value: Any) -> bool:
    """Arrays / tensors with at least two dimensions are logged as images."""
    ndim = getattr(value, "ndim", None)
    return isinstance(ndim, int) and ndim >= 2


class Run:
    """A run started with :func:`init`. Wraps a :class:`nuviz.Logger`."""

    def __init__(
        self,
        project: str | None = None,
        name: str | None = None,
        config: Any = None,
        tags: list[str] | None = None,
        seed: int | None = None,
        max_steps: int | None = None,
        logger_config: NuvizConfig | None = None,
    ) -> None:
        self._logger = Logger(
            name=name,
            project=project,
            hparams=config,
            tags=tags,
            seed=seed,
            max_steps=max_steps,
            config=logger_config,
        )
        self._step = 0
        self._pending: dict[str, Any] = {}
        self._warned: set[str] = set()

    @property
    def logger(self) -> Logger:
        """The underlying :class:`nuviz.Logger`."""
        return self._logger

    @property
    def name(self) -> str:
        return self._logger.name

    @property
    def id(self) -> str:
        return self._logger.run_id

    @property
    def dir(self) -> str:
        return str(self._logger.experiment_dir)

    @property
    def step(self) -> int:
        """The step the next ``log()`` call without ``step=`` will use."""
        return self._step

    def define_metric(
        self, name: str, summary: str | None = None, goal: str | None = None
    ) -> None:
        """Declare a metric's direction: ``summary="min"`` or ``"max"``.

        ``goal="minimize"`` / ``"maximize"`` is accepted as well. Other
        wandb summary modes (``mean``, ``last``, ...) are ignored.
        """
        direction = goal or summary
        if direction in ("min", "max", "minimize", "maximize"):
            self._logger.define_metric(name, direction)

    def log(self, data: Mapping[str, Any], step: int | None = None, commit: bool = True) -> None:
        """Log a dict of values.

        Scalars become metrics; nested dicts are flattened to ``a/b`` keys;
        2-D+ arrays are saved as images. Without ``step`` an internal
        counter is used and advanced on every committed call, as in wandb.
        With ``commit=False`` values are held and written with the next
        committed call.
        """
        if step is not None:
            step = int(step)
            if step != self._step and self._pending:
                self._flush_pending()
            self._step = step

        self._pending.update(_flatten(data))
        if commit:
            self._flush_pending()
            self._step += 1

    def _flush_pending(self) -> None:
        pending, self._pending = self._pending, {}
        scalars: dict[str, Any] = {}
        for name, value in pending.items():
            if _is_image(value):
                self._logger.image(name.replace("/", "_"), value, step=self._step)
                continue
            try:
                scalars[name] = to_number(value)
            except TypeError:
                if name not in self._warned:
                    self._warned.add(name)
                    warn(f"'{name}' is not a scalar or an image; ignored")
        if scalars:
            self._logger.step(self._step, **scalars)

    def finish(self, exit_code: int | None = None) -> None:
        """Finish the run; a non-zero ``exit_code`` marks it failed."""
        if self._pending:
            self._flush_pending()
        if exit_code:
            self._logger.finish(status="failed", error=f"exit code {exit_code}")
        else:
            self._logger.finish()

    def __enter__(self) -> Run:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._pending:
            self._flush_pending()
        self._logger.__exit__(exc_type, exc, tb)


_current: Run | None = None


def init(
    project: str | None = None,
    name: str | None = None,
    config: Any = None,
    tags: list[str] | None = None,
    seed: int | None = None,
    max_steps: int | None = None,
    logger_config: NuvizConfig | None = None,
    **_ignored: Any,
) -> Run:
    """Start a run and make it the target of :func:`log` / :func:`finish`.

    Unknown keyword arguments (``entity``, ``mode``, ``reinit``, ...) are
    accepted and ignored so existing ``wandb.init(...)`` calls keep working.
    """
    global _current
    if _current is not None:
        _current.finish()
    _current = Run(
        project=project,
        name=name,
        config=config,
        tags=tags,
        seed=seed,
        max_steps=max_steps,
        logger_config=logger_config,
    )
    return _current


def log(data: Mapping[str, Any], step: int | None = None, commit: bool = True) -> None:
    """Log to the run started by :func:`init`."""
    if _current is None:
        raise RuntimeError("call nuviz.init() before nuviz.log()")
    _current.log(data, step=step, commit=commit)


def finish(exit_code: int | None = None) -> None:
    """Finish the run started by :func:`init` (no-op if there is none)."""
    global _current
    if _current is not None:
        _current.finish(exit_code=exit_code)
        _current = None
