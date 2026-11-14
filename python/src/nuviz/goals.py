"""Metric direction ("is lower or higher better?").

Explicit goals set with ``Logger.define_metric()`` are stored in ``meta.json``
and always win. Otherwise a name-based heuristic is used. The Rust CLI
implements the same heuristic (``cli/src/data/goal.rs``); both are checked
against ``spec/fixtures/metric_goals.json``.
"""

from __future__ import annotations

import re
from typing import Literal

Goal = Literal["min", "max"]

# Matched anywhere in the lower-cased name (e.g. "val_loss", "l1loss").
_MIN_SUBSTRINGS = ("loss", "lpips", "error", "mse", "mae")
# Matched only as a whole token, because they are too short to match safely
# as substrings ("wer" is in "power", "cer" in "certainty", ...).
_MIN_TOKENS = frozenset(
    {"err", "rmse", "fid", "kid", "ppl", "perplexity", "wer", "cer", "nll", "kl", "bpd"}
)
_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")


def default_goal(name: str) -> Goal:
    """Heuristic goal for a metric name."""
    lower = name.lower()
    if any(k in lower for k in _MIN_SUBSTRINGS):
        return "min"
    if any(tok in _MIN_TOKENS for tok in _TOKEN_SPLIT.split(lower)):
        return "min"
    return "max"


def validate_goal(goal: str) -> Goal:
    if goal in ("min", "minimize", "lower"):
        return "min"
    if goal in ("max", "maximize", "higher"):
        return "max"
    raise ValueError(f"goal must be 'min' or 'max', got {goal!r}")
