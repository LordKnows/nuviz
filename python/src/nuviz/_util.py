"""Small shared helpers."""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


def to_number(value: Any) -> int | float:
    """Coerce a scalar-like value to a plain Python ``int`` or ``float``.

    Accepts Python numbers, numpy scalars, and 0-dim tensors/arrays (anything
    exposing ``.item()`` or ``__float__``). Raises ``TypeError`` for values that
    are not scalar numbers.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        # Normalise int/float subclasses (e.g. numpy.float64) to builtins
        return int(value) if isinstance(value, int) else float(value)

    item = getattr(value, "item", None)
    if callable(item):
        try:
            extracted = item()
        except Exception as e:  # e.g. tensors with more than one element
            raise TypeError(f"expected a scalar, got {type(value).__name__}: {e}") from None
        if isinstance(extracted, bool):
            return int(extracted)
        if isinstance(extracted, (int, float)):
            return extracted
        raise TypeError(f"expected a number, got {type(value).__name__}")

    if isinstance(value, (str, bytes)) or not hasattr(value, "__float__"):
        raise TypeError(f"expected a number, got {type(value).__name__}")
    return float(value)


def is_finite(value: int | float) -> bool:
    return not isinstance(value, float) or math.isfinite(value)


def warn(message: str) -> None:
    """Print a nuviz warning to stderr. Never raises."""
    with contextlib.suppress(Exception):
        print(f"[nuviz] Warning: {message}", file=sys.stderr)


def atomic_write_json(path: Path, data: Any, indent: int | None = 2) -> None:
    """Write JSON to ``path`` atomically (temp file + rename)."""
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            with contextlib.suppress(OSError):
                tmp.unlink()
