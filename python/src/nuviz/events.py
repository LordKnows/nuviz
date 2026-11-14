"""Append-only event stream (``events.jsonl``).

Events carry things that are not per-step scalars: anomaly alerts and
references to saved images and point clouds. The CLI tails this file to show
alerts live.
"""

from __future__ import annotations

import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from nuviz._util import warn


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


class EventWriter:
    """Synchronous, thread-safe, never-raising appender for events.jsonl."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._warned = False

    def emit(self, event_type: str, step: int | None = None, **fields: Any) -> None:
        record: dict[str, Any] = {"timestamp": time.time(), "type": event_type}
        if step is not None:
            record["step"] = int(step)
        record.update(_json_safe(fields))
        try:
            line = json.dumps(record, separators=(",", ":"), allow_nan=False, default=str)
            with self._lock, open(self._path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except (OSError, TypeError, ValueError) as e:
            if not self._warned:
                self._warned = True
                warn(f"failed to write event: {e}")
