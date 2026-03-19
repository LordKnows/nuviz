"""Synchronous writer for per-scene metrics (scenes.jsonl)."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

from nuviz.types import SceneRecord


class SceneWriter:
    """Appends SceneRecord entries to scenes.jsonl.

    Unlike JsonlWriter, this is synchronous — scene evaluations are
    infrequent and typically happen after training, so background
    buffering is unnecessary.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._closed = False

    def write(self, record: SceneRecord) -> None:
        """Append a scene record to scenes.jsonl."""
        if self._closed:
            return
        try:
            with open(self._path, "a") as f:
                data = asdict(record)
                f.write(json.dumps(data, separators=(",", ":")) + "\n")
                f.flush()
        except OSError as e:
            print(f"[nuviz] Warning: failed to write scene record: {e}", file=sys.stderr)

    def close(self) -> None:
        """Mark writer as closed."""
        self._closed = True
