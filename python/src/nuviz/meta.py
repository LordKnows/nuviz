"""Read-modify-write access to ``meta.json``.

``meta.json`` is shared with the CLI (``nuviz tag`` writes ``tags``), so every
update merges into whatever is on disk and replaces the file atomically.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from nuviz._util import atomic_write_json, warn

SCHEMA_VERSION = 1


class MetaStore:
    """Thread-safe, never-raising updater for an experiment's ``meta.json``."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> dict[str, Any]:
        """Return the current contents, or ``{}`` if missing or unreadable."""
        try:
            with open(self._path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def update(self, fields: dict[str, Any]) -> bool:
        """Merge ``fields`` into meta.json, keeping keys written by others.

        Returns ``False`` (after a warning) if the file could not be written.
        """
        with self._lock:
            try:
                atomic_write_json(self._path, {**self.read(), **fields})
                return True
            except (OSError, TypeError, ValueError) as e:
                warn(f"failed to update {self._path.name}: {e}")
                return False
