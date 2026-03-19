"""Background JSONL writer with buffered, non-blocking writes."""

from __future__ import annotations

import atexit
import json
import sys
import threading
from collections import deque
from dataclasses import asdict
from pathlib import Path

from nuviz.types import MetricRecord


def _sanitize_floats(data: dict) -> None:  # type: ignore[type-arg]
    """Replace NaN/Inf values with None in-place for valid JSON output."""
    import math
    for key, value in data.items():
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            data[key] = None
        elif isinstance(value, dict):
            _sanitize_floats(value)


class JsonlWriter:
    """Thread-safe, buffered JSONL writer.

    Writes are buffered and flushed either when the buffer reaches
    `flush_count` records, or every `flush_interval` seconds, whichever
    comes first. All I/O happens on a background daemon thread.

    Supports file rotation: when the current file exceeds `max_file_size_bytes`
    or `max_file_lines`, it is renamed to `metrics.1.jsonl` (shifting existing
    numbered files up), and a new `metrics.jsonl` is created.
    """

    def __init__(
        self,
        path: Path,
        flush_interval: float = 2.0,
        flush_count: int = 50,
        max_file_size_bytes: int = 50_000_000,
        max_file_lines: int = 500_000,
    ) -> None:
        self._path = path
        self._flush_interval = flush_interval
        self._flush_count = flush_count
        self._max_file_size_bytes = max_file_size_bytes
        self._max_file_lines = max_file_lines
        self._lines_written: int = 0

        self._buffer: deque[MetricRecord] = deque()
        self._lock = threading.Lock()
        self._closed = False
        self._close_event = threading.Event()

        self._thread = threading.Thread(target=self._run, daemon=True, name="nuviz-writer")
        self._thread.start()
        atexit.register(self.close)

    def write(self, record: MetricRecord) -> None:
        """Enqueue a record for writing. Non-blocking, thread-safe."""
        if self._closed:
            return
        with self._lock:
            self._buffer.append(record)
            should_flush = len(self._buffer) >= self._flush_count
        if should_flush:
            self._close_event.set()  # Wake writer thread early

    def flush(self) -> None:
        """Force flush all buffered records to disk."""
        records = self._drain_buffer()
        if records:
            self._write_records(records)

    def close(self) -> None:
        """Flush remaining records and stop the writer thread."""
        if self._closed:
            return
        self._closed = True
        self._close_event.set()
        self._thread.join(timeout=5.0)
        # Final flush in case thread missed anything
        self.flush()

    def _drain_buffer(self) -> list[MetricRecord]:
        with self._lock:
            records = list(self._buffer)
            self._buffer.clear()
        return records

    def _run(self) -> None:
        """Background thread: periodically flush the buffer."""
        while not self._closed:
            self._close_event.wait(timeout=self._flush_interval)
            self._close_event.clear()
            records = self._drain_buffer()
            if records:
                self._write_records(records)

    def _write_records(self, records: list[MetricRecord]) -> None:
        """Write records to the JSONL file. Never raises."""
        try:
            for record in records:
                self._maybe_rotate()
                data = asdict(record)
                # Remove None gpu field for cleaner output
                if data.get("gpu") is None:
                    del data["gpu"]
                # Sanitize NaN/Inf to null for valid JSON (Rust serde compat)
                _sanitize_floats(data)
                with open(self._path, "a") as f:
                    f.write(json.dumps(data, separators=(",", ":")) + "\n")
                    f.flush()
                self._lines_written += 1
        except OSError as e:
            print(f"[nuviz] Warning: failed to write metrics: {e}", file=sys.stderr)

    def _maybe_rotate(self) -> None:
        """Rotate the JSONL file if it exceeds size or line limits."""
        if not self._path.exists():
            return

        try:
            file_size = self._path.stat().st_size
        except OSError:
            return

        should_rotate = (
            file_size >= self._max_file_size_bytes
            or self._lines_written >= self._max_file_lines
        )
        if not should_rotate:
            return

        self._rotate_files()
        self._lines_written = 0

    def _rotate_files(self) -> None:
        """Shift numbered files up and rename current file to .1."""
        parent = self._path.parent
        stem = self._path.stem  # e.g., "metrics"
        suffix = self._path.suffix  # e.g., ".jsonl"

        # Find existing numbered files and shift them up
        existing_numbers: list[int] = []
        for f in parent.iterdir():
            name = f.name
            prefix = f"{stem}."
            if name.startswith(prefix) and name.endswith(suffix):
                middle = name[len(prefix) : -len(suffix)]
                if middle.isdigit():
                    existing_numbers.append(int(middle))

        # Shift in descending order to avoid collisions
        for num in sorted(existing_numbers, reverse=True):
            old_name = parent / f"{stem}.{num}{suffix}"
            new_name = parent / f"{stem}.{num + 1}{suffix}"
            try:
                old_name.rename(new_name)
            except OSError as e:
                print(f"[nuviz] Warning: rotation rename failed: {e}", file=sys.stderr)

        # Rename current file to .1
        rotated = parent / f"{stem}.1{suffix}"
        try:
            self._path.rename(rotated)
        except OSError as e:
            print(f"[nuviz] Warning: rotation failed: {e}", file=sys.stderr)
