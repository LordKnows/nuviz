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
    """

    def __init__(
        self,
        path: Path,
        flush_interval: float = 2.0,
        flush_count: int = 50,
    ) -> None:
        self._path = path
        self._flush_interval = flush_interval
        self._flush_count = flush_count

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
            with open(self._path, "a") as f:
                for record in records:
                    data = asdict(record)
                    # Remove None gpu field for cleaner output
                    if data.get("gpu") is None:
                        del data["gpu"]
                    # Sanitize NaN/Inf to null for valid JSON (Rust serde compat)
                    _sanitize_floats(data)
                    f.write(json.dumps(data, separators=(",", ":")) + "\n")
                f.flush()
        except OSError as e:
            print(f"[nuviz] Warning: failed to write metrics: {e}", file=sys.stderr)
