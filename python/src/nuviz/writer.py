"""Background JSONL writer with buffered, non-blocking writes."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import threading
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

from nuviz._util import to_number, warn
from nuviz.types import MetricRecord


def _sanitize_metrics(data: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-safe copy of a metrics dict.

    Values are coerced to plain numbers; NaN/Inf become ``None`` (JSON ``null``)
    so every line is strictly valid JSON. Raises ``TypeError`` for values that
    are not scalar numbers.
    """
    out: dict[str, Any] = {}
    for key, value in data.items():
        if value is None:
            out[key] = None
        elif isinstance(value, dict):
            out[key] = _sanitize_metrics(value)
        else:
            number = to_number(value)
            if isinstance(number, float) and (number != number or number in (_INF, -_INF)):
                out[key] = None
            else:
                out[key] = number
    return out


_INF = float("inf")


def serialize_record(record: MetricRecord) -> str:
    """Serialize a record to one JSONL line (including the trailing newline)."""
    data: dict[str, Any] = {
        "step": int(record.step),
        "timestamp": float(record.timestamp),
        "metrics": _sanitize_metrics(record.metrics),
    }
    if record.gpu is not None:
        data["gpu"] = _sanitize_metrics(record.gpu)
    return json.dumps(data, separators=(",", ":"), allow_nan=False) + "\n"


class JsonlWriter:
    """Thread-safe, buffered JSONL writer.

    Records are queued by ``write()`` and written by a single background
    daemon thread, either when the queue reaches ``flush_count`` records or
    every ``flush_interval`` seconds, whichever comes first. The background
    thread is the *only* code that touches the file, so ordering is always
    preserved and rotation can never race.

    The queue is bounded by ``max_queue``. When it is full, ``write()`` waits
    briefly for the writer to catch up (backpressure); if the writer is stuck
    (e.g. a hung filesystem) new records are dropped and counted instead of
    blocking the training loop or growing memory without bound.

    Supports file rotation: when the current file exceeds
    ``max_file_size_bytes`` or ``max_file_lines``, it is renamed to
    ``metrics.1.jsonl`` (shifting existing numbered files up), and a new
    ``metrics.jsonl`` is started.
    """

    def __init__(
        self,
        path: Path,
        flush_interval: float = 2.0,
        flush_count: int = 50,
        max_file_size_bytes: int = 50_000_000,
        max_file_lines: int = 500_000,
        max_queue: int = 100_000,
        close_timeout: float = 30.0,
        on_tick: Callable[[], None] | None = None,
    ) -> None:
        self._path = path
        self._flush_interval = flush_interval
        self._flush_count = max(1, flush_count)
        self._max_file_size_bytes = max_file_size_bytes
        self._max_file_lines = max_file_lines
        self._max_queue = max(1, max_queue)
        self._close_timeout = close_timeout
        self._on_tick = on_tick
        self._block_timeout = 1.0  # max time write() waits for queue space

        # State shared between producers and the writer thread (guarded by _cond)
        self._cond = threading.Condition()
        self._buffer: deque[MetricRecord] = deque()
        self._enqueued = 0  # records accepted by write()
        self._processed = 0  # records written or discarded by the writer thread
        self._wake = False
        self._closed = False
        self._degraded = False  # writer is stuck; drop instead of waiting
        self._dropped = 0

        # State owned exclusively by the writer thread
        self._file: IO[str] | None = None
        self._file_bytes = 0
        self._file_lines = 0
        self._warned: set[str] = set()

        self._thread = threading.Thread(target=self._run, daemon=True, name="nuviz-writer")
        self._thread.start()
        atexit.register(self.close)

    # ------------------------------------------------------------------ API

    @property
    def dropped(self) -> int:
        """Number of records dropped because the queue was full."""
        return self._dropped

    def write(self, record: MetricRecord) -> bool:
        """Enqueue a record for writing. Thread-safe.

        Returns ``False`` if the record was not accepted (writer closed, or
        the queue is full and the writer is not draining it).
        """
        with self._cond:
            if self._closed:
                return False

            if len(self._buffer) >= self._max_queue:
                if not self._degraded:
                    self._wake = True
                    self._cond.notify_all()
                    drained = self._cond.wait_for(
                        lambda: len(self._buffer) < self._max_queue or self._closed,
                        timeout=self._block_timeout,
                    )
                    if not drained:
                        self._degraded = True
                if self._closed:
                    return False
                if len(self._buffer) >= self._max_queue:
                    self._dropped += 1
                    if self._dropped == 1:
                        warn(
                            f"metrics queue is full ({self._max_queue} records); "
                            "dropping new records until the writer catches up"
                        )
                    return False
            self._degraded = False

            self._buffer.append(record)
            self._enqueued += 1
            if len(self._buffer) >= self._flush_count:
                self._wake = True
                self._cond.notify_all()
        return True

    def flush(self, timeout: float | None = 10.0) -> bool:
        """Block until every record enqueued so far has been written.

        Returns ``False`` if the timeout expired first.
        """
        if threading.current_thread() is self._thread:
            return False
        with self._cond:
            target = self._enqueued
            self._wake = True
            self._cond.notify_all()
            return self._cond.wait_for(
                lambda: self._processed >= target or not self._thread.is_alive(),
                timeout=timeout,
            )

    def close(self) -> None:
        """Flush remaining records and stop the writer thread."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
            self._wake = True
            self._cond.notify_all()

        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=self._close_timeout)
            if self._thread.is_alive():
                with self._cond:
                    pending = self._enqueued - self._processed
                warn(
                    f"writer did not finish within {self._close_timeout:.0f}s; "
                    f"{pending} record(s) may not have been written"
                )
        if self._dropped:
            warn(f"{self._dropped} metric record(s) were dropped because the queue was full")
        with contextlib.suppress(Exception):
            atexit.unregister(self.close)

    # ------------------------------------------------------- writer thread

    def _run(self) -> None:
        """Background thread: drain the queue and write batches."""
        while True:
            with self._cond:
                if not self._wake and not self._closed:
                    self._cond.wait_for(
                        lambda: self._wake or self._closed, timeout=self._flush_interval
                    )
                self._wake = False
                records = list(self._buffer)
                self._buffer.clear()
                closing = self._closed
                self._cond.notify_all()  # producers waiting for space

            if records:
                try:
                    self._write_batch(records)
                except Exception as e:  # never let the writer thread die
                    warn(f"failed to write metrics: {e}")

            if self._on_tick is not None:
                try:
                    self._on_tick()
                except Exception as e:
                    self._warn_once("tick", f"background tick failed: {e}")

            with self._cond:
                self._processed += len(records)
                self._cond.notify_all()
                if closing and not self._buffer:
                    break

        self._close_file()

    def _write_batch(self, records: list[MetricRecord]) -> None:
        pending: list[str] = []
        pending_bytes = 0

        for record in records:
            try:
                line = serialize_record(record)
            except (TypeError, ValueError, OverflowError) as e:
                self._warn_once(
                    f"serialize:{e}", f"skipping record at step {record.step}: {e}"
                )
                continue

            if self._should_rotate(len(pending), pending_bytes):
                self._emit(pending)
                pending, pending_bytes = [], 0
                self._rotate_files()

            pending.append(line)
            pending_bytes += len(line.encode("utf-8"))

        self._emit(pending)

    def _should_rotate(self, pending_lines: int, pending_bytes: int) -> bool:
        if self._file is None and not pending_lines:
            self._open_file()
        size = self._file_bytes + pending_bytes
        lines = self._file_lines + pending_lines
        if size == 0:
            return False
        return size >= self._max_file_size_bytes or lines >= self._max_file_lines

    def _emit(self, lines: list[str]) -> None:
        """Append lines to the current file with a single write call."""
        if not lines:
            return
        try:
            self._ensure_current_file()
            if self._file is None:
                raise OSError(f"cannot open {self._path}")
            payload = "".join(lines)
            self._file.write(payload)
            self._file.flush()
            self._file_bytes += len(payload.encode("utf-8"))
            self._file_lines += len(lines)
        except OSError as e:
            self._close_file()
            warn(f"failed to write {len(lines)} metric record(s): {e}")

    def _open_file(self) -> None:
        try:
            self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
            self._file_bytes = os.fstat(self._file.fileno()).st_size
        except OSError:
            self._file = None
            self._file_bytes = 0

    def _ensure_current_file(self) -> None:
        """Open the file if needed, re-opening if it was moved or deleted."""
        if self._file is not None:
            try:
                on_disk = os.stat(self._path)
                ours = os.fstat(self._file.fileno())
                if (on_disk.st_ino, on_disk.st_dev) == (ours.st_ino, ours.st_dev):
                    return
            except OSError:
                pass
            self._close_file()
            self._file_lines = 0
        self._file = open(self._path, "a", encoding="utf-8")  # noqa: SIM115
        self._file_bytes = os.fstat(self._file.fileno()).st_size

    def _close_file(self) -> None:
        if self._file is not None:
            with contextlib.suppress(OSError):
                self._file.close()
            self._file = None

    def _rotate_files(self) -> None:
        """Shift numbered files up and rename the current file to ``.1``."""
        self._close_file()
        if not self._path.exists():
            self._file_bytes = 0
            self._file_lines = 0
            return

        parent = self._path.parent
        stem = self._path.stem  # e.g., "metrics"
        suffix = self._path.suffix  # e.g., ".jsonl"

        existing_numbers: list[int] = []
        prefix = f"{stem}."
        try:
            names = [f.name for f in parent.iterdir()]
        except OSError:
            names = []
        for name in names:
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
                warn(f"rotation rename failed: {e}")

        try:
            self._path.rename(parent / f"{stem}.1{suffix}")
        except OSError as e:
            warn(f"rotation failed: {e}")

        self._file_bytes = 0
        self._file_lines = 0

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        if len(self._warned) < 100:
            self._warned.add(key)
        warn(message)
