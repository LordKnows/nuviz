"""GPU metrics collection via nvidia-smi polling."""

from __future__ import annotations

import subprocess
import threading


class GpuCollector:
    """Collects GPU metrics by polling nvidia-smi in a background thread.

    Falls back gracefully to None if nvidia-smi is not available.
    Thread-safe: latest() can be called from any thread.
    """

    def __init__(self, poll_interval: float = 5.0, device_index: int = 0) -> None:
        self._poll_interval = poll_interval
        self._device_index = device_index
        self._lock = threading.Lock()
        self._latest: dict[str, int | float] | None = None
        self._available: bool | None = None  # None = not yet checked
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="nuviz-gpu")
        self._thread.start()

    def latest(self) -> dict[str, int | float] | None:
        """Return the most recent GPU reading, or None if unavailable."""
        with self._lock:
            return self._latest

    def stop(self) -> None:
        """Stop the polling thread."""
        self._stop_event.set()
        self._thread.join(timeout=3.0)

    def _run(self) -> None:
        """Background thread: poll nvidia-smi at interval."""
        self._probe()
        if not self._available:
            return  # nvidia-smi not found, thread exits

        while not self._stop_event.is_set():
            self._poll()
            self._stop_event.wait(timeout=self._poll_interval)

    def _probe(self) -> None:
        """Check if nvidia-smi is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            self._available = result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            self._available = False

    def _poll(self) -> None:
        """Poll nvidia-smi and update latest reading."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={self._device_index}",
                    "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=3.0,
            )
            if result.returncode != 0:
                return

            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) >= 4:
                reading: dict[str, int | float] = {
                    "gpu_util": int(parts[0]),
                    "mem_used_mb": int(parts[1]),
                    "mem_total_mb": int(parts[2]),
                    "temperature_c": int(parts[3]),
                }
                with self._lock:
                    self._latest = reading
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError, ValueError):
            pass  # Silently skip failed polls
