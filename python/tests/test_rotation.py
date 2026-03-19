"""Tests for JSONL file rotation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nuviz.types import MetricRecord
from nuviz.writer import JsonlWriter


@pytest.fixture()
def writer_dir(tmp_path: Path) -> Path:
    d = tmp_path / "test-rotation"
    d.mkdir()
    return d


class TestJsonlRotation:
    def test_no_rotation_under_limit(self, writer_dir: Path) -> None:
        path = writer_dir / "metrics.jsonl"
        writer = JsonlWriter(
            path=path,
            flush_interval=100,
            flush_count=1,
            max_file_size_bytes=10_000,
            max_file_lines=1000,
        )

        for i in range(5):
            writer.write(MetricRecord(step=i, timestamp=float(i), metrics={"loss": 1.0 / (i + 1)}))

        writer.close()

        assert path.exists()
        assert not (writer_dir / "metrics.1.jsonl").exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_rotation_by_lines(self, writer_dir: Path) -> None:
        path = writer_dir / "metrics.jsonl"
        writer = JsonlWriter(
            path=path,
            flush_interval=100,
            flush_count=1,  # Flush every record for test
            max_file_size_bytes=10_000_000,  # Won't trigger
            max_file_lines=5,  # Rotate after 5 lines
        )

        for i in range(12):
            writer.write(MetricRecord(step=i, timestamp=float(i), metrics={"loss": 1.0 / (i + 1)}))

        writer.close()

        # Should have rotated: metrics.jsonl + metrics.1.jsonl (at least)
        assert path.exists()
        rotated = writer_dir / "metrics.1.jsonl"
        assert rotated.exists()

        # All records should be preserved across files
        all_records = _read_all_records(writer_dir)
        assert len(all_records) == 12

    def test_rotation_by_size(self, writer_dir: Path) -> None:
        path = writer_dir / "metrics.jsonl"
        writer = JsonlWriter(
            path=path,
            flush_interval=100,
            flush_count=1,
            max_file_size_bytes=200,  # Very small limit
            max_file_lines=1_000_000,
        )

        for i in range(20):
            writer.write(
                MetricRecord(step=i, timestamp=float(i), metrics={"loss": 0.1, "psnr": 25.0})
            )

        writer.close()

        # Should have at least one rotated file
        rotated = writer_dir / "metrics.1.jsonl"
        assert rotated.exists()

        all_records = _read_all_records(writer_dir)
        assert len(all_records) == 20

    def test_multiple_rotations(self, writer_dir: Path) -> None:
        path = writer_dir / "metrics.jsonl"
        writer = JsonlWriter(
            path=path,
            flush_interval=100,
            flush_count=1,
            max_file_size_bytes=10_000_000,
            max_file_lines=3,  # Rotate every 3 lines
        )

        for i in range(10):
            writer.write(MetricRecord(step=i, timestamp=float(i), metrics={"loss": float(i)}))

        writer.close()

        # Should have multiple rotated files
        assert path.exists()
        assert (writer_dir / "metrics.1.jsonl").exists()

        all_records = _read_all_records(writer_dir)
        assert len(all_records) == 10

        # Verify chronological order (steps should be increasing)
        steps = [r["step"] for r in all_records]
        assert steps == sorted(steps)

    def test_rotation_file_numbering(self, writer_dir: Path) -> None:
        path = writer_dir / "metrics.jsonl"
        writer = JsonlWriter(
            path=path,
            flush_interval=100,
            flush_count=1,
            max_file_size_bytes=10_000_000,
            max_file_lines=2,  # Rotate every 2 lines
        )

        for i in range(7):
            writer.write(MetricRecord(step=i, timestamp=float(i), metrics={"v": float(i)}))

        writer.close()

        # Verify rotation file naming
        all_files = sorted(writer_dir.glob("metrics*.jsonl"))
        assert len(all_files) >= 2


def _read_all_records(directory: Path) -> list[dict]:  # type: ignore[type-arg]
    """Read all records from main + rotated JSONL files in order."""
    records: list[dict] = []  # type: ignore[type-arg]

    # Read rotated files (highest number first = oldest data)
    rotated: list[tuple[int, Path]] = []
    for f in directory.iterdir():
        name = f.name
        if name.startswith("metrics.") and name.endswith(".jsonl") and name != "metrics.jsonl":
            num_str = name[len("metrics.") : -len(".jsonl")]
            if num_str.isdigit():
                rotated.append((int(num_str), f))

    rotated.sort(reverse=True)  # Highest number = oldest
    for _, path in rotated:
        records.extend(_read_jsonl(path))

    # Read main file (newest data)
    main = directory / "metrics.jsonl"
    if main.exists():
        records.extend(_read_jsonl(main))

    return records


def _read_jsonl(path: Path) -> list[dict]:  # type: ignore[type-arg]
    records = []
    for line in path.read_text().strip().split("\n"):
        if line.strip():
            records.append(json.loads(line))
    return records
