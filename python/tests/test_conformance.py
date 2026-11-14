"""Cross-language wire-format conformance.

``spec/fixtures/records.json`` holds logical records and
``spec/fixtures/metrics.jsonl`` their golden wire encoding. The Rust CLI is
tested against the same two files (``cli/tests/conformance_test.rs``), so the
writer and the reader cannot drift apart silently.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from nuviz.types import MetricRecord
from nuviz.writer import JsonlWriter, serialize_record

FIXTURES = Path(__file__).resolve().parents[2] / "spec" / "fixtures"
_SPECIAL = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}


def _logical_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = json.loads((FIXTURES / "records.json").read_text("utf-8"))
    return records


def _to_record(raw: dict[str, Any]) -> MetricRecord:
    metrics = {
        k: _SPECIAL[v] if isinstance(v, str) else v for k, v in raw["metrics"].items()
    }
    return MetricRecord(
        step=raw["step"], timestamp=raw["timestamp"], metrics=metrics, gpu=raw.get("gpu")
    )


def test_serializer_matches_golden_file() -> None:
    golden = (FIXTURES / "metrics.jsonl").read_text("utf-8").splitlines(keepends=True)
    produced = [serialize_record(_to_record(r)) for r in _logical_records()]
    assert produced == golden


def test_writer_output_matches_golden_file(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    writer = JsonlWriter(path, flush_interval=10.0, flush_count=3)
    for raw in _logical_records():
        writer.write(_to_record(raw))
    writer.close()
    assert path.read_bytes() == (FIXTURES / "metrics.jsonl").read_bytes()


def test_every_golden_line_is_strict_json() -> None:
    def reject(token: str) -> None:
        raise AssertionError(f"non-standard JSON constant {token!r} in wire format")

    for line in (FIXTURES / "metrics.jsonl").read_text("utf-8").splitlines():
        data = json.loads(line, parse_constant=reject)
        assert set(data) <= {"step", "timestamp", "metrics", "gpu"}
        assert isinstance(data["step"], int)
        assert isinstance(data["metrics"], dict)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_values_become_null(value: float) -> None:
    line = serialize_record(MetricRecord(step=0, timestamp=0.0, metrics={"x": value}))
    assert json.loads(line)["metrics"]["x"] is None
