"""Tests for experiment naming and directory resolution."""

from __future__ import annotations

import re
from pathlib import Path

from nuviz.naming import _sanitize_name, resolve_experiment_dir


class TestSanitizeName:
    def test_clean_name_unchanged(self) -> None:
        assert _sanitize_name("exp-001") == "exp-001"

    def test_spaces_replaced(self) -> None:
        assert _sanitize_name("my experiment") == "my_experiment"

    def test_special_chars_replaced(self) -> None:
        assert _sanitize_name("exp@#$%001") == "exp_001"

    def test_multiple_underscores_collapsed(self) -> None:
        assert _sanitize_name("a___b") == "a_b"

    def test_dots_and_hyphens_preserved(self) -> None:
        assert _sanitize_name("lr1e-4.run1") == "lr1e-4.run1"

    def test_leading_trailing_underscores_stripped(self) -> None:
        assert _sanitize_name("__name__") == "name"


class TestResolveExperimentDir:
    def test_user_provided_name(self, tmp_base_dir: Path) -> None:
        result = resolve_experiment_dir("exp-001", None, tmp_base_dir)
        assert result.name == "exp-001"
        assert result.parent == tmp_base_dir
        assert result.exists()

    def test_timestamp_fallback(self, tmp_base_dir: Path) -> None:
        result = resolve_experiment_dir(None, None, tmp_base_dir)
        assert re.match(r"\d{8}_\d{6}", result.name)
        assert result.exists()

    def test_project_nesting(self, tmp_base_dir: Path) -> None:
        result = resolve_experiment_dir("exp-001", "my_project", tmp_base_dir)
        assert result == tmp_base_dir / "my_project" / "exp-001"
        assert result.exists()

    def test_name_sanitization(self, tmp_base_dir: Path) -> None:
        result = resolve_experiment_dir("bad name!@#", None, tmp_base_dir)
        assert result.name == "bad_name"
        assert result.exists()

    def test_collision_handling(self, tmp_base_dir: Path) -> None:
        # Create the first one
        first = resolve_experiment_dir("exp", None, tmp_base_dir)
        assert first.name == "exp"

        # Second should get suffix _1
        second = resolve_experiment_dir("exp", None, tmp_base_dir)
        assert second.name == "exp_1"
        assert second.exists()

        # Third should get suffix _2
        third = resolve_experiment_dir("exp", None, tmp_base_dir)
        assert third.name == "exp_2"

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        deep = tmp_path / "a" / "b" / "c"
        result = resolve_experiment_dir("exp", None, deep)
        assert result.exists()

    def test_project_name_sanitized(self, tmp_base_dir: Path) -> None:
        result = resolve_experiment_dir("exp", "bad project!", tmp_base_dir)
        assert result.parent.name == "bad_project"
