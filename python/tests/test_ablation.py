"""Tests for the Ablation class."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nuviz.ablation import Ablation, _config_hash, _deep_set
from nuviz.logger import Logger


class TestDeepSet:
    def test_simple_key(self) -> None:
        result = _deep_set({"a": 1}, "a", 2)
        assert result == {"a": 2}

    def test_dotted_key(self) -> None:
        result = _deep_set({"model": {"lr": 1e-4}}, "model.lr", 1e-3)
        assert result == {"model": {"lr": 1e-3}}

    def test_creates_nested_dicts(self) -> None:
        result = _deep_set({}, "model.optimizer.lr", 0.01)
        assert result == {"model": {"optimizer": {"lr": 0.01}}}

    def test_does_not_mutate_original(self) -> None:
        original = {"a": 1, "b": {"c": 2}}
        result = _deep_set(original, "b.c", 99)
        assert original["b"]["c"] == 2
        assert result["b"]["c"] == 99


class TestConfigHash:
    def test_deterministic(self) -> None:
        config = {"lr": 1e-4, "sh": 2, "dense": True}
        assert _config_hash(config) == _config_hash(config)

    def test_order_independent(self) -> None:
        a = {"lr": 1e-4, "sh": 2}
        b = {"sh": 2, "lr": 1e-4}
        assert _config_hash(a) == _config_hash(b)

    def test_different_configs_differ(self) -> None:
        a = {"lr": 1e-4}
        b = {"lr": 1e-3}
        assert _config_hash(a) != _config_hash(b)


class TestAblationVary:
    def test_single_param(self) -> None:
        ab = Ablation("test", base_config={"lr": 1e-4})
        ab.vary("lr", [1e-3, 1e-4, 1e-5])
        configs = ab.generate()
        assert len(configs) == 3
        lr_values = {c["lr"] for c in configs}
        assert lr_values == {1e-3, 1e-4, 1e-5}

    def test_cartesian_product(self) -> None:
        ab = Ablation("test", base_config={"lr": 1e-4, "sh": 2})
        ab.vary("lr", [1e-3, 1e-4])
        ab.vary("sh", [0, 1, 2])
        configs = ab.generate()
        assert len(configs) == 6  # 2 * 3

    def test_dotted_key(self) -> None:
        ab = Ablation("test", base_config={"model": {"lr": 1e-4}})
        ab.vary("model.lr", [1e-3, 1e-4])
        configs = ab.generate()
        assert len(configs) == 2
        assert configs[0]["model"]["lr"] == 1e-3
        assert configs[1]["model"]["lr"] == 1e-4

    def test_empty_values_raises(self) -> None:
        ab = Ablation("test", base_config={})
        with pytest.raises(ValueError, match="at least one value"):
            ab.vary("lr", [])

    def test_method_chaining(self) -> None:
        ab = Ablation("test", base_config={})
        result = ab.vary("lr", [1e-3]).vary("sh", [2, 3])
        assert result is ab

    def test_no_variations_returns_base(self) -> None:
        ab = Ablation("test", base_config={"lr": 1e-4})
        configs = ab.generate()
        assert len(configs) == 1
        assert configs[0]["lr"] == 1e-4
        assert "_config_hash" in configs[0]

    def test_base_config_not_mutated(self) -> None:
        base = {"lr": 1e-4, "sh": 2}
        ab = Ablation("test", base_config=base)
        ab.vary("lr", [1e-3, 1e-5])
        ab.generate()
        assert base == {"lr": 1e-4, "sh": 2}


class TestAblationToggle:
    def test_toggle_bool(self) -> None:
        ab = Ablation("test", base_config={"dense": True})
        ab.toggle("dense")
        configs = ab.generate()
        assert len(configs) == 2
        values = {c["dense"] for c in configs}
        assert values == {True, False}

    def test_toggle_custom_values(self) -> None:
        ab = Ablation("test", base_config={})
        ab.toggle("mode", "fast", "slow")
        configs = ab.generate()
        assert {c["mode"] for c in configs} == {"fast", "slow"}


class TestAblationConfigHash:
    def test_all_configs_have_hash(self) -> None:
        ab = Ablation("test", base_config={"lr": 1e-4})
        ab.vary("lr", [1e-3, 1e-4])
        for config in ab.generate():
            assert "_config_hash" in config
            assert len(config["_config_hash"]) == 16

    def test_different_configs_different_hashes(self) -> None:
        ab = Ablation("test", base_config={"lr": 1e-4})
        ab.vary("lr", [1e-3, 1e-4])
        configs = ab.generate()
        hashes = {c["_config_hash"] for c in configs}
        assert len(hashes) == 2

    def test_hash_is_deterministic(self) -> None:
        ab = Ablation("test", base_config={"lr": 1e-4})
        ab.vary("lr", [1e-3, 1e-4])
        hashes1 = [c["_config_hash"] for c in ab.generate()]
        hashes2 = [c["_config_hash"] for c in ab.generate()]
        assert hashes1 == hashes2


class TestAblationExport:
    def test_export_creates_yaml_files(self, tmp_path: Path) -> None:
        ab = Ablation("test", base_config={"lr": 1e-4, "sh": 2})
        ab.vary("lr", [1e-3, 1e-4])
        paths = ab.export(tmp_path / "configs")

        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".yaml"

    def test_export_yaml_content(self, tmp_path: Path) -> None:
        import yaml

        ab = Ablation("test", base_config={"lr": 1e-4})
        ab.vary("lr", [1e-3])
        paths = ab.export(tmp_path / "configs")

        data = yaml.safe_load(paths[0].read_text())
        assert data["lr"] == 1e-3
        assert data["_ablation_name"] == "test"
        assert "_config_hash" in data

    def test_export_creates_directory(self, tmp_path: Path) -> None:
        out = tmp_path / "new" / "nested" / "dir"
        ab = Ablation("test", base_config={"x": 1})
        ab.export(out)
        assert out.exists()


class TestAblationYamlLoad:
    def test_load_yaml_base_config(self, tmp_path: Path) -> None:
        import yaml

        config_path = tmp_path / "base.yaml"
        config_path.write_text(yaml.dump({"lr": 1e-4, "sh": 2}))

        ab = Ablation("test", base_config=config_path)
        ab.vary("lr", [1e-3])
        configs = ab.generate()
        assert configs[0]["lr"] == 1e-3
        assert configs[0]["sh"] == 2

    def test_load_yaml_string_path(self, tmp_path: Path) -> None:
        import yaml

        config_path = tmp_path / "base.yaml"
        config_path.write_text(yaml.dump({"lr": 1e-4}))

        ab = Ablation("test", base_config=str(config_path))
        configs = ab.generate()
        assert configs[0]["lr"] == 1e-4

    def test_none_base_config(self) -> None:
        ab = Ablation("test")
        ab.vary("lr", [1e-3, 1e-4])
        configs = ab.generate()
        assert len(configs) == 2


class TestAblationSeedMeta:
    def test_seed_and_config_hash_in_meta(self, tmp_path: Path) -> None:
        from nuviz.config import NuvizConfig

        config = NuvizConfig(
            base_dir=tmp_path,
            flush_interval_seconds=0.1,
            flush_count=5,
            enable_alerts=False,
            enable_snapshot=False,
        )

        log = Logger(
            "test",
            config=config,
            snapshot=False,
            seed=42,
            config_hash="abc123",
        )
        log.step(0, loss=1.0)
        log.finish()

        meta = json.loads((log.experiment_dir / "meta.json").read_text())
        assert meta["seed"] == 42
        assert meta["config_hash"] == "abc123"

    def test_no_seed_no_field(self, tmp_path: Path) -> None:
        from nuviz.config import NuvizConfig

        config = NuvizConfig(
            base_dir=tmp_path,
            flush_interval_seconds=0.1,
            flush_count=5,
            enable_alerts=False,
            enable_snapshot=False,
        )

        log = Logger("test", config=config, snapshot=False)
        log.step(0, loss=1.0)
        log.finish()

        meta = json.loads((log.experiment_dir / "meta.json").read_text())
        assert "seed" not in meta
        assert "config_hash" not in meta
