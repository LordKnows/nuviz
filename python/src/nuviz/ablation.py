"""Declarative ablation experiment configuration generator."""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any


def _deep_set(d: dict[str, Any], dotted_key: str, value: Any) -> dict[str, Any]:
    """Set a value in a nested dict using dot-notation key. Returns a new dict."""
    result = copy.deepcopy(d)
    keys = dotted_key.split(".")
    current = result
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    return result


def _config_hash(config: dict[str, Any]) -> str:
    """Compute a deterministic SHA256 hash for a config dict."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class Ablation:
    """Declarative ablation experiment definition.

    Generates the cartesian product of all varied parameters, merged into
    deep copies of a base configuration. Does not launch experiments —
    only generates and exports configs.

    Usage::

        ab = Ablation("3dgs_ablation", base_config={"lr": 1e-4, "sh": 2})
        ab.vary("lr", [1e-3, 1e-4, 1e-5])
        ab.vary("sh", [0, 1, 2, 3])

        configs = ab.generate()  # list of config dicts
        ab.export("configs/ablation/")  # writes YAML files
    """

    def __init__(
        self,
        name: str,
        base_config: dict[str, Any] | str | Path | None = None,
    ) -> None:
        self._name = name
        self._variations: list[tuple[str, list[Any]]] = []

        if base_config is None:
            self._base_config: dict[str, Any] = {}
        elif isinstance(base_config, dict):
            self._base_config = copy.deepcopy(base_config)
        else:
            self._base_config = self._load_yaml(Path(base_config))

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        """Load a YAML file. Requires PyYAML."""
        try:
            import yaml
        except ImportError:
            msg = (
                "PyYAML is required to load YAML config files. "
                "Install it with: pip install nuviz[yaml]"
            )
            raise ImportError(msg) from None

        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            msg = f"Expected YAML file to contain a mapping, got {type(data).__name__}"
            raise ValueError(msg)
        return data

    @property
    def name(self) -> str:
        """Ablation study name."""
        return self._name

    def vary(self, param: str, values: list[Any]) -> Ablation:
        """Register a parameter sweep.

        Args:
            param: Parameter name. Use dot notation for nested keys
                   (e.g., "model.lr", "training.batch_size").
            values: List of values to sweep.

        Returns:
            Self for method chaining.
        """
        if not values:
            msg = f"vary() requires at least one value for param '{param}'"
            raise ValueError(msg)
        self._variations.append((param, list(values)))
        return self

    def toggle(
        self,
        param: str,
        true_val: Any = True,
        false_val: Any = False,
    ) -> Ablation:
        """Register a binary toggle parameter.

        Convenience method equivalent to ``vary(param, [true_val, false_val])``.

        Returns:
            Self for method chaining.
        """
        return self.vary(param, [true_val, false_val])

    def generate(self) -> list[dict[str, Any]]:
        """Generate the cartesian product of all varied parameters.

        Each config dict is a deep copy of the base config with the
        varied parameters overridden. A ``_config_hash`` field is added
        for deterministic experiment grouping.

        Returns:
            List of config dicts.
        """
        if not self._variations:
            config = copy.deepcopy(self._base_config)
            config["_config_hash"] = _config_hash(self._base_config)
            return [config]

        param_names = [v[0] for v in self._variations]
        value_lists = [v[1] for v in self._variations]

        configs: list[dict[str, Any]] = []
        for combo in itertools.product(*value_lists):
            config = copy.deepcopy(self._base_config)
            for param, value in zip(param_names, combo, strict=True):
                config = _deep_set(config, param, value)
            config["_config_hash"] = _config_hash(
                {k: v for k, v in config.items() if k != "_config_hash"}
            )
            configs.append(config)

        return configs

    def export(self, directory: str | Path) -> list[Path]:
        """Export generated configs as YAML files.

        Args:
            directory: Output directory. Created if it doesn't exist.

        Returns:
            List of written file paths.

        Raises:
            ImportError: If PyYAML is not installed.
        """
        try:
            import yaml
        except ImportError:
            msg = (
                "PyYAML is required to export YAML config files. "
                "Install it with: pip install nuviz[yaml]"
            )
            raise ImportError(msg) from None

        out_dir = Path(directory)
        out_dir.mkdir(parents=True, exist_ok=True)

        configs = self.generate()
        paths: list[Path] = []

        for config in configs:
            filename = self._config_filename(config)
            path = out_dir / filename
            # Write config without _config_hash in the YAML body
            export_config = {k: v for k, v in config.items() if k != "_config_hash"}
            export_config["_ablation_name"] = self._name
            export_config["_config_hash"] = config["_config_hash"]
            with open(path, "w") as f:
                yaml.dump(export_config, f, default_flow_style=False, sort_keys=False)
            paths.append(path)

        return paths

    def _config_filename(self, config: dict[str, Any]) -> str:
        """Generate a descriptive filename from varied parameter values."""
        parts = [self._name]
        for param, _ in self._variations:
            keys = param.split(".")
            value = config
            for k in keys:
                value = value[k]
            # Format value for filename
            if isinstance(value, float):
                parts.append(f"{param.replace('.', '-')}_{value:g}")
            else:
                parts.append(f"{param.replace('.', '-')}_{value}")
        return "_".join(parts) + ".yaml"
