"""Small, strict YAML configuration for M0; later-stage knobs are not invented."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class VisualizationConfig:
    application_id: str = "roomdelta-m0-synthetic"
    point_spacing_m: float = 0.2

    def __post_init__(self) -> None:
        if not isinstance(self.application_id, str) or not self.application_id.strip():
            raise ValueError("visualization.application_id must be a nonempty string")
        if (isinstance(self.point_spacing_m, bool) or not isinstance(self.point_spacing_m, (int, float))
                or not np.isfinite(self.point_spacing_m) or not 0.02 <= self.point_spacing_m <= 1):
            raise ValueError("visualization.point_spacing_m must be between 0.02 and 1 metre")


@dataclass(frozen=True)
class RunConfig:
    schema_version: int = 1
    random_seed: int = 0
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported config schema_version (expected 1)")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")


def _mapping(value: Any, allowed: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"unknown {name} keys: {', '.join(sorted(unknown))}")
    return value


def load_config(path: Path | None = None) -> RunConfig:
    """Load safe YAML, rejecting misspelled options; None uses built-in defaults."""
    if path is None:
        return RunConfig()
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML in {path}: {exc}") from exc
    values = _mapping(raw, {"schema_version", "random_seed", "visualization"}, "config")
    visual = _mapping(values.get("visualization", {}), {"application_id", "point_spacing_m"}, "visualization")
    return RunConfig(
        schema_version=values.get("schema_version", 1),
        random_seed=values.get("random_seed", 0),
        visualization=VisualizationConfig(**visual),
    )
