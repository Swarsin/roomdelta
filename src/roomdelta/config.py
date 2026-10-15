"""Strict YAML configuration for M0 demonstrations and M1 frame preparation."""

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
class FrameExtractionConfig:
    frames_per_recording: int = 12
    candidate_stride: int = 5
    min_blur_score: float = 20.0
    min_time_separation_s: float = 0.5
    diversity_weight: float = 0.35
    target_width: int | None = 640
    target_height: int | None = 480
    resize_mode: str = "letterbox"
    aruco_enabled: bool = False
    aruco_dictionary: str = "DICT_4X4_50"
    marker_side_m: float | None = None

    def __post_init__(self) -> None:
        for name in ("frames_per_recording", "candidate_stride"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"extraction.{name} must be a positive integer")
        for name in ("min_blur_score", "min_time_separation_s", "diversity_weight"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"extraction.{name} must be finite and nonnegative")
        if self.diversity_weight > 1:
            raise ValueError("extraction.diversity_weight must be between 0 and 1")
        if (self.target_width is None) != (self.target_height is None):
            raise ValueError("extraction.target_width and target_height must both be set or both be null")
        for value in (self.target_width, self.target_height):
            if value is not None and (type(value) is not int or not 1 <= value <= 8192):
                raise ValueError("extraction target dimensions must be integers between 1 and 8192")
        if self.resize_mode not in ("letterbox", "stretch", "center-crop"):
            raise ValueError("extraction.resize_mode must be letterbox, stretch or center-crop")
        dictionaries = {f"DICT_{n}X{n}_{count}" for n in (4, 5, 6, 7) for count in (50, 100, 250, 1000)}
        if (type(self.aruco_enabled) is not bool or not isinstance(self.aruco_dictionary, str)
                or self.aruco_dictionary not in dictionaries | {"DICT_ARUCO_ORIGINAL"}):
            raise ValueError("invalid extraction ArUco enable flag or dictionary name")
        if self.marker_side_m is not None:
            if (isinstance(self.marker_side_m, bool) or not isinstance(self.marker_side_m, (int, float))
                    or not np.isfinite(self.marker_side_m) or self.marker_side_m <= 0):
                raise ValueError("extraction.marker_side_m must be a finite positive measured length in metres")
            if not self.aruco_enabled:
                raise ValueError("extraction.marker_side_m requires aruco_enabled: true")


@dataclass(frozen=True)
class RunConfig:
    schema_version: int = 1
    random_seed: int = 0
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    extraction: FrameExtractionConfig = field(default_factory=FrameExtractionConfig)

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
    values = _mapping(raw, {"schema_version", "random_seed", "visualization", "extraction"}, "config")
    visual = _mapping(values.get("visualization", {}), {"application_id", "point_spacing_m"}, "visualization")
    extraction = _mapping(values.get("extraction", {}), set(FrameExtractionConfig.__dataclass_fields__), "extraction")
    return RunConfig(
        schema_version=values.get("schema_version", 1),
        random_seed=values.get("random_seed", 0),
        visualization=VisualizationConfig(**visual),
        extraction=FrameExtractionConfig(**extraction),
    )
