"""Validated shared contracts; no reconstruction or change inference is performed.

Arrays are NumPy arrays on CPU. Dataclasses are frozen, but array buffers are not;
consumers must treat validated arrays as read-only and revalidate after editing.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray

from roomdelta.geometry import (
    FloatArray, finite_array, validate_intrinsics, validate_pixel_transform, validate_pose,
)

SCHEMA_VERSION = "roomdelta/1"
COORDINATE_SYSTEM = "right-handed world: +X right, +Y forward, +Z up; OpenCV camera"
Timepoint = Literal["before", "after"]


class ScaleStatus(str, Enum):
    UNCALIBRATED = "uncalibrated"
    MEASURED = "measured"
    SYNTHETIC = "synthetic"


class CandidateType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    APPEARANCE = "appearance"
    UNKNOWN = "unknown"


class EventType(str, Enum):
    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    REPLACEMENT_CANDIDATE = "replacement-candidate"
    UNKNOWN = "unknown"


class EvidenceStatus(str, Enum):
    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    ABSTAINED = "abstained"


class VoxelState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2
    CONFLICTING = 3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _nonnegative(value: float, name: str) -> None:
    _require(bool(np.isfinite(value) and value >= 0), f"{name} must be finite and nonnegative")


def _size_wh(size: tuple[int, int], name: str) -> None:
    _require(len(size) == 2 and all(type(v) is int and v > 0 for v in size),
             f"{name} must contain positive integer width and height")


def _view_ids(ids: tuple[int, ...]) -> None:
    _require(all(type(i) is int and i >= 0 for i in ids) and len(set(ids)) == len(ids),
             "supporting view IDs must be unique nonnegative integers")


def _bounds(value: object, name: str) -> FloatArray:
    bounds = finite_array(value, (2, 3), name)
    _require(bool((bounds[1] >= bounds[0]).all()), f"{name} max must be >= min")
    return bounds


@dataclass(frozen=True)
class FrameRecord:
    """Manifest entry; dimensions are (W,H), timestamp is seconds from source start.

    pixel_transform[3,3] maps original pixel coordinates to processed coordinates.
    index is the stable zero-based index into every reconstruction frame array.
    """

    scenario: str
    timepoint: Timepoint
    file_path: Path
    source_filename: str
    timestamp_s: float
    original_size_wh: tuple[int, int]
    processed_size_wh: tuple[int, int]
    pixel_transform: FloatArray
    index: int
    valid: bool = True
    blur_score: float | None = None
    quality_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require(bool(self.scenario and self.source_filename), "scenario and source_filename are required")
        _require(self.timepoint in ("before", "after"), "timepoint must be before or after")
        _require(isinstance(self.file_path, Path), "file_path must be a pathlib.Path")
        _require(type(self.index) is int and self.index >= 0, "frame index must be a nonnegative integer")
        _require(type(self.valid) is bool, "frame valid must be bool")
        _nonnegative(self.timestamp_s, "timestamp_s")
        _size_wh(self.original_size_wh, "original_size_wh")
        _size_wh(self.processed_size_wh, "processed_size_wh")
        object.__setattr__(self, "pixel_transform", validate_pixel_transform(self.pixel_transform))
        if self.blur_score is not None:
            _nonnegative(self.blur_score, "blur_score")


@dataclass(frozen=True)
class Calibration:
    """Model-to-world Sim(3), scale label and relative scale uncertainty (fraction).

    model_to_world applies to raw model points, not to already converted arrays.
    MEASURED and SYNTHETIC contracts use metres; UNCALIBRATED uses arbitrary units.
    No scale estimator is implemented in M0.
    """

    status: ScaleStatus
    model_to_world: FloatArray
    method: str
    relative_scale_uncertainty: float | None = None

    def __post_init__(self) -> None:
        _require(isinstance(self.status, ScaleStatus), "status must be a ScaleStatus")
        _require(bool(self.method.strip()), "calibration method is required")
        transform = finite_array(self.model_to_world, (4, 4), "model_to_world")
        scale = float(np.cbrt(np.linalg.det(transform[:3, :3])))
        _require(scale > 0, "model_to_world must have positive scale and no reflection")
        rigid = transform.copy()
        rigid[:3, :3] /= scale
        validate_pose(rigid)
        object.__setattr__(self, "model_to_world", transform)
        if self.relative_scale_uncertainty is not None:
            _nonnegative(self.relative_scale_uncertainty, "relative_scale_uncertainty")
        if self.status is ScaleStatus.MEASURED:
            _require(self.relative_scale_uncertainty is not None,
                     "measured calibration must report scale uncertainty")

    @property
    def unit(self) -> Literal["m", "arbitrary"]:
        return "arbitrary" if self.status is ScaleStatus.UNCALIBRATED else "m"


@dataclass(frozen=True)
class ModelInfo:
    """Provenance to retain in cache metadata and future cache-key inputs.

    Confidence interpretation is model-specific, never assumed to be probability.
    input_sha256 maps source identifiers to hashes; M2 must populate real provenance.
    """

    data_kind: Literal["real", "synthetic"]
    model_name: str | None
    checkpoint_revision: str | None
    execution_device: str
    random_seed: int
    preprocessing: Mapping[str, Any] = field(default_factory=dict)
    dependency_versions: Mapping[str, str] = field(default_factory=dict)
    input_sha256: Mapping[str, str] = field(default_factory=dict)
    confidence_interpretation: str | None = None

    def __post_init__(self) -> None:
        _require(self.data_kind in ("real", "synthetic"), "data_kind must be real or synthetic")
        _require(bool(self.execution_device), "execution_device is required")
        _require(type(self.random_seed) is int and self.random_seed >= 0, "random_seed must be nonnegative integer")
        _require((self.model_name is None) == (self.checkpoint_revision is None),
                 "model name and checkpoint revision must both be provided or both be absent")
        for digest in self.input_sha256.values():
            _require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
                     "input hashes must be lowercase SHA-256 hex digests")


@dataclass(frozen=True)
class ReconstructionCache:
    """Normalized cameras/depths; units follow calibration, all arrays are CPU-side.

    K[N,3,3] matches depth[N,H,W] and each frame's processed dimensions.
    T_cw[N,4,4] is rigid, world-to-camera; depth is camera-z, not ray length.
    Invalid depth pixels are finite placeholders (usually 0) with a false mask.
    Optional points[P,3] are world-space; confidence has no assumed [0,1] range.
    """

    frames: tuple[FrameRecord, ...]
    K: FloatArray
    T_cw: FloatArray
    depth: FloatArray
    depth_valid: NDArray[np.bool_]
    calibration: Calibration
    model_info: ModelInfo
    depth_confidence: FloatArray | None = None
    points_world: FloatArray | None = None
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require(self.schema_version == SCHEMA_VERSION, "unsupported reconstruction schema_version")
        count = len(self.frames)
        _require(count > 0, "reconstruction must contain frames")
        _require(tuple(f.index for f in self.frames) == tuple(range(count)),
                 "frame indices must be contiguous and match array ordering")
        _require(len({f.scenario for f in self.frames}) == 1, "cache frames must belong to one scenario")
        sizes = {f.processed_size_wh for f in self.frames}
        _require(len(sizes) == 1, "stored depth arrays require a common processed resolution")
        width, height = self.frames[0].processed_size_wh
        object.__setattr__(self, "K", finite_array(self.K, (count, 3, 3), "K"))
        object.__setattr__(self, "T_cw", finite_array(self.T_cw, (count, 4, 4), "T_cw"))
        for K, pose in zip(self.K, self.T_cw, strict=True):
            validate_intrinsics(K)
            validate_pose(pose)
        shape = (count, height, width)
        object.__setattr__(self, "depth", finite_array(self.depth, shape, "depth"))
        mask = np.asarray(self.depth_valid)
        _require(mask.shape == shape and mask.dtype == np.bool_, "depth_valid must be bool with shape [N,H,W]")
        _require(bool((self.depth[mask] > 0).all()), "valid depth values must be positive")
        for index, frame in enumerate(self.frames):
            _require(frame.valid or not bool(mask[index].any()), "invalid frame cannot contain valid depth")
        object.__setattr__(self, "depth_valid", mask)
        if self.depth_confidence is not None:
            _require(bool(self.model_info.confidence_interpretation), "confidence interpretation is required")
            object.__setattr__(self, "depth_confidence", finite_array(self.depth_confidence, shape, "depth_confidence"))
        if self.points_world is not None:
            points = np.asarray(self.points_world, dtype=np.float64)
            _require(points.ndim == 2 and points.shape[1] == 3 and bool(np.isfinite(points).all()),
                     "points_world must be finite with shape [P,3]")
            object.__setattr__(self, "points_world", points)
        _require((self.calibration.status is ScaleStatus.SYNTHETIC) == (self.model_info.data_kind == "synthetic"),
                 "synthetic calibration and synthetic provenance must agree")

    @property
    def timepoints(self) -> tuple[Timepoint, ...]:
        return tuple(frame.timepoint for frame in self.frames)


@dataclass(frozen=True)
class VoxelEvidence:
    """Compact [nx,ny,nz] counts/labels, not a voxel-by-frame dense tensor.

    Counts alone are not independent observations. view_groups and quality metadata
    preserve provenance; reason_flags is a uint bit field whose meanings M3 defines.
    """

    occupied_views: NDArray[np.integer]
    free_views: NDArray[np.integer]
    state: NDArray[np.integer]
    reason_flags: NDArray[np.unsignedinteger]
    view_groups: Mapping[int, str]
    quality: Mapping[str, Any]

    def validate(self, shape: tuple[int, int, int]) -> None:
        for name in ("occupied_views", "free_views", "state", "reason_flags"):
            array = np.asarray(getattr(self, name))
            _require(array.shape == shape and np.issubdtype(array.dtype, np.integer),
                     f"{name} must be an integer array with voxel grid shape")
            _require(bool((array >= 0).all()), f"{name} cannot contain negative values")
        _require(bool(np.isin(self.state, [v.value for v in VoxelState]).all()), "invalid voxel state code")
        _require(np.issubdtype(np.asarray(self.reason_flags).dtype, np.unsignedinteger),
                 "reason_flags must use an unsigned integer dtype")
        _view_ids(tuple(self.view_groups))


@dataclass(frozen=True)
class VoxelGrid:
    """Shared metric lattice; origin is the minimum corner, indices are (x,y,z).

    Centre(i) = origin_m + (i + 0.5) * voxel_size_m. Bounds cover the entire lattice.
    Relative-unit reconstructions must be calibrated before constructing this grid.
    """

    origin_m: FloatArray
    bounds_m: FloatArray
    voxel_size_m: float
    shape: tuple[int, int, int]
    evidence: Mapping[Timepoint, VoxelEvidence]
    calibration: Calibration

    def __post_init__(self) -> None:
        origin = finite_array(self.origin_m, (3,), "origin_m")
        bounds = _bounds(self.bounds_m, "bounds_m")
        _require(bool(np.isfinite(self.voxel_size_m) and self.voxel_size_m > 0), "voxel_size_m must be positive")
        _require(len(self.shape) == 3 and all(type(v) is int and v > 0 for v in self.shape), "invalid grid shape")
        _require(self.calibration.unit == "m", "VoxelGrid requires calibrated metric coordinates")
        _require(bool(np.allclose(bounds[0], origin) and np.allclose(bounds[1], origin + np.array(self.shape) * self.voxel_size_m)),
                 "bounds must match origin, shape and voxel size")
        _require(set(self.evidence) == {"before", "after"}, "grid requires before and after evidence")
        for evidence in self.evidence.values():
            evidence.validate(self.shape)
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "bounds_m", bounds)


@dataclass(frozen=True)
class ChangeCandidate:
    """Unverified region; voxel_indices[M,3] and metric centroid/bounds[2,3]."""

    id: str
    type: CandidateType
    voxel_indices: NDArray[np.integer]
    centroid_m: FloatArray
    bounds_m: FloatArray
    supporting_view_ids: tuple[int, ...]
    static_alignment_diagnostics: Mapping[str, Any]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _require(bool(self.id) and isinstance(self.type, CandidateType), "candidate ID and CandidateType are required")
        indices = np.asarray(self.voxel_indices)
        _require(indices.ndim == 2 and indices.shape[1] == 3 and len(indices) > 0
                 and np.issubdtype(indices.dtype, np.integer) and bool((indices >= 0).all()),
                 "voxel_indices must be nonnegative integer [M,3], M > 0")
        centroid = finite_array(self.centroid_m, (3,), "centroid_m")
        bounds = _bounds(self.bounds_m, "bounds_m")
        _require(bool(((centroid >= bounds[0]) & (centroid <= bounds[1])).all()), "centroid must lie inside bounds")
        _view_ids(self.supporting_view_ids)
        object.__setattr__(self, "voxel_indices", indices)
        object.__setattr__(self, "centroid_m", centroid)
        object.__setattr__(self, "bounds_m", bounds)

    def validate_context(self, grid_shape: tuple[int, int, int], frame_count: int) -> None:
        """Check indices against the actual originating grid and frame manifest."""
        _require(bool((self.voxel_indices < np.array(grid_shape)).all()), "voxel index outside grid")
        _require(all(i < frame_count for i in self.supporting_view_ids), "supporting view outside frame manifest")


@dataclass(frozen=True)
class ChangeEvent:
    """Event interface, not an inference result; confidence is not probability."""

    id: str
    type: EventType
    status: EvidenceStatus
    size_m: FloatArray
    evidence_reasons: tuple[str, ...]
    supporting_view_ids: tuple[int, ...] = ()
    semantic_label: str = "unknown object"
    before_centroid_m: FloatArray | None = None
    after_centroid_m: FloatArray | None = None
    disposition: str = "unverified"
    crop_paths: tuple[Path, ...] = ()
    rerun_entity_id: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        _require(bool(self.id and self.semantic_label and self.disposition), "event ID, label and disposition are required")
        _require(isinstance(self.type, EventType) and isinstance(self.status, EvidenceStatus), "invalid event type/status")
        size = finite_array(self.size_m, (3,), "size_m")
        _require(bool((size >= 0).all()), "size_m must be nonnegative")
        object.__setattr__(self, "size_m", size)
        _view_ids(self.supporting_view_ids)
        for name in ("before_centroid_m", "after_centroid_m"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, finite_array(value, (3,), name))
        if self.type in (EventType.REMOVED, EventType.MOVED):
            _require(self.before_centroid_m is not None, "removed/moved event requires before centroid")
        if self.type in (EventType.ADDED, EventType.MOVED):
            _require(self.after_centroid_m is not None, "added/moved event requires after centroid")
        _require(bool(self.evidence_reasons), "events must record evidence/abstention reasons")
        if self.status is EvidenceStatus.CONFIRMED:
            _require(self.type not in (EventType.UNKNOWN, EventType.REPLACEMENT_CANDIDATE)
                     and bool(self.supporting_view_ids), "confirmed events require a supported concrete event type")
        if self.confidence is not None:
            _nonnegative(self.confidence, "confidence (not calibrated probability)")
        _require(all(isinstance(path, Path) for path in self.crop_paths), "crop_paths must contain pathlib.Paths")


@dataclass(frozen=True)
class ChangesDocument:
    """Future changes.json top-level contract; M0 does not produce change results."""

    source_scenario: str
    model_info: ModelInfo
    calibration: Calibration
    events: tuple[ChangeEvent, ...]
    schema_version: str = SCHEMA_VERSION
    coordinate_system: str = COORDINATE_SYSTEM
    unit: Literal["m"] = "m"

    def __post_init__(self) -> None:
        _require(self.schema_version == SCHEMA_VERSION and self.coordinate_system == COORDINATE_SYSTEM,
                 "unsupported changes schema/coordinate system")
        _require(self.unit == "m" and self.calibration.unit == "m", "metric events require metric calibration")
        _require(bool(self.source_scenario), "source_scenario is required")
        _require(len({event.id for event in self.events}) == len(self.events), "event IDs must be unique")
        _require((self.calibration.status is ScaleStatus.SYNTHETIC) == (self.model_info.data_kind == "synthetic"),
                 "synthetic calibration and synthetic provenance must agree")
