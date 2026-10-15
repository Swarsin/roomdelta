"""Malformed inputs must fail before later stages can interpret them as evidence."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from roomdelta.contracts import (
    Calibration, CandidateType, ChangeCandidate, ChangeEvent, ChangesDocument,
    EvidenceStatus, EventType, FrameRecord, ModelInfo, ReconstructionCache,
    ScaleStatus, VoxelEvidence, VoxelGrid,
)


@pytest.fixture
def calibration() -> Calibration:
    return Calibration(ScaleStatus.SYNTHETIC, np.eye(4), "analytic fixture", 0)


@pytest.fixture
def model_info() -> ModelInfo:
    return ModelInfo("synthetic", None, None, "cpu", 0)


@pytest.fixture
def cache(calibration: Calibration, model_info: ModelInfo) -> ReconstructionCache:
    frames = tuple(FrameRecord("test", t, Path(f"{t}.png"), f"{t}.mp4", 0, (4, 3), (4, 3), np.eye(3), i)
                   for i, t in enumerate(("before", "after")))
    return ReconstructionCache(frames, np.tile(np.eye(3), (2, 1, 1)), np.tile(np.eye(4), (2, 1, 1)),
                               np.ones((2, 3, 4)), np.ones((2, 3, 4), dtype=bool), calibration, model_info)


def test_valid_cache_contract_and_explicit_synthetic_units(cache: ReconstructionCache) -> None:
    assert cache.timepoints == ("before", "after")
    assert cache.calibration.unit == "m"
    assert cache.model_info.model_name is None


@pytest.mark.parametrize("fault", ["nan_depth", "negative_depth", "mask_shape", "mask_dtype", "K_shape", "pose_reflection", "ordering", "resolution", "invalid_frame", "schema", "confidence"])
def test_invalid_reconstruction_contracts(cache: ReconstructionCache, fault: str) -> None:
    changes: dict[str, object] = {}
    if fault in ("nan_depth", "negative_depth"):
        depth = cache.depth.copy()
        depth[0, 0, 0] = np.nan if fault == "nan_depth" else -1
        changes["depth"] = depth
    elif fault == "mask_shape":
        changes["depth_valid"] = np.ones((2, 4, 3), dtype=bool)
    elif fault == "mask_dtype":
        changes["depth_valid"] = np.ones((2, 3, 4), dtype=np.uint8)
    elif fault == "K_shape":
        changes["K"] = np.eye(3)
    elif fault == "pose_reflection":
        poses = cache.T_cw.copy()
        poses[0, 0, 0] = -1
        changes["T_cw"] = poses
    elif fault == "ordering":
        changes["frames"] = cache.frames[::-1]
    elif fault == "resolution":
        changes["frames"] = tuple(replace(f, processed_size_wh=(8, 6)) for f in cache.frames)
    elif fault == "invalid_frame":
        changes["frames"] = (replace(cache.frames[0], valid=False), cache.frames[1])
    elif fault == "schema":
        changes["schema_version"] = "unsupported"
    else:
        changes["depth_confidence"] = np.ones_like(cache.depth)
    with pytest.raises(ValueError):
        replace(cache, **changes)


def test_zero_depth_is_allowed_only_when_masked_invalid(cache: ReconstructionCache) -> None:
    depth = cache.depth.copy()
    mask = cache.depth_valid.copy()
    depth[0, 0, 0] = 0
    mask[0, 0, 0] = False
    valid = replace(cache, depth=depth, depth_valid=mask)
    assert not valid.depth_valid[0, 0, 0]
    with pytest.raises(ValueError, match="positive"):
        replace(cache, depth=depth)


def test_uncalibrated_cache_keeps_arbitrary_units(cache: ReconstructionCache) -> None:
    calibration = Calibration(ScaleStatus.UNCALIBRATED, np.eye(4), "unsolved")
    info = ModelInfo("real", "not-executed-contract-example", "revision-placeholder", "cpu", 0)
    contract = replace(cache, calibration=calibration, model_info=info)
    assert contract.calibration.unit == "arbitrary"
    with pytest.raises(ValueError, match="metric"):
        ChangesDocument("test", info, calibration, ())


def test_similarity_accepts_uniform_scale_but_rejects_nonuniform(calibration: Calibration) -> None:
    similarity = np.eye(4)
    similarity[:3, :3] *= 2
    similarity[:3, 3] = [1, 2, 3]
    valid = replace(calibration, model_to_world=similarity)
    assert valid.model_to_world[0, 0] == 2
    similarity[0, 0] = 3
    with pytest.raises(ValueError, match="orthonormal"):
        replace(calibration, model_to_world=similarity)


def test_measured_calibration_requires_uncertainty() -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        Calibration(ScaleStatus.MEASURED, np.eye(4), "measurement")


def test_synthetic_labels_cannot_be_silently_mixed(cache: ReconstructionCache) -> None:
    with pytest.raises(ValueError, match="provenance"):
        replace(cache, model_info=replace(cache.model_info, data_kind="real"))


def test_compact_voxel_contract_and_bounds(calibration: Calibration) -> None:
    shape = (2, 3, 4)
    zero = np.zeros(shape, dtype=np.uint16)
    evidence = VoxelEvidence(zero, zero, zero.astype(np.uint8), zero, {0: "group-a"}, {"status": "synthetic"})
    grid = VoxelGrid(np.zeros(3), np.array([[0, 0, 0], [1, 1.5, 2]]), 0.5, shape,
                     {"before": evidence, "after": evidence}, calibration)
    assert grid.evidence["before"].state.shape == shape
    with pytest.raises(ValueError, match="bounds"):
        replace(grid, bounds_m=np.array([[0, 0, 0], [2, 2, 2]]))
    with pytest.raises(ValueError, match="metric"):
        replace(grid, calibration=Calibration(ScaleStatus.UNCALIBRATED, np.eye(4), "unsolved"))
    with pytest.raises(ValueError, match="state code"):
        replace(grid, evidence={"before": replace(evidence, state=np.full(shape, 99)), "after": evidence})


def test_candidate_is_unverified_and_context_checks_indices() -> None:
    candidate = ChangeCandidate("fixture", CandidateType.UNKNOWN, np.array([[1, 2, 3]]),
                                np.ones(3), np.array([[0, 0, 0], [2, 2, 2]]), (0,), {}, ("synthetic contract",))
    candidate.validate_context((2, 3, 4), 1)
    with pytest.raises(ValueError, match="voxel index"):
        candidate.validate_context((1, 3, 4), 1)
    with pytest.raises(ValueError, match="supporting view"):
        candidate.validate_context((2, 3, 4), 0)
    with pytest.raises(ValueError, match="centroid"):
        replace(candidate, centroid_m=np.array([9, 9, 9]))


def test_unknown_event_abstains_and_cannot_be_confirmed() -> None:
    event = ChangeEvent("unknown", EventType.UNKNOWN, EvidenceStatus.ABSTAINED,
                        np.ones(3), ("insufficient positive evidence",))
    assert event.semantic_label == "unknown object"
    with pytest.raises(ValueError, match="concrete"):
        replace(event, status=EvidenceStatus.CONFIRMED, supporting_view_ids=(0,))


def test_moved_event_requires_both_positions() -> None:
    with pytest.raises(ValueError, match="before centroid"):
        ChangeEvent("move", EventType.MOVED, EvidenceStatus.CANDIDATE, np.ones(3), ("fixture",))


def test_changes_document_disallows_duplicate_ids(calibration: Calibration, model_info: ModelInfo) -> None:
    event = ChangeEvent("id", EventType.UNKNOWN, EvidenceStatus.ABSTAINED, np.ones(3), ("fixture",))
    with pytest.raises(ValueError, match="unique"):
        ChangesDocument("test", model_info, calibration, (event, event))
