"""Analytic image/video fixtures only: no real camera motion/metric measurements."""

from dataclasses import replace
from importlib.resources import files
import json
from pathlib import Path
import subprocess
import sys

import cv2
import jsonschema
import numpy as np
import pytest

from roomdelta.config import FrameExtractionConfig, load_config
from roomdelta.frames import (
    FrameCandidate, VideoPreparationError, appearance_distance, detect_markers, frame_quality,
    ground_truth_template, load_frame_records, prepare_frames, preprocess_image, scan_video, select_frames, sha256_file,
)
from roomdelta.geometry import transform_intrinsics

ROOT = Path(__file__).resolve().parents[1]


def synthetic_image(index: int) -> np.ndarray:
    """Moving 2D graphics and an ArUco marker; this is not a 3D camera trajectory."""
    image = np.full((240, 320, 3), 245, dtype=np.uint8)
    for y in range(0, 240, 12):
        for x in range(0, 140, 12):
            if (x // 12 + y // 12) % 2:
                image[y:y + 12, x:x + 12] = [35, 80, 110]
    cv2.circle(image, (40 + index % 90, 60 + index % 110), 22, (60, 160, 240), -1)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 80)
    image[65:165, 185:285] = 255
    image[75:155, 195:275] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, f"SYNTHETIC {index}", (145, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return image


def make_video(path: Path, count: int = 90, offset: int = 0, blank: bool = False) -> None:
    """Write an actual MJPEG AVI fixture at 10 FPS; fail if encoder unavailable."""
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    assert writer.isOpened(), "MJPEG writer unavailable in this OpenCV build"
    try:
        for index in range(count):
            image = np.full((240, 320, 3), 128, dtype=np.uint8) if blank else synthetic_image(index + offset)
            if not blank and index < 10:
                image = cv2.GaussianBlur(image, (41, 41), 12)
            writer.write(image)
    finally:
        writer.release()


@pytest.fixture(scope="module")
def videos(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    directory = tmp_path_factory.mktemp("synthetic_videos")
    before, after = directory / "before.avi", directory / "after.avi"
    make_video(before)
    make_video(after, offset=13)
    return before, after


def test_quality_rejects_blurred_image() -> None:
    image = synthetic_image(20)
    sharp, descriptor = frame_quality(image)
    blurred, _ = frame_quality(cv2.GaussianBlur(image, (41, 41), 12))
    assert sharp > blurred * 10
    assert descriptor.shape == (1024,)
    assert np.isfinite(descriptor).all()


@pytest.mark.parametrize("mode,resized,padding,crop", [
    ("letterbox", (128, 96), (0, 16, 0, 16), (0, 0, 0, 0)),
    ("stretch", (128, 128), (0, 0, 0, 0), (0, 0, 0, 0)),
    ("center-crop", (171, 128), (0, 0, 0, 0), (21, 0, 22, 0)),
])
def test_preprocessing_records_actual_integer_geometry(mode: str, resized: tuple[int, int], padding: tuple[int, ...], crop: tuple[int, ...]) -> None:
    processed = preprocess_image(synthetic_image(0), FrameExtractionConfig(target_width=128, target_height=128, resize_mode=mode))
    assert processed.image.shape == (128, 128, 3)
    assert processed.resized_size_wh == resized
    assert processed.padding_ltrb == padding
    assert processed.crop_ltrb == crop
    sx, sy = resized[0] / 320, resized[1] / 240
    original = np.array([195, 75, 1.0])
    mapped = processed.pixel_transform @ original
    np.testing.assert_allclose(mapped[:2], [sx * 195 + (sx - 1) / 2 + padding[0] - crop[0],
                                          sy * 75 + (sy - 1) / 2 + padding[1] - crop[1]])
    K = np.array([[200, 0, 159.5], [0, 200, 119.5], [0, 0, 1]])
    adjusted = transform_intrinsics(K, processed.pixel_transform)
    np.testing.assert_allclose(adjusted[0, 0], 200 * sx)
    np.testing.assert_allclose(adjusted[1, 1], 200 * sy)
    if mode == "letterbox":
        assert (processed.image[:16] == 0).all()


def test_linear_resize_coordinate_ramp_matches_half_pixel_affine() -> None:
    # Independent sampled-image check of the affine, rather than just recomputing it.
    ramp = np.tile(np.arange(100, dtype=np.uint8)[None, :, None], (60, 1, 3))
    processed = preprocess_image(ramp, FrameExtractionConfig(target_width=50, target_height=30, resize_mode="stretch"))
    for column in (3, 10, 25, 40):
        original_x = np.linalg.inv(processed.pixel_transform) @ [column, 10, 1]
        assert abs(float(processed.image[10, column, 0]) - original_x[0]) <= 0.51


def test_native_resolution_is_identity() -> None:
    image = synthetic_image(1)
    processed = preprocess_image(image, FrameExtractionConfig(target_width=None, target_height=None))
    np.testing.assert_array_equal(processed.image, image)
    np.testing.assert_array_equal(processed.pixel_transform, np.eye(3))


def candidates(scores: list[float], times: list[float] | None = None, values: list[float] | None = None) -> tuple[FrameCandidate, ...]:
    times = times if times is not None else [float(i) for i in range(len(scores))]
    values = values if values is not None else [0.0] * len(scores)
    return tuple(FrameCandidate(i, t, (320, 240), s, np.full(1024, v)) for i, (s, t, v) in enumerate(zip(scores, times, values, strict=True)))


def test_temporal_bins_choose_sharp_frames_across_sequence() -> None:
    config = FrameExtractionConfig(frames_per_recording=3, min_blur_score=1, min_time_separation_s=0, diversity_weight=0)
    selected = select_frames(candidates([10, 100, 10, 100, 10, 100]), config)
    assert [candidate.source_frame_index for candidate in selected] == [1, 3, 5]


def test_thumbnail_difference_prioritizes_appearance_not_claimed_pose() -> None:
    config = FrameExtractionConfig(frames_per_recording=2, min_blur_score=1, min_time_separation_s=0, diversity_weight=1)
    pool = candidates([100] * 4, values=[0, 0, 0, 1])
    selected = select_frames(pool, config)
    assert [candidate.source_frame_index for candidate in selected] == [0, 3]
    assert appearance_distance(pool[0], pool[3]) == 1


def test_stationary_recording_not_rejected_for_image_similarity() -> None:
    config = FrameExtractionConfig(frames_per_recording=3, min_blur_score=1)
    selected = select_frames(candidates([100] * 6), config)
    assert len(selected) == 3
    assert all(appearance_distance(a, b) == 0 for a, b in zip(selected, selected[1:]))


def test_spacing_fallback_finds_feasible_count_after_bad_greedy_choice() -> None:
    config = FrameExtractionConfig(frames_per_recording=3, min_blur_score=1, min_time_separation_s=1, diversity_weight=0)
    selected = select_frames(candidates([20, 100, 20, 20], times=[0, 0.5, 1, 2]), config)
    assert [candidate.timestamp_s for candidate in selected] == [0, 1, 2]


def test_too_few_sharp_or_separated_frames_are_errors() -> None:
    config = FrameExtractionConfig(frames_per_recording=3, min_blur_score=20)
    with pytest.raises(VideoPreparationError, match="too few usable frames"):
        select_frames(candidates([0, 5, 100, 100]), config)
    with pytest.raises(VideoPreparationError, match="separated"):
        select_frames(candidates([100] * 4, times=[0, 0.1, 0.2, 0.3]), config)


@pytest.mark.parametrize("changes", [
    {"frames_per_recording": 0}, {"frames_per_recording": True}, {"candidate_stride": -2},
    {"min_blur_score": float("nan")}, {"min_time_separation_s": -1}, {"diversity_weight": 2},
    {"target_width": None}, {"target_width": 0}, {"target_height": 1.5}, {"resize_mode": "guess"},
    {"aruco_dictionary": "TYPO"}, {"aruco_dictionary": []}, {"marker_side_m": 0}, {"marker_side_m": 0.1}, {"aruco_enabled": "true"},
])
def test_invalid_settings_rejected(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        FrameExtractionConfig(**changes)


def test_yaml_extraction_options_are_strict(tmp_path: Path) -> None:
    path = tmp_path / "settings.yaml"
    path.write_text("extraction:\n  typo: 10\n")
    with pytest.raises(ValueError, match="unknown extraction"):
        load_config(path)


def test_optional_aruco_corners_and_measured_size_do_not_create_scale() -> None:
    image = synthetic_image(0)
    config = FrameExtractionConfig(aruco_enabled=True, marker_side_m=0.08, target_width=128, target_height=128)
    processed = preprocess_image(image, config)
    detections = detect_markers(image, processed, config)
    assert len(detections) == 1
    marker = detections[0]
    assert marker["marker_id"] == 7
    assert marker["measured_side_m"] == 0.08
    np.testing.assert_allclose(marker["corners_original_px"], [[195, 75], [274, 75], [274, 154], [195, 154]], atol=1)
    original = np.column_stack((marker["corners_original_px"], np.ones(4)))
    np.testing.assert_allclose(marker["corners_processed_px"], (original @ processed.pixel_transform.T)[:, :2])
    assert detect_markers(image, processed, replace(config, marker_side_m=None))[0]["measured_side_m"] is None
    assert detect_markers(image, processed, FrameExtractionConfig()) == []


def test_cropped_marker_retains_corners_and_flags_outside_image() -> None:
    image = synthetic_image(0)
    config = FrameExtractionConfig(aruco_enabled=True, target_width=50, target_height=240, resize_mode="center-crop")
    marker = detect_markers(image, preprocess_image(image, config), config)[0]
    assert marker["all_corners_in_processed_image"] is False


def test_full_preparation_preserves_sources_timestamps_contracts_and_hashes(videos: tuple[Path, Path], tmp_path: Path) -> None:
    # MJPEG compression adds high-frequency artifacts to blurred fixtures;
    # use an explicit test threshold, not a claim that the default is calibrated.
    config = FrameExtractionConfig(aruco_enabled=True, marker_side_m=0.08, min_blur_score=500)
    output = tmp_path / "prepared"
    manifest = prepare_frames(*videos, output, "synthetic_test", config, "synthetic")
    records = load_frame_records(output / "frames_manifest.json")
    assert len(records) == 24
    assert [record.index for record in records] == list(range(24))
    assert [r.timepoint for r in records] == ["before"] * 12 + ["after"] * 12
    assert manifest["data_kind"] == "synthetic"
    assert manifest["calibration_status"] == "uncalibrated"
    assert "do not guarantee different camera positions" in manifest["selection_caveat"]
    for timepoint, path in zip(("before", "after"), videos, strict=True):
        source = manifest["sources"][timepoint]
        assert source["decoded_frame_count"] == 90
        assert source["source_sha256"] == sha256_file(path)
        assert source["blur_rejected_count"] > 0
        selected = [record for record in records if record.timepoint == timepoint]
        assert all(r.source_frame_index >= 10 for r in selected)
        assert all(b.timestamp_s - a.timestamp_s >= 0.5 - 1e-9 for a, b in zip(selected, selected[1:]))
        assert (output / f"contact_sheet_{timepoint}.jpg").is_file()
    for record, data in zip(records, manifest["frames"], strict=True):
        assert record.file_path.is_file()
        image = cv2.imdecode(np.frombuffer(record.file_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        assert image.shape == (480, 640, 3)
        assert record.timestamp_s == pytest.approx(record.source_frame_index / 10, abs=1e-6)
        assert data["image_sha256"] == sha256_file(record.file_path)
        assert data["aruco_detections"][0]["marker_id"] == 7
    template = json.loads((output / "ground_truth.template.json").read_text())
    assert template["annotations_status"] == "pending"
    assert not (output / "ground_truth.json").exists()


def test_output_collision_preserves_existing_annotations(videos: tuple[Path, Path], tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    truth = output / "ground_truth.json"
    truth.write_text("user-owned annotations")
    with pytest.raises(VideoPreparationError, match="already exists"):
        prepare_frames(*videos, output, "basic", FrameExtractionConfig())
    assert truth.read_text() == "user-owned annotations"


def test_missing_corrupt_and_unusable_videos_do_not_publish_outputs(videos: tuple[Path, Path], tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.avi"
    corrupt.write_bytes(b"not a video")
    for before in (tmp_path / "missing.avi", corrupt):
        with pytest.raises(VideoPreparationError):
            prepare_frames(before, videos[1], tmp_path / "out", "basic", FrameExtractionConfig())
        assert not (tmp_path / "out").exists()
    blank = tmp_path / "blank.avi"
    make_video(blank, count=20, blank=True)
    with pytest.raises(VideoPreparationError, match="too few usable"):
        prepare_frames(blank, videos[1], tmp_path / "out", "basic", FrameExtractionConfig())
    assert not (tmp_path / "out").exists()


def test_invalid_scenario_and_identical_source_paths(videos: tuple[Path, Path], tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scenario"):
        prepare_frames(*videos, tmp_path / "out", "../escape", FrameExtractionConfig())
    with pytest.raises(ValueError, match="different recording"):
        prepare_frames(videos[0], videos[0], tmp_path / "out", "basic", FrameExtractionConfig())


class FakeCapture:
    def __init__(self, timestamps: list[float], fps: float = 10, advertised: int | None = None) -> None:
        self.timestamps = timestamps
        self.fps = fps
        self.advertised = len(timestamps) if advertised is None else advertised
        self.index = 0
        self.released = False

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.timestamps):
            return False, None
        self.index += 1
        return True, synthetic_image(self.index)

    def get(self, property_id: int) -> float:
        if property_id == cv2.CAP_PROP_POS_MSEC:
            return self.timestamps[max(0, self.index - 1)] * 1000 if self.timestamps else 0
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.advertised,
                cv2.CAP_PROP_ORIENTATION_META: 0}.get(property_id, 0)

    def getBackendName(self) -> str:
        return "synthetic_mock"

    def release(self) -> None:
        self.released = True


def test_bad_backend_timestamps_use_labelled_fps_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    capture = FakeCapture([0, 0, 0])
    monkeypatch.setattr("roomdelta.frames._open_video", lambda path: capture)
    scan = scan_video(Path("mock.avi"), FrameExtractionConfig(candidate_stride=1))
    assert scan.metadata["timestamp_basis"] == "frame_index_over_fps_approximate"
    assert [c.timestamp_s for c in scan.candidates] == [0, 0.1, 0.2]
    assert scan.metadata["warnings"]
    assert capture.released


@pytest.mark.parametrize("timestamps,fps,advertised,message", [
    ([0, 0, 0], 0, 3, "unusable timestamps"),
    ([0, 0.1, 0.2], 10, 10, "failed read before advertised end"),
    ([], 10, 10, "no frames"),
])
def test_failed_reads_and_invalid_time_metadata(monkeypatch: pytest.MonkeyPatch, timestamps: list[float], fps: float, advertised: int, message: str) -> None:
    capture = FakeCapture(timestamps, fps, advertised)
    monkeypatch.setattr("roomdelta.frames._open_video", lambda path: capture)
    with pytest.raises(VideoPreparationError, match=message):
        scan_video(Path("mock.avi"), FrameExtractionConfig(candidate_stride=1))
    assert capture.released


def test_second_pass_failure_cleans_staging_and_never_publishes(videos: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise VideoPreparationError("second-pass failed read")
    monkeypatch.setattr("roomdelta.frames._extract_selected", fail)
    with pytest.raises(VideoPreparationError, match="second-pass"):
        prepare_frames(*videos, tmp_path / "out", "basic", FrameExtractionConfig())
    assert not (tmp_path / "out").exists()
    assert not list(tmp_path.glob(".roomdelta-frames-*"))


@pytest.mark.parametrize("fault", ["ordering", "path", "windows_path", "source_index", "scenario", "source"])
def test_manifest_reader_rejects_malformed_entries(videos: tuple[Path, Path], tmp_path: Path, fault: str) -> None:
    output = tmp_path / "out"
    manifest = prepare_frames(*videos, output, "test", FrameExtractionConfig(frames_per_recording=2), "synthetic")
    first = manifest["frames"][0]
    if fault == "ordering":
        first["index"] = 7
    elif fault == "path":
        first["file_path"] = "../escape.png"
    elif fault == "windows_path":
        first["file_path"] = "C:/private/frame.png"
    elif fault == "source_index":
        first["source_frame_index"] = 1000
    elif fault == "scenario":
        first["scenario"] = "different"
    else:
        first["source_recording_id"] = "after"
    path = output / "frames_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_frame_records(path)


@pytest.fixture
def truth_schema() -> dict[str, object]:
    schema = json.loads(files("roomdelta").joinpath("schemas/ground_truth.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return schema


def test_ground_truth_template_is_pending_and_schema_valid(truth_schema: dict[str, object]) -> None:
    template = ground_truth_template("basic", "synthetic", Path("before.avi"), Path("after.avi"))
    jsonschema.validate(template, truth_schema)
    assert template["events"] == []
    assert template["annotations_status"] == "pending"
    template["annotations_status"] = "complete"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(template, truth_schema)


def test_ground_truth_moved_measurement_requires_uncertainty(truth_schema: dict[str, object]) -> None:
    truth = ground_truth_template("basic", "synthetic", Path("before.avi"), Path("after.avi"))
    truth.update(annotations_status="complete", annotator="synthetic-test")
    event = {"id": "fixture-1", "object_label": "synthetic square", "type": "moved", "before_centroid": None,
             "after_centroid": None, "measured_displacement": {"distance_m": 1, "vector_m": None, "uncertainty_m": 0.1, "method": "synthetic schema fixture"}, "notes": "Not a real measurement"}
    truth["events"] = [event]
    jsonschema.validate(truth, truth_schema)
    del event["measured_displacement"]["uncertainty_m"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(truth, truth_schema)


def test_ground_truth_positions_require_declared_reference_frame(truth_schema: dict[str, object]) -> None:
    truth = ground_truth_template("basic", "synthetic", Path("before.avi"), Path("after.avi"))
    truth["events"] = [{"id": "fixture", "object_label": "fixture", "type": "added", "before_centroid": None,
                         "after_centroid": {"xyz_m": [0, 0, 1], "uncertainty_m": 0.1, "method": "analytic fixture"},
                         "measured_displacement": None, "notes": "Not a real-world measurement"}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(truth, truth_schema)
    truth["origin_description"] = "Synthetic fixture origin; X right, Y forward, Z up"
    jsonschema.validate(truth, truth_schema)


def test_missing_scenario_is_a_cli_error(videos: tuple[Path, Path], tmp_path: Path) -> None:
    result = subprocess.run([sys.executable, "-m", "roomdelta.pipeline", "--stage", "extract-frames",
                             "--before", str(videos[0]), "--after", str(videos[1]), "--output", str(tmp_path / "out")],
                            capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == 2
    assert "requires --scenario" in result.stderr
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("entrypoint", ["module", "script"])
def test_cli_pair_preparation_and_effective_overrides(videos: tuple[Path, Path], tmp_path: Path, entrypoint: str) -> None:
    command = [sys.executable, "-m", "roomdelta.pipeline", "--stage", "extract-frames"] if entrypoint == "module" else [sys.executable, str(ROOT / "scripts/prepare_frames.py")]
    command += ["--before", str(videos[0]), "--after", str(videos[1]), "--output", str(tmp_path / "out"), "--scenario", "synthetic_test",
                "--data-kind", "synthetic", "--frames-per-recording", "3", "--candidate-stride", "2", "--target-size", "128", "128", "--aruco", "--marker-side-m", "0.08"]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=60, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    output = tmp_path / "out"
    config = load_config(output / "effective_config.yaml")
    assert config.extraction.frames_per_recording == 3
    assert config.extraction.target_width == 128
    assert config.extraction.candidate_stride == 2
    run = json.loads((output / "run_manifest.json").read_text())
    assert run["milestone"] == "M1"
    assert run["data_kind"] == "synthetic"
    assert run["inference_performed"] is False
    assert len(load_frame_records(output / "frames_manifest.json")) == 6


@pytest.mark.parametrize("options,message,code", [
    (["--frames-per-recording", "0"], "positive integer", 1),
    (["--target-size", "0", "480"], "target dimensions", 1),
    (["--native-resolution", "--target-size", "128", "128"], "cannot be combined", 2),
    (["--marker-side-m", "-0.1", "--aruco"], "finite positive", 1),
])
def test_cli_invalid_options_are_helpful(videos: tuple[Path, Path], tmp_path: Path, options: list[str], message: str, code: int) -> None:
    result = subprocess.run([sys.executable, "-m", "roomdelta.pipeline", "--stage", "extract-frames", "--before", str(videos[0]), "--after", str(videos[1]),
                             "--scenario", "test", "--output", str(tmp_path / "out"), *options], capture_output=True, text=True, check=False, timeout=60)
    assert result.returncode == code
    assert message in result.stderr
    assert not (tmp_path / "out").exists()
