"""Rotation semantics, raw-pixel geometry and decoder ownership regressions."""

import json
from pathlib import Path
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from roomdelta.config import FrameExtractionConfig
from roomdelta.contracts import FrameRecord
from roomdelta.frames import (
    VideoPreparationError, _open_video, detect_markers, load_frame_records, prepare_frames,
    preprocess_image, rotation_clockwise, rotation_pixel_transform, upright_image,
)
from roomdelta.geometry import transform_intrinsics, validate_pixel_transform


@pytest.mark.parametrize("degrees,convention,expected", [
    (90, "opencv_orientation_meta", 90), (90, "ffmpeg_displaymatrix", 270),
    (-90, "opencv_orientation_meta", 270), (-90, "ffmpeg_displaymatrix", 90),
    (180, "opencv_orientation_meta", 180), (180, "ffmpeg_displaymatrix", 180),
    (270, "opencv_orientation_meta", 270), (270, "ffmpeg_displaymatrix", 90),
    (-270, "opencv_orientation_meta", 90), (450, "ffmpeg_displaymatrix", 270),
])
def test_metadata_sign_depends_on_declared_semantics(degrees: float, convention: str, expected: int) -> None:
    assert rotation_clockwise(degrees, convention) == expected


@pytest.mark.parametrize("degrees", [45, 90.5, np.nan, np.inf])
def test_unsupported_rotation_metadata_is_not_guessed(degrees: float) -> None:
    with pytest.raises(VideoPreparationError, match="unsupported display rotation"):
        rotation_clockwise(degrees, "opencv_orientation_meta")


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_rotation_affine_maps_every_raw_pixel_to_actual_rotated_pixel(angle: int) -> None:
    raw = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    matrix, size = rotation_pixel_transform((5, 3), angle)
    upright = upright_image(raw, angle)
    np.testing.assert_array_equal(upright, np.rot90(raw, k=-angle // 90))
    assert (upright.shape[1], upright.shape[0]) == size
    for y in range(3):
        for x in range(5):
            u, v, _ = (matrix @ [x, y, 1]).astype(int)
            np.testing.assert_array_equal(upright[v, u], raw[y, x])
            np.testing.assert_allclose(np.linalg.inv(matrix) @ [u, v, 1], [x, y, 1])


def test_phone_portrait_rotation_resize_and_padding_have_complete_affine() -> None:
    raw = np.zeros((576, 1024, 3), dtype=np.uint8)
    rotation, size = rotation_pixel_transform((1024, 576), 90)
    assert size == (576, 1024)
    resized = preprocess_image(upright_image(raw, 90), FrameExtractionConfig(target_width=480, target_height=640))
    assert resized.image.shape == (640, 480, 3)
    assert resized.resized_size_wh == (360, 640)
    assert resized.padding_ltrb == (60, 0, 60, 0)
    complete = resized.pixel_transform @ rotation
    np.testing.assert_allclose(complete, [[0, -0.625, 419.1875], [0.625, 0, -0.1875], [0, 0, 1]])
    raw_point = np.array([200, 300, 1])
    np.testing.assert_allclose(complete @ raw_point, resized.pixel_transform @ (rotation @ raw_point))


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_frame_contract_accepts_rotation_without_claiming_canonical_rotated_K(angle: int) -> None:
    transform, _ = rotation_pixel_transform((320, 240), angle)
    validate_pixel_transform(transform)
    record = FrameRecord("test", "before", Path("frame.png"), "source.mp4", 0,
                         (320, 240), (480, 640), transform, 0)
    np.testing.assert_allclose(record.pixel_transform, transform)
    with pytest.raises(ValueError, match="camera-axis conversion"):
        transform_intrinsics(np.eye(3), transform)


@pytest.mark.parametrize("bad", [
    [[-1, 0, 5], [0, 1, 0], [0, 0, 1]],
    [[1, 0.1, 0], [0, 1, 0], [0, 0, 1]],
])
def test_extended_pixel_contract_still_rejects_reflection_and_shear(bad: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="no shear/reflection"):
        validate_pixel_transform(bad)


def marker_image() -> np.ndarray:
    raw = np.full((240, 320, 3), 255, dtype=np.uint8)
    marker = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 7, 80)
    raw[75:155, 195:275] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
    raw[:40, :60] = (255, 0, 0)  # Deliberately asymmetric content for decoder comparisons.
    return raw


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_aruco_raw_upright_and_final_corners_agree_with_final_image(angle: int) -> None:
    raw = marker_image()
    rotation, _ = rotation_pixel_transform((320, 240), angle)
    upright = upright_image(raw, angle)
    config = FrameExtractionConfig(aruco_enabled=True, target_width=480, target_height=640)
    processed = preprocess_image(upright, config)
    marker = detect_markers(upright, processed, config, rotation)[0]
    raw_corners = np.asarray(marker["corners_raw_decoded_px"])
    np.testing.assert_allclose(raw_corners, [[195, 75], [274, 75], [274, 154], [195, 154]], atol=1)
    np.testing.assert_allclose(marker["corners_original_px"], raw_corners)
    homogeneous = np.column_stack((raw_corners, np.ones(4)))
    np.testing.assert_allclose(marker["corners_upright_px"], (homogeneous @ rotation.T)[:, :2])
    complete = processed.pixel_transform @ rotation
    np.testing.assert_allclose(marker["corners_processed_px"], (homogeneous @ complete.T)[:, :2])
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    actual, ids, _ = detector.detectMarkers(processed.image)
    assert ids.ravel().tolist() == [7]
    np.testing.assert_allclose(marker["corners_processed_px"], actual[0].reshape(4, 2), atol=2)


class AutoCapture:
    """Backend starts auto-correcting by default, as OpenCV FFMPEG does."""

    def __init__(self, angle: int, can_disable: bool = True) -> None:
        self.angle, self.can_disable = angle, can_disable
        self.auto = True
        self.index = 0
        self.auto_on_reads: list[bool] = []
        self.raw = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def getBackendName(self) -> str:
        return "FFMPEG"

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_ORIENTATION_AUTO and self.can_disable:
            self.auto = bool(value)
            return True
        return False

    def get(self, prop: int) -> float:
        return {cv2.CAP_PROP_ORIENTATION_AUTO: float(self.auto), cv2.CAP_PROP_ORIENTATION_META: self.angle,
                cv2.CAP_PROP_FPS: 5, cv2.CAP_PROP_FRAME_COUNT: 8,
                cv2.CAP_PROP_POS_MSEC: max(0, self.index - 1) * 200}.get(prop, 0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.auto_on_reads.append(self.auto)
        if self.index == 8:
            return False, None
        self.index += 1
        return True, np.rot90(self.raw, -self.angle // 90).copy() if self.auto else self.raw.copy()

    def release(self) -> None:
        self.released = True


@pytest.mark.parametrize("angle", [90, 180, 270])
def test_pipeline_disables_default_auto_and_rotates_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, angle: int,
) -> None:
    captures = []

    def open_capture(*args: object) -> AutoCapture:
        capture = AutoCapture(angle)
        captures.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    before, after = tmp_path / "before.mp4", tmp_path / "after.mp4"
    before.write_bytes(b"synthetic backend fixture before")
    after.write_bytes(b"synthetic backend fixture after")
    output = tmp_path / "frames"
    config = FrameExtractionConfig(frames_per_recording=2, candidate_stride=1, min_blur_score=0,
                                   min_time_separation_s=0, target_width=None, target_height=None)
    manifest = prepare_frames(before, after, output, "synthetic", config, "synthetic")
    assert len(captures) == 4  # scan/save per source
    assert all(c.released and not any(c.auto_on_reads) for c in captures)
    records = load_frame_records(output / "frames_manifest.json")
    for record, entry in zip(records, manifest["frames"], strict=True):
        actual = cv2.imdecode(np.frombuffer(record.file_path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        expected = np.rot90(captures[0].raw, -angle // 90)
        np.testing.assert_array_equal(actual, expected)
        assert entry["raw_decoded_size_wh"] == (5, 3)
        assert entry["upright_size_wh"] == (expected.shape[1], expected.shape[0])
        assert manifest["sources"][record.timepoint]["automatic_rotation"] is False
    # Reader remains compatible with the legacy schema envelope.
    manifest["schema_version"] = "roomdelta-frames/1"
    (output / "legacy.json").write_text(json.dumps(manifest))
    assert len(load_frame_records(output / "legacy.json")) == 4


@pytest.mark.parametrize("fault", ["raw_size", "upright_size", "composite", "source_angle", "source_auto"])
def test_v2_manifest_rejects_inconsistent_orientation_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: AutoCapture(90))
    before, after = tmp_path / "before.mp4", tmp_path / "after.mp4"
    before.write_bytes(b"synthetic before")
    after.write_bytes(b"synthetic after")
    output = tmp_path / "frames"
    config = FrameExtractionConfig(frames_per_recording=1, candidate_stride=1, min_blur_score=0)
    manifest = prepare_frames(before, after, output, "synthetic", config, "synthetic")
    if fault in ("raw_size", "upright_size"):
        manifest["frames"][0]["raw_decoded_size_wh" if fault == "raw_size" else "upright_size_wh"] = [99, 99]
    elif fault == "composite":
        manifest["frames"][0]["pixel_transform"][0][2] += 5
    elif fault == "source_angle":
        manifest["sources"]["before"]["orientation_metadata_degrees"] = 270
    else:
        manifest["sources"]["before"]["automatic_rotation"] = True
    path = output / "frames_manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="orientation/dimensions/transforms"):
        load_frame_records(path)


def test_refuse_double_rotation_when_backend_cannot_disable_auto(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic backend fixture")
    capture = AutoCapture(90, can_disable=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: capture)
    with pytest.raises(VideoPreparationError, match="cannot disable automatic video rotation"):
        _open_video(source)
    assert capture.released
    assert capture.auto_on_reads == []


@pytest.fixture(scope="module")
def encoded_rotation_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("encoded_display_rotation")
    source = directory / "sensor.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 6, (320, 240))
    assert writer.isOpened(), "MP4V encoder unavailable"
    try:
        for _ in range(12):
            writer.write(marker_image())
    finally:
        writer.release()
    return source


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="FFmpeg CLI needed to encode actual display matrices")
@pytest.mark.parametrize("displaymatrix_degrees", [90, 180, 270])
def test_real_encoded_metadata_matches_ffprobe_and_opencv_auto_without_double_rotation(
    encoded_rotation_source: Path, tmp_path: Path, displaymatrix_degrees: int,
) -> None:
    source = tmp_path / "rotated.mp4"
    # Modern FFmpeg writes an actual display matrix via this input override;
    # the legacy `-metadata rotate=...` tag no longer writes one in FFmpeg 8.
    result = subprocess.run(["ffmpeg", "-v", "error", "-display_rotation:v:0", str(displaymatrix_degrees),
                             "-noautorotate", "-i", str(encoded_rotation_source), "-c", "copy", str(source)],
                            capture_output=True, text=True, timeout=30)
    if result.returncode and "Unrecognized option" in result.stderr:
        result = subprocess.run(["ffmpeg", "-v", "error", "-i", str(encoded_rotation_source), "-c", "copy",
                                 "-metadata:s:v:0", f"rotate={displaymatrix_degrees}", str(source)],
                                capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0",
                        "-show_entries", "stream_side_data=rotation", "-of", "json", str(source)], text=True, timeout=30))
    angle = probe["streams"][0]["side_data_list"][0]["rotation"]
    assert angle % 360 == displaymatrix_degrees
    auto_capture = cv2.VideoCapture(str(source))
    auto_capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    expected_frames = []
    while True:
        ok, upright = auto_capture.read()
        if not ok:
            break
        expected_frames.append(upright)
    opencv_angle = auto_capture.get(cv2.CAP_PROP_ORIENTATION_META)
    auto_capture.release()
    assert len(expected_frames) == 12
    expected_cw = rotation_clockwise(angle, "ffmpeg_displaymatrix")
    assert rotation_clockwise(opencv_angle, "opencv_orientation_meta") == expected_cw
    # Two genuinely encoded inputs, not a monkeypatched decoder.
    after = tmp_path / "after.mp4"
    shutil.copyfile(source, after)
    output = tmp_path / "prepared"
    config = FrameExtractionConfig(frames_per_recording=2, candidate_stride=1, min_blur_score=0,
                                   min_time_separation_s=0, target_width=None, target_height=None, aruco_enabled=True)
    manifest = prepare_frames(source, after, output, "synthetic", config, "synthetic")
    entry = manifest["frames"][0]
    actual = cv2.imdecode(np.frombuffer((output / entry["file_path"]).read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
    # Compare the same selected source index, not frame zero: inter-frame codec
    # quantization can differ even for a fixture with identical source images.
    np.testing.assert_array_equal(actual, expected_frames[entry["source_frame_index"]])
    assert entry["aruco_detections"][0]["marker_id"] == 7
    assert manifest["sources"]["before"]["rotation_degrees_clockwise"] == expected_cw
    assert manifest["sources"]["before"]["raw_decoded_size_wh"] == (320, 240)
    assert len(load_frame_records(output / "frames_manifest.json")) == 4
