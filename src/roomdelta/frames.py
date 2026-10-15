"""M1 CPU video preparation. Selection heuristics do not estimate camera poses."""

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

from roomdelta.config import FrameExtractionConfig
from roomdelta.contracts import FrameRecord, Timepoint
from roomdelta.geometry import FloatArray

LOGGER = logging.getLogger(__name__)
MANIFEST_VERSION = "roomdelta-frames/2"
Image = NDArray[np.uint8]


class VideoPreparationError(RuntimeError):
    """Video decoding or preparation failed; no successful manifest was produced."""


@dataclass(frozen=True)
class FrameCandidate:
    source_frame_index: int
    timestamp_s: float
    original_size_wh: tuple[int, int]
    blur_score: float
    descriptor: FloatArray


@dataclass(frozen=True)
class PreprocessedImage:
    """BGR image and original-to-processed pixel affine, with actual integer geometry."""

    image: Image
    pixel_transform: FloatArray
    resized_size_wh: tuple[int, int]
    padding_ltrb: tuple[int, int, int, int]
    crop_ltrb: tuple[int, int, int, int]
    content_rect_xywh: tuple[int, int, int, int]


@dataclass(frozen=True)
class VideoScan:
    candidates: tuple[FrameCandidate, ...]
    metadata: dict[str, Any]


def rotation_clockwise(
    degrees: float, convention: Literal["opencv_orientation_meta", "ffmpeg_displaymatrix"]
) -> int:
    """Normalize known metadata semantics to clockwise quarter-turns in image space.

    OpenCV FFMPEG CAP_PROP_ORIENTATION_META is clockwise; FFmpeg's
    av_display_rotation_get/ffprobe side-data rotation is counterclockwise.
    Do not use container/display-matrix angles as OpenCV angles without conversion.
    """
    if not np.isfinite(degrees) or not np.isclose(degrees / 90, round(degrees / 90), atol=1e-6, rtol=0):
        raise VideoPreparationError(f"unsupported display rotation {degrees}; expected finite multiples of 90 degrees")
    if convention not in ("opencv_orientation_meta", "ffmpeg_displaymatrix"):
        raise VideoPreparationError(f"unsupported rotation convention: {convention}")
    return int(round(degrees) * (1 if convention == "opencv_orientation_meta" else -1)) % 360


def rotation_pixel_transform(raw_size_wh: tuple[int, int], clockwise: int) -> tuple[FloatArray, tuple[int, int]]:
    """Return raw-to-upright A and upright (W,H); integer pixel centres, no resampling."""
    width, height = raw_size_wh
    if any(type(v) is not int or v <= 0 for v in raw_size_wh) or clockwise not in (0, 90, 180, 270):
        raise ValueError("rotation requires positive raw dimensions and a clockwise quarter-turn")
    matrices = {
        0: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        90: [[0, -1, height - 1], [1, 0, 0], [0, 0, 1]],
        180: [[-1, 0, width - 1], [0, -1, height - 1], [0, 0, 1]],
        270: [[0, 1, 0], [-1, 0, width - 1], [0, 0, 1]],
    }
    return np.array(matrices[clockwise], dtype=np.float64), ((height, width) if clockwise % 180 else (width, height))


def upright_image(raw_image: Image, clockwise: int) -> Image:
    """Apply the one explicit display rotation to a known unrotated decoded image."""
    if clockwise == 0:
        return raw_image
    flags = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    if clockwise not in flags:
        raise ValueError("clockwise rotation must be 0, 90, 180 or 270")
    return cv2.rotate(raw_image, flags[clockwise])


def _capture_rotation(capture: cv2.VideoCapture) -> tuple[float, int]:
    """Require raw decoding, then interpret metadata in the verified backend convention.

    See OpenCV 5.0 cap_interface.hpp applyMetadataRotation and cap_ffmpeg_impl.hpp
    get_rotation_angle. Unknown backends with nonzero rotation fail rather than guess.
    """
    if capture.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 0:
        raise VideoPreparationError("OpenCV automatic orientation is still enabled; refusing a possible double rotation")
    degrees = float(capture.get(cv2.CAP_PROP_ORIENTATION_META))
    clockwise = rotation_clockwise(degrees, "opencv_orientation_meta")
    if clockwise and capture.getBackendName() != "FFMPEG":
        raise VideoPreparationError(f"rotation semantics are not verified for backend {capture.getBackendName()}; use OpenCV's FFMPEG backend")
    return degrees, clockwise


def _image(image: Image) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) <= 0:
        raise VideoPreparationError("decoder must return a nonempty uint8 BGR image[H,W,3]")


def frame_quality(image: Image) -> tuple[float, FloatArray]:
    """Variance of 3x3 Laplacian on grey (max dimension 320); descriptor is grey 32x32.

    Blur score units are squared 8-bit intensity derivatives. The descriptor is
    [1024] in [0,1]; distances reflect appearance, not physical baseline/independence.
    """
    _image(image)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = grey.shape
    if max(width, height) > 320:
        scale = 320 / max(width, height)
        grey = cv2.resize(grey, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
    score = float(cv2.Laplacian(grey, cv2.CV_64F, ksize=3).var())
    descriptor = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float64).ravel() / 255
    return score, descriptor


def preprocess_image(image: Image, config: FrameExtractionConfig) -> PreprocessedImage:
    """Resize then pad/crop. A includes OpenCV's half-pixel resize offsets.

    With sx = resized_W/original_W: u' = sx*(u+0.5)-0.5 + pad_left-crop_left.
    Therefore K_processed = A @ K_original in the existing integer-centre convention.
    Intrinsics themselves are neither estimated nor invented in M1.
    """
    _image(image)
    height, width = image.shape[:2]
    target_width, target_height = config.target_width, config.target_height
    if target_width is None or target_height is None:
        return PreprocessedImage(image.copy(), np.eye(3), (width, height), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, width, height))
    if config.resize_mode == "stretch":
        resized_width, resized_height = target_width, target_height
    else:
        ratios = (target_width / width, target_height / height)
        scale = min(ratios) if config.resize_mode == "letterbox" else max(ratios)
        resized_width, resized_height = max(1, round(width * scale)), max(1, round(height * scale))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x, pad_y = max(0, target_width - resized_width), max(0, target_height - resized_height)
    crop_x, crop_y = max(0, resized_width - target_width), max(0, resized_height - target_height)
    padding = (pad_x // 2, pad_y // 2, pad_x - pad_x // 2, pad_y - pad_y // 2)
    crop = (crop_x // 2, crop_y // 2, crop_x - crop_x // 2, crop_y - crop_y // 2)
    left, top, right, bottom = crop
    resized = resized[top:resized_height - bottom, left:resized_width - right]
    result = cv2.copyMakeBorder(resized, padding[1], padding[3], padding[0], padding[2], cv2.BORDER_CONSTANT, value=(0, 0, 0))
    sx, sy = resized_width / width, resized_height / height
    affine = np.array([[sx, 0, (sx - 1) / 2 + padding[0] - crop[0]],
                       [0, sy, (sy - 1) / 2 + padding[1] - crop[1]], [0, 0, 1]])
    return PreprocessedImage(result, affine, (resized_width, resized_height), padding, crop,
                             (padding[0], padding[1], result.shape[1] - pad_x, result.shape[0] - pad_y))


def _open_video(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise VideoPreparationError(f"video does not exist or is not a file: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoPreparationError(f"cannot open video: {path}; check codec support and file integrity")
    # Before any read: exactly one owner of orientation, our explicit raw-to-upright step.
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    if capture.get(cv2.CAP_PROP_ORIENTATION_AUTO) != 0:
        capture.release()
        raise VideoPreparationError(f"cannot disable automatic video rotation: {path}; export an explicitly oriented video first")
    return capture


def scan_video(path: Path, config: FrameExtractionConfig) -> VideoScan:
    """Sequential decode; retain only candidate metrics/descriptors, not video images.

    POS_MSEC is trusted only if finite, nonnegative and strictly increasing on
    every decoded frame. Otherwise the entire recording uses index/FPS and is
    labelled approximate. Decode failures before advertised EOF are errors.
    """
    capture = _open_video(path)
    candidates: list[FrameCandidate] = []
    count = 0
    timestamps_ok = True
    previous_timestamp = -1.0
    original_size: tuple[int, int] | None = None
    try:
        fps_raw = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps_raw if np.isfinite(fps_raw) and fps_raw > 0 else None
        advertised_raw = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        advertised = round(advertised_raw) if np.isfinite(advertised_raw) and advertised_raw > 0 else None
        orientation, clockwise = _capture_rotation(capture)
        backend = capture.getBackendName()
        while True:
            success, image = capture.read()
            if not success:
                break
            if image is None:
                raise VideoPreparationError(f"decoder returned no image at frame {count}: {path}")
            _image(image)
            size = (image.shape[1], image.shape[0])
            if original_size is not None and size != original_size:
                raise VideoPreparationError(f"video resolution changed at frame {count}: {path}")
            original_size = size
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000
            if not np.isfinite(timestamp) or timestamp < 0 or timestamp <= previous_timestamp:
                timestamps_ok = False
            previous_timestamp = timestamp
            if count % config.candidate_stride == 0:
                score, descriptor = frame_quality(upright_image(image, clockwise))
                candidates.append(FrameCandidate(count, timestamp, size, score, descriptor))
            count += 1
    finally:
        capture.release()
    if count == 0:
        raise VideoPreparationError(f"no frames could be decoded from video: {path}")
    if advertised is not None and count < advertised:
        raise VideoPreparationError(f"failed read before advertised end of {path}: decoded {count} of {advertised} frames; file may be truncated")
    warnings: list[str] = []
    if not timestamps_ok:
        if fps is None:
            raise VideoPreparationError(f"video has unusable timestamps and no valid FPS for fallback: {path}")
        candidates = [replace(candidate, timestamp_s=candidate.source_frame_index / fps) for candidate in candidates]
        warnings.append("Backend timestamps unusable; timestamps are approximate frame_index/FPS (not exact variable-rate PTS).")
    if advertised is None:
        warnings.append("Frame count unavailable: OpenCV cannot distinguish normal EOF from a late decode failure.")
    raw_to_upright, display_size = rotation_pixel_transform(original_size, clockwise)
    metadata = {
        "source_filename": path.name, "decoded_frame_count": count, "advertised_frame_count": advertised,
        "fps": fps, "timestamp_basis": "opencv_pos_msec" if timestamps_ok else "frame_index_over_fps_approximate",
        "original_size_wh": original_size, "raw_decoded_size_wh": original_size,
        "upright_size_wh": display_size, "decoder_backend": backend,
        "orientation_metadata_degrees": orientation,
        "orientation_metadata_convention": "opencv_orientation_meta_clockwise",
        "rotation_degrees_clockwise": clockwise, "rotation_applied_by": "roomdelta",
        "orientation_policy": "disable_opencv_auto_then_rotate_once",
        "raw_to_upright": raw_to_upright.tolist(), "automatic_rotation": False, "warnings": warnings,
    }
    return VideoScan(tuple(candidates), metadata)


def appearance_distance(first: FrameCandidate, second: FrameCandidate) -> float:
    """Grey thumbnail RMS difference in [0,1], not camera position distance."""
    return float(np.sqrt(np.mean((first.descriptor - second.descriptor) ** 2)))


def select_frames(candidates: tuple[FrameCandidate, ...], config: FrameExtractionConfig) -> tuple[FrameCandidate, ...]:
    """Deterministic temporal-bin selection weighted by sharpness/appearance novelty.

    Fill empty bins from remaining usable candidates, respecting minimum timestamp
    spacing. Similar images are not hard-rejected: stationary/unchanged controls
    remain possible. Exact requested count is required, never silently reduced.
    """
    usable = [candidate for candidate in candidates if candidate.blur_score >= config.min_blur_score]
    target = config.frames_per_recording
    if len(usable) < target:
        raise VideoPreparationError(f"too few usable frames: {len(usable)} pass blur threshold {config.min_blur_score} "
                                    f"from {len(candidates)} sampled candidates; requested {target}. "
                                    "Use a sharper/longer video, reduce candidate_stride, frames_per_recording or min_blur_score.")
    selected: list[FrameCandidate] = []
    max_score = max(candidate.blur_score for candidate in usable)

    def eligible(candidate: FrameCandidate) -> bool:
        return all(candidate.source_frame_index != other.source_frame_index and
                   abs(candidate.timestamp_s - other.timestamp_s) + 1e-9 >= config.min_time_separation_s for other in selected)

    def priority(candidate: FrameCandidate) -> tuple[float, float, int]:
        novelty = min((appearance_distance(candidate, other) for other in selected), default=0.0)
        sharpness = candidate.blur_score / max(max_score, 1e-12)
        weighted = (1 - config.diversity_weight) * sharpness + config.diversity_weight * novelty
        return weighted, candidate.blur_score, -candidate.source_frame_index

    start, end = min(c.timestamp_s for c in usable), max(c.timestamp_s for c in usable)
    edges = np.linspace(start, end + 1e-9, target + 1)
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        available = [c for c in usable if lower <= c.timestamp_s < upper and eligible(c)]
        if available:
            selected.append(max(available, key=priority))
    while len(selected) < target:
        available = [candidate for candidate in usable if eligible(candidate)]
        if not available:
            # Earliest-first is maximum-cardinality for a fixed time separation;
            # avoids a sharp-but-central greedy choice falsely implying insufficient input.
            selected = []
            for candidate in sorted(usable, key=lambda c: (c.timestamp_s, c.source_frame_index)):
                if eligible(candidate):
                    selected.append(candidate)
                if len(selected) == target:
                    LOGGER.warning("Using earliest-spaced sharp-enough fallback; temporal/appearance choices could not fill the request")
                    return tuple(sorted(selected, key=lambda c: c.source_frame_index))
            raise VideoPreparationError(f"too few usable separated frames: at most {len(selected)} of {target} with "
                                        f"min_time_separation_s={config.min_time_separation_s}; "
                                        "record a longer sequence or reduce spacing/frame count")
        selected.append(max(available, key=priority))
    return tuple(sorted(selected, key=lambda candidate: candidate.source_frame_index))


def detect_markers(
    image: Image, processed: PreprocessedImage, config: FrameExtractionConfig,
    raw_to_upright: FloatArray | None = None,
) -> list[dict[str, Any]]:
    """Detect in upright image; retain raw, upright and final corner coordinates.

    processed.pixel_transform maps upright to final pixels. corners_original_px
    remains a backward-compatible alias for raw decoded coordinates.
    """
    if not config.aruco_enabled:
        return []
    if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "ArucoDetector"):
        raise VideoPreparationError("ArUco requires the pinned opencv-contrib-python-headless package; check conflicting OpenCV installations")
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, config.aruco_dictionary))
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return []
    detections = []
    width, height = processed.image.shape[1], processed.image.shape[0]
    for corner, marker_id in zip(corners, ids.ravel(), strict=True):
        upright = np.asarray(corner, dtype=np.float64).reshape(4, 2)
        homogeneous = np.column_stack((upright, np.ones(4)))
        original = (homogeneous @ np.linalg.inv(np.eye(3) if raw_to_upright is None else raw_to_upright).T)[:, :2]
        mapped = homogeneous @ processed.pixel_transform.T
        xy = mapped[:, :2]
        detections.append({
            "marker_id": int(marker_id), "dictionary": config.aruco_dictionary,
            "corners_original_px": original.tolist(), "corners_raw_decoded_px": original.tolist(),
            "corners_upright_px": upright.tolist(), "corners_processed_px": xy.tolist(),
            "all_corners_in_processed_image": bool(((xy[:, 0] >= 0) & (xy[:, 0] < width)
                                                   & (xy[:, 1] >= 0) & (xy[:, 1] < height)).all()),
            "measured_side_m": config.marker_side_m,
        })
    return sorted(detections, key=lambda detection: detection["marker_id"])


def sha256_file(path: Path) -> str:
    """Stream a file hash without retaining videos/images in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_image(path: Path, image: Image) -> None:
    # imencode + pathlib also supports Windows Unicode output paths.
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise VideoPreparationError(f"could not encode extracted image: {path}")
    path.write_bytes(encoded.tobytes())


def _contact_sheet(previews: list[tuple[Image, str]], path: Path) -> None:
    portrait = previews[0][0].shape[0] > previews[0][0].shape[1]
    columns, tile_width, image_height = 3, (180 if portrait else 240), (240 if portrait else 180)
    tile_height = image_height + 30
    sheet = np.full(((len(previews) + columns - 1) // columns * tile_height, columns * tile_width, 3), 245, dtype=np.uint8)
    for index, (image, label) in enumerate(previews):
        thumb = preprocess_image(image, FrameExtractionConfig(target_width=tile_width, target_height=image_height)).image
        x, y = (index % columns) * tile_width, (index // columns) * tile_height
        sheet[y:y + image_height, x:x + tile_width] = thumb
        cv2.putText(sheet, label, (x + 4, y + image_height + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.3 if portrait else 0.35, (0, 0, 0), 1, cv2.LINE_AA)
    _write_image(path, sheet)


def _extract_selected(
    path: Path, selected: tuple[FrameCandidate, ...], timepoint: Timepoint,
    scenario: str, config: FrameExtractionConfig, stage: Path, start_index: int,
    timestamp_basis: str, rotation_degrees_clockwise: int,
) -> list[dict[str, Any]]:
    capture = _open_video(path)
    by_index = {candidate.source_frame_index: candidate for candidate in selected}
    records: list[dict[str, Any]] = []
    previews: list[tuple[Image, str]] = []
    (stage / timepoint).mkdir()
    try:
        _, clockwise = _capture_rotation(capture)
        if clockwise != rotation_degrees_clockwise:
            raise VideoPreparationError(f"second-pass orientation mismatch: {path}")
        for source_index in range(selected[-1].source_frame_index + 1):
            success, image = capture.read()
            if not success or image is None:
                raise VideoPreparationError(f"failed second-pass read at source frame {source_index}: {path}")
            if source_index not in by_index:
                continue
            candidate = by_index[source_index]
            if (image.shape[1], image.shape[0]) != candidate.original_size_wh:
                raise VideoPreparationError(f"second-pass resolution mismatch: {path}")
            raw_to_upright, upright_size = rotation_pixel_transform(candidate.original_size_wh, clockwise)
            upright = upright_image(image, clockwise)
            processed = preprocess_image(upright, config)
            complete_transform = processed.pixel_transform @ raw_to_upright
            relative = Path(timepoint) / f"frame_{source_index:08d}.png"
            record = FrameRecord(scenario, timepoint, relative, path.name, candidate.timestamp_s,
                                 candidate.original_size_wh, (processed.image.shape[1], processed.image.shape[0]),
                                 complete_transform, start_index + len(records), blur_score=candidate.blur_score,
                                 quality_reasons=("passed_blur_threshold", "temporal_appearance_heuristic_not_pose_evidence"),
                                 source_frame_index=source_index)
            data = asdict(record)
            data["file_path"] = relative.as_posix()
            data["pixel_transform"] = record.pixel_transform.tolist()
            data.update({
                "source_recording_id": timepoint, "timestamp_basis": timestamp_basis,
                "raw_decoded_size_wh": candidate.original_size_wh, "upright_size_wh": upright_size,
                "raw_to_upright": raw_to_upright.tolist(),
                "upright_to_processed": processed.pixel_transform.tolist(),
                "rotation_degrees_clockwise": clockwise,
                "preprocessing": {"mode": config.resize_mode if config.target_width is not None else "identity",
                                  "resized_size_wh": processed.resized_size_wh,
                                  "padding_ltrb": processed.padding_ltrb, "crop_ltrb": processed.crop_ltrb,
                                  "content_rect_xywh": processed.content_rect_xywh,
                                  "interpolation": "opencv_INTER_LINEAR", "pixel_centres": "integer_half_pixel_resize"},
                "aruco_detections": detect_markers(upright, processed, config, raw_to_upright),
            })
            _write_image(stage / relative, processed.image)
            data["image_sha256"] = sha256_file(stage / relative)
            records.append(data)
            preview = processed.image.copy()
            for detection in data["aruco_detections"]:
                polygon = np.rint(detection["corners_processed_px"]).astype(np.int32)
                cv2.polylines(preview, [polygon], True, (0, 255, 0), 2)
            previews.append((preview, f"idx {source_index}  t={candidate.timestamp_s:.3f}s  blur={candidate.blur_score:.0f}"))
    finally:
        capture.release()
    _contact_sheet(previews, stage / f"contact_sheet_{timepoint}.jpg")
    return records


def ground_truth_template(scenario: str, data_kind: str, before: Path, after: Path) -> dict[str, Any]:
    """Unannotated template, not an empty-scene assertion or fabricated measurements."""
    return {"schema_version": "roomdelta-ground-truth/1", "scenario": scenario, "data_kind": data_kind,
            "annotations_status": "pending", "annotator": None,
            "source_recordings": {"before": before.name, "after": after.name},
            "coordinate_system": "right-handed world: +Z up", "origin_description": None,
            "unit": "m", "events": [],
            "notes": "UNANNOTATED: empty events does not mean unchanged. Enter actual observations and ruler measurements; unknown measurements stay null."}


def prepare_frames(
    before: Path, after: Path, output: Path, scenario: str, config: FrameExtractionConfig,
    data_kind: Literal["real", "synthetic"] = "real",
) -> dict[str, Any]:
    """Prepare a pair in a fresh output directory; stage all files before publication.

    Source indices are per recording; manifest index is globally contiguous with
    before frames first, then after, each ordered by source index. Source hashes
    and decoded counts are retained. No intrinsics, 3D scale or inference is produced.
    """
    if not isinstance(scenario, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", scenario):
        raise ValueError("scenario must be 1–80 letters/digits/underscores/hyphens, starting with a letter/digit")
    if data_kind not in ("real", "synthetic"):
        raise ValueError("data_kind must be real or synthetic")
    if output.exists():
        raise VideoPreparationError(f"output already exists: {output}; choose a new directory to preserve frames and annotations")
    for video in (before, after):
        if not video.is_file():
            raise VideoPreparationError(f"video does not exist or is not a file: {video}")
    if before.resolve() == after.resolve():
        raise ValueError("before and after must be different recording files")
    if config.aruco_enabled and config.marker_side_m is None:
        LOGGER.warning("ArUco detection enabled without a measured marker size; corners only, metric calibration unavailable")
    try:
        sources: dict[str, Any] = {}
        selections: dict[str, tuple[FrameCandidate, ...]] = {}
        stats: dict[str, tuple[int, int]] = {}
        for timepoint, path in (("before", before), ("after", after)):
            stat = path.stat()
            stats[timepoint] = (stat.st_size, stat.st_mtime_ns)
            digest = sha256_file(path)
            scan = scan_video(path, config)
            selections[timepoint] = select_frames(scan.candidates, config)
            try:
                source_path = Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()
                source_path_kind = "relative_to_manifest"
            except ValueError:  # Different Windows drives cannot be relativized.
                source_path, source_path_kind = path.resolve().as_posix(), "absolute"
            sources[timepoint] = dict(scan.metadata, source_recording_id=timepoint, source_path=source_path,
                                     source_path_kind=source_path_kind, source_sha256=digest,
                                     sampled_candidate_count=len(scan.candidates),
                                     blur_rejected_count=sum(c.blur_score < config.min_blur_score for c in scan.candidates),
                                     selected_frame_count=len(selections[timepoint]))
            LOGGER.info("%s: decoded %s frames; selected %s of %s candidates", timepoint,
                        scan.metadata["decoded_frame_count"], len(selections[timepoint]), len(scan.candidates))
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".roomdelta-frames-", dir=output.parent) as temporary:
            stage = Path(temporary) / "prepared"
            stage.mkdir()
            frames: list[dict[str, Any]] = []
            for timepoint, path in (("before", before), ("after", after)):
                frames.extend(_extract_selected(path, selections[timepoint], timepoint, scenario, config, stage,
                                                len(frames), sources[timepoint]["timestamp_basis"],
                                                sources[timepoint]["rotation_degrees_clockwise"]))
                stat = path.stat()
                if (stat.st_size, stat.st_mtime_ns) != stats[timepoint]:
                    raise VideoPreparationError(f"source video changed during preparation: {path}; retry with a stable file")
            manifest = {
                "schema_version": MANIFEST_VERSION, "scenario": scenario, "data_kind": data_kind,
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "opencv_version": cv2.__version__,
                "path_base": "manifest_directory", "frame_order": "before_then_after_source_index_ascending",
                "coordinate_unit": "pixels_only_no_3d_calibration", "calibration_status": "uncalibrated",
                "selection_method": "temporal_bins_sharpness_thumbnail_rms_with_earliest_spaced_fallback",
                "selection_caveat": "Appearance/time spacing do not guarantee different camera positions or independent views. Inspect overlap manually.",
                "extraction_config": asdict(config), "sources": sources, "frames": frames,
            }
            (stage / "frames_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            template = ground_truth_template(scenario, data_kind, before, after)
            (stage / "ground_truth.template.json").write_text(json.dumps(template, indent=2) + "\n", encoding="utf-8")
            if output.exists():
                raise VideoPreparationError(f"output appeared during preparation: {output}; choose a new directory")
            stage.rename(output)
        return manifest
    except cv2.error as exc:
        raise VideoPreparationError(f"OpenCV preparation failed: {exc}") from exc


def load_frame_records(manifest_path: Path) -> tuple[FrameRecord, ...]:
    """Read M1 manifest entries into existing contracts; image paths resolve from manifest.

    This is a frame-manifest reader, not an M2 reconstruction/cache reader.
    """
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["schema_version"] not in ("roomdelta-frames/1", MANIFEST_VERSION):
            raise ValueError("unsupported frame manifest schema_version")
        records = []
        last_source_index: dict[str, int] = {}
        for index, frame in enumerate(manifest["frames"]):
            relative = Path(frame["file_path"])
            if relative.is_absolute() or PureWindowsPath(frame["file_path"]).drive or ".." in relative.parts or "\\" in frame["file_path"]:
                raise ValueError("manifest image paths must be portable relative paths without traversal")
            record = FrameRecord(
                scenario=frame["scenario"], timepoint=frame["timepoint"], file_path=manifest_path.parent / relative,
                source_filename=frame["source_filename"], timestamp_s=frame["timestamp_s"],
                original_size_wh=tuple(frame["original_size_wh"]), processed_size_wh=tuple(frame["processed_size_wh"]),
                pixel_transform=np.asarray(frame["pixel_transform"]), index=frame["index"], valid=frame["valid"],
                blur_score=frame["blur_score"], quality_reasons=tuple(frame["quality_reasons"]),
                source_frame_index=frame["source_frame_index"],
            )
            source = manifest["sources"][frame["source_recording_id"]]
            if manifest["schema_version"] == MANIFEST_VERSION:
                expected_rotation, upright_size = rotation_pixel_transform(record.original_size_wh, frame["rotation_degrees_clockwise"])
                if (tuple(frame["raw_decoded_size_wh"]) != record.original_size_wh
                        or tuple(frame["upright_size_wh"]) != upright_size
                        or tuple(source["raw_decoded_size_wh"]) != record.original_size_wh
                        or tuple(source["upright_size_wh"]) != upright_size
                        or not np.allclose(frame["raw_to_upright"], expected_rotation)
                        or not np.allclose(source["raw_to_upright"], expected_rotation)
                        or not np.allclose(record.pixel_transform, np.asarray(frame["upright_to_processed"]) @ expected_rotation)
                        or frame["rotation_degrees_clockwise"] != source["rotation_degrees_clockwise"]
                        or rotation_clockwise(source["orientation_metadata_degrees"], "opencv_orientation_meta") != source["rotation_degrees_clockwise"]
                        or source["orientation_metadata_convention"] != "opencv_orientation_meta_clockwise"
                        or source["automatic_rotation"] is not False):
                    raise ValueError("manifest orientation/dimensions/transforms are inconsistent")
            if (record.index != index or record.scenario != manifest["scenario"]
                    or frame["source_recording_id"] != record.timepoint
                    or record.source_filename != source["source_filename"]
                    or record.source_frame_index is None
                    or not last_source_index.get(record.timepoint, -1) < record.source_frame_index < source["decoded_frame_count"]):
                raise ValueError("manifest frame ordering/source metadata are inconsistent")
            last_source_index[record.timepoint] = record.source_frame_index
            records.append(record)
        if not records or {record.timepoint for record in records} != {"before", "after"}:
            raise ValueError("frame manifest requires before and after entries")
        if [r.timepoint for r in records] != sorted([r.timepoint for r in records], key=lambda t: t == "after"):
            raise ValueError("manifest must put all before frames ahead of after frames")
        return tuple(records)
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid frame manifest: {exc}") from exc
