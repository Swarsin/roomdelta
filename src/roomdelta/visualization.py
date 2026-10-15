"""M0 deterministic synthetic geometry, not model predictions or detected events."""

from pathlib import Path

import numpy as np

from roomdelta.config import VisualizationConfig
from roomdelta.geometry import FloatArray, invert_pose


def synthetic_camera_pose() -> FloatArray:
    """T_cw for a camera at (0,-1,1.2)m looking along world +Y with world +Z up."""
    rotation = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
    centre = np.array([0., -1., 1.2])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = -rotation @ centre
    return pose


def _surface_points(spacing: float) -> FloatArray:
    """Analytic floor and three walls in metres; returns points[P,3]."""
    x = np.linspace(-2, 2, int(np.ceil(4 / spacing)) + 1)
    y = np.linspace(0, 4, int(np.ceil(4 / spacing)) + 1)
    z = np.linspace(0, 2.5, int(np.ceil(2.5 / spacing)) + 1)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    floor = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
    xx, zz = np.meshgrid(x, z, indexing="ij")
    back = np.column_stack((xx.ravel(), np.full(xx.size, 4), zz.ravel()))
    yy, zz = np.meshgrid(y, z, indexing="ij")
    left = np.column_stack((np.full(yy.size, -2), yy.ravel(), zz.ravel()))
    right = np.column_stack((np.full(yy.size, 2), yy.ravel(), zz.ravel()))
    return np.vstack((floor, back, left, right))


def save_synthetic_scene(path: Path, config: VisualizationConfig) -> Path:
    """Write a headless .rrd fixture with metric geometry, axes and a camera frustum.

    Rerun is imported only here. No CUDA, model weights or display are required.
    A matching rerun-sdk viewer is needed for manual inspection.
    """
    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:
        raise RuntimeError("Synthetic visualization needs: python -m pip install -e '.[visualization]'") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    recording = rr.RecordingStream(config.application_id)
    blueprint = rrb.Horizontal(
        rrb.Spatial3DView(origin="world", name="SYNTHETIC fixture (metres)"),
        rrb.TextDocumentView(origin="description", name="Provenance"),
        column_shares=[3, 1],
    )
    recording.save(path, default_blueprint=blueprint)
    try:
        recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        recording.log("description", rr.TextDocument(
            "# RoomDelta M0 — SYNTHETIC ONLY\n\n"
            "Analytic floor, walls, box and camera. No phone recording, model inference, "
            "change detection or evaluation.\n\n"
            "World: right-handed, +Z up, metres by construction. "
            "Axes: X red, Y green, Z blue. Camera: OpenCV right/down/forward.\n\n"
            "The orange box is a fixture, not an added-object prediction.",
            media_type="text/markdown",
        ), static=True)
        recording.log("world/axes", rr.Arrows3D(
            origins=np.zeros((3, 3)), vectors=np.eye(3) * 0.75,
            colors=[[255, 60, 60], [60, 220, 60], [60, 120, 255]],
            labels=["+X", "+Y", "+Z (up)"],
        ), static=True)
        recording.log("world/room", rr.Points3D(
            _surface_points(config.point_spacing_m), colors=[160, 180, 195], radii=0.015,
        ), static=True)
        recording.log("world/fixture_box", rr.Boxes3D(
            centers=[[0.5, 2, 0.4]], sizes=[[0.8, 0.6, 0.8]],
            colors=[255, 160, 40], labels=["SYNTHETIC box; not a prediction"],
        ), static=True)
        T_wc = invert_pose(synthetic_camera_pose())
        recording.log("world/camera", rr.Transform3D(
            translation=T_wc[:3, 3], mat3x3=T_wc[:3, :3],
        ), static=True)
        recording.log("world/camera/image", rr.Pinhole(
            image_from_camera=[[400, 0, 320], [0, 400, 240], [0, 0, 1]],
            resolution=[640, 480], camera_xyz=rr.ViewCoordinates.RDF,
            image_plane_distance=0.5,
        ), static=True)
        recording.flush()
    finally:
        recording.disconnect()
    return path
