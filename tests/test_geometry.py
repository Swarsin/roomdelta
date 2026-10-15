"""Analytic camera/pose checks; these do not validate real reconstruction."""

import numpy as np
import pytest

from roomdelta.geometry import invert_pose, project_points, transform_intrinsics, transform_points, validate_pose
from roomdelta.visualization import synthetic_camera_pose


def test_nontrivial_pose_round_trip_and_camera_centre() -> None:
    pose = synthetic_camera_pose()
    points = np.array([[0, 0, 0], [1, 2, 3], [-2, 4, 0.5]], dtype=float)
    camera = transform_points(points, pose)
    np.testing.assert_allclose(transform_points(camera, invert_pose(pose)), points, atol=1e-12)
    np.testing.assert_allclose(invert_pose(pose)[:3, 3], [0, -1, 1.2])
    np.testing.assert_allclose(transform_points([[0, -1, 1.2]], pose), [[0, 0, 0]])
    np.testing.assert_allclose(pose @ invert_pose(pose), np.eye(4), atol=1e-12)


def test_camera_axes_and_world_up() -> None:
    points = [[0, 1, 1.2], [1, 1, 1.2], [0, 1, 2.2]]
    np.testing.assert_allclose(transform_points(points, synthetic_camera_pose()), [[0, 0, 2], [1, 0, 2], [0, -1, 2]])


def test_pinhole_projection_known_pixels_and_camera_z() -> None:
    K = np.array([[100, 0, 50], [0, 200, 40], [0, 0, 1]])
    result = project_points([[0, 0, 2], [1, 0.5, 4]], K, np.eye(4), (100, 80))
    np.testing.assert_allclose(result.pixels, [[50, 40], [75, 65]])
    np.testing.assert_allclose(result.camera_z, [2, 4])
    np.testing.assert_array_equal(result.valid, [True, True])
    assert result.camera_z[1] != np.linalg.norm([1, 0.5, 4])


def test_projection_with_nonidentity_world_to_camera() -> None:
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]])
    result = project_points([[0, 1, 1.2], [1, 1, 2.2]], K, synthetic_camera_pose(), (640, 480))
    np.testing.assert_allclose(result.pixels, [[320, 240], [520, 40]])
    np.testing.assert_allclose(result.camera_z, [2, 2])
    assert result.valid.all()


def test_projection_bounds_and_behind_camera() -> None:
    K = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = project_points([[0, 0, 1], [10, 0, 1], [0, 8, 1], [-1, 0, 1], [0, 0, 0], [0, 0, -1]], K, np.eye(4), (10, 8))
    np.testing.assert_array_equal(result.valid, [True, False, False, False, False, False])
    assert np.isnan(result.pixels[-2:]).all()


def test_resize_crop_changes_intrinsics_and_projected_pixels_consistently() -> None:
    K = np.array([[400, 0, 320], [0, 400, 240], [0, 0, 1]])
    affine = np.array([[0.5, 0, -20], [0, 0.5, -10], [0, 0, 1]])
    processed_K = transform_intrinsics(K, affine)
    np.testing.assert_allclose(processed_K, [[200, 0, 140], [0, 200, 110], [0, 0, 1]])
    original = project_points([[1, 0.5, 4]], K, np.eye(4), (640, 480))
    processed = project_points([[1, 0.5, 4]], processed_K, np.eye(4), (280, 220))
    expected = affine @ np.r_[original.pixels[0], 1]
    np.testing.assert_allclose(processed.pixels[0], expected[:2])


def test_consistent_known_scale_preserves_pixels_and_scales_camera_z() -> None:
    # Analytic fixture only, not an ArUco scale estimator.
    K = np.array([[100, 0, 50], [0, 100, 40], [0, 0, 1]])
    pose = synthetic_camera_pose()
    points = np.array([[0.2, 2, 1.2]])
    original = project_points(points, K, pose, (100, 80))
    metric_pose = pose.copy()
    metric_pose[:3, 3] *= 2
    scaled = project_points(points * 2, K, metric_pose, (100, 80))
    np.testing.assert_allclose(scaled.pixels, original.pixels)
    np.testing.assert_allclose(scaled.camera_z, original.camera_z * 2)


@pytest.mark.parametrize("kind", ["reflection", "scale", "bad_last_row", "nan"])
def test_invalid_poses_are_rejected(kind: str) -> None:
    pose = np.eye(4)
    if kind == "reflection":
        pose[0, 0] = -1
    elif kind == "scale":
        pose[0, 0] = 2
    elif kind == "bad_last_row":
        pose[3, 0] = 1
    else:
        pose[0, 0] = np.nan
    with pytest.raises(ValueError):
        validate_pose(pose)


@pytest.mark.parametrize("points", [[[np.nan, 0, 1]], [[0, 1]], [0, 0, 1]])
def test_invalid_points_are_rejected(points: object) -> None:
    with pytest.raises(ValueError, match="points"):
        transform_points(points, np.eye(4))


@pytest.mark.parametrize("size", [(0, 10), (10, -1), (10.5, 10), (True, 10)])
def test_invalid_image_size_is_rejected(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="image_size"):
        project_points([[0, 0, 1]], np.eye(3), np.eye(4), size)


def test_empty_points_are_supported() -> None:
    result = project_points(np.empty((0, 3)), np.eye(3), np.eye(4), (10, 10))
    assert result.pixels.shape == (0, 2)
    assert result.valid.shape == (0,)


def test_invalid_intrinsics_and_pixel_transform() -> None:
    bad_K = np.eye(3)
    bad_K[0, 0] = -1
    with pytest.raises(ValueError, match="focal"):
        project_points([[0, 0, 1]], bad_K, np.eye(4), (10, 10))
    with pytest.raises(ValueError, match="resize/crop"):
        transform_intrinsics(np.eye(3), [[1, 0.5, 0], [0, 1, 0], [0, 0, 1]])
