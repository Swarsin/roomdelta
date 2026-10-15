"""Camera geometry: column-vector transforms; world +Z up; OpenCV cameras."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def finite_array(value: object, shape: tuple[int, ...], name: str) -> FloatArray:
    """Convert to float64 and require an exact shape and finite entries."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}, got {array.shape}")
    return array


def validate_pose(T_cw: object) -> FloatArray:
    """Validate a rigid world-to-camera 4x4 pose (no scale or reflection)."""
    pose = finite_array(T_cw, (4, 4), "T_cw")
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6, rtol=0):
        raise ValueError("T_cw must have homogeneous last row [0, 0, 0, 1]")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("T_cw rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-6, rtol=0):
        raise ValueError("T_cw rotation must be right-handed (determinant +1)")
    return pose


def validate_intrinsics(K: object) -> FloatArray:
    """Validate OpenCV K[3,3] in pixels, allowing horizontal skew."""
    matrix = finite_array(K, (3, 3), "K")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("K focal lengths must be positive")
    if not np.allclose(matrix[2], [0, 0, 1], atol=1e-9, rtol=0) or matrix[1, 0] != 0:
        raise ValueError("K must use the standard OpenCV pinhole form")
    return matrix


def validate_pixel_transform(transform: object) -> FloatArray:
    """Validate a positive axis-aligned resize/crop affine in pixel coordinates."""
    affine = finite_array(transform, (3, 3), "pixel_transform")
    if (
        affine[0, 0] <= 0 or affine[1, 1] <= 0
        or affine[0, 1] != 0 or affine[1, 0] != 0
        or not np.allclose(affine[2], [0, 0, 1], atol=1e-9, rtol=0)
    ):
        raise ValueError("pixel_transform must represent positive axis-aligned resize/crop")
    return affine


def invert_pose(T_cw: object) -> FloatArray:
    """Return T_wc, the rigid inverse of T_cw; translations retain input units."""
    pose = validate_pose(T_cw)
    inverse = np.eye(4)
    inverse[:3, :3] = pose[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ pose[:3, 3]
    return inverse


def transform_points(points: object, transform: object) -> FloatArray:
    """Apply a rigid 4x4 transform to finite points[N,3], preserving units."""
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("points must be finite with shape [N,3]")
    pose = validate_pose(transform)
    return xyz @ pose[:3, :3].T + pose[:3, 3]


def transform_intrinsics(K: object, pixel_transform: object) -> FloatArray:
    """Return K_processed = A @ K_original for the recorded resize/crop A."""
    return validate_intrinsics(validate_pixel_transform(pixel_transform) @ validate_intrinsics(K))


@dataclass(frozen=True)
class Projection:
    """Pixels[N,2] (u,v), camera-z[N] in input units, and in-image validity[N]."""

    pixels: FloatArray
    camera_z: FloatArray
    valid: NDArray[np.bool_]


def project_points(
    points_world: object, K: object, T_cw: object, image_size_wh: tuple[int, int]
) -> Projection:
    """Project world points using camera-z depth, not Euclidean ray distance.

    Pixel centres have integer coordinates. Bounds are 0 <= u < W, 0 <= v < H.
    Non-positive-z points have NaN pixels and are invalid; positive-z points
    outside the image retain their pixel coordinates but are also invalid.
    """
    width, height = image_size_wh
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in (width, height)):
        raise ValueError("image_size_wh must contain positive integer width and height")
    camera = transform_points(points_world, T_cw)
    intrinsics = validate_intrinsics(K)
    positive = camera[:, 2] > 0
    pixels = np.full((len(camera), 2), np.nan)
    projected = camera[positive] @ intrinsics.T
    pixels[positive] = projected[:, :2] / projected[:, 2, None]
    valid = (
        positive & np.isfinite(pixels).all(axis=1)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    return Projection(pixels=pixels, camera_z=camera[:, 2], valid=valid)
