"""Internal coordinates: right-handed Z-up, metres, radians, xyzw quaternions."""

import math

import numpy as np


def yaw_to_xyzw(yaw: float) -> list[float]:
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def xyzw_to_yaw(rotation) -> float:
    q = np.asarray(rotation, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError("Expected a finite, nonzero xyzw quaternion")
    x, y, z, w = q / np.linalg.norm(q)
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def stage_to_world(up_axis: str, meters_per_unit: float) -> np.ndarray:
    """Rigid axis conversion and scale, applied to column vectors."""
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("meters_per_unit must be positive and finite")
    if up_axis.upper() == "Z":
        rotation = np.eye(3)
    elif up_axis.upper() == "Y":
        # +90 degrees about X: (x, y, z) -> (x, -z, y).
        rotation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    else:
        raise ValueError(f"Unsupported up axis: {up_axis}")
    transform = np.eye(4)
    transform[:3, :3] = rotation * meters_per_unit
    return transform


def transform_points(points, transform) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    return points @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]
