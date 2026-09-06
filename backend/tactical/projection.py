"""Calibrated pixel-to-ground projection boundaries."""

from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import numpy as np

from .detector import PixelDetection, TimestampedFrame
from .schema import Observation


def _finite_array(value: object, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {shape} matrix")
    return array


def _bounds(value: object) -> tuple[tuple[float, float], tuple[float, float]]:
    array = _finite_array(value, (2, 2), "ground_bounds")
    if np.any(array[:, 0] >= array[:, 1]):
        raise ValueError("ground_bounds must have positive extent")
    return tuple(tuple(float(item) for item in row) for row in array)  # type: ignore[return-value]


def _inside(x: float, z: float, bounds: tuple[tuple[float, float], tuple[float, float]]) -> bool:
    return bounds[0][0] <= x <= bounds[0][1] and bounds[1][0] <= z <= bounds[1][1]


def _pixel_covariance(detection: PixelDetection) -> np.ndarray:
    if detection.covariance_px is not None:
        return np.asarray(detection.covariance_px, dtype=np.float64)
    scale = max(1.0, (detection.x_max - detection.x_min) * 0.03)
    variance = scale * scale / max(detection.confidence, 0.05)
    return np.eye(2, dtype=np.float64) * variance


def _observation(
    frame: TimestampedFrame,
    detection: PixelDetection,
    xyz: tuple[float, float, float],
    covariance_xz: np.ndarray,
    calibration_variance: float,
) -> Observation:
    covariance = np.zeros((3, 3), dtype=np.float64)
    covariance[np.ix_([0, 2], [0, 2])] = (
        covariance_xz + np.eye(2, dtype=np.float64) * calibration_variance
    )
    covariance[1, 1] = calibration_variance
    return Observation(
        t=frame.t,
        sensor_id=frame.sensor_id,
        xyz=xyz,
        conf=detection.confidence,
        sequence=frame.sequence,
        covariance=tuple(tuple(float(item) for item in row) for row in covariance),
    )


@dataclass(frozen=True, slots=True)
class HomographyCalibration:
    """Fixed aerial calibration mapping image ``(u, v)`` to ground ``(x, z)``."""

    image_to_ground_xz: tuple[tuple[float, float, float], ...]
    ground_y: float = 0.0
    ground_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-100.0, 100.0),
        (-100.0, 100.0),
    )
    calibration_variance: float = 0.01

    def __post_init__(self) -> None:
        matrix = _finite_array(self.image_to_ground_xz, (3, 3), "image_to_ground_xz")
        if abs(float(np.linalg.det(matrix))) <= 1e-12:
            raise ValueError("image_to_ground_xz must be invertible")
        object.__setattr__(
            self,
            "image_to_ground_xz",
            tuple(tuple(float(item) for item in row) for row in matrix),
        )
        ground_y = float(self.ground_y)
        variance = float(self.calibration_variance)
        if not math.isfinite(ground_y):
            raise ValueError("ground_y must be finite")
        if not math.isfinite(variance) or variance < 0.0:
            raise ValueError("calibration_variance must be finite and non-negative")
        object.__setattr__(self, "ground_y", ground_y)
        object.__setattr__(self, "ground_bounds", _bounds(self.ground_bounds))
        object.__setattr__(self, "calibration_variance", variance)


class FixedAerialProjector:
    def __init__(self, calibration: HomographyCalibration) -> None:
        self.calibration = calibration
        self._matrix = np.asarray(calibration.image_to_ground_xz, dtype=np.float64)

    def project(
        self, frame: TimestampedFrame, detection: PixelDetection
    ) -> Observation:
        detection.validate_frame_bounds(frame)
        u, v = detection.bottom_center
        numerator = self._matrix @ np.array([u, v, 1.0], dtype=np.float64)
        denominator = float(numerator[2])
        if abs(denominator) <= 1e-10:
            raise ValueError("homography projects pixel to infinity")
        x, z = float(numerator[0] / denominator), float(numerator[1] / denominator)
        if not math.isfinite(x) or not math.isfinite(z):
            raise ValueError("homography produced a non-finite ground point")
        if not _inside(x, z, self.calibration.ground_bounds):
            raise ValueError("projected point lies outside calibrated ground bounds")
        h = self._matrix
        jacobian = np.array(
            [
                [
                    (h[0, 0] * denominator - numerator[0] * h[2, 0]) / denominator**2,
                    (h[0, 1] * denominator - numerator[0] * h[2, 1]) / denominator**2,
                ],
                [
                    (h[1, 0] * denominator - numerator[1] * h[2, 0]) / denominator**2,
                    (h[1, 1] * denominator - numerator[1] * h[2, 1]) / denominator**2,
                ],
            ],
            dtype=np.float64,
        )
        covariance_xz = jacobian @ _pixel_covariance(detection) @ jacobian.T
        return _observation(
            frame,
            detection,
            (x, self.calibration.ground_y, z),
            covariance_xz,
            self.calibration.calibration_variance,
        )


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        for name in ("fx", "fy", "cx", "cy"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name in {"fx", "fy"} and value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class CameraPose:
    """Camera-to-world rotation and camera origin in map coordinates."""

    rotation: tuple[tuple[float, float, float], ...]
    xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        rotation = _finite_array(self.rotation, (3, 3), "rotation")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6):
            raise ValueError("rotation must be orthonormal")
        if float(np.linalg.det(rotation)) < 0.999999:
            raise ValueError("rotation must be right-handed")
        xyz = np.asarray(self.xyz, dtype=np.float64)
        if xyz.shape != (3,) or not np.isfinite(xyz).all():
            raise ValueError("xyz must be a finite 3-vector")
        object.__setattr__(self, "rotation", tuple(tuple(float(item) for item in row) for row in rotation))
        object.__setattr__(self, "xyz", tuple(float(item) for item in xyz))


@runtime_checkable
class PoseProvider(Protocol):
    """Future boundary for timestamped moving-camera poses."""

    def pose_at(self, sensor_id: str, t: float) -> CameraPose:
        ...


@dataclass(frozen=True, slots=True)
class FixedPoseProvider:
    pose: CameraPose

    def pose_at(self, sensor_id: str, t: float) -> CameraPose:
        if not sensor_id or not math.isfinite(float(t)):
            raise ValueError("sensor_id and t must identify a valid frame")
        return self.pose


@dataclass(frozen=True, slots=True)
class RayGroundCalibration:
    intrinsics: CameraIntrinsics
    ground_y: float = 0.0
    ground_bounds: tuple[tuple[float, float], tuple[float, float]] = (
        (-100.0, 100.0),
        (-100.0, 100.0),
    )
    calibration_variance: float = 0.01

    def __post_init__(self) -> None:
        ground_y = float(self.ground_y)
        variance = float(self.calibration_variance)
        if not math.isfinite(ground_y):
            raise ValueError("ground_y must be finite")
        if not math.isfinite(variance) or variance < 0.0:
            raise ValueError("calibration_variance must be finite and non-negative")
        object.__setattr__(self, "ground_y", ground_y)
        object.__setattr__(self, "ground_bounds", _bounds(self.ground_bounds))
        object.__setattr__(self, "calibration_variance", variance)


class RayGroundProjector:
    def __init__(
        self, calibration: RayGroundCalibration, poses: PoseProvider
    ) -> None:
        self.calibration = calibration
        self.poses = poses

    def _ground_xz(
        self, pose: CameraPose, u: float, v: float
    ) -> tuple[float, float]:
        intrinsics = self.calibration.intrinsics
        camera_ray = np.array(
            [(u - intrinsics.cx) / intrinsics.fx, (v - intrinsics.cy) / intrinsics.fy, 1.0],
            dtype=np.float64,
        )
        direction = np.asarray(pose.rotation, dtype=np.float64) @ camera_ray
        origin = np.asarray(pose.xyz, dtype=np.float64)
        if direction[1] >= -1e-10:
            raise ValueError("camera ray does not intersect the ground in front")
        distance = (self.calibration.ground_y - origin[1]) / direction[1]
        if not math.isfinite(float(distance)) or distance <= 0.0:
            raise ValueError("camera ray has no forward ground intersection")
        point = origin + distance * direction
        x, z = float(point[0]), float(point[2])
        if not _inside(x, z, self.calibration.ground_bounds):
            raise ValueError("projected point lies outside calibrated ground bounds")
        return x, z

    def project(
        self, frame: TimestampedFrame, detection: PixelDetection
    ) -> Observation:
        detection.validate_frame_bounds(frame)
        pose = self.poses.pose_at(frame.sensor_id, frame.t)
        u, v = detection.bottom_center
        x, z = self._ground_xz(pose, u, v)
        epsilon = 0.25
        jacobian = np.empty((2, 2), dtype=np.float64)
        for axis, delta in enumerate(((epsilon, 0.0), (0.0, epsilon))):
            plus = self._ground_xz(pose, u + delta[0], v + delta[1])
            minus = self._ground_xz(pose, u - delta[0], v - delta[1])
            jacobian[:, axis] = (np.asarray(plus) - np.asarray(minus)) / (2.0 * epsilon)
        covariance_xz = jacobian @ _pixel_covariance(detection) @ jacobian.T
        return _observation(
            frame,
            detection,
            (x, self.calibration.ground_y, z),
            covariance_xz,
            self.calibration.calibration_variance,
        )
