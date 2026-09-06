"""Strict image-detector boundary without a model implementation.

Detector adapters receive immutable timestamped frames and return pixel-space
detections.  Metric projection deliberately lives in ``projection.py``.
"""

from dataclasses import dataclass
import math
from typing import Protocol, Sequence, runtime_checkable


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TimestampedFrame:
    sensor_id: str
    t: float
    sequence: int
    width: int
    height: int
    pixels: object

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise ValueError("sensor_id must be a non-empty string")
        object.__setattr__(self, "t", _finite(self.t, "t"))
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.pixels is None:
            raise ValueError("pixels must not be None")


@dataclass(frozen=True, slots=True)
class PixelDetection:
    """One confidence-scored pixel bounding box in ``xyxy`` order."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    class_id: str = "entity"
    covariance_px: tuple[tuple[float, float], tuple[float, float]] | None = None

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.x_min < 0.0 or self.y_min < 0.0:
            raise ValueError("bounding-box coordinates must be non-negative")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("bounding box must have positive extent")
        confidence = _finite(self.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", confidence)
        if not isinstance(self.class_id, str) or not self.class_id:
            raise ValueError("class_id must be a non-empty string")
        if self.covariance_px is not None:
            rows = tuple(tuple(_finite(item, "covariance_px") for item in row) for row in self.covariance_px)
            if len(rows) != 2 or any(len(row) != 2 for row in rows):
                raise ValueError("covariance_px must be a finite 2x2 matrix")
            if abs(rows[0][1] - rows[1][0]) > 1e-10:
                raise ValueError("covariance_px must be symmetric")
            trace = rows[0][0] + rows[1][1]
            determinant = rows[0][0] * rows[1][1] - rows[0][1] ** 2
            if rows[0][0] < 0.0 or rows[1][1] < 0.0 or trace < 0.0 or determinant < -1e-10:
                raise ValueError("covariance_px must be positive semidefinite")
            object.__setattr__(self, "covariance_px", rows)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, self.y_max)

    def validate_frame_bounds(self, frame: TimestampedFrame) -> None:
        if self.x_max > frame.width or self.y_max > frame.height:
            raise ValueError("detection bounding box lies outside the frame")


@runtime_checkable
class Detector(Protocol):
    """Boundary implemented later by a concrete detector adapter."""

    def detect(self, frame: TimestampedFrame) -> Sequence[PixelDetection]:
        ...
