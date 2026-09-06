"""Validated schemas shared by the tactical belief core."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Iterable

import numpy as np


def _finite_float(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _vector(values: Iterable[float], size: int, name: str) -> tuple[float, ...]:
    result = tuple(_finite_float(value, name) for value in values)
    if len(result) != size:
        raise ValueError(f"{name} must contain {size} values")
    return result


def _covariance(
    value: Iterable[Iterable[float]] | None,
) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("covariance must be a finite 3x3 matrix")
    if not np.allclose(array, array.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if float(np.linalg.eigvalsh(array).min()) < -1e-10:
        raise ValueError("covariance must be positive semidefinite")
    return tuple(tuple(float(item) for item in row) for row in array)


class EvidenceState(str, Enum):
    OBSERVED = "observed"
    STALE = "stale"
    CONFLICTING = "conflicting"


class LifecycleState(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"


@dataclass(frozen=True, slots=True)
class Observation:
    """A time-stamped metric position observation.

    Coordinates are right-handed ``(x=east, y=up, z=north)``.  Covariance,
    when supplied, follows the same xyz order.
    """

    t: float
    sensor_id: str
    xyz: tuple[float, float, float]
    conf: float
    sequence: int
    covariance: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _finite_float(self.t, "t"))
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise ValueError("sensor_id must be a non-empty string")
        object.__setattr__(self, "xyz", _vector(self.xyz, 3, "xyz"))
        confidence = _finite_float(self.conf, "conf")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("conf must be in [0, 1]")
        object.__setattr__(self, "conf", confidence)
        if isinstance(self.sequence, bool) or not isinstance(self.sequence, int):
            raise ValueError("sequence must be an integer")
        if self.sequence < 0:
            raise ValueError("sequence must be non-negative")
        object.__setattr__(self, "covariance", _covariance(self.covariance))


@dataclass(frozen=True, slots=True)
class SensorSpec:
    """Static sensor geometry used only for expected-visibility accounting."""

    sensor_id: str
    xyz: tuple[float, float, float]
    forward_xz: tuple[float, float] = (0.0, 1.0)
    horizontal_fov_degrees: float = 360.0
    max_range: float = 50.0

    def __post_init__(self) -> None:
        if not isinstance(self.sensor_id, str) or not self.sensor_id:
            raise ValueError("sensor_id must be a non-empty string")
        object.__setattr__(self, "xyz", _vector(self.xyz, 3, "xyz"))
        forward = _vector(self.forward_xz, 2, "forward_xz")
        norm = math.hypot(*forward)
        if norm <= 1e-12:
            raise ValueError("forward_xz must be non-zero")
        object.__setattr__(
            self, "forward_xz", (forward[0] / norm, forward[1] / norm)
        )
        fov = _finite_float(self.horizontal_fov_degrees, "horizontal_fov_degrees")
        if not 0.0 < fov <= 360.0:
            raise ValueError("horizontal_fov_degrees must be in (0, 360]")
        object.__setattr__(self, "horizontal_fov_degrees", fov)
        distance = _finite_float(self.max_range, "max_range")
        if distance <= 0.0:
            raise ValueError("max_range must be positive")
        object.__setattr__(self, "max_range", distance)


@dataclass(frozen=True, slots=True)
class TrackSnapshot:
    """Immutable, serialization-friendly view of one estimated track."""

    track_id: int
    t: float
    xyz: tuple[float, float, float]
    velocity_xz: tuple[float, float]
    covariance: tuple[tuple[float, float, float, float], ...]
    evidence: EvidenceState
    lifecycle: LifecycleState
    hits: int
    misses: int
    expected_visible_misses: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id < 1
        ):
            raise ValueError("track_id must be a positive integer")
        object.__setattr__(self, "t", _finite_float(self.t, "t"))
        object.__setattr__(self, "xyz", _vector(self.xyz, 3, "xyz"))
        object.__setattr__(
            self, "velocity_xz", _vector(self.velocity_xz, 2, "velocity_xz")
        )
        covariance = np.asarray(self.covariance, dtype=np.float64)
        if covariance.shape != (4, 4) or not np.isfinite(covariance).all():
            raise ValueError("covariance must be a finite 4x4 matrix")
        if not np.allclose(covariance, covariance.T, atol=1e-10):
            raise ValueError("covariance must be symmetric")
        if float(np.linalg.eigvalsh(covariance).min()) < -1e-10:
            raise ValueError("covariance must be positive semidefinite")
        object.__setattr__(
            self,
            "covariance",
            tuple(tuple(float(item) for item in row) for row in covariance),
        )
        if not isinstance(self.evidence, EvidenceState):
            raise ValueError("evidence must be an EvidenceState")
        if not isinstance(self.lifecycle, LifecycleState):
            raise ValueError("lifecycle must be a LifecycleState")
        for name in ("hits", "misses", "expected_visible_misses"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


@dataclass(slots=True)
class TrackState:
    """Mutable package-internal filter and lifecycle state."""

    track_id: int
    t: float
    state: np.ndarray
    covariance: np.ndarray
    y: float
    lifecycle: LifecycleState = LifecycleState.TENTATIVE
    evidence: EvidenceState = EvidenceState.OBSERVED
    hits: int = 1
    misses: int = 0
    expected_visible_misses: int = 0
    last_observed_t: float = 0.0
    sensor_ids: set[str] = field(default_factory=set)
