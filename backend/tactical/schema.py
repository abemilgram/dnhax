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


class IntentKind(str, Enum):
    """The complete set of authored public navigation intents."""

    HOLD = "HOLD"
    CROSS = "CROSS"
    FLANK = "FLANK"


class RiskBand(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


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


@dataclass(frozen=True, slots=True)
class ActorState:
    """Current navigation state for the actor receiving a route cue."""

    t: float
    xyz: tuple[float, float, float]
    velocity_xz: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _finite_float(self.t, "t"))
        object.__setattr__(self, "xyz", _vector(self.xyz, 3, "xyz"))
        object.__setattr__(
            self, "velocity_xz", _vector(self.velocity_xz, 2, "velocity_xz")
        )


@dataclass(frozen=True, slots=True)
class TimedPoint:
    """One serialization-friendly point on a candidate route."""

    t: float
    xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _finite_float(self.t, "t"))
        object.__setattr__(self, "xyz", _vector(self.xyz, 3, "xyz"))


@dataclass(frozen=True, slots=True)
class ScoreComponents:
    """Coarse geometric route metrics; all fractions are in ``[0, 1]``."""

    route_length: float
    turn_cost: float
    los_fraction: float
    exposure_fraction: float
    time_in_open: float
    open_fraction: float
    uncertainty_risk: float
    risk_std: float
    exposure_cvar90: float
    reach_probability: float
    progress: float
    invalid_fraction: float
    risk_score: float

    def __post_init__(self) -> None:
        for name in (
            "route_length",
            "turn_cost",
            "time_in_open",
            "risk_std",
        ):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "los_fraction",
            "exposure_fraction",
            "open_fraction",
            "uncertainty_risk",
            "exposure_cvar90",
            "reach_probability",
            "progress",
            "invalid_fraction",
            "risk_score",
        ):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class TrajectoryCandidate:
    """Rankable intent route with no automatic-control semantics."""

    intent: IntentKind
    valid: bool
    utility: float
    score: ScoreComponents
    route: tuple[TimedPoint, ...]
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.intent, IntentKind):
            raise ValueError("intent must be an IntentKind")
        if not isinstance(self.valid, bool):
            raise ValueError("valid must be a bool")
        object.__setattr__(self, "utility", _finite_float(self.utility, "utility"))
        route = tuple(self.route)
        if not all(isinstance(point, TimedPoint) for point in route):
            raise ValueError("route must contain TimedPoint values")
        if any(b.t < a.t for a, b in zip(route, route[1:])):
            raise ValueError("route timestamps must be non-decreasing")
        object.__setattr__(self, "route", route)
        reasons = tuple(self.reasons)
        if not all(isinstance(reason, str) and reason for reason in reasons):
            raise ValueError("reasons must contain non-empty strings")
        object.__setattr__(self, "reasons", reasons)


@dataclass(frozen=True, slots=True)
class PlanRanking:
    """A deterministic best-first ranking for one planning cycle."""

    t: float
    cycle_index: int
    candidates: tuple[TrajectoryCandidate, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "t", _finite_float(self.t, "t"))
        if (
            isinstance(self.cycle_index, bool)
            or not isinstance(self.cycle_index, int)
            or self.cycle_index < 0
        ):
            raise ValueError("cycle_index must be a non-negative integer")
        candidates = tuple(self.candidates)
        if not candidates:
            raise ValueError("candidates must not be empty")
        if not all(isinstance(item, TrajectoryCandidate) for item in candidates):
            raise ValueError("candidates must contain TrajectoryCandidate values")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class Cue:
    """Change-only navigation cue emitted by the deterministic reducer."""

    sequence: int
    t: float
    intent: IntentKind
    risk_band: RiskBand
    risk: float
    route_valid: bool
    tracking_degraded: bool
    reasons: tuple[str, ...]
    route: tuple[TimedPoint, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence, bool)
            or not isinstance(self.sequence, int)
            or self.sequence < 1
        ):
            raise ValueError("sequence must be a positive integer")
        object.__setattr__(self, "t", _finite_float(self.t, "t"))
        if not isinstance(self.intent, IntentKind):
            raise ValueError("intent must be an IntentKind")
        if not isinstance(self.risk_band, RiskBand):
            raise ValueError("risk_band must be a RiskBand")
        risk = _finite_float(self.risk, "risk")
        if not 0.0 <= risk <= 1.0:
            raise ValueError("risk must be in [0, 1]")
        object.__setattr__(self, "risk", risk)
        if not isinstance(self.route_valid, bool):
            raise ValueError("route_valid must be a bool")
        if not isinstance(self.tracking_degraded, bool):
            raise ValueError("tracking_degraded must be a bool")
        object.__setattr__(self, "reasons", tuple(self.reasons))
        object.__setattr__(self, "route", tuple(self.route))


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
