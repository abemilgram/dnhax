"""Geometric route metrics and coarse observer-belief risk scoring."""

import math
from typing import Iterable, Sequence

import numpy as np

from .map import TacticalMap
from .schema import EvidenceState, ScoreComponents, TimedPoint, TrackSnapshot


Point = tuple[float, float, float]


def polyline_length(points: Iterable[Point]) -> float:
    route = tuple(points)
    return sum(_distance(first, second) for first, second in zip(route, route[1:]))


def turn_cost(points: Iterable[Point]) -> float:
    """Return total absolute horizontal turn in radians."""

    route = tuple(points)
    total = 0.0
    for first, middle, last in zip(route, route[1:], route[2:]):
        incoming = (middle[0] - first[0], middle[2] - first[2])
        outgoing = (last[0] - middle[0], last[2] - middle[2])
        in_norm = math.hypot(*incoming)
        out_norm = math.hypot(*outgoing)
        if in_norm <= 1e-12 or out_norm <= 1e-12:
            continue
        cosine = sum(a * b for a, b in zip(incoming, outgoing)) / (
            in_norm * out_norm
        )
        total += math.acos(max(-1.0, min(1.0, cosine)))
    return total


def interpolate_polyline(points: Sequence[Point], distance: float) -> Point:
    if not points:
        raise ValueError("points must not be empty")
    distance = float(distance)
    if not math.isfinite(distance):
        raise ValueError("distance must be finite")
    if distance <= 0.0 or len(points) == 1:
        return tuple(float(value) for value in points[0])
    remaining = distance
    for start, end in zip(points, points[1:]):
        length = _distance(start, end)
        if remaining <= length and length > 1e-12:
            fraction = remaining / length
            return tuple(
                float(a + fraction * (b - a)) for a, b in zip(start, end)
            )
        remaining -= length
    return tuple(float(value) for value in points[-1])


def observer_reliability(track: TrackSnapshot, t: float) -> float:
    age = max(0.0, float(t) - track.last_observed_t)
    decay = math.exp(-age / 1.5)
    if track.evidence is EvidenceState.OBSERVED:
        return decay
    if track.evidence is EvidenceState.STALE:
        return 0.75 * decay
    return 0.5 * decay


def has_line_of_sight(
    tactical_map: TacticalMap,
    route_point: Point,
    observer_point: Point,
    *,
    route_height: float = 1.0,
) -> bool:
    """Test static-map LOS without assigning an action to the observer."""

    origin = (route_point[0], route_point[1] + route_height, route_point[2])
    return not tactical_map.is_occluded(origin, observer_point)


def evaluate_route(
    tactical_map: TacticalMap,
    route: Sequence[TimedPoint],
    observers: Iterable[TrackSnapshot],
) -> ScoreComponents:
    """Score one public timed route against observer mean beliefs."""

    if not route:
        return _invalid_score()
    times = np.asarray([point.t for point in route], dtype=np.float64)
    positions = np.asarray([point.xyz for point in route], dtype=np.float64)[None, :, :]
    tracks = tuple(sorted(observers, key=lambda item: item.track_id))[:16]
    observer_positions = np.empty((1, len(tracks), len(route), 3), dtype=np.float64)
    reliabilities = np.empty(len(tracks), dtype=np.float64)
    uncertainty = np.empty(len(tracks), dtype=np.float64)
    for index, track in enumerate(tracks):
        elapsed = times - track.t
        observer_positions[0, index, :, 0] = (
            track.xyz[0] + elapsed * track.velocity_xz[0]
        )
        observer_positions[0, index, :, 1] = track.xyz[1]
        observer_positions[0, index, :, 2] = (
            track.xyz[2] + elapsed * track.velocity_xz[1]
        )
        reliabilities[index] = observer_reliability(track, float(times[0]))
        covariance = np.asarray(track.covariance, dtype=np.float64)
        uncertainty[index] = math.sqrt(
            max(0.0, float(covariance[0, 0] + covariance[1, 1]))
        )
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 0.0
    points = tuple(point.xyz for point in route)
    return score_rollouts(
        tactical_map,
        positions,
        observer_positions,
        reliabilities,
        uncertainty,
        dt=max(0.0, dt),
        route_length=polyline_length(points),
        route_turn_cost=turn_cost(points),
        reach_probability=1.0,
        progress=1.0,
        invalid_fraction=0.0,
    )


def score_rollouts(
    tactical_map: TacticalMap,
    actor_positions: np.ndarray,
    observer_positions: np.ndarray,
    observer_reliabilities: np.ndarray,
    observer_uncertainties: np.ndarray,
    *,
    dt: float,
    route_length: float,
    route_turn_cost: float,
    reach_probability: float,
    progress: float,
    invalid_fraction: float,
) -> ScoreComponents:
    """Aggregate rollout geometry into bounded route-risk components."""

    actors = np.asarray(actor_positions, dtype=np.float64)
    observers = np.asarray(observer_positions, dtype=np.float64)
    if actors.ndim != 3 or actors.shape[2] != 3:
        raise ValueError("actor_positions must have shape (rollouts, steps, 3)")
    if observers.shape != (actors.shape[0], len(observer_reliabilities), actors.shape[1], 3):
        raise ValueError(
            "observer_positions must have shape (rollouts, observers, steps, 3)"
        )
    rollout_count, step_count, _ = actors.shape
    los_values = np.zeros((rollout_count, step_count), dtype=np.float64)
    exposure_values = np.zeros_like(los_values)
    openness_values = np.zeros_like(los_values)
    for rollout in range(rollout_count):
        for step in range(step_count):
            actor = tuple(float(value) for value in actors[rollout, step])
            openness_values[rollout, step] = tactical_map.openness_at(actor)
            best_exposure = 0.0
            any_los = False
            for observer_index in range(observers.shape[1]):
                observer = tuple(
                    float(value)
                    for value in observers[rollout, observer_index, step]
                )
                if has_line_of_sight(tactical_map, actor, observer):
                    any_los = True
                    distance = _distance(actor, observer)
                    weighted = float(observer_reliabilities[observer_index]) * math.exp(
                        -distance / 25.0
                    )
                    best_exposure = max(best_exposure, weighted)
            los_values[rollout, step] = float(any_los)
            exposure_values[rollout, step] = best_exposure

    rollout_exposure = exposure_values.mean(axis=1)
    los_fraction = float(los_values.mean())
    exposure_fraction = float(exposure_values.mean())
    open_fraction = float(openness_values.mean())
    time_in_open = float(
        openness_values.sum(axis=1).mean() * max(0.0, dt)
    )
    tail_count = max(1, math.ceil(rollout_count * 0.1))
    exposure_cvar90 = float(np.sort(rollout_exposure)[-tail_count:].mean())
    if len(observer_uncertainties):
        reliability_total = max(1e-12, float(np.sum(observer_reliabilities)))
        weighted_sigma = float(
            np.dot(observer_uncertainties, observer_reliabilities) / reliability_total
        )
        uncertainty_risk = min(1.0, weighted_sigma / 5.0)
    else:
        uncertainty_risk = 0.0
    risk_std = float(np.std(rollout_exposure)) + 0.1 * uncertainty_risk
    risk_score = min(
        1.0,
        0.40 * los_fraction
        + 0.30 * exposure_fraction
        + 0.10 * open_fraction
        + 0.10 * exposure_cvar90
        + 0.20 * uncertainty_risk,
    )
    return ScoreComponents(
        route_length=max(0.0, float(route_length)),
        turn_cost=max(0.0, float(route_turn_cost)),
        los_fraction=_unit(los_fraction),
        exposure_fraction=_unit(exposure_fraction),
        time_in_open=max(0.0, time_in_open),
        open_fraction=_unit(open_fraction),
        uncertainty_risk=_unit(uncertainty_risk),
        risk_std=max(0.0, risk_std),
        exposure_cvar90=_unit(exposure_cvar90),
        reach_probability=_unit(reach_probability),
        progress=_unit(progress),
        invalid_fraction=_unit(invalid_fraction),
        risk_score=_unit(risk_score),
    )


def _invalid_score() -> ScoreComponents:
    return ScoreComponents(
        route_length=0.0,
        turn_cost=0.0,
        los_fraction=0.0,
        exposure_fraction=0.0,
        time_in_open=0.0,
        open_fraction=0.0,
        uncertainty_risk=0.0,
        risk_std=0.0,
        exposure_cvar90=0.0,
        reach_probability=0.0,
        progress=0.0,
        invalid_fraction=1.0,
        risk_score=1.0,
    )


def _distance(first: Point, second: Point) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
