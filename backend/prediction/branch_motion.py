"""Bounded stochastic trajectory branches on a validated navigation prior.

The generator is a source-time simulation.  It samples one start from the
last pixel-derived observation and its 2-D covariance for each requested
particle, then simulates one event schedule through validated navigation
edges.  It never initializes from ``Track.mean`` or an unconstrained Kalman
state, and it never creates additional samples while exploring alternatives.

The small v1 behavior model is deliberately disclosed here: exploration uses
probability 0.20, stopping has rate 0.12/s, reversing has rate 0.04/s, a
stopped agent waits uniformly for 0.5--2.0 s, and each branch has probability
0.20 of using half speed.  These are branch assumptions, not learned human
motion statistics.  ``config`` may provide the three ``branch_*`` fields;
the stated values are used while older EngineConfig objects are present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_EPS = 1e-10
_SNAP_RADIUS_M = 3.0
_DEFAULT_EXPLORATION = 0.20
_DEFAULT_STOP_RATE = 0.12
_DEFAULT_REVERSE_RATE = 0.04
_SLOW_PROBABILITY = 0.20
_SLOW_FACTOR = 0.50
_MIN_HOLD_S = 0.5
_MAX_HOLD_S = 2.0
_DEFAULT_MAX_EVENTS = 128
_MAX_EVENTS_HARD = 512
_VALID_MODES = {"moving", "slow", "holding", "returned", "arrived"}


def _value(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _finite_scalar(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _finite_xyz(value: Any, name: str = "xyz") -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3-vector") from exc
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    return result.copy()


def _sample_covariance(observation: Any) -> np.ndarray:
    raw = _value(observation, "covariance_xy")
    try:
        covariance = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("last observation covariance_xy must be a finite 2x2 matrix") from exc
    if covariance.shape != (2, 2) or not np.isfinite(covariance).all():
        raise ValueError("last observation covariance_xy must be a finite 2x2 matrix")
    covariance = (covariance + covariance.T) * 0.5
    values, vectors = np.linalg.eigh(covariance)
    if not np.isfinite(values).all() or float(np.min(values)) < -1e-7:
        raise ValueError("last observation covariance_xy must be positive semidefinite")
    values = np.maximum(values, 1e-8)
    return (vectors * values) @ vectors.T


def _config_float(config: Any, name: str, default: float, low: float, high: float) -> float:
    value = _finite_scalar(getattr(config, name, default), name)
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _config_int(config: Any, name: str, default: int, low: int, high: int) -> int:
    raw = getattr(config, name, default)
    if isinstance(raw, (bool, np.bool_)) or not isinstance(raw, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(raw)
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _surface_for_point(geometry: Any, point: np.ndarray) -> Any:
    surface_fn = getattr(geometry, "_surface_for_point", None)
    if not callable(surface_fn):
        raise TypeError("geometry must be a WorldGeometry with validated surfaces")
    return surface_fn(point)


def _safe_start_nodes(geometry: Any, start: np.ndarray) -> list[tuple[str, float]]:
    """Return actual validated nodes reachable by a <=3 m walkable connector."""

    if _surface_for_point(geometry, start) is None:
        return []
    nodes = getattr(geometry, "_nodes", None)
    if not isinstance(nodes, dict):
        raise TypeError("geometry must expose validated navigation nodes")
    same_surface = getattr(geometry, "_same_surface_or_elevation", None)
    lift = getattr(geometry, "_lift_for_navigation", None)
    segment_blocked = getattr(geometry, "_segment_blocked", None)
    same_level_walkable = getattr(geometry, "_same_level_walkable", None)
    if not all(callable(fn) for fn in (same_surface, lift, segment_blocked, same_level_walkable)):
        raise TypeError("geometry lacks validated navigation connector helpers")
    candidates: list[tuple[str, float]] = []
    for node_id, node in nodes.items():
        node_xyz = np.asarray(node, dtype=float)
        gap = float(np.linalg.norm(node_xyz - start))
        if not np.isfinite(gap) or gap > _SNAP_RADIUS_M + _EPS:
            continue
        if _surface_for_point(geometry, node_xyz) is None or not same_surface(start, node_xyz):
            continue
        if segment_blocked(lift(start), lift(node_xyz), include_surfaces=False):
            continue
        if abs(float(start[2]) - float(node_xyz[2])) <= 0.35 and not same_level_walkable(start, node_xyz):
            continue
        candidates.append((str(node_id), gap))
    return sorted(candidates, key=lambda item: (item[1], item[0]))


def _sample_node_path(
    geometry: Any,
    start: np.ndarray,
    goal_node: str,
    goal_nodes: set[str],
    rng: Any,
    exploration_probability: float,
) -> np.ndarray | None:
    """Choose one bounded graph walk using Dijkstra guidance at each node.

    Dijkstra supplies the shortest remaining distance, while exploration may
    select any other validated outgoing edge.  A dead-end therefore appears
    as a real walk back through the same edge, rather than being discarded as
    an impossible simple path.  The walk length is bounded independently of
    graph density.
    """

    candidates = _safe_start_nodes(geometry, start)
    nodes = geometry._nodes
    adjacency = geometry._adjacency
    if not candidates or goal_node not in nodes:
        return None
    start_node = candidates[0][0]  # nearest safe connector, deterministic tie break
    distances, _ = geometry._dijkstra_to(goal_node)
    if not np.isfinite(distances.get(start_node, float("inf"))):
        return None

    node_path = [start_node]
    explored_at: set[str] = set()
    max_hops = min(128, max(32, 4 * max(1, len(nodes))))
    for _ in range(max_hops):
        current = node_path[-1]
        if current == goal_node or current in goal_nodes:
            break
        neighbors = sorted(adjacency.get(current, []), key=lambda item: str(item[0]))
        if not neighbors:
            return None
        preferred = min(
            neighbors,
            key=lambda item: (float(item[1]) + float(distances.get(str(item[0]), float("inf"))), float(item[1]), str(item[0])),
        )[0]
        if len(neighbors) > 1 and current not in explored_at and float(rng.random()) < exploration_probability:
            alternatives = [str(neighbor) for neighbor, _ in neighbors if str(neighbor) != str(preferred)]
            next_node = str(rng.choice(alternatives))
            explored_at.add(current)
        else:
            next_node = str(preferred)
        node_path.append(next_node)
    else:
        return None
    if node_path[-1] != goal_node and node_path[-1] not in goal_nodes:
        return None

    points = [start.copy()]
    for node_id in node_path:
        node_point = np.asarray(nodes[node_id], dtype=float)
        if float(np.linalg.norm(node_point - points[-1])) > _EPS:
            points.append(node_point.copy())
    return np.asarray(points, dtype=float)


def _polyline_cumulative(path: np.ndarray) -> tuple[np.ndarray, float]:
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return cumulative, float(cumulative[-1])


def _polyline_position(path: np.ndarray, cumulative: np.ndarray, distance: float) -> np.ndarray:
    distance = float(np.clip(distance, 0.0, float(cumulative[-1])))
    if len(path) == 1:
        return path[0].copy()
    segment = int(np.searchsorted(cumulative, distance, side="right") - 1)
    segment = int(np.clip(segment, 0, len(path) - 2))
    length = float(cumulative[segment + 1] - cumulative[segment])
    fraction = 0.0 if length <= _EPS else (distance - float(cumulative[segment])) / length
    return path[segment] + fraction * (path[segment + 1] - path[segment])


def _mode_for_motion(slow: bool) -> str:
    return "slow" if slow else "moving"


def _goal_hits(path: np.ndarray, geometry: Any, goal_nodes: set[str]) -> list[tuple[float, str]]:
    """Map modeled goal nodes that occur on this exact graph polyline."""

    cumulative, _ = _polyline_cumulative(path)
    nodes = geometry._nodes
    hits: list[tuple[float, str]] = []
    for position, point in zip(cumulative, path):
        for goal_node in sorted(goal_nodes):
            if goal_node in nodes and np.linalg.norm(point - np.asarray(nodes[goal_node], dtype=float)) <= 1e-8:
                hits.append((float(position), goal_node))
    return sorted(hits, key=lambda item: (item[0], item[1]))


def _next_vertex_distance(cumulative: np.ndarray, distance: float, direction: float) -> float:
    """Return the next polyline vertex in the current travel direction."""

    if direction > 0:
        index = int(np.searchsorted(cumulative, distance, side="right"))
        index = int(np.clip(index, 1, len(cumulative) - 1))
    else:
        index = int(np.searchsorted(cumulative, distance, side="left") - 1)
        index = int(np.clip(index, 0, len(cumulative) - 2))
    return float(cumulative[index])


def _simulate_route(
    path: np.ndarray,
    horizon: float,
    speed: float,
    rng: Any,
    stop_rate: float,
    reverse_rate: float,
    max_events: int,
    target_goal_node: str,
    goal_hits: list[tuple[float, str]],
) -> tuple[np.ndarray, np.ndarray, list[str], float | None, int, str | None, float | None]:
    cumulative, total = _polyline_cumulative(path)
    goal_hits = list(goal_hits)

    def goal_at(distance: float) -> str | None:
        for goal_distance, goal_node in goal_hits:
            if abs(float(goal_distance) - float(distance)) <= 1e-8:
                return goal_node
        return None

    initial_goal = goal_at(0.0)
    if total <= _EPS or initial_goal is not None:
        first_arrival = 0.0 if initial_goal == target_goal_node else None
        return (
            np.asarray([0.0, horizon]),
            np.vstack([path[0], path[0]]),
            ["arrived", "arrived"] if initial_goal is not None else ["holding", "holding"],
            first_arrival,
            0,
            initial_goal,
            0.0 if initial_goal is not None else None,
        )

    slow = float(rng.random()) < _SLOW_PROBABILITY
    direction = 1.0
    distance = 0.0
    elapsed = 0.0
    reverse_count = 0
    first_arrival: float | None = None
    reached_goal_node: str | None = None
    reached_goal_s: float | None = None
    mode = _mode_for_motion(slow)
    times: list[float] = [0.0]
    positions: list[np.ndarray] = [path[0].copy()]
    modes: list[str] = [mode]
    event_count = 0

    def push(at: float, point: np.ndarray, next_mode: str) -> None:
        at = float(np.clip(at, 0.0, horizon))
        if at <= times[-1] + _EPS:
            # Zero-duration boundaries are not useful breakpoints. Preserve
            # the later mode/point when roundoff makes two events coincide.
            times[-1] = at
            positions[-1] = np.asarray(point, dtype=float).copy()
            modes[-1] = next_mode
        else:
            times.append(at)
            positions.append(np.asarray(point, dtype=float).copy())
            modes.append(next_mode)

    while elapsed < horizon - _EPS and event_count < max_events:
        if reached_goal_node is not None:
            push(horizon, positions[-1], "arrived")
            elapsed = horizon
            break
        if distance >= total - _EPS and direction > 0:
            distance = total
            reached_goal_node = goal_at(distance)
            reached_goal_s = elapsed if reached_goal_node is not None else None
            if reached_goal_node == target_goal_node:
                first_arrival = elapsed
            if reached_goal_node is None:
                raise ValueError("Route ended without reaching a modeled goal node")
            push(elapsed, path[-1], "arrived")
            continue
        if distance <= _EPS and direction < 0:
            # Returning to the observed start is a turn-around boundary, not
            # a new random reversal and not a shortcut through geometry.
            distance = 0.0
            direction = 1.0
            mode = _mode_for_motion(slow)
            push(elapsed, path[0], "returned")
            continue

        motion_speed = speed * (_SLOW_FACTOR if slow else 1.0)
        boundary = _next_vertex_distance(cumulative, distance, direction)
        segment_distance = abs(boundary - distance)
        edge_time = segment_distance / motion_speed
        stop_wait = float(rng.exponential(1.0 / stop_rate)) if stop_rate > 0 else float("inf")
        reverse_wait = float(rng.exponential(1.0 / reverse_rate)) if reverse_rate > 0 else float("inf")
        wait = min(edge_time, stop_wait, reverse_wait, horizon - elapsed)
        if not np.isfinite(wait) or wait < 0:
            wait = max(0.0, horizon - elapsed)
        reaches_vertex = edge_time <= wait + _EPS
        distance += direction * motion_speed * wait
        if reaches_vertex:
            # Avoid accumulating speed*dt roundoff just below a vertex, which
            # could create a zero-time loop at the same edge boundary.
            distance = boundary
        distance = float(np.clip(distance, 0.0, total))
        elapsed += wait
        point = _polyline_position(path, cumulative, distance)

        hit_goal = goal_at(distance)
        if hit_goal is not None:
            reached_goal_node = hit_goal
            reached_goal_s = elapsed
            if hit_goal == target_goal_node:
                first_arrival = elapsed
            push(elapsed, point, "arrived")
            push(horizon, point, "arrived")
            elapsed = horizon
            break
        if elapsed >= horizon - _EPS:
            push(horizon, point, mode)
            elapsed = horizon
            break
        if reaches_vertex:
            # Exact node/turn breakpoint.  The next loop chooses the next
            # validated edge from the graph polyline; no corner is cut.
            event_count += 1
            push(elapsed, point, mode)
            if mode == "returned":
                mode = _mode_for_motion(slow)
                modes[-1] = mode
            continue
        event_count += 1
        if stop_wait <= reverse_wait:
            push(elapsed, point, "holding")
            hold_duration = float(rng.uniform(_MIN_HOLD_S, _MAX_HOLD_S))
            resume_at = min(horizon, elapsed + hold_duration)
            push(resume_at, point, _mode_for_motion(slow))
            elapsed = resume_at
            mode = _mode_for_motion(slow)
        else:
            reverse_count += 1
            direction *= -1.0
            mode = "returned"
            push(elapsed, point, mode)

    if elapsed < horizon - _EPS:
        raise ValueError("trajectory event cap exceeded")
    if times[-1] < horizon - _EPS:
        push(horizon, positions[-1], modes[-1])
    times[-1] = horizon
    return (
        np.asarray(times, dtype=float),
        np.asarray(positions, dtype=float),
        modes,
        first_arrival,
        reverse_count,
        reached_goal_node,
        reached_goal_s,
    )


@dataclass(frozen=True)
class BranchTrajectory:
    """One event-broken trajectory hypothesis in elapsed source seconds."""

    times: np.ndarray
    xyz: np.ndarray
    modes: list[str]
    speed: float
    first_arrival_s: float | None
    reverse_count: int
    reached_goal_node: str | None = None
    reached_goal_s: float | None = None

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=float)
        xyz = np.asarray(self.xyz, dtype=float)
        if times.ndim != 1 or xyz.ndim != 2 or xyz.shape[1:] != (3,) or len(times) != len(xyz) or len(times) < 2:
            raise ValueError("BranchTrajectory needs matching times and Nx3 xyz arrays")
        if not np.isfinite(times).all() or not np.isfinite(xyz).all() or np.any(np.diff(times) <= _EPS):
            raise ValueError("BranchTrajectory event times/positions must be finite and increasing")
        if abs(float(times[0])) > _EPS or times[-1] <= _EPS:
            raise ValueError("BranchTrajectory times must start at 0 and end at a positive horizon")
        if len(self.modes) != len(times) or any(mode not in _VALID_MODES for mode in self.modes):
            raise ValueError("BranchTrajectory modes must align with event times")
        speed = _finite_scalar(self.speed, "speed")
        if speed <= 0:
            raise ValueError("BranchTrajectory speed must be positive")
        if isinstance(self.reverse_count, (bool, np.bool_)) or not isinstance(self.reverse_count, (int, np.integer)) or self.reverse_count < 0:
            raise ValueError("reverse_count must be a nonnegative integer")
        if self.first_arrival_s is not None:
            arrival = _finite_scalar(self.first_arrival_s, "first_arrival_s")
            if arrival < -_EPS or arrival > float(times[-1]) + _EPS:
                raise ValueError("first_arrival_s must lie within the trajectory horizon")
        if (self.reached_goal_node is None) != (self.reached_goal_s is None):
            raise ValueError("reached_goal_node and reached_goal_s must be provided together")
        if self.reached_goal_node is not None:
            if not isinstance(self.reached_goal_node, str) or not self.reached_goal_node:
                raise ValueError("reached_goal_node must be a nonempty string")
            reached = _finite_scalar(self.reached_goal_s, "reached_goal_s")
            if reached < -_EPS or reached > float(times[-1]) + _EPS:
                raise ValueError("reached_goal_s must lie within the trajectory horizon")
        object.__setattr__(self, "times", times.copy())
        object.__setattr__(self, "xyz", xyz.copy())
        object.__setattr__(self, "modes", list(self.modes))
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "reverse_count", int(self.reverse_count))

    def positions(self, elapsed: float | np.ndarray) -> np.ndarray:
        """Piecewise-linear positions, clamped to the event horizon."""

        query = np.asarray(elapsed, dtype=float)
        if not np.isfinite(query).all():
            raise ValueError("elapsed must contain only finite values")
        clamped = np.clip(query, 0.0, float(self.times[-1]))
        flat = clamped.reshape(-1)
        segments = np.searchsorted(self.times, flat, side="right") - 1
        segments = np.clip(segments, 0, len(self.times) - 2).astype(int)
        lengths = self.times[segments + 1] - self.times[segments]
        fraction = np.divide(
            flat - self.times[segments],
            lengths,
            out=np.zeros_like(flat),
            where=lengths > _EPS,
        )
        result = self.xyz[segments] + fraction[:, None] * (self.xyz[segments + 1] - self.xyz[segments])
        return result.reshape(query.shape + (3,))


def _hold_trajectory(start: np.ndarray, horizon: float, speed: float) -> BranchTrajectory:
    return BranchTrajectory(
        times=np.asarray([0.0, horizon], dtype=float),
        xyz=np.vstack([start, start]),
        modes=["holding", "holding"],
        speed=speed,
        first_arrival_s=None,
        reverse_count=0,
    )


def generate_trajectories(geometry: Any, config: Any, track: Any, intent: Any, rng: Any) -> list[BranchTrajectory]:
    """Generate at most ``config.particles`` bounded branches.

    Each loop iteration is one attempted particle.  Invalid covariance/start
    samples, missing walkable connectors, and disconnected goals are omitted;
    no replacement samples or synthetic fallback paths are fabricated.
    """

    if rng is None or not all(callable(getattr(rng, name, None)) for name in ("random", "normal", "multivariate_normal")):
        raise TypeError("rng must be a NumPy-compatible random generator")
    horizon = _config_float(config, "horizon_s", 10.0, 1e-6, 120.0)
    particles = _config_int(config, "particles", 64, 1, 4096)
    base_speed = _config_float(config, "speed_mps", 2.5, 1e-6, 100.0)
    speed_std = _config_float(config, "speed_std_mps", 0.6, 0.0, 100.0)
    exploration = _config_float(config, "branch_exploration_probability", _DEFAULT_EXPLORATION, 0.0, 1.0)
    stop_rate = _config_float(config, "branch_stop_rate_per_s", _DEFAULT_STOP_RATE, 0.0, 100.0)
    reverse_rate = _config_float(config, "branch_reverse_rate_per_s", _DEFAULT_REVERSE_RATE, 0.0, 100.0)
    max_events = _config_int(config, "max_events_per_sample", _DEFAULT_MAX_EVENTS, 1, _MAX_EVENTS_HARD)

    observation = _value(track, "last_observation")
    if observation is None:
        return []
    observed_xyz = _finite_xyz(_value(observation, "xyz"), "last_observation.xyz")
    covariance = _sample_covariance(observation)
    intent_kind = _value(intent, "kind")
    if intent_kind not in {"hold", "route"}:
        raise ValueError("intent kind must be hold or route")
    goal_node = _value(intent, "goal_node")
    if intent_kind == "route" and not isinstance(goal_node, str):
        raise ValueError("route intent needs a goal_node")
    modeled_goal_nodes: set[str] = {str(goal_node)} if intent_kind == "route" else set()
    prior = getattr(geometry, "prior", None)
    for prior_intent in getattr(prior, "intents", []):
        if _value(prior_intent, "kind") == "route":
            prior_goal = _value(prior_intent, "goal_node")
            if isinstance(prior_goal, str) and prior_goal:
                modeled_goal_nodes.add(prior_goal)

    trajectories: list[BranchTrajectory] = []
    for _ in range(particles):
        sampled_xy = np.asarray(rng.multivariate_normal(observed_xyz[:2], covariance), dtype=float)
        if sampled_xy.shape != (2,) or not np.isfinite(sampled_xy).all():
            continue
        start = np.r_[sampled_xy, observed_xyz[2]]
        speed_draw = float(rng.normal(base_speed, speed_std))
        if not np.isfinite(speed_draw):
            continue
        speed = float(np.clip(speed_draw, 0.15, 8.0))
        # Every branch starts on authored walkable geometry and has a safe
        # nearby graph-node connector. Routes then use only validated edges.
        if not _safe_start_nodes(geometry, start):
            continue
        if intent_kind == "hold":
            trajectories.append(_hold_trajectory(start, horizon, speed))
            continue
        path = _sample_node_path(
            geometry,
            start,
            goal_node,
            modeled_goal_nodes,
            rng,
            exploration,
        )
        if path is None:
            continue
        try:
            goal_hits = _goal_hits(path, geometry, modeled_goal_nodes)
            times, xyz, modes, arrival, reversals, reached_goal_node, reached_goal_s = _simulate_route(
                path,
                horizon,
                speed,
                rng,
                stop_rate,
                reverse_rate,
                max_events,
                goal_node,
                goal_hits,
            )
            trajectories.append(
                BranchTrajectory(
                    times,
                    xyz,
                    modes,
                    speed,
                    arrival,
                    reversals,
                    reached_goal_node,
                    reached_goal_s,
                )
            )
        except (ValueError, FloatingPointError):
            # A malformed sampled branch is simply an unsuccessful attempt;
            # it must not produce an unconstrained replacement trajectory.
            continue
    return trajectories


__all__ = ["BranchTrajectory", "generate_trajectories"]
