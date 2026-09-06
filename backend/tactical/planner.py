"""Seeded, CPU-light simulation of fixed authored navigation intents."""

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np

from .map import TacticalMap
from .navgraph import NavGraph
from .schema import (
    ActorState,
    IntentKind,
    PlanRanking,
    ScoreComponents,
    TimedPoint,
    TrackSnapshot,
    TrajectoryCandidate,
)
from .scoring import (
    interpolate_polyline,
    observer_reliability,
    polyline_length,
    score_rollouts,
    turn_cost,
)


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    horizon: float = 9.0
    step: float = 0.15
    rollouts_per_intent: int = 64
    scenario_seed: int | str = 0
    preferred_speed: float = 4.0
    response_time: float = 0.45
    max_acceleration: float = 4.0
    acceleration_noise: float = 0.7
    max_observers: int = 16
    max_graph_snap_distance: float = 2.5
    actor_horizontal_clearance: float = 0.35
    goal_radius: float = 0.25

    def __post_init__(self) -> None:
        for name in (
            "horizon",
            "step",
            "preferred_speed",
            "response_time",
            "max_acceleration",
            "max_graph_snap_distance",
            "goal_radius",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)
        noise = float(self.acceleration_noise)
        if not math.isfinite(noise) or noise < 0.0:
            raise ValueError("acceleration_noise must be finite and non-negative")
        object.__setattr__(self, "acceleration_noise", noise)
        clearance = float(self.actor_horizontal_clearance)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                "actor_horizontal_clearance must be finite and non-negative"
            )
        object.__setattr__(self, "actor_horizontal_clearance", clearance)
        for name in ("rollouts_per_intent", "max_observers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_observers > 16:
            raise ValueError("max_observers must not exceed 16")
        if not isinstance(self.scenario_seed, (int, str)):
            raise ValueError("scenario_seed must be an int or string")


class TacticalPlanner:
    """Rank HOLD/CROSS/FLANK using only authored routes and coarse risk."""

    def __init__(
        self,
        tactical_map: TacticalMap,
        config: PlannerConfig | None = None,
    ) -> None:
        self.map = tactical_map
        self.config = config or PlannerConfig()
        self.graph = NavGraph(
            tactical_map,
            actor_horizontal_clearance=self.config.actor_horizontal_clearance,
        )

    def plan(
        self,
        actor: ActorState,
        observers: Iterable[TrackSnapshot] = (),
        *,
        cycle_index: int = 0,
    ) -> PlanRanking:
        if (
            isinstance(cycle_index, bool)
            or not isinstance(cycle_index, int)
            or cycle_index < 0
        ):
            raise ValueError("cycle_index must be a non-negative integer")
        nearest = self.graph.nearest_node(actor.xyz)
        snap_distance = _distance(actor.xyz, nearest.xyz)
        if snap_distance > self.config.max_graph_snap_distance:
            candidates = tuple(
                self._invalid_candidate(actor.t, kind, "actor_outside_corridor")
                for kind in IntentKind
            )
            return PlanRanking(actor.t, cycle_index, candidates)
        if self.map.segment_collides(
            actor.xyz,
            nearest.xyz,
            horizontal_clearance=self.config.actor_horizontal_clearance,
        ):
            candidates = tuple(
                self._invalid_candidate(actor.t, kind, "actor_connector_blocked")
                for kind in IntentKind
            )
            return PlanRanking(actor.t, cycle_index, candidates)

        tracks = self._select_observers(tuple(observers), actor)
        rng = np.random.default_rng(self._stable_seed(cycle_index))
        steps = max(1, int(round(self.config.horizon / self.config.step))) + 1
        acceleration_noise = rng.standard_normal(
            (self.config.rollouts_per_intent, steps - 1)
        )
        observer_positions, reliabilities, uncertainties = self._sample_observers(
            tracks, actor.t, steps, rng
        )

        candidates = []
        for kind in IntentKind:
            try:
                node_path = self.graph.intent_path(kind, nearest.node_id)
                route_points = self.graph.path_points(node_path)
                if snap_distance > 1e-12:
                    route_points = (actor.xyz,) + route_points
            except ValueError:
                candidates.append(
                    self._invalid_candidate(actor.t, kind, "route_unavailable")
                )
                continue
            candidates.append(
                self._simulate(
                    actor,
                    kind,
                    route_points,
                    acceleration_noise,
                    observer_positions,
                    reliabilities,
                    uncertainties,
                )
            )
        priority = {
            IntentKind.HOLD: 0,
            IntentKind.FLANK: 1,
            IntentKind.CROSS: 2,
        }
        candidates.sort(
            key=lambda item: (-item.utility, priority[item.intent])
        )
        return PlanRanking(actor.t, cycle_index, tuple(candidates))

    def _select_observers(
        self, tracks: tuple[TrackSnapshot, ...], actor: ActorState
    ) -> tuple[TrackSnapshot, ...]:
        return tuple(
            sorted(
                tracks,
                key=lambda track: (
                    -observer_reliability(track, actor.t),
                    _distance(actor.xyz, track.xyz),
                    track.track_id,
                ),
            )[: self.config.max_observers]
        )

    def _sample_observers(
        self,
        tracks: tuple[TrackSnapshot, ...],
        start_t: float,
        steps: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rollouts = self.config.rollouts_per_intent
        positions = np.empty((rollouts, len(tracks), steps, 3), dtype=np.float64)
        reliabilities = np.empty(len(tracks), dtype=np.float64)
        uncertainties = np.empty(len(tracks), dtype=np.float64)
        standard_samples = rng.standard_normal((rollouts, len(tracks), 4))
        elapsed = np.arange(steps, dtype=np.float64) * self.config.step
        for index, track in enumerate(tracks):
            covariance = np.asarray(track.covariance, dtype=np.float64)
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            transform = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
            mean = np.array(
                [
                    track.xyz[0],
                    track.xyz[2],
                    track.velocity_xz[0],
                    track.velocity_xz[1],
                ],
                dtype=np.float64,
            )
            samples = mean + standard_samples[:, index, :] @ transform.T
            positions[:, index, :, 0] = (
                samples[:, 0, None] + samples[:, 2, None] * elapsed
            )
            positions[:, index, :, 1] = track.xyz[1]
            positions[:, index, :, 2] = (
                samples[:, 1, None] + samples[:, 3, None] * elapsed
            )
            reliabilities[index] = observer_reliability(track, start_t)
            uncertainties[index] = math.sqrt(
                max(0.0, float(covariance[0, 0] + covariance[1, 1]))
            )
        return positions, reliabilities, uncertainties

    def _simulate(
        self,
        actor: ActorState,
        kind: IntentKind,
        route_points: tuple[tuple[float, float, float], ...],
        common_noise: np.ndarray,
        observer_positions: np.ndarray,
        reliabilities: np.ndarray,
        uncertainties: np.ndarray,
    ) -> TrajectoryCandidate:
        route_length = polyline_length(route_points)
        rollout_count = self.config.rollouts_per_intent
        steps = common_noise.shape[1] + 1
        positions = np.empty((rollout_count, steps, 3), dtype=np.float64)
        distances = np.zeros((rollout_count, steps), dtype=np.float64)
        speed = np.full(
            rollout_count,
            min(self.config.preferred_speed, math.hypot(*actor.velocity_xz)),
            dtype=np.float64,
        )
        for step_index in range(1, steps):
            commanded = (self.config.preferred_speed - speed) / self.config.response_time
            acceleration = np.clip(
                commanded + self.config.acceleration_noise * common_noise[:, step_index - 1],
                -self.config.max_acceleration,
                self.config.max_acceleration,
            )
            speed = np.clip(
                speed + acceleration * self.config.step,
                0.0,
                self.config.preferred_speed,
            )
            distances[:, step_index] = np.minimum(
                route_length,
                distances[:, step_index - 1] + speed * self.config.step,
            )
        for rollout in range(rollout_count):
            for step_index in range(steps):
                positions[rollout, step_index] = interpolate_polyline(
                    route_points, float(distances[rollout, step_index])
                )

        if route_length <= self.config.goal_radius:
            reached = np.ones(rollout_count, dtype=bool)
            progress = 1.0
        else:
            reached = distances[:, -1] >= route_length - self.config.goal_radius
            progress = float(np.mean(np.minimum(1.0, distances[:, -1] / route_length)))
        score = score_rollouts(
            self.map,
            positions,
            observer_positions,
            reliabilities,
            uncertainties,
            dt=self.config.step,
            route_length=route_length,
            route_turn_cost=turn_cost(route_points),
            reach_probability=float(np.mean(reached)),
            progress=progress,
            invalid_fraction=0.0,
        )
        utility = (
            1.25 * score.progress
            + 0.75 * score.reach_probability
            - 1.10 * score.los_fraction
            - 0.90 * score.exposure_fraction
            - 0.45 * score.open_fraction
            - 0.45 * score.exposure_cvar90
            - 1.50 * score.invalid_fraction
            - 0.10 * score.turn_cost
            - 0.20 * score.uncertainty_risk
        )
        median_distances = np.median(distances, axis=0)
        public_route = tuple(
            TimedPoint(
                t=actor.t + index * self.config.step,
                xyz=interpolate_polyline(route_points, float(distance)),
            )
            for index, distance in enumerate(median_distances)
        )
        reasons = ["route_valid"]
        if score.risk_score >= 0.55:
            reasons.append("geometric_risk_high")
        elif score.risk_score >= 0.25:
            reasons.append("geometric_risk_medium")
        else:
            reasons.append("geometric_risk_low")
        return TrajectoryCandidate(
            intent=kind,
            valid=True,
            utility=float(utility),
            score=score,
            route=public_route,
            reasons=tuple(reasons),
        )

    def _invalid_candidate(
        self, t: float, kind: IntentKind, reason: str
    ) -> TrajectoryCandidate:
        score = ScoreComponents(
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
        return TrajectoryCandidate(
            intent=kind,
            valid=False,
            utility=-1.5,
            score=score,
            route=(),
            reasons=(reason,),
        )

    def _stable_seed(self, cycle_index: int) -> int:
        payload = f"{self.config.scenario_seed}:{cycle_index}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def plan_trajectories(
    tactical_map: TacticalMap,
    actor: ActorState,
    observers: Iterable[TrackSnapshot] = (),
    *,
    config: PlannerConfig | None = None,
    cycle_index: int = 0,
) -> PlanRanking:
    return TacticalPlanner(tactical_map, config).plan(
        actor, observers, cycle_index=cycle_index
    )


def _distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))
