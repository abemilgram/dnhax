"""Deterministic gated assignment for small track sets."""

from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Sequence

import numpy as np

from .kalman import innovation
from .schema import Observation, TrackState


@dataclass(frozen=True, slots=True)
class AssociationResult:
    assignments: tuple[tuple[int, int], ...]
    unassigned_tracks: tuple[int, ...]
    unassigned_observations: tuple[int, ...]
    conflicting_tracks: tuple[int, ...]
    conflicting_observations: tuple[int, ...]
    costs: tuple[tuple[float, ...], ...]


def _ambiguous_candidates(
    costs: np.ndarray, gate: float, margin: float
) -> tuple[set[int], set[int]]:
    conflicting_tracks: set[int] = set()
    conflicting_observations: set[int] = set()
    for track_index, row in enumerate(costs):
        candidates = sorted(
            (float(cost), observation_index)
            for observation_index, cost in enumerate(row)
            if cost <= gate
        )
        if len(candidates) >= 2 and candidates[1][0] - candidates[0][0] <= margin:
            conflicting_tracks.add(track_index)
            conflicting_observations.update((candidates[0][1], candidates[1][1]))
    for observation_index in range(costs.shape[1]):
        candidates = sorted(
            (float(costs[track_index, observation_index]), track_index)
            for track_index in range(costs.shape[0])
            if costs[track_index, observation_index] <= gate
        )
        if len(candidates) >= 2 and candidates[1][0] - candidates[0][0] <= margin:
            conflicting_observations.add(observation_index)
            conflicting_tracks.update((candidates[0][1], candidates[1][1]))
    return conflicting_tracks, conflicting_observations


def associate(
    tracks: Sequence[TrackState],
    observations: Sequence[Observation],
    *,
    base_measurement_variance: float,
    gate_mahalanobis_squared: float = 9.21,
    miss_cost: float | None = None,
    ambiguity_margin: float = 0.5,
) -> AssociationResult:
    """Find a deterministic minimum-cost assignment including track misses.

    The dynamic program is exact for at most 16 observations. Ambiguous
    near-equal alternatives are withheld instead of being arbitrarily forced.
    """

    if len(observations) > 16 or len(tracks) > 16:
        raise ValueError("association supports at most 16 tracks and observations")
    if gate_mahalanobis_squared <= 0.0 or not math.isfinite(
        gate_mahalanobis_squared
    ):
        raise ValueError("gate_mahalanobis_squared must be finite and positive")
    if ambiguity_margin < 0.0 or not math.isfinite(ambiguity_margin):
        raise ValueError("ambiguity_margin must be finite and non-negative")
    miss_cost = (
        gate_mahalanobis_squared + 1.0 if miss_cost is None else float(miss_cost)
    )
    if miss_cost <= 0.0 or not math.isfinite(miss_cost):
        raise ValueError("miss_cost must be finite and positive")

    costs = np.full((len(tracks), len(observations)), np.inf, dtype=np.float64)
    for track_index, track in enumerate(tracks):
        for observation_index, observation in enumerate(observations):
            _, _, distance = innovation(
                track.state,
                track.covariance,
                observation,
                base_measurement_variance,
            )
            costs[track_index, observation_index] = distance

    conflicting_tracks, conflicting_observations = _ambiguous_candidates(
        costs, gate_mahalanobis_squared, ambiguity_margin
    )

    @lru_cache(maxsize=None)
    def solve(track_index: int, used: int) -> tuple[float, tuple[int, ...]]:
        if track_index == len(tracks):
            return 0.0, ()
        missed_cost, missed_tail = solve(track_index + 1, used)
        best = (miss_cost + missed_cost, (-1,) + missed_tail)
        if track_index in conflicting_tracks:
            return best
        for observation_index in range(len(observations)):
            if (
                used & (1 << observation_index)
                or observation_index in conflicting_observations
                or costs[track_index, observation_index]
                > gate_mahalanobis_squared
            ):
                continue
            tail_cost, tail = solve(
                track_index + 1, used | (1 << observation_index)
            )
            candidate = (
                float(costs[track_index, observation_index]) + tail_cost,
                (observation_index,) + tail,
            )
            if candidate[0] < best[0] - 1e-12 or (
                abs(candidate[0] - best[0]) <= 1e-12 and candidate[1] < best[1]
            ):
                best = candidate
        return best

    _, choices = solve(0, 0)
    assignments = tuple(
        (track_index, observation_index)
        for track_index, observation_index in enumerate(choices)
        if observation_index >= 0
    )
    assigned_tracks = {track for track, _ in assignments}
    assigned_observations = {observation for _, observation in assignments}
    return AssociationResult(
        assignments=assignments,
        unassigned_tracks=tuple(
            index for index in range(len(tracks)) if index not in assigned_tracks
        ),
        unassigned_observations=tuple(
            index
            for index in range(len(observations))
            if index not in assigned_observations
        ),
        conflicting_tracks=tuple(sorted(conflicting_tracks)),
        conflicting_observations=tuple(sorted(conflicting_observations)),
        costs=tuple(tuple(float(value) for value in row) for row in costs),
    )
