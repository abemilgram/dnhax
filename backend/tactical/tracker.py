"""Occlusion-aware deterministic lifecycle management around the Kalman core."""

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .association import associate
from .kalman import observation_noise, predict, update
from .map import TacticalMap
from .schema import (
    EvidenceState,
    LifecycleState,
    Observation,
    SensorSpec,
    TrackSnapshot,
    TrackState,
)


@dataclass(frozen=True, slots=True)
class TrackerConfig:
    confirmation_hits: int = 3
    tentative_miss_limit: int = 2
    visible_miss_limit: int = 3
    max_coast_seconds: float = 8.0
    acceleration_variance: float = 2.0
    base_measurement_variance: float = 0.16
    initial_velocity_variance: float = 9.0
    gate_mahalanobis_squared: float = 9.21
    ambiguity_margin: float = 0.5

    def __post_init__(self) -> None:
        for name in (
            "confirmation_hits",
            "tentative_miss_limit",
            "visible_miss_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "max_coast_seconds",
            "base_measurement_variance",
            "initial_velocity_variance",
            "gate_mahalanobis_squared",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if (
            not math.isfinite(self.acceleration_variance)
            or self.acceleration_variance < 0.0
        ):
            raise ValueError("acceleration_variance must be finite and non-negative")
        if not math.isfinite(self.ambiguity_margin) or self.ambiguity_margin < 0.0:
            raise ValueError("ambiguity_margin must be finite and non-negative")


class TacticalTracker:
    """Track geometric beliefs without producing tactical control actions."""

    def __init__(
        self,
        tactical_map: TacticalMap,
        sensors: Iterable[SensorSpec],
        config: TrackerConfig | None = None,
    ) -> None:
        self.map = tactical_map
        sensor_list = tuple(sensors)
        self.sensors = {sensor.sensor_id: sensor for sensor in sensor_list}
        if len(self.sensors) != len(sensor_list):
            raise ValueError("sensor ids must be unique")
        self.config = config or TrackerConfig()
        self._tracks: list[TrackState] = []
        self._next_track_id = 1
        self._last_t: float | None = None

    def _birth(self, observation: Observation) -> None:
        position_noise = observation_noise(
            observation, self.config.base_measurement_variance
        )
        covariance = np.zeros((4, 4), dtype=np.float64)
        covariance[:2, :2] = position_noise
        covariance[2:, 2:] = (
            np.eye(2, dtype=np.float64) * self.config.initial_velocity_variance
        )
        lifecycle = (
            LifecycleState.CONFIRMED
            if self.config.confirmation_hits <= 1
            else LifecycleState.TENTATIVE
        )
        self._tracks.append(
            TrackState(
                track_id=self._next_track_id,
                t=observation.t,
                state=np.array(
                    [observation.xyz[0], observation.xyz[2], 0.0, 0.0],
                    dtype=np.float64,
                ),
                covariance=covariance,
                y=observation.xyz[1],
                lifecycle=lifecycle,
                last_observed_t=observation.t,
                sensor_ids={observation.sensor_id},
            )
        )
        self._next_track_id += 1

    def _expected_visible(self, track: TrackState) -> bool:
        point = (float(track.state[0]), track.y, float(track.state[1]))
        return any(self.map.is_visible(sensor, point) for sensor in self.sensors.values())

    def step(
        self, t: float, observations: Iterable[Observation] = ()
    ) -> tuple[TrackSnapshot, ...]:
        """Advance to ``t``, ingest observations, and return stable snapshots."""

        t = float(t)
        if not math.isfinite(t):
            raise ValueError("t must be finite")
        if self._last_t is not None and t < self._last_t:
            raise ValueError("tracker time must be monotonically non-decreasing")
        ordered = sorted(
            tuple(observations),
            key=lambda item: (item.sensor_id, item.sequence, item.xyz),
        )
        seen_sequences: set[tuple[str, int]] = set()
        for observation in ordered:
            if observation.t != t:
                raise ValueError("all observation timestamps must equal step time")
            if observation.sensor_id not in self.sensors:
                raise ValueError(f"unknown sensor_id {observation.sensor_id!r}")
            key = (observation.sensor_id, observation.sequence)
            if key in seen_sequences:
                raise ValueError("sensor observation sequences must be unique per step")
            seen_sequences.add(key)

        for track in self._tracks:
            dt = t - track.t
            track.state, track.covariance = predict(
                track.state,
                track.covariance,
                dt,
                self.config.acceleration_variance,
            )
            track.t = t

        result = associate(
            self._tracks,
            ordered,
            base_measurement_variance=self.config.base_measurement_variance,
            gate_mahalanobis_squared=self.config.gate_mahalanobis_squared,
            ambiguity_margin=self.config.ambiguity_margin,
        )
        assignments = dict(result.assignments)
        conflicting_tracks = set(result.conflicting_tracks)
        conflicting_observations = set(result.conflicting_observations)

        for track_index, track in enumerate(self._tracks):
            observation_index = assignments.get(track_index)
            if observation_index is not None:
                observation = ordered[observation_index]
                track.state, track.covariance = update(
                    track.state,
                    track.covariance,
                    observation,
                    self.config.base_measurement_variance,
                )
                track.y += observation.conf * (observation.xyz[1] - track.y)
                track.evidence = EvidenceState.OBSERVED
                track.hits += 1
                track.misses = 0
                track.expected_visible_misses = 0
                track.last_observed_t = t
                track.sensor_ids.add(observation.sensor_id)
                if track.hits >= self.config.confirmation_hits:
                    track.lifecycle = LifecycleState.CONFIRMED
            elif track_index in conflicting_tracks:
                track.evidence = EvidenceState.CONFLICTING
            else:
                track.evidence = EvidenceState.STALE
                track.misses += 1
                if self._expected_visible(track):
                    track.expected_visible_misses += 1

        for observation_index in result.unassigned_observations:
            if observation_index not in conflicting_observations:
                self._birth(ordered[observation_index])

        retained = []
        for track in self._tracks:
            if track.lifecycle is LifecycleState.TENTATIVE:
                expired = (
                    track.misses >= self.config.tentative_miss_limit
                    or t - track.last_observed_t > self.config.max_coast_seconds
                )
            else:
                expired = (
                    track.expected_visible_misses >= self.config.visible_miss_limit
                    or t - track.last_observed_t > self.config.max_coast_seconds
                )
            if not expired:
                retained.append(track)
        self._tracks = retained
        self._last_t = t
        return self.snapshots()

    def snapshots(self) -> tuple[TrackSnapshot, ...]:
        return tuple(
            TrackSnapshot(
                track_id=track.track_id,
                t=track.t,
                xyz=(float(track.state[0]), track.y, float(track.state[1])),
                velocity_xz=(float(track.state[2]), float(track.state[3])),
                covariance=tuple(
                    tuple(float(value) for value in row) for row in track.covariance
                ),
                evidence=track.evidence,
                lifecycle=track.lifecycle,
                hits=track.hits,
                misses=track.misses,
                expected_visible_misses=track.expected_visible_misses,
            )
            for track in sorted(self._tracks, key=lambda item: item.track_id)
        )
