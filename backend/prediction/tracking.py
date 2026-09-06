"""Bounded, anonymous tracking for pixel-derived world observations.

The tracker deliberately knows nothing about game entities.  It keeps a small
set of constant-velocity hypotheses on the ground plane and uses only the
metric pixel-projection result in :class:`Observation`.

The association and update code is intentionally conservative.  In
particular, observations from different sensors arriving at the same time
are fused with covariance intersection because their projection errors may be
correlated.  This is less sharp than assuming independent cameras, but avoids
claiming unrealistic certainty when they share calibration, scene geometry,
or detector errors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .schema import EngineConfig, Observation


_EPS = 1e-9
_OBS_TOL = 1e-7
_SURFACE_Z_TOL = 0.40
_CLOSE_XY_M = 1.25


def _as_float(value: Any, name: str) -> float:
    """Convert a scalar and reject bool, non-finite, and non-real values."""

    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        out = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(out):
        raise ValueError(f"{name} must be a finite number")
    return out


def _validate_spd(matrix: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    """Return a symmetric positive-definite copy of a covariance matrix."""

    try:
        value = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric {shape[0]}x{shape[1]} matrix") from exc
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite {shape[0]}x{shape[1]} matrix")
    if not np.allclose(value, value.T, rtol=0.0, atol=1e-8):
        raise ValueError(f"{name} must be symmetric")
    value = (value + value.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(value)
    if np.min(eigenvalues) <= _EPS:
        raise ValueError(f"{name} must be positive definite")
    return value


def _validate_observation(observation: Observation) -> tuple[np.ndarray, np.ndarray, float, float, str, str, str]:
    """Validate an Observation at the tracker boundary and return arrays."""

    if observation is None:
        raise ValueError("observation must not be None")
    try:
        xyz = np.asarray(observation.xyz, dtype=float)
        covariance = _validate_spd(observation.covariance_xy, (2, 2), "observation covariance_xy")
    except AttributeError as exc:
        raise ValueError("observation must implement the Observation contract") from exc
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("observation xyz/covariance_xy is invalid") from exc
    if xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
        raise ValueError("observation xyz must be three finite numbers")
    t = _as_float(observation.t, "observation t")
    if t < 0:
        raise ValueError("observation t must be non-negative")
    confidence = _as_float(observation.confidence, "observation confidence")
    if confidence < 0 or confidence > 1:
        raise ValueError("observation confidence must be between 0 and 1")
    sensor = getattr(observation, "sensor_id", None)
    frame = getattr(observation, "frame_id", None)
    surface = getattr(observation, "surface_id", None)
    if not isinstance(sensor, str) or not sensor:
        raise ValueError("observation sensor_id must be a non-empty string")
    if not isinstance(frame, str) or not frame:
        raise ValueError("observation frame_id must be a non-empty string")
    if not isinstance(surface, str) or not surface:
        raise ValueError("observation surface_id must be a non-empty string")
    return xyz, covariance, t, confidence, sensor, frame, surface


def _project_psd(matrix: np.ndarray, floor: float = 1e-10) -> np.ndarray:
    """Symmetrize and remove round-off negative eigenvalues."""

    value = (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T) * 0.5
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not np.all(np.isfinite(value)):
        raise ValueError("internal covariance became invalid")
    eigenvalues, eigenvectors = np.linalg.eigh(value)
    if np.min(eigenvalues) < -1e-7:
        raise ValueError("internal covariance became non-PSD")
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


def _covariance_intersection(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse two 2D observations without assuming independent errors.

    A small deterministic grid is sufficient here and avoids bringing an
    optimizer dependency into the backend.  The selected covariance minimizes
    trace, with the first grid point winning exact ties for deterministic
    behavior.
    """

    inverse_a = np.linalg.inv(covariance_a)
    inverse_b = np.linalg.inv(covariance_b)
    best_mean: np.ndarray | None = None
    best_covariance: np.ndarray | None = None
    best_objective = float("inf")
    # Include both endpoints: when one source is much less certain, the
    # conservative optimum may be to retain only the better covariance.
    for weight in np.linspace(0.0, 1.0, 21):
        information = weight * inverse_a + (1.0 - weight) * inverse_b
        covariance = _project_psd(np.linalg.inv(information))
        mean = covariance @ (
            weight * inverse_a @ mean_a + (1.0 - weight) * inverse_b @ mean_b
        )
        objective = float(np.trace(covariance))
        if objective < best_objective - 1e-12:
            best_objective = objective
            best_mean = mean
            best_covariance = covariance
    if best_mean is None or best_covariance is None:
        raise ValueError("covariance intersection failed")
    return best_mean, best_covariance


@dataclass
class Track:
    """One anonymous 2D constant-velocity hypothesis."""

    id: str
    mean: np.ndarray
    covariance: np.ndarray
    z: float
    surface_id: str
    t: float
    last_seen: float
    last_sensor: str
    last_observation: Observation
    status: str
    revision: int

    # State needed to re-run a same-time update after conservative CI fusion.
    # These fields are private implementation details and retain the public
    # dataclass constructor/API requested by the prediction engine.
    _fusion_t: float | None = field(default=None, repr=False, compare=False)
    _fusion_mean: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fusion_covariance: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fusion_prior_mean: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fusion_prior_covariance: np.ndarray | None = field(default=None, repr=False, compare=False)
    _fusion_keys: set[tuple[str, str]] = field(default_factory=set, repr=False, compare=False)
    _fusion_newborn: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("track id must be a non-empty string")
        self.mean = np.asarray(self.mean, dtype=float).copy()
        self.covariance = np.asarray(self.covariance, dtype=float).copy()
        if self.mean.shape != (4,) or not np.all(np.isfinite(self.mean)):
            raise ValueError("track mean must be four finite numbers")
        if self.covariance.shape != (4, 4) or not np.all(np.isfinite(self.covariance)):
            raise ValueError("track covariance must be a finite 4x4 matrix")
        self.covariance = _project_psd(self.covariance)
        self.z = _as_float(self.z, "track z")
        self.t = _as_float(self.t, "track t")
        self.last_seen = _as_float(self.last_seen, "track last_seen")
        if self.t < 0 or self.last_seen < 0:
            raise ValueError("track timestamps must be non-negative")
        if not isinstance(self.surface_id, str) or not self.surface_id:
            raise ValueError("track surface_id must be a non-empty string")
        if not isinstance(self.last_sensor, str) or not self.last_sensor:
            raise ValueError("track last_sensor must be a non-empty string")
        if self.status not in {"observed", "stale", "conflicting"}:
            raise ValueError("track status must be observed, stale, or conflicting")
        if not isinstance(self.revision, (int, np.integer)) or self.revision < 0:
            raise ValueError("track revision must be a non-negative integer")

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable public state for API responses."""

        return {
            "id": self.id,
            "xyz": [float(self.mean[0]), float(self.mean[1]), float(self.z)],
            "velocity_xy": [float(self.mean[2]), float(self.mean[3])],
            "covariance_xy": [
                [float(self.covariance[0, 0]), float(self.covariance[0, 1])],
                [float(self.covariance[1, 0]), float(self.covariance[1, 1])],
            ],
            "surface_id": self.surface_id,
            "t": float(self.t),
            "status": self.status,
            "last_seen": float(self.last_seen),
            "revision": int(self.revision),
        }


class Tracker:
    """Bounded, monotonic, anonymous tracker driven by pixel observations."""

    def __init__(self, config: EngineConfig, transition_validator=None):
        if config is None:
            raise ValueError("config is required")
        self.config = config
        self.transition_validator = transition_validator
        self.tracks: dict[str, Track] = {}
        self.now = 0.0
        self._next_id = 1

    def _validate_time(self, t: Any, name: str = "t") -> float:
        value = _as_float(t, name)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
        return value

    def _process_noise(self, dt: float) -> np.ndarray:
        acceleration_variance = float(self.config.acceleration_std_mps2) ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        base = acceleration_variance * np.array(
            [[dt4 / 4.0, dt3 / 2.0], [dt3 / 2.0, dt2]], dtype=float
        )
        noise = np.zeros((4, 4), dtype=float)
        noise[np.ix_([0, 2], [0, 2])] = base
        noise[np.ix_([1, 3], [1, 3])] = base
        return noise

    def advance(self, t: float) -> None:
        """Propagate every live track to monotonic time ``t`` and expire old ones."""

        target = self._validate_time(t)
        if target < self.now:
            raise ValueError(f"timestamp {target} is earlier than tracker time {self.now}")
        for track in list(self.tracks.values()):
            dt = target - track.t
            if dt < 0:
                raise ValueError("track timestamp is ahead of tracker time")
            if dt > _OBS_TOL:
                transition = np.eye(4, dtype=float)
                transition[0, 2] = dt
                transition[1, 3] = dt
                track.mean = transition @ track.mean
                track.covariance = _project_psd(
                    transition @ track.covariance @ transition.T + self._process_noise(dt)
                )
                track.t = target
                track.revision += 1
            else:
                track.t = target
            age = target - track.last_seen
            if age > float(self.config.expire_after_s):
                del self.tracks[track.id]
                continue
            if age > float(self.config.stale_after_s):
                track.status = "stale"
            elif track.status != "conflicting":
                track.status = "observed"
        self.now = target

    def _surface_compatible(self, track: Track, xyz: np.ndarray, surface_id: str) -> bool:
        dz = abs(float(xyz[2]) - track.z)
        if dz > _SURFACE_Z_TOL:
            return bool(self.transition_validator and self.transition_validator(track, xyz, surface_id, self.now))
        if track.surface_id == surface_id:
            # Same declared surface is the normal case.  The z consistency
            # check above still protects against malformed projections.
            return True
        distance = float(np.linalg.norm(xyz[:2] - track.mean[:2]))
        return distance <= _CLOSE_XY_M

    def _candidate_cost(self, track: Track, xyz: np.ndarray, covariance: np.ndarray, surface_id: str) -> float | None:
        if not self._surface_compatible(track, xyz, surface_id):
            return None
        innovation = xyz[:2] - track.mean[:2]
        innovation_covariance = _project_psd(track.covariance[:2, :2] + covariance)
        try:
            cost = float(innovation @ np.linalg.solve(innovation_covariance, innovation))
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(cost) or cost > float(self.config.association_gate):
            return None
        return max(cost, 0.0)

    def _update_with_measurement(
        self,
        prior_mean: np.ndarray,
        prior_covariance: np.ndarray,
        measurement: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        h = np.zeros((2, 4), dtype=float)
        h[0, 0] = 1.0
        h[1, 1] = 1.0
        innovation = measurement - h @ prior_mean
        innovation_covariance = _project_psd(h @ prior_covariance @ h.T + measurement_covariance)
        try:
            gain = prior_covariance @ h.T @ np.linalg.inv(innovation_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("internal Kalman innovation covariance is invalid") from exc
        mean = prior_mean + gain @ innovation
        identity_minus_gain = np.eye(4) - gain @ h
        covariance = (
            identity_minus_gain @ prior_covariance @ identity_minus_gain.T
            + gain @ measurement_covariance @ gain.T
        )
        return mean, _project_psd(covariance)

    def _fuse_observation(self, track: Track, observation: Observation, xyz: np.ndarray, covariance: np.ndarray, t: float) -> None:
        key = (str(observation.sensor_id), str(observation.frame_id))
        if track._fusion_t is not None and abs(track._fusion_t - t) <= _OBS_TOL:
            if key in track._fusion_keys:
                # Retransmission of one packet must not count as a second sensor.
                track.last_observation = observation
                track.last_sensor = str(observation.sensor_id)
                return
            if track._fusion_mean is None or track._fusion_covariance is None:
                raise ValueError("internal fusion state is incomplete")
            fused_mean, fused_covariance = _covariance_intersection(
                track._fusion_mean,
                track._fusion_covariance,
                xyz[:2],
                covariance,
            )
            track._fusion_mean = fused_mean
            track._fusion_covariance = fused_covariance
            track._fusion_keys.add(key)
            if track._fusion_newborn:
                # A newborn track's position already represents its first
                # observation.  Replacing that block with the CI result keeps
                # the independent velocity prior and avoids counting the
                # first camera's posterior as a second measurement.
                track.mean[:2] = fused_mean
                track.covariance[:2, :2] = fused_covariance
                track.covariance = _project_psd(track.covariance)
            else:
                if track._fusion_prior_mean is None or track._fusion_prior_covariance is None:
                    raise ValueError("internal fusion prior is incomplete")
                track.mean, track.covariance = self._update_with_measurement(
                    track._fusion_prior_mean,
                    track._fusion_prior_covariance,
                    fused_mean,
                    fused_covariance,
                )
        else:
            track._fusion_t = t
            track._fusion_prior_mean = track.mean.copy()
            track._fusion_prior_covariance = track.covariance.copy()
            track._fusion_mean = xyz[:2].copy()
            track._fusion_covariance = covariance.copy()
            track._fusion_keys = {key}
            track._fusion_newborn = False
            track.mean, track.covariance = self._update_with_measurement(
                track._fusion_prior_mean,
                track._fusion_prior_covariance,
                xyz[:2],
                covariance,
            )
        track.z = float(xyz[2])
        track.surface_id = str(observation.surface_id)
        track.last_seen = t
        track.t = t
        track.last_sensor = str(observation.sensor_id)
        track.last_observation = observation
        track.status = "observed"
        track.revision += 1

    def _new_track(self, observation: Observation, xyz: np.ndarray, covariance: np.ndarray, t: float) -> Track:
        track_id = f"track-{self._next_id:04d}"
        self._next_id += 1
        velocity_variance = max(float(self.config.speed_std_mps) ** 2, 0.25)
        # A newborn track is initialized directly from the first observation;
        # its velocity remains an independent prior until a later frame.
        state_mean = np.array([xyz[0], xyz[1], 0.0, 0.0], dtype=float)
        state_covariance = np.zeros((4, 4), dtype=float)
        state_covariance[:2, :2] = covariance
        state_covariance[2:, 2:] = np.eye(2) * velocity_variance
        track = Track(
            id=track_id,
            mean=state_mean,
            covariance=state_covariance,
            z=float(xyz[2]),
            surface_id=str(observation.surface_id),
            t=t,
            last_seen=t,
            last_sensor=str(observation.sensor_id),
            last_observation=observation,
            status="observed",
            revision=1,
        )
        track._fusion_t = t
        track._fusion_prior_mean = None
        track._fusion_prior_covariance = None
        track._fusion_mean = xyz[:2].copy()
        track._fusion_covariance = covariance.copy()
        track._fusion_keys = {(str(observation.sensor_id), str(observation.frame_id))}
        track._fusion_newborn = True
        return track

    def _ambiguity_margin(self, first: float, second: float) -> bool:
        margin = float(self.config.hysteresis_margin)
        return second - first <= margin * max(1.0, first, second) + 1e-9

    def ingest(self, observations: list[Observation], t: float, sensor_id: str) -> dict[str, list[str]]:
        """Advance and associate one frame of observations.

        The call is monotonic: ``t`` and every observation timestamp must be at
        or after the tracker's previous time, and no observation may be from
        after the frame being ingested.  Ambiguous gated matches are reported
        in ``conflicts`` and left unassigned so a crossing cannot silently
        exchange anonymous identities.
        """

        frame_time = self._validate_time(t)
        if frame_time < self.now:
            raise ValueError(f"timestamp {frame_time} is earlier than tracker time {self.now}")
        if not isinstance(sensor_id, str) or not sensor_id:
            raise ValueError("sensor_id must be a non-empty string")
        if observations is None:
            raise ValueError("observations must be a list")
        try:
            observation_list = list(observations)
        except TypeError as exc:
            raise ValueError("observations must be a list") from exc
        previous_time = self.now
        validated: list[tuple[Observation, np.ndarray, np.ndarray, float, str]] = []
        for observation in observation_list:
            xyz, covariance, observation_time, confidence, observation_sensor, _, _ = _validate_observation(observation)
            if observation_sensor != sensor_id:
                raise ValueError("observation sensor_id must match ingest sensor_id")
            if observation_time < previous_time:
                raise ValueError("observation timestamp is earlier than tracker time")
            if abs(observation_time - frame_time) > _OBS_TOL:
                raise ValueError("observation timestamp must match ingest time")
            if confidence >= float(self.config.min_detection_confidence):
                validated.append((observation, xyz, covariance, confidence, observation_sensor))
        self.advance(frame_time)
        result: dict[str, list[str]] = {"matched": [], "created": [], "conflicts": []}
        if not validated:
            return result
        created_this_ingest: set[str] = set()

        live_tracks = list(self.tracks.values())
        costs: list[list[tuple[int, float]]] = []
        for observation, xyz, covariance, _, _ in validated:
            edges: list[tuple[int, float]] = []
            surface_id = str(observation.surface_id)
            for track_index, track in enumerate(live_tracks):
                cost = self._candidate_cost(track, xyz, covariance, surface_id)
                if cost is not None:
                    edges.append((track_index, cost))
            edges.sort(key=lambda item: (item[1], live_tracks[item[0]].id))
            costs.append(edges)

        ambiguous_observations: set[int] = set()
        ambiguous_tracks: set[int] = set()
        for observation_index, edges in enumerate(costs):
            if len(edges) >= 2 and self._ambiguity_margin(edges[0][1], edges[1][1]):
                ambiguous_observations.add(observation_index)
                for track_index, _ in edges:
                    ambiguous_tracks.add(track_index)
        for track_index in range(len(live_tracks)):
            edges = sorted(
                ((observation_index, edge_cost)
                 for observation_index, observation_edges in enumerate(costs)
                 for candidate_track_index, edge_cost in observation_edges
                 if candidate_track_index == track_index),
                key=lambda item: (item[1], item[0]),
            )
            if len(edges) >= 2 and self._ambiguity_margin(edges[0][1], edges[1][1]):
                ambiguous_tracks.add(track_index)
                for observation_index, _ in edges:
                    ambiguous_observations.add(observation_index)

        for observation_index in sorted(ambiguous_observations):
            for track_index, _ in costs[observation_index]:
                result["conflicts"].append(live_tracks[track_index].id)
        result["conflicts"] = list(dict.fromkeys(result["conflicts"]))
        for track_id in result["conflicts"]:
            track = self.tracks.get(track_id)
            if track is not None:
                track.status = "conflicting"
                track.revision += 1

        # Choose the lowest-cost remaining edge repeatedly.  The ambiguity pass
        # above removes close alternatives before this deterministic one-to-one
        # assignment, so a tie can never silently pick a swap.
        all_edges = sorted(
            (
                edge_cost,
                live_tracks[track_index].id,
                observation_index,
                track_index,
            )
            for observation_index, observation_edges in enumerate(costs)
            if observation_index not in ambiguous_observations
            for track_index, edge_cost in observation_edges
            if track_index not in ambiguous_tracks
        )
        assigned_observations: set[int] = set()
        assigned_tracks: set[int] = set()
        assignments: list[tuple[int, int]] = []
        for _, _, observation_index, track_index in all_edges:
            if observation_index in assigned_observations or track_index in assigned_tracks:
                continue
            assigned_observations.add(observation_index)
            assigned_tracks.add(track_index)
            assignments.append((observation_index, track_index))
        assigned_track_ids = {live_tracks[index].id for index in assigned_tracks}

        for observation_index, track_index in assignments:
            observation, xyz, covariance, _, _ = validated[observation_index]
            track = live_tracks[track_index]
            self._fuse_observation(track, observation, xyz, covariance, frame_time)
            result["matched"].append(track.id)

        # Create hypotheses for unassigned, unambiguous observations.  A pair
        # of same-time sensors with no existing track is coalesced by proximity
        # to avoid manufacturing duplicate anonymous identities at startup.
        for observation_index, (observation, xyz, covariance, _, _) in enumerate(validated):
            if observation_index in assigned_observations or observation_index in ambiguous_observations:
                continue
            merged = False
            for track in list(self.tracks.values()):
                if track.id in assigned_track_ids:
                    continue
                if track.last_seen != frame_time or track.last_sensor == str(observation.sensor_id):
                    continue
                if not self._surface_compatible(track, xyz, str(observation.surface_id)):
                    continue
                delta = xyz[:2] - track.mean[:2]
                merge_covariance = _project_psd(track.covariance[:2, :2] + covariance)
                distance = float(delta @ np.linalg.solve(merge_covariance, delta))
                if distance <= float(self.config.association_gate):
                    self._fuse_observation(track, observation, xyz, covariance, frame_time)
                    if track.id not in created_this_ingest:
                        result["matched"].append(track.id)
                    merged = True
                    break
            if merged:
                continue
            if len(self.tracks) >= int(self.config.max_tracks):
                continue
            track = self._new_track(observation, xyz, covariance, frame_time)
            self.tracks[track.id] = track
            created_this_ingest.add(track.id)
            result["created"].append(track.id)

        result["matched"] = list(dict.fromkeys(result["matched"]))
        result["created"] = list(dict.fromkeys(result["created"]))
        return result


__all__ = ["Track", "Tracker"]
