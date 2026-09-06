"""Deterministic in-memory tactical orchestration with lazy replay time."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import threading
import time
from typing import Callable, Protocol

from .cues import CueConfig, CueReducer
from .map import TacticalMap, load_map
from .planner import PlannerConfig, TacticalPlanner
from .schema import ActorState, Cue, EvidenceState, PlanRanking
from .tape import (
    ActorEvent,
    FeedEvent,
    FeedState,
    ObservationEvent,
    TacticalTape,
    load_tape,
)
from .tracker import TacticalTracker, TrackerConfig


DEFAULT_TAPE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "fixtures"
    / "tactical"
    / "golden_a_site.json"
)


class TacticalPersistence(Protocol):
    def save_session(
        self,
        session_id: str,
        tape_id: str,
        status: str,
        replay_time: float,
        revision: int,
    ) -> None:
        ...

    def save_cue(
        self,
        session_id: str,
        revision: int,
        cue: dict[str, object],
    ) -> None:
        ...


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    tracker: TrackerConfig = TrackerConfig(
        confirmation_hits=1,
        visible_miss_limit=3,
        max_coast_seconds=8.0,
        acceleration_variance=1.0,
        base_measurement_variance=0.08,
    )
    planner: PlannerConfig = PlannerConfig(
        horizon=9.0,
        step=0.2,
        rollouts_per_intent=16,
        scenario_seed="tactical-service",
    )
    cues: CueConfig = CueConfig(
        switch_margin=0.05,
        consecutive_cycles=1,
        minimum_dwell=0.5,
    )


class TacticalService:
    """Own hot belief state and expose immutable JSON snapshots.

    The deterministic core advances only on explicit calls.  An injected
    monotonic clock is used solely to translate a playing transport into a
    requested replay position.
    """

    tracking_hz = 10
    planning_hz = 2

    def __init__(
        self,
        tape: TacticalTape,
        *,
        tactical_map: TacticalMap | None = None,
        clock: Callable[[], float] = time.monotonic,
        persistence: TacticalPersistence | None = None,
        session_id: str | None = None,
        config: ServiceConfig | None = None,
    ) -> None:
        self.tape = tape
        self.map = tactical_map or load_map()
        self.clock = clock
        self.persistence = persistence
        self.session_id = session_id or f"replay:{tape.tape_id}"
        self.config = config or ServiceConfig()
        self._lock = threading.RLock()
        self._revision = 0
        self._playing = False
        self._wall_anchor = float(self.clock())
        self._replay_anchor = 0.0
        self._persisted_cue_sequences: set[int] = set()
        self._reset_components()
        self._rebuild(0.0)
        self._persist_session()

    @classmethod
    def from_path(
        cls,
        path: str | Path = DEFAULT_TAPE_PATH,
        **kwargs: object,
    ) -> "TacticalService":
        return cls(load_tape(path), **kwargs)

    def _reset_components(self) -> None:
        self._tracker = TacticalTracker(
            self.map,
            self.tape.sensors,
            self.config.tracker,
        )
        planner_config = self.config.planner
        if planner_config.scenario_seed == "tactical-service":
            planner_config = PlannerConfig(
                **{
                    **asdict(planner_config),
                    "scenario_seed": self.tape.tape_id,
                }
            )
        self._planner = TacticalPlanner(self.map, planner_config)
        self._cue_reducer = CueReducer(self.config.cues)
        self._actor = self.tape.initial_actor
        self._feeds = dict(self.tape.initial_feeds)
        self._tracks = ()
        self._ranking: PlanRanking | None = None
        self._cue_history: list[dict[str, object]] = []
        self._position_tick = -1
        self._snapshot_json = "{}"

    def _rebuild(self, replay_time: float) -> None:
        destination_tick = self._validate_position(replay_time)
        self._reset_components()
        for tick in range(destination_tick + 1):
            self._process_tick(tick)
        self._position_tick = destination_tick
        self._publish_snapshot()

    def _validate_position(self, replay_time: float) -> int:
        value = float(replay_time)
        if not math.isfinite(value) or value < 0.0 or value > self.tape.duration:
            raise ValueError("replay position must be inside the tape duration")
        tick = round(value * self.tracking_hz)
        if abs(tick / self.tracking_hz - value) > 1e-9:
            raise ValueError("replay position must align to the 10 Hz clock")
        return tick

    def _process_tick(self, tick: int) -> None:
        t = tick / self.tracking_hz
        observations = []
        for event in self.tape.index.at_tick(tick):
            if isinstance(event, FeedEvent):
                self._feeds[event.sensor_id] = event.state
            elif isinstance(event, ActorEvent):
                self._actor = event.actor
            elif isinstance(event, ObservationEvent):
                if self._feeds[event.observation.sensor_id] is not FeedState.ONLINE:
                    raise ValueError("observations require an online feed")
                observations.append(event.observation)
        visible_sensors = tuple(
            sensor_id
            for sensor_id, state in self._feeds.items()
            if state is FeedState.ONLINE
        )
        self._tracks = self._tracker.step(
            t,
            observations,
            visible_sensor_ids=visible_sensors,
        )
        if tick % (self.tracking_hz // self.planning_hz) == 0:
            actor = ActorState(t, self._actor.xyz, self._actor.velocity_xz)
            self._ranking = self._planner.plan(
                actor,
                self._tracks,
                cycle_index=tick // (self.tracking_hz // self.planning_hz),
            )
            degraded = any(
                track.evidence is not EvidenceState.OBSERVED for track in self._tracks
            ) or any(state is not FeedState.ONLINE for state in self._feeds.values())
            cue = self._cue_reducer.update(
                self._ranking,
                tracking_degraded=degraded,
            )
            if cue is not None:
                self._revision += 1
                self._record_cue(cue, self._revision)
        self._position_tick = tick
        self._revision += 1

    def _record_cue(self, cue: Cue, revision: int) -> None:
        payload = {"revision": revision, **asdict(cue)}
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        normalized = json.loads(encoded)
        self._cue_history.append(normalized)
        if self.persistence is not None and cue.sequence not in self._persisted_cue_sequences:
            self.persistence.save_cue(self.session_id, revision, normalized)
            self._persisted_cue_sequences.add(cue.sequence)

    def _map_json(self) -> dict[str, object]:
        return {
            "name": self.map.name,
            "coordinate_system": self.map.coordinate_system,
            "obstacles": [
                {
                    "id": obstacle.name,
                    "min": obstacle.minimum,
                    "max": obstacle.maximum,
                }
                for obstacle in self.map.obstacles
            ],
            "nodes": list(self.map.nav_nodes),
            "edges": list(self.map.nav_edges),
            "zones": list(self.map.zones),
        }

    def _publish_snapshot(self) -> None:
        active_tracks = [
            asdict(track)
            for track in self._tracks
            if track.evidence is not EvidenceState.STALE
        ]
        ghosts = [
            asdict(track)
            for track in self._tracks
            if track.evidence is EvidenceState.STALE
        ]
        snapshot = {
            "schema_version": 1,
            "revision": self._revision,
            "replay": {
                "id": self.tape.tape_id,
                "duration": self.tape.duration,
                "position": max(0, self._position_tick) / self.tracking_hz,
                "playing": self._playing,
            },
            "map": self._map_json(),
            "actor": asdict(
                ActorState(
                    max(0, self._position_tick) / self.tracking_hz,
                    self._actor.xyz,
                    self._actor.velocity_xz,
                )
            ),
            "feeds": [
                {"sensor_id": sensor_id, "state": state.value}
                for sensor_id, state in sorted(self._feeds.items())
            ],
            "tracks": active_tracks,
            "ghosts": ghosts,
            "ranking": asdict(self._ranking) if self._ranking is not None else None,
            "cue_history": list(self._cue_history),
        }
        self._snapshot_json = json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def _persist_session(self) -> None:
        if self.persistence is not None:
            self.persistence.save_session(
                self.session_id,
                self.tape.tape_id,
                "playing" if self._playing else "paused",
                max(0, self._position_tick) / self.tracking_hz,
                self._revision,
            )

    def _advance_from_clock(self) -> None:
        if not self._playing:
            return
        elapsed = max(0.0, float(self.clock()) - self._wall_anchor)
        requested = min(self.tape.duration, self._replay_anchor + elapsed)
        destination = min(self.tape.final_tick, int(math.floor(requested * 10.0 + 1e-9)))
        if destination > self._position_tick:
            for tick in range(self._position_tick + 1, destination + 1):
                self._process_tick(tick)
            self._publish_snapshot()
        if destination >= self.tape.final_tick:
            self._playing = False
            self._revision += 1
            self._publish_snapshot()
            self._persist_session()

    def current(self) -> dict[str, object]:
        with self._lock:
            self._advance_from_clock()
            return json.loads(self._snapshot_json)

    def advance_to(self, replay_time: float) -> dict[str, object]:
        """Explicit deterministic advancement, primarily for tests/adapters."""

        with self._lock:
            destination = self._validate_position(replay_time)
            if destination < self._position_tick:
                self._rebuild(replay_time)
            else:
                for tick in range(self._position_tick + 1, destination + 1):
                    self._process_tick(tick)
                self._publish_snapshot()
            self._replay_anchor = destination / self.tracking_hz
            self._wall_anchor = float(self.clock())
            self._persist_session()
            return json.loads(self._snapshot_json)

    def start(self) -> dict[str, object]:
        with self._lock:
            self._advance_from_clock()
            if self._position_tick >= self.tape.final_tick:
                self._rebuild(0.0)
            if not self._playing:
                self._playing = True
                self._wall_anchor = float(self.clock())
                self._replay_anchor = max(0, self._position_tick) / self.tracking_hz
                self._revision += 1
                self._publish_snapshot()
                self._persist_session()
            return json.loads(self._snapshot_json)

    def pause(self) -> dict[str, object]:
        with self._lock:
            self._advance_from_clock()
            if self._playing:
                self._playing = False
                self._revision += 1
                self._publish_snapshot()
                self._persist_session()
            return json.loads(self._snapshot_json)

    def seek(self, replay_time: float) -> dict[str, object]:
        with self._lock:
            was_playing = self._playing
            self._rebuild(replay_time)
            self._playing = was_playing
            self._wall_anchor = float(self.clock())
            self._replay_anchor = replay_time
            self._revision += 1
            self._publish_snapshot()
            self._persist_session()
            return json.loads(self._snapshot_json)

    def restart(self) -> dict[str, object]:
        with self._lock:
            self._rebuild(0.0)
            self._playing = True
            self._wall_anchor = float(self.clock())
            self._replay_anchor = 0.0
            self._revision += 1
            self._publish_snapshot()
            self._persist_session()
            return json.loads(self._snapshot_json)

    def events_since(
        self, revision: int
    ) -> tuple[tuple[str, int, dict[str, object]], ...]:
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("revision must be a non-negative integer")
        with self._lock:
            snapshot = self.current()
            events: list[tuple[str, int, dict[str, object]]] = []
            for cue in snapshot["cue_history"]:  # type: ignore[index]
                cue_revision = int(cue["revision"])
                if cue_revision > revision:
                    events.append(("cue", cue_revision, cue))
            current_revision = int(snapshot["revision"])
            if current_revision > revision:
                events.append(("snapshot", current_revision, snapshot))
            return tuple(events)
