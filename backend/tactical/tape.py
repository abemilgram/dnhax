"""Validated deterministic tactical event tapes and replay indexing."""

from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable

from .schema import ActorState, Observation, SensorSpec


class FeedState(str, Enum):
    ONLINE = "online"
    OCCLUDED = "occluded"
    OFFLINE = "offline"


@dataclass(frozen=True, slots=True)
class ObservationEvent:
    t: float
    observation: Observation


@dataclass(frozen=True, slots=True)
class ActorEvent:
    t: float
    actor: ActorState


@dataclass(frozen=True, slots=True)
class FeedEvent:
    t: float
    sensor_id: str
    state: FeedState


TapeEvent = ObservationEvent | ActorEvent | FeedEvent


def _tick(t: float) -> int:
    tick = round(float(t) * 10.0)
    if abs(tick / 10.0 - float(t)) > 1e-9:
        raise ValueError("event timestamps must align to the 10 Hz tracking clock")
    return tick


@dataclass(frozen=True, slots=True)
class ReplayIndex:
    """Immutable tick index shared by replay and future live adapters."""

    by_tick: MappingProxyType
    final_tick: int

    def at_tick(self, tick: int) -> tuple[TapeEvent, ...]:
        if isinstance(tick, bool) or not isinstance(tick, int):
            raise ValueError("tick must be an integer")
        return self.by_tick.get(tick, ())

    def through(self, t: float) -> tuple[TapeEvent, ...]:
        final = _tick(t)
        return tuple(
            event
            for tick in range(final + 1)
            for event in self.at_tick(tick)
        )


@dataclass(frozen=True, slots=True)
class TacticalTape:
    tape_id: str
    duration: float
    sensors: tuple[SensorSpec, ...]
    initial_actor: ActorState
    initial_feeds: MappingProxyType
    events: tuple[TapeEvent, ...]
    index: ReplayIndex

    @property
    def final_tick(self) -> int:
        return self.index.final_tick


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _strict_keys(
    value: dict[str, Any],
    *,
    required: Iterable[str],
    optional: Iterable[str] = (),
    name: str,
) -> None:
    required_set = set(required)
    allowed = required_set | set(optional)
    missing = required_set.difference(value)
    extra = set(value).difference(allowed)
    if missing:
        raise ValueError(f"{name} is missing {sorted(missing)[0]!r}")
    if extra:
        raise ValueError(f"{name} contains unknown field {sorted(extra)[0]!r}")


def _finite(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _parse_sensor(value: Any, index: int) -> SensorSpec:
    item = _object(value, f"sensors[{index}]")
    _strict_keys(
        item,
        required=("sensor_id", "xyz"),
        optional=("forward_xz", "horizontal_fov_degrees", "max_range"),
        name=f"sensors[{index}]",
    )
    return SensorSpec(
        sensor_id=item["sensor_id"],
        xyz=item["xyz"],
        forward_xz=item.get("forward_xz", (0.0, 1.0)),
        horizontal_fov_degrees=item.get("horizontal_fov_degrees", 360.0),
        max_range=item.get("max_range", 50.0),
    )


def _parse_actor(value: Any, t: float, name: str) -> ActorState:
    item = _object(value, name)
    _strict_keys(
        item,
        required=("xyz",),
        optional=("velocity_xz",),
        name=name,
    )
    return ActorState(t=t, xyz=item["xyz"], velocity_xz=item.get("velocity_xz", (0.0, 0.0)))


def _parse_event(
    value: Any,
    index: int,
    *,
    duration: float,
    sensor_ids: frozenset[str],
) -> TapeEvent:
    name = f"events[{index}]"
    item = _object(value, name)
    kind = item.get("type")
    t = _finite(item.get("t"), f"{name}.t")
    _tick(t)
    if t < 0.0 or t > duration:
        raise ValueError(f"{name}.t must be inside the tape duration")
    if kind == "observation":
        _strict_keys(
            item,
            required=("type", "t", "sensor_id", "xyz", "conf", "sequence"),
            optional=("covariance",),
            name=name,
        )
        observation = Observation(
            t=t,
            sensor_id=item["sensor_id"],
            xyz=item["xyz"],
            conf=item["conf"],
            sequence=item["sequence"],
            covariance=item.get("covariance"),
        )
        if observation.sensor_id not in sensor_ids:
            raise ValueError(f"{name} references an unknown sensor")
        return ObservationEvent(t, observation)
    if kind == "actor":
        _strict_keys(
            item,
            required=("type", "t", "xyz"),
            optional=("velocity_xz",),
            name=name,
        )
        return ActorEvent(
            t,
            ActorState(
                t=t,
                xyz=item["xyz"],
                velocity_xz=item.get("velocity_xz", (0.0, 0.0)),
            ),
        )
    if kind == "feed":
        _strict_keys(
            item,
            required=("type", "t", "sensor_id", "state"),
            name=name,
        )
        if item["sensor_id"] not in sensor_ids:
            raise ValueError(f"{name} references an unknown sensor")
        try:
            state = FeedState(item["state"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.state is invalid") from exc
        return FeedEvent(t, item["sensor_id"], state)
    raise ValueError(f"{name}.type must be observation, actor, or feed")


def load_tape(path: str | Path) -> TacticalTape:
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load tactical tape {path}") from exc
    root = _object(data, "tape")
    _strict_keys(
        root,
        required=("version", "id", "duration", "sensors", "initial", "events"),
        name="tape",
    )
    if root["version"] != 1:
        raise ValueError("tape.version must equal 1")
    tape_id = root["id"]
    if not isinstance(tape_id, str) or not tape_id:
        raise ValueError("tape.id must be a non-empty string")
    duration = _finite(root["duration"], "tape.duration")
    if duration <= 0.0 or duration >= 120.0:
        raise ValueError("tape.duration must be greater than 0 and under 120 seconds")
    final_tick = _tick(duration)
    if not isinstance(root["sensors"], list) or not root["sensors"]:
        raise ValueError("tape.sensors must be a non-empty list")
    sensors = tuple(
        _parse_sensor(value, index) for index, value in enumerate(root["sensors"])
    )
    sensor_ids = frozenset(sensor.sensor_id for sensor in sensors)
    if len(sensor_ids) != len(sensors):
        raise ValueError("tape sensor ids must be unique")
    initial = _object(root["initial"], "tape.initial")
    _strict_keys(
        initial,
        required=("actor", "feeds"),
        name="tape.initial",
    )
    initial_actor = _parse_actor(initial["actor"], 0.0, "tape.initial.actor")
    feeds = _object(initial["feeds"], "tape.initial.feeds")
    if set(feeds) != set(sensor_ids):
        raise ValueError("tape.initial.feeds must define every sensor exactly once")
    try:
        initial_feeds = {
            sensor_id: FeedState(feeds[sensor_id]) for sensor_id in sorted(feeds)
        }
    except (TypeError, ValueError) as exc:
        raise ValueError("tape.initial.feeds contains an invalid state") from exc
    if not isinstance(root["events"], list):
        raise ValueError("tape.events must be a list")
    events = tuple(
        _parse_event(value, index, duration=duration, sensor_ids=sensor_ids)
        for index, value in enumerate(root["events"])
    )
    if any(second.t < first.t for first, second in zip(events, events[1:])):
        raise ValueError("tape.events must be sorted by timestamp")
    sequences: set[tuple[str, int]] = set()
    grouped: dict[int, list[TapeEvent]] = {}
    replay_feeds = dict(initial_feeds)
    for event in events:
        if isinstance(event, FeedEvent):
            replay_feeds[event.sensor_id] = event.state
        elif isinstance(event, ObservationEvent):
            if replay_feeds[event.observation.sensor_id] is not FeedState.ONLINE:
                raise ValueError(
                    "observations may only occur while their feed is online"
                )
            key = (event.observation.sensor_id, event.observation.sequence)
            if key in sequences:
                raise ValueError("observation sequences must be unique per sensor")
            sequences.add(key)
        grouped.setdefault(_tick(event.t), []).append(event)
    index = ReplayIndex(
        MappingProxyType(
            {tick: tuple(values) for tick, values in sorted(grouped.items())}
        ),
        final_tick,
    )
    return TacticalTape(
        tape_id=tape_id,
        duration=duration,
        sensors=sensors,
        initial_actor=initial_actor,
        initial_feeds=MappingProxyType(initial_feeds),
        events=events,
        index=index,
    )
