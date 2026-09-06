"""Small validated metric map and deterministic line-of-sight queries."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .schema import SensorSpec


DEFAULT_MAP_PATH = Path(__file__).with_name("maps") / "a_site.json"


def _point(value: Any, name: str) -> tuple[float, float, float]:
    try:
        point = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite xyz point") from exc
    if len(point) != 3 or not all(math.isfinite(item) for item in point):
        raise ValueError(f"{name} must be a finite xyz point")
    return point


@dataclass(frozen=True, slots=True)
class AABB:
    name: str
    minimum: tuple[float, float, float]
    maximum: tuple[float, float, float]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("obstacle name must be non-empty")
        minimum = _point(self.minimum, "minimum")
        maximum = _point(self.maximum, "maximum")
        if any(low >= high for low, high in zip(minimum, maximum)):
            raise ValueError(f"obstacle {self.name!r} must have positive extent")
        object.__setattr__(self, "minimum", minimum)
        object.__setattr__(self, "maximum", maximum)

    def intersects_segment(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        epsilon: float = 1e-9,
    ) -> bool:
        """Return whether the closed segment enters this box (slab method)."""

        direction = np.subtract(end, start, dtype=np.float64)
        enter, leave = 0.0, 1.0
        for axis in range(3):
            origin = start[axis]
            delta = float(direction[axis])
            low, high = self.minimum[axis], self.maximum[axis]
            if abs(delta) <= epsilon:
                if origin < low - epsilon or origin > high + epsilon:
                    return False
                continue
            near = (low - origin) / delta
            far = (high - origin) / delta
            if near > far:
                near, far = far, near
            enter = max(enter, near)
            leave = min(leave, far)
            if enter > leave + epsilon:
                return False
        return leave >= epsilon and enter <= 1.0 - epsilon

    def intersects_actor_segment(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        horizontal_clearance: float,
        epsilon: float = 1e-9,
    ) -> bool:
        """Return whether a closed actor segment enters the horizontally inflated box."""

        minimum = (
            self.minimum[0] - horizontal_clearance,
            self.minimum[1],
            self.minimum[2] - horizontal_clearance,
        )
        maximum = (
            self.maximum[0] + horizontal_clearance,
            self.maximum[1],
            self.maximum[2] + horizontal_clearance,
        )
        direction = np.subtract(end, start, dtype=np.float64)
        enter, leave = 0.0, 1.0
        for axis in range(3):
            origin = start[axis]
            delta = float(direction[axis])
            low, high = minimum[axis], maximum[axis]
            if abs(delta) <= epsilon:
                if origin < low - epsilon or origin > high + epsilon:
                    return False
                continue
            near = (low - origin) / delta
            far = (high - origin) / delta
            if near > far:
                near, far = far, near
            enter = max(enter, near)
            leave = min(leave, far)
            if enter > leave + epsilon:
                return False
        return leave >= -epsilon and enter <= 1.0 + epsilon


@dataclass(frozen=True, slots=True)
class TacticalMap:
    name: str
    coordinate_system: str
    obstacles: tuple[AABB, ...]
    nav_nodes: tuple[dict[str, Any], ...]
    nav_edges: tuple[dict[str, Any], ...]
    zones: tuple[dict[str, Any], ...]
    intents: tuple[dict[str, Any], ...]

    def is_occluded(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> bool:
        start = _point(start, "start")
        end = _point(end, "end")
        return any(box.intersects_segment(start, end) for box in self.obstacles)

    def segment_collides(
        self,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        *,
        horizontal_clearance: float = 0.0,
    ) -> bool:
        """Query movement collision with horizontal actor clearance."""

        start = _point(start, "start")
        end = _point(end, "end")
        clearance = float(horizontal_clearance)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError("horizontal_clearance must be finite and non-negative")
        return any(
            box.intersects_actor_segment(start, end, clearance)
            for box in self.obstacles
        )

    def openness_at(self, xyz: tuple[float, float, float]) -> float:
        """Return authored openness, preferring the most covered overlapping zone."""

        point = _point(xyz, "xyz")
        matching = [
            float(zone["openness"])
            for zone in self.zones
            if all(
                float(zone["min"][axis]) <= point[axis] <= float(zone["max"][axis])
                for axis in range(3)
            )
        ]
        return min(matching, default=1.0)

    def ray_occluded(
        self,
        origin: tuple[float, float, float],
        direction: tuple[float, float, float],
        max_distance: float,
    ) -> bool:
        """Query a finite ray, expressed as origin, direction, and distance."""

        origin = _point(origin, "origin")
        direction = _point(direction, "direction")
        norm = math.sqrt(sum(value * value for value in direction))
        if norm <= 1e-12:
            raise ValueError("direction must be non-zero")
        max_distance = float(max_distance)
        if not math.isfinite(max_distance) or max_distance <= 0.0:
            raise ValueError("max_distance must be finite and positive")
        end = tuple(
            origin[axis] + direction[axis] * max_distance / norm for axis in range(3)
        )
        return self.is_occluded(origin, end)

    def is_visible(
        self,
        sensor: SensorSpec,
        target: tuple[float, float, float],
    ) -> bool:
        target = _point(target, "target")
        dx = target[0] - sensor.xyz[0]
        dz = target[2] - sensor.xyz[2]
        distance_xz = math.hypot(dx, dz)
        distance_3d = math.sqrt(
            dx * dx + (target[1] - sensor.xyz[1]) ** 2 + dz * dz
        )
        if distance_3d > sensor.max_range or distance_xz <= 1e-12:
            return distance_3d <= sensor.max_range and not self.is_occluded(
                sensor.xyz, target
            )
        if sensor.horizontal_fov_degrees < 360.0:
            direction = (dx / distance_xz, dz / distance_xz)
            cosine = sum(a * b for a, b in zip(sensor.forward_xz, direction))
            threshold = math.cos(math.radians(sensor.horizontal_fov_degrees / 2.0))
            if cosine < threshold - 1e-12:
                return False
        return not self.is_occluded(sensor.xyz, target)


def _validate_named_entries(
    entries: Any, label: str, *, point_key: str | None = None
) -> tuple[dict[str, Any], ...]:
    if not isinstance(entries, list):
        raise ValueError(f"{label} must be a list")
    result = []
    names = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str):
            raise ValueError(f"each {label} entry must have a string id")
        if entry["id"] in names:
            raise ValueError(f"duplicate {label} id {entry['id']!r}")
        names.add(entry["id"])
        copy = dict(entry)
        if point_key is not None:
            copy[point_key] = _point(copy.get(point_key), f"{label}.{point_key}")
        result.append(copy)
    return tuple(result)


def load_map(path: str | Path = DEFAULT_MAP_PATH) -> TacticalMap:
    """Load and validate an authored tactical map JSON file."""

    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load tactical map {path}") from exc
    if not isinstance(data, dict):
        raise ValueError("map root must be an object")
    if data.get("coordinate_system") != "right-handed:x-east,y-up,z-north":
        raise ValueError("map coordinate_system must be right-handed:x-east,y-up,z-north")
    obstacles_data = data.get("obstacles")
    if not isinstance(obstacles_data, list):
        raise ValueError("obstacles must be a list")
    obstacles = tuple(
        AABB(
            name=entry.get("id", ""),
            minimum=entry.get("min"),
            maximum=entry.get("max"),
        )
        for entry in obstacles_data
        if isinstance(entry, dict)
    )
    if len(obstacles) != len(obstacles_data):
        raise ValueError("each obstacle must be an object")
    nodes = _validate_named_entries(data.get("nav_nodes"), "nav_nodes", point_key="xyz")
    node_ids = {node["id"] for node in nodes}
    edges = _validate_named_entries(data.get("nav_edges"), "nav_edges")
    for edge in edges:
        if edge.get("from") not in node_ids or edge.get("to") not in node_ids:
            raise ValueError(f"nav edge {edge['id']!r} references an unknown node")
        cost = edge.get("cost")
        if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
            raise ValueError(f"nav edge {edge['id']!r} needs a positive finite cost")
    zones = _validate_named_entries(data.get("zones"), "zones")
    for zone in zones:
        zone["min"] = _point(zone.get("min"), "zones.min")
        zone["max"] = _point(zone.get("max"), "zones.max")
        if any(a >= b for a, b in zip(zone["min"], zone["max"])):
            raise ValueError(f"zone {zone['id']!r} must have positive extent")
        openness = zone.get("openness")
        if (
            isinstance(openness, bool)
            or not isinstance(openness, (int, float))
            or not math.isfinite(float(openness))
            or not 0.0 <= float(openness) <= 1.0
        ):
            raise ValueError(f"zone {zone['id']!r} openness must be in [0, 1]")
        zone["openness"] = float(openness)
    intents = _validate_named_entries(data.get("intents"), "intents")
    return TacticalMap(
        name=str(data.get("name", "")),
        coordinate_system=data["coordinate_system"],
        obstacles=obstacles,
        nav_nodes=nodes,
        nav_edges=edges,
        zones=zones,
        intents=intents,
    )
