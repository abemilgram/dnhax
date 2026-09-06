"""Validated deterministic navigation graph and authored intent routes."""

from dataclasses import dataclass
import heapq
import math
from typing import Iterable

from .map import TacticalMap
from .schema import IntentKind


@dataclass(frozen=True, slots=True)
class NavNode:
    node_id: str
    xyz: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class IntentDefinition:
    """Authored route semantics. Objectives never come from observer tracks."""

    kind: IntentKind
    goal: str
    waypoints: tuple[str, ...] = ()


class NavGraph:
    """An immutable view of a ``TacticalMap`` navigation graph."""

    def __init__(
        self,
        tactical_map: TacticalMap,
        *,
        actor_horizontal_clearance: float = 0.0,
    ) -> None:
        clearance = float(actor_horizontal_clearance)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                "actor_horizontal_clearance must be finite and non-negative"
            )
        nodes = tuple(
            NavNode(node["id"], tuple(float(value) for value in node["xyz"]))
            for node in tactical_map.nav_nodes
        )
        if not nodes:
            raise ValueError("navigation graph must contain at least one node")
        self._nodes = {node.node_id: node for node in nodes}
        if len(self._nodes) != len(nodes):
            raise ValueError("navigation node ids must be unique")

        adjacency: dict[str, list[tuple[str, float, str]]] = {
            node.node_id: [] for node in nodes
        }
        edge_ids: set[str] = set()
        edge_pairs: set[tuple[str, str]] = set()
        heuristic_scales: list[float] = []
        for edge in tactical_map.nav_edges:
            edge_id = edge.get("id")
            start = edge.get("from")
            end = edge.get("to")
            cost = edge.get("cost")
            if not isinstance(edge_id, str) or not edge_id or edge_id in edge_ids:
                raise ValueError("navigation edge ids must be unique non-empty strings")
            edge_ids.add(edge_id)
            if start not in self._nodes or end not in self._nodes:
                raise ValueError(f"navigation edge {edge_id!r} references unknown node")
            if start == end:
                raise ValueError(f"navigation edge {edge_id!r} is a self-loop")
            if (
                isinstance(cost, bool)
                or not isinstance(cost, (int, float))
                or not math.isfinite(float(cost))
                or float(cost) <= 0.0
            ):
                raise ValueError(f"navigation edge {edge_id!r} has invalid cost")
            directed = edge.get("directed", False)
            if not isinstance(directed, bool):
                raise ValueError(f"navigation edge {edge_id!r} directed must be a bool")
            pair = (start, end) if directed else tuple(sorted((start, end)))
            if pair in edge_pairs:
                raise ValueError(f"duplicate navigation connection {pair!r}")
            edge_pairs.add(pair)
            edge_cost = float(cost)
            if not tactical_map.segment_collides(
                self._nodes[start].xyz,
                self._nodes[end].xyz,
                horizontal_clearance=clearance,
            ):
                adjacency[start].append((end, edge_cost, edge_id))
                if not directed:
                    adjacency[end].append((start, edge_cost, edge_id))
            distance = _distance(self._nodes[start].xyz, self._nodes[end].xyz)
            if distance > 1e-12:
                heuristic_scales.append(edge_cost / distance)

        self._adjacency = {
            node_id: tuple(sorted(neighbors, key=lambda item: (item[0], item[2])))
            for node_id, neighbors in adjacency.items()
        }
        self._edge_costs = {
            (start, end): cost
            for start, neighbors in self._adjacency.items()
            for end, cost, _ in neighbors
        }
        self._heuristic_scale = min(heuristic_scales, default=0.0)
        self._intents = self._parse_intents(tactical_map.intents)

    @classmethod
    def from_map(
        cls,
        tactical_map: TacticalMap,
        *,
        actor_horizontal_clearance: float = 0.0,
    ) -> "NavGraph":
        return cls(
            tactical_map,
            actor_horizontal_clearance=actor_horizontal_clearance,
        )

    @property
    def nodes(self) -> tuple[NavNode, ...]:
        return tuple(self._nodes[key] for key in sorted(self._nodes))

    @property
    def intents(self) -> tuple[IntentDefinition, ...]:
        return tuple(self._intents[kind] for kind in IntentKind)

    def _parse_intents(
        self, entries: Iterable[dict[str, object]]
    ) -> dict[IntentKind, IntentDefinition]:
        parsed: dict[IntentKind, IntentDefinition] = {}
        for entry in entries:
            try:
                kind = IntentKind(entry.get("kind"))
            except ValueError as exc:
                raise ValueError(
                    "configured intent kinds must be exactly HOLD, CROSS, and FLANK"
                ) from exc
            if kind in parsed:
                raise ValueError(f"duplicate configured intent kind {kind.value}")
            if kind is IntentKind.HOLD:
                goal = entry.get("anchor")
                waypoints: tuple[str, ...] = ()
            else:
                goal = entry.get("goal")
                raw_waypoints = entry.get("waypoints", ())
                if not isinstance(raw_waypoints, (list, tuple)) or not all(
                    isinstance(item, str) for item in raw_waypoints
                ):
                    raise ValueError(f"{kind.value} waypoints must be node ids")
                waypoints = tuple(raw_waypoints)
                if kind is IntentKind.FLANK and not waypoints:
                    raise ValueError("FLANK must contain an authored corridor waypoint")
            if not isinstance(goal, str) or goal not in self._nodes:
                raise ValueError(f"{kind.value} must reference a known goal node")
            unknown = [item for item in waypoints if item not in self._nodes]
            if unknown:
                raise ValueError(f"{kind.value} references unknown waypoint {unknown[0]!r}")
            parsed[kind] = IntentDefinition(kind, goal, waypoints)
        if set(parsed) != set(IntentKind):
            raise ValueError(
                "configured intent kinds must be exactly HOLD, CROSS, and FLANK"
            )
        return parsed

    def nearest_node(self, xyz: tuple[float, float, float]) -> NavNode:
        point = tuple(float(value) for value in xyz)
        if len(point) != 3 or not all(math.isfinite(value) for value in point):
            raise ValueError("xyz must be a finite three-dimensional point")
        return min(
            self._nodes.values(),
            key=lambda node: (_distance(point, node.xyz), node.node_id),
        )

    def astar(self, start: str, goal: str) -> tuple[str, ...]:
        """Find a minimum-cost path, breaking all ties lexicographically."""

        if start not in self._nodes or goal not in self._nodes:
            raise ValueError("start and goal must reference known nodes")
        if start == goal:
            return (start,)
        initial_path = (start,)
        queue: list[tuple[float, float, tuple[str, ...], str]] = [
            (self._heuristic(start, goal), 0.0, initial_path, start)
        ]
        best: dict[str, tuple[float, tuple[str, ...]]] = {
            start: (0.0, initial_path)
        }
        epsilon = 1e-12
        while queue:
            _, cost, path, node = heapq.heappop(queue)
            known_cost, known_path = best[node]
            if cost > known_cost + epsilon or (
                abs(cost - known_cost) <= epsilon and path != known_path
            ):
                continue
            if node == goal:
                return path
            for neighbor, edge_cost, _ in self._adjacency[node]:
                candidate_cost = cost + edge_cost
                candidate_path = path + (neighbor,)
                previous = best.get(neighbor)
                if previous is not None and (
                    candidate_cost > previous[0] + epsilon
                    or (
                        abs(candidate_cost - previous[0]) <= epsilon
                        and candidate_path >= previous[1]
                    )
                ):
                    continue
                best[neighbor] = (candidate_cost, candidate_path)
                priority = candidate_cost + self._heuristic(neighbor, goal)
                heapq.heappush(
                    queue, (priority, candidate_cost, candidate_path, neighbor)
                )
        raise ValueError(f"no navigation path from {start!r} to {goal!r}")

    def _heuristic(self, node: str, goal: str) -> float:
        return (
            _distance(self._nodes[node].xyz, self._nodes[goal].xyz)
            * self._heuristic_scale
        )

    def intent_path(
        self,
        intent: IntentKind | str,
        start: str | tuple[float, float, float],
    ) -> tuple[str, ...]:
        kind = intent if isinstance(intent, IntentKind) else IntentKind(intent)
        start_id = (
            start
            if isinstance(start, str)
            else self.nearest_node(start).node_id
        )
        if start_id not in self._nodes:
            raise ValueError("start must reference a known node")
        definition = self._intents[kind]
        stops = definition.waypoints + (definition.goal,)
        result = (start_id,)
        current = start_id
        for stop in stops:
            segment = self.astar(current, stop)
            result += segment[1:]
            current = stop
        return result

    path_for_intent = intent_path

    def path_points(self, path: Iterable[str]) -> tuple[tuple[float, float, float], ...]:
        ids = tuple(path)
        if not ids:
            raise ValueError("path must not be empty")
        try:
            points = tuple(self._nodes[node_id].xyz for node_id in ids)
        except KeyError as exc:
            raise ValueError(f"path references unknown node {exc.args[0]!r}") from exc
        if not self.is_valid_path(ids):
            raise ValueError("path contains a non-adjacent node pair")
        return points

    def is_valid_path(self, path: Iterable[str]) -> bool:
        ids = tuple(path)
        return bool(ids) and all(
            (start, end) in self._edge_costs or start == end
            for start, end in zip(ids, ids[1:])
        )

    def path_cost(self, path: Iterable[str]) -> float:
        ids = tuple(path)
        if not self.is_valid_path(ids):
            raise ValueError("path is not valid")
        return sum(
            self._edge_costs[(start, end)]
            for start, end in zip(ids, ids[1:])
            if start != end
        )


def _distance(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))
