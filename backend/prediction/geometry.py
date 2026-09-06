"""Bounded metric geometry for the prediction engine.

The coordinate contract here is deliberately small: X/Y lie on the ground
plane, Z is up, and camera poses use OpenCV's right/down/forward axes.  This
module consumes the typed models in :mod:`backend.prediction.schema`; it does
not consume radar coordinates or game entity transforms.
"""

from __future__ import annotations

import heapq
from typing import Any, Iterable

import numpy as np

from .schema import Camera, FramePacket, Observation, ScenePrior, Surface


_EPS = 1e-9
_Z_TOLERANCE = 0.35
_NAV_HEIGHT_M = 0.8
_SNAP_RADIUS_M = 3.0
_MAX_COVARIANCE_STD_M = 5.0


def _as_vec(value: Any, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def _polygon_area(polygon: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(polygon[:, 0] * np.roll(polygon[:, 1], -1))
        - np.sum(polygon[:, 1] * np.roll(polygon[:, 0], -1))
    )


def point_in_polygon(xy: Any, polygon: Any) -> bool:
    """Return whether a 2D point is inside a polygon, including its boundary."""

    point = _as_vec(xy, 2, "xy")
    poly = np.asarray(polygon, dtype=float)
    if poly.ndim != 2 or poly.shape[1] != 2 or len(poly) < 3 or not np.isfinite(poly).all():
        return False

    # Boundary inclusion is useful for exact synthetic fixtures, while the
    # small tolerance avoids classifying a numerically close point as outside.
    scale = max(1.0, float(np.max(np.abs(poly))), float(np.max(np.abs(point))))
    tol = 1e-9 * scale
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        cross = float(edge[0] * (point[1] - a[1]) - edge[1] * (point[0] - a[0]))
        if abs(cross) <= tol:
            dot = float(np.dot(point - a, point - b))
            if dot <= tol * tol:
                return True

    # Standard half-open ray crossing test.  The boundary branch above keeps
    # vertices and horizontal edges deterministic.
    inside = False
    x, y = point
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ya, yb = float(a[1]), float(b[1])
        if (ya > y) != (yb > y):
            x_intersection = float(a[0] + (y - ya) * (b[0] - a[0]) / (yb - ya))
            if x < x_intersection:
                inside = not inside
    return inside


def _segment_aabb_interval(a: np.ndarray, b: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> tuple[float, float] | None:
    """Return the inclusive segment parameter interval inside an AABB."""

    direction = b - a
    low, high = 0.0, 1.0
    for axis in range(3):
        value = float(a[axis])
        delta = float(direction[axis])
        lo, hi = float(minimum[axis]), float(maximum[axis])
        if abs(delta) <= _EPS:
            if value < lo - _EPS or value > hi + _EPS:
                return None
            continue
        t0 = (lo - value) / delta
        t1 = (hi - value) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        low = max(low, t0)
        high = min(high, t1)
        if low > high + _EPS:
            return None
    return low, high


def project_world(camera: Camera, xyz: Any) -> np.ndarray[2] | None:
    """Project a world point through an OpenCV pinhole camera.

    ``camera.camera_to_world`` maps camera coordinates to world coordinates.
    The result is returned in calibrated full-frame pixel coordinates and may
    be outside the image; callers such as :meth:`WorldGeometry.visible` apply
    the image bounds appropriate to their use.
    """

    try:
        world = _as_vec(xyz, 3, "xyz")
    except ValueError:
        return None
    k = np.asarray(camera.intrinsics, dtype=float)
    pose = np.asarray(camera.camera_to_world, dtype=float)
    if k.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
        return None
    camera_xyz = pose[:3, :3].T @ (world - pose[:3, 3])
    if not np.isfinite(camera_xyz).all() or camera_xyz[2] <= 1e-8:
        return None
    pixels = k @ camera_xyz
    if abs(float(pixels[2])) <= _EPS or not np.isfinite(pixels).all():
        return None
    return np.asarray(pixels[:2] / pixels[2], dtype=float)


class WorldGeometry:
    """Validated surfaces, walls, and a bounded navigation graph."""

    def __init__(self, prior: ScenePrior):
        self.prior = prior
        self._surfaces = list(prior.surfaces)
        self._walls = list(prior.walls)
        self._nodes = {node.id: np.asarray(node.xyz, dtype=float) for node in prior.nodes}
        self._edges: list[tuple[str, str]] = []
        self._adjacency: dict[str, list[tuple[str, float]]] = {node_id: [] for node_id in self._nodes}
        self._validate_prior()

    def _validate_prior(self) -> None:
        # ScenePrior checks most references, but this constructor is also the
        # boundary for callers constructing equivalent model-like fixtures.
        if getattr(self.prior, "units", "meters") != "meters":
            raise ValueError("WorldGeometry requires meter units")
        surface_ids = [surface.id for surface in self._surfaces]
        wall_ids = [wall.id for wall in self._walls]
        if len(surface_ids) != len(set(surface_ids)):
            raise ValueError("Surface IDs must be unique")
        if len(wall_ids) != len(set(wall_ids)):
            raise ValueError("Wall IDs must be unique")
        for surface in self._surfaces:
            polygon = np.asarray(surface.polygon, dtype=float)
            if polygon.ndim != 2 or polygon.shape[1] != 2 or len(polygon) < 3:
                raise ValueError(f"Surface {surface.id!r} polygon is degenerate")
            if not np.isfinite(polygon).all() or not np.isfinite(float(surface.z)):
                raise ValueError(f"Surface {surface.id!r} polygon is non-finite")
            if abs(_polygon_area(polygon)) <= 1e-9:
                raise ValueError(f"Surface {surface.id!r} polygon is degenerate")
        for wall in self._walls:
            minimum = _as_vec(wall.minimum, 3, f"Wall {wall.id} minimum")
            maximum = _as_vec(wall.maximum, 3, f"Wall {wall.id} maximum")
            if np.any(minimum >= maximum):
                raise ValueError(f"Wall {wall.id!r} has invalid bounds")
        for node_id, xyz in self._nodes.items():
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                raise ValueError(f"Navigation node {node_id!r} must be finite")

        for edge in self.prior.edges:
            a_id, b_id = edge.a, edge.b
            if a_id not in self._nodes or b_id not in self._nodes or a_id == b_id:
                raise ValueError("Navigation edges must reference distinct existing nodes")
            a, b = self._nodes[a_id], self._nodes[b_id]
            if self._surface_for_point(a) is None or self._surface_for_point(b) is None:
                raise ValueError(f"Navigation edge {a_id!r}->{b_id!r} has an unsupported endpoint")
            if self._nav_edge_blocked(a, b):
                raise ValueError(f"Navigation edge {a_id!r}->{b_id!r} passes through a wall")
            distance = float(np.linalg.norm(b - a))
            if distance <= _EPS:
                raise ValueError("Navigation edges must have positive 3D length")
            self._edges.append((a_id, b_id))
            self._adjacency[a_id].append((b_id, distance))
            self._adjacency[b_id].append((a_id, distance))

    def _surface_for_point(self, xyz: np.ndarray, z_tolerance: float = _Z_TOLERANCE) -> Surface | None:
        candidates = []
        for surface in self._surfaces:
            polygon = np.asarray(surface.polygon, dtype=float)
            if point_in_polygon(xyz[:2], polygon) and abs(float(xyz[2]) - float(surface.z)) <= z_tolerance:
                candidates.append((abs(float(xyz[2]) - float(surface.z)), surface))
        return min(candidates, key=lambda item: item[0])[1] if candidates else None

    def _surface_z_near(self, xyz: np.ndarray) -> float:
        surface = self._surface_for_point(xyz)
        return float(surface.z) if surface is not None else float(xyz[2])

    def _lift_for_navigation(self, xyz: np.ndarray) -> np.ndarray:
        result = np.asarray(xyz, dtype=float).copy()
        result[2] = self._surface_z_near(result) + _NAV_HEIGHT_M
        return result

    def _nav_edge_blocked(self, a: np.ndarray, b: np.ndarray) -> bool:
        # Evaluate a human-height path above each endpoint's local floor.  The
        # linearly changing offset handles stairs/elevation links while still
        # catching a wall that reaches the ground boundary.
        a_lift = self._lift_for_navigation(np.asarray(a, dtype=float))
        b_lift = self._lift_for_navigation(np.asarray(b, dtype=float))
        if self._segment_blocked(a_lift, b_lift, include_surfaces=False):
            return True
        # Same-level links must stay in the walkable surface union. An edge
        # with a real elevation change is an explicit stair/ramp connection;
        # this prior does not claim to model its intermediate walkable mesh.
        if abs(float(a[2]) - float(b[2])) <= _Z_TOLERANCE and not self._same_level_walkable(a, b):
            return True
        return False

    def _same_level_walkable(self, a: np.ndarray, b: np.ndarray) -> bool:
        # The status of a straight segment changes only at polygon boundary
        # intersections. Collect those exact breakpoints, then test one point
        # in every resulting interval; this catches arbitrarily narrow gaps.
        start_xy, end_xy = a[:2], b[:2]
        direction = end_xy - start_xy
        direction_norm = float(np.dot(direction, direction))
        cuts = [0.0, 1.0]
        if direction_norm <= _EPS:
            return self._surface_for_point(a) is not None
        for surface in self._surfaces:
            if abs(float(surface.z) - float(a[2])) > _Z_TOLERANCE:
                continue
            polygon = np.asarray(surface.polygon, dtype=float)
            for vertex_a, vertex_b in zip(polygon, np.roll(polygon, -1, axis=0)):
                edge = vertex_b - vertex_a
                cross = float(direction[0] * edge[1] - direction[1] * edge[0])
                offset = vertex_a - start_xy
                if abs(cross) > _EPS:
                    fraction = float((offset[0] * edge[1] - offset[1] * edge[0]) / cross)
                    edge_fraction = float((offset[0] * direction[1] - offset[1] * direction[0]) / cross)
                    if -_EPS <= fraction <= 1.0 + _EPS and -_EPS <= edge_fraction <= 1.0 + _EPS:
                        cuts.append(float(np.clip(fraction, 0.0, 1.0)))
                else:
                    # Collinear boundary: its endpoints still delimit changes
                    # between adjacent polygon pieces and gaps.
                    if abs(float(offset[0] * direction[1] - offset[1] * direction[0])) <= _EPS:
                        cuts.extend([
                            float(np.clip(np.dot(vertex_a - start_xy, direction) / direction_norm, 0.0, 1.0)),
                            float(np.clip(np.dot(vertex_b - start_xy, direction) / direction_norm, 0.0, 1.0)),
                        ])
        cuts = np.unique(np.asarray(cuts, dtype=float))
        for left, right in zip(cuts[:-1], cuts[1:]):
            if right - left <= _EPS:
                continue
            fraction = float((left + right) * 0.5)
            point = a + fraction * (b - a)
            if self._surface_for_point(point) is None:
                return False
        return True

    def _segment_blocked(self, a: np.ndarray, b: np.ndarray, include_surfaces: bool = True) -> bool:
        for wall in self._walls:
            minimum = np.asarray(wall.minimum, dtype=float)
            maximum = np.asarray(wall.maximum, dtype=float)
            if _segment_aabb_interval(a, b, minimum, maximum) is not None:
                return True
        if include_surfaces and self._segment_blocked_by_surface(a, b):
            return True
        return False

    def _segment_blocked_by_surface(self, a: np.ndarray, b: np.ndarray) -> bool:
        dz = float(b[2] - a[2])
        if abs(dz) <= _EPS:
            return False
        for surface in self._surfaces:
            fraction = (float(surface.z) - float(a[2])) / dz
            # Endpoints on a floor are allowed: the floor is the observed
            # support plane, rather than an occluder of its own endpoint.
            if fraction <= 1e-8 or fraction >= 1.0 - 1e-8:
                continue
            crossing = a + fraction * (b - a)
            if point_in_polygon(crossing[:2], surface.polygon):
                return True
        return False

    def line_of_sight(self, a: Any, b: Any) -> bool:
        """Return whether an axis-aligned wall blocks the segment ``a``--``b``."""

        first, second = _as_vec(a, 3, "a"), _as_vec(b, 3, "b")
        return not self._segment_blocked(first, second)

    def line_of_sight_many(self, a: Any, b: Any) -> np.ndarray:
        """Vectorized wall/floor line-of-sight for one point to many points."""

        origin = _as_vec(a, 3, "a")
        targets = np.asarray(b, dtype=float)
        if targets.ndim != 2 or targets.shape[1] != 3 or not np.isfinite(targets).all():
            raise ValueError("b must be a finite [N,3] array")
        blocked = np.zeros(len(targets), dtype=bool)
        direction = targets - origin
        for wall in self._walls:
            minimum = np.asarray(wall.minimum, dtype=float)
            maximum = np.asarray(wall.maximum, dtype=float)
            low = np.zeros(len(targets), dtype=float)
            high = np.ones(len(targets), dtype=float)
            valid = np.ones(len(targets), dtype=bool)
            for axis in range(3):
                delta = direction[:, axis]
                value = origin[axis]
                parallel = np.abs(delta) <= _EPS
                valid &= (~parallel) | ((value >= minimum[axis] - _EPS) & (value <= maximum[axis] + _EPS))
                safe_delta = np.where(parallel, 1.0, delta)
                enter = (minimum[axis] - value) / safe_delta
                leave = (maximum[axis] - value) / safe_delta
                lower = np.minimum(enter, leave)
                upper = np.maximum(enter, leave)
                low = np.maximum(low, np.where(parallel, 0.0, lower))
                high = np.minimum(high, np.where(parallel, 1.0, upper))
                valid &= low <= high + _EPS
            blocked |= valid
        dz = direction[:, 2]
        nonflat = np.abs(dz) > _EPS
        for surface in self._surfaces:
            fractions = np.divide(float(surface.z) - origin[2], dz, out=np.zeros(len(targets)), where=nonflat)
            candidates = np.flatnonzero(~blocked & nonflat & (fractions > 1e-8) & (fractions < 1.0 - 1e-8))
            crossing = origin + fractions[candidates, None] * direction[candidates]
            inside = np.array([point_in_polygon(point[:2], surface.polygon) for point in crossing], dtype=bool)
            blocked[candidates] |= inside
        return ~blocked

    def visible(self, camera: Camera, xyz: Any) -> bool:
        """Check full-frame projection and wall visibility for a torso point."""

        if camera.coordinate_frame != self.prior.coordinate_frame:
            return False
        try:
            world = _as_vec(xyz, 3, "xyz")
        except ValueError:
            return False
        pixels = project_world(camera, world)
        if pixels is None:
            return False
        if not (0.0 <= pixels[0] < float(camera.width) and 0.0 <= pixels[1] < float(camera.height)):
            return False
        return self.line_of_sight(np.asarray(camera.camera_to_world, dtype=float)[:3, 3], world)

    def visible_many(self, camera: Camera, points: Any) -> np.ndarray:
        """Vectorized equivalent of :meth:`visible` for torso samples."""

        worlds = np.asarray(points, dtype=float)
        if worlds.ndim != 2 or worlds.shape[1] != 3 or not np.isfinite(worlds).all():
            raise ValueError("points must be a finite [N,3] array")
        visible = np.zeros(len(worlds), dtype=bool)
        if camera.coordinate_frame != self.prior.coordinate_frame:
            return visible
        k = np.asarray(camera.intrinsics, dtype=float)
        pose = np.asarray(camera.camera_to_world, dtype=float)
        camera_xyz = (worlds - pose[:3, 3]) @ pose[:3, :3]
        positive = camera_xyz[:, 2] > 1e-8
        if positive.any():
            pixels_h = camera_xyz[positive] @ k.T
            pixels = pixels_h[:, :2] / pixels_h[:, 2, None]
            in_frame = (
                (pixels[:, 0] >= 0.0)
                & (pixels[:, 0] < float(camera.width))
                & (pixels[:, 1] >= 0.0)
                & (pixels[:, 1] < float(camera.height))
            )
            candidate_indices = np.flatnonzero(positive)
            candidate_indices = candidate_indices[in_frame]
            if len(candidate_indices):
                visible[candidate_indices] = self.line_of_sight_many(
                    pose[:3, 3], worlds[candidate_indices]
                )
        return visible

    @staticmethod
    def _ray_plane(origin: np.ndarray, direction: np.ndarray, surface: Surface, pixel: np.ndarray) -> tuple[float, np.ndarray] | None:
        dz = float(direction[2])
        # A ray near parallel to a horizontal floor produces an unbounded,
        # unstable intersection even when it technically has a hit.
        if abs(dz) < 0.02:
            return None
        t = (float(surface.z) - float(origin[2])) / dz
        if not np.isfinite(t) or t <= 1e-7:
            return None
        point = origin + t * direction
        if not np.isfinite(point).all() or not point_in_polygon(point[:2], surface.polygon):
            return None
        return float(t), point

    def _ray_surface_hit(self, camera: Camera, pixel: np.ndarray, surface_index: int | None = None, origin: np.ndarray | None = None, check_walls: bool = True) -> tuple[int, float, np.ndarray] | None:
        k = np.asarray(camera.intrinsics, dtype=float)
        pose = np.asarray(camera.camera_to_world, dtype=float)
        if k.shape != (3, 3) or pose.shape != (4, 4):
            return None
        try:
            ray_camera = np.linalg.solve(k, np.asarray([pixel[0], pixel[1], 1.0], dtype=float))
        except np.linalg.LinAlgError:
            return None
        direction = pose[:3, :3] @ ray_camera
        ray_origin = pose[:3, 3] if origin is None else np.asarray(origin, dtype=float)
        if not np.isfinite(direction).all() or not np.isfinite(ray_origin).all():
            return None

        indices: Iterable[int] = range(len(self._surfaces)) if surface_index is None else (surface_index,)
        hits: list[tuple[float, int, np.ndarray]] = []
        for index in indices:
            hit = self._ray_plane(ray_origin, direction, self._surfaces[index], np.asarray(pixel, dtype=float))
            if hit is not None:
                t, point = hit
                hits.append((t, index, point))
        if not hits:
            return None
        t_surface, index, point = min(hits, key=lambda item: item[0])
        if check_walls and self._ray_blocked_by_wall(ray_origin, direction, t_surface):
            return None
        return index, t_surface, point

    def _ray_blocked_by_wall(self, origin: np.ndarray, direction: np.ndarray, t_surface: float) -> bool:
        for wall in self._walls:
            interval = _line_aabb_interval(origin, direction, np.asarray(wall.minimum, dtype=float), np.asarray(wall.maximum, dtype=float))
            if interval is None:
                continue
            entry, exit_ = interval
            if exit_ >= 1e-8 and entry <= float(t_surface) + 1e-8:
                return True
        return False

    def _finite_difference_covariance(self, camera: Camera, pixel: np.ndarray, surface_index: int, center: np.ndarray) -> np.ndarray | None:
        pixel_std = float(camera.pixel_std)
        position_std = float(camera.position_std_m)
        pose = np.asarray(camera.camera_to_world, dtype=float)
        origin = pose[:3, 3]

        derivatives: list[np.ndarray] = []

        def point_at(sample_pixel: np.ndarray, sample_origin: np.ndarray) -> np.ndarray | None:
            hit = self._ray_surface_hit(camera, sample_pixel, surface_index=surface_index, origin=sample_origin, check_walls=True)
            return None if hit is None else hit[2]

        for axis in range(2):
            plus_pixel = pixel.copy()
            minus_pixel = pixel.copy()
            plus_pixel[axis] += pixel_std
            minus_pixel[axis] -= pixel_std
            plus = point_at(plus_pixel, origin)
            minus = point_at(minus_pixel, origin)
            if plus is None or minus is None:
                return None
            derivatives.append((plus[:2] - minus[:2]) / (2.0 * pixel_std))

        # Camera position uncertainty is modeled isotropically in world axes.
        # A finite difference keeps this valid for arbitrary camera rotations
        # and oblique views without assuming a closed-form ground projection.
        for axis in range(3):
            plus_origin = origin.copy()
            minus_origin = origin.copy()
            plus_origin[axis] += position_std
            minus_origin[axis] -= position_std
            plus = point_at(pixel, plus_origin)
            minus = point_at(pixel, minus_origin)
            if plus is None or minus is None:
                return None
            derivatives.append((plus[:2] - minus[:2]) / (2.0 * position_std))

        covariance = np.zeros((2, 2), dtype=float)
        for derivative in derivatives[:2]:
            covariance += (pixel_std * pixel_std) * np.outer(derivative, derivative)
        for derivative in derivatives[2:]:
            covariance += (position_std * position_std) * np.outer(derivative, derivative)
        covariance = 0.5 * (covariance + covariance.T)
        if not np.isfinite(covariance).all():
            return None
        eigenvalues = np.linalg.eigvalsh(covariance)
        if not np.isfinite(eigenvalues).all() or float(np.min(eigenvalues)) < -1e-7:
            return None
        if float(np.max(eigenvalues)) > _MAX_COVARIANCE_STD_M ** 2:
            return None
        if float(np.min(eigenvalues)) < 0.0:
            # Remove only numerical negative eigenvalues while retaining valid
            # negative off-diagonal covariance terms.
            values, vectors = np.linalg.eigh(covariance)
            values = np.maximum(values, 0.0)
            covariance = (vectors * values) @ vectors.T
            covariance = 0.5 * (covariance + covariance.T)
        return covariance

    @staticmethod
    def _diagnostic(index: int, detection: Any, accepted: bool, reason: str, **extra: Any) -> dict:
        result = {
            "detection_index": index,
            "box": [float(value) for value in detection.box],
            "confidence": float(detection.confidence),
            "accepted": bool(accepted),
            "reason": reason,
        }
        result.update(extra)
        return result

    def project(self, packet: FramePacket) -> tuple[list[Observation], list[dict]]:
        """Project detector box feet onto the nearest visible floor surface.

        The second return value contains diagnostics only for rejected
        detections. Geometric acceptance is independent of detector
        confidence; the tracking layer owns any minimum-confidence policy.
        """

        observations: list[Observation] = []
        diagnostics: list[dict] = []
        camera = packet.camera
        if camera.coordinate_frame != self.prior.coordinate_frame:
            for index, detection in enumerate(packet.detections):
                diagnostics.append(self._diagnostic(index, detection, False, "frame_mismatch"))
            return observations, diagnostics

        width, height = float(camera.width), float(camera.height)
        pose = np.asarray(camera.camera_to_world, dtype=float)
        for index, detection in enumerate(packet.detections):
            left, top, right, bottom = (float(value) for value in detection.box)
            if not bool(getattr(detection, "ground_contact_visible", False)):
                diagnostics.append(self._diagnostic(index, detection, False, "ground_contact_unknown"))
                continue
            # A detector box touching the calibrated image boundary may have a
            # missing foot or side.  Reject it before ray casting.
            if left <= 0.0 or top <= 0.0 or right >= width or bottom >= height:
                diagnostics.append(self._diagnostic(index, detection, False, "truncated_or_edge_box"))
                continue
            pixel = np.asarray([(left + right) * 0.5, bottom], dtype=float)
            k = np.asarray(camera.intrinsics, dtype=float)
            try:
                direction = pose[:3, :3] @ np.linalg.solve(k, np.asarray([pixel[0], pixel[1], 1.0], dtype=float))
            except np.linalg.LinAlgError:
                diagnostics.append(self._diagnostic(index, detection, False, "invalid_camera"))
                continue
            if not np.isfinite(direction).all() or abs(float(direction[2])) < 0.02:
                diagnostics.append(self._diagnostic(index, detection, False, "grazing_ray"))
                continue
            hit = self._ray_surface_hit(camera, pixel, check_walls=True)
            if hit is None:
                # Distinguish a wall from an absent surface where possible;
                # both are intentionally rejected.
                raw_hit = self._ray_surface_hit(camera, pixel, check_walls=False)
                reason = "occluded_by_wall" if raw_hit is not None else "no_surface"
                diagnostics.append(self._diagnostic(index, detection, False, reason))
                continue
            surface_index, distance, world_point = hit
            covariance = self._finite_difference_covariance(camera, pixel, surface_index, world_point)
            if covariance is None:
                diagnostics.append(self._diagnostic(index, detection, False, "excessive_uncertainty"))
                continue
            surface = self._surfaces[surface_index]
            try:
                observation = Observation(
                    sensor_id=camera.sensor_id,
                    frame_id=packet.frame_id,
                    detection_index=index,
                    t=float(packet.t),
                    xyz=tuple(float(value) for value in world_point),
                    covariance_xy=covariance.tolist(),
                    confidence=float(detection.confidence),
                    surface_id=surface.id,
                )
            except Exception as exc:
                diagnostics.append(self._diagnostic(index, detection, False, "invalid_observation", detail=str(exc)))
                continue
            observations.append(observation)
        return observations, diagnostics

    def _same_surface_or_elevation(self, start: np.ndarray, node: np.ndarray) -> bool:
        start_surface = self._surface_for_point(start)
        node_surface = self._surface_for_point(node)
        if start_surface is not None and node_surface is not None:
            return start_surface.id == node_surface.id or abs(float(start[2]) - float(node[2])) <= _Z_TOLERANCE
        return abs(float(start[2]) - float(node[2])) <= _Z_TOLERANCE

    def _dijkstra_to(self, goal_node: str) -> tuple[dict[str, float], dict[str, str | None]]:
        distances = {node_id: float("inf") for node_id in self._nodes}
        previous: dict[str, str | None] = {node_id: None for node_id in self._nodes}
        distances[goal_node] = 0.0
        queue: list[tuple[float, str]] = [(0.0, goal_node)]
        while queue:
            distance, node_id = heapq.heappop(queue)
            if distance > distances[node_id] + _EPS:
                continue
            for neighbor, edge_length in self._adjacency[node_id]:
                candidate = distance + edge_length
                if candidate < distances[neighbor] - _EPS:
                    distances[neighbor] = candidate
                    previous[neighbor] = node_id
                    heapq.heappush(queue, (candidate, neighbor))
        return distances, previous

    def route_distance(self, start: Any, end: Any) -> float | None:
        """Conservative graph distance between two supported points, including safe snaps."""
        end_xyz = _as_vec(end, 3, "end")
        if self._surface_for_point(end_xyz) is None:
            return None
        best = float("inf")
        for identity, node in self._nodes.items():
            gap = float(np.linalg.norm(node - end_xyz))
            if gap > _SNAP_RADIUS_M or not self._same_surface_or_elevation(end_xyz, node):
                continue
            if self._nav_edge_blocked(end_xyz, node):
                continue
            path = self.route(start, identity)
            if path is not None:
                best = min(best, float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum()) + gap)
        return best if np.isfinite(best) else None

    def route(self, start: Any, goal_node: str) -> np.ndarray | None:
        """Route from a world point to a graph node using 3D edge lengths."""

        if goal_node not in self._nodes:
            return None
        start_xyz = _as_vec(start, 3, "start")
        # A graph snap from empty space would let a route jump across gaps in
        # the authored walkable geometry. Require a known floor/elevation at
        # the observed start before considering any graph node.
        if self._surface_for_point(start_xyz) is None:
            return None
        distances, previous = self._dijkstra_to(goal_node)
        candidates: list[tuple[float, str]] = []
        for node_id, node_xyz in self._nodes.items():
            gap = float(np.linalg.norm(node_xyz - start_xyz))
            if self._surface_for_point(node_xyz) is None:
                continue
            if gap > _SNAP_RADIUS_M or not self._same_surface_or_elevation(start_xyz, node_xyz):
                continue
            if not np.isfinite(distances[node_id]):
                continue
            # The snap segment uses the same human-height collision check as
            # graph edges, so a floor-level segment cannot slip beneath a wall.
            if self._segment_blocked(
                self._lift_for_navigation(start_xyz),
                self._lift_for_navigation(node_xyz),
                include_surfaces=False,
            ):
                continue
            if abs(float(start_xyz[2]) - float(node_xyz[2])) <= _Z_TOLERANCE and not self._same_level_walkable(start_xyz, node_xyz):
                continue
            candidates.append((gap, node_id))
        if not candidates:
            return None
        _, snapped_id = min(candidates, key=lambda item: (item[0], item[1]))
        node_path = [snapped_id]
        while node_path[-1] != goal_node:
            next_node = previous[node_path[-1]]
            if next_node is None:
                return None
            node_path.append(next_node)
        points = [start_xyz]
        for node_id in node_path:
            node_xyz = self._nodes[node_id]
            if float(np.linalg.norm(points[-1] - node_xyz)) > 1e-9:
                points.append(node_xyz)
        return np.asarray(points, dtype=float)


def _line_aabb_interval(origin: np.ndarray, direction: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> tuple[float, float] | None:
    """Return the unbounded ray parameter interval inside an AABB."""

    low, high = -float("inf"), float("inf")
    for axis in range(3):
        value = float(origin[axis])
        delta = float(direction[axis])
        lo, hi = float(minimum[axis]), float(maximum[axis])
        if abs(delta) <= _EPS:
            if value < lo - _EPS or value > hi + _EPS:
                return None
            continue
        t0 = (lo - value) / delta
        t1 = (hi - value) / delta
        if t0 > t1:
            t0, t1 = t1, t0
        low = max(low, t0)
        high = min(high, t1)
        if low > high + _EPS:
            return None
    return float(low), float(high)


def sample_path(path: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """Piecewise-linear samples at path distances, clamped at both ends."""

    points = np.asarray(path, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0 or not np.isfinite(points).all():
        raise ValueError("path must be a non-empty finite [N,3] array")
    query = np.asarray(distances, dtype=float)
    if not np.isfinite(query).all():
        raise ValueError("distances must be finite")
    if len(points) == 1:
        return np.broadcast_to(points[0], query.shape + (3,)).copy()
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    clamped = np.clip(query, 0.0, float(cumulative[-1]))
    flat = clamped.reshape(-1)
    # One searchsorted/interpolation pass handles scalar, vector, and tensor
    # query arrays alike; this is on the particle rollout hot path.
    segments = np.searchsorted(cumulative, flat, side="right") - 1
    segments = np.clip(segments, 0, len(points) - 2).astype(int)
    lengths = segment_lengths[segments]
    fractions = np.divide(
        flat - cumulative[segments],
        lengths,
        out=np.zeros_like(flat),
        where=lengths > _EPS,
    )
    result = points[segments] + fractions[:, None] * (points[segments + 1] - points[segments])
    return result.reshape(query.shape + (3,))
