import numpy as np
import pytest
from backend.prediction.schema import Camera, Detection, FramePacket, Intent, NavEdge, NavNode, ScenePrior, Surface, Wall
from backend.prediction.geometry import WorldGeometry, point_in_polygon, project_world, sample_path

def camera(position=(0, 0, 2), width=100, height=100):
    # Camera right/down/forward maps to world right/-Y/-Z: it looks down.
    pose = np.eye(4)
    pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[:3, 3] = position
    return Camera(
        sensor_id="cam",
        coordinate_frame="room",
        calibration_id="fixture-cal",
        calibration_provenance="synthetic calibration fixture",
        width=width,
        height=height,
        intrinsics=[[100, 0, width / 2], [0, 100, height / 2], [0, 0, 1]],
        camera_to_world=pose.tolist(),
        position_std_m=0.1,
        pixel_std=1,
        available_t=0,
    )


def prior(*, surfaces=None, walls=None, nodes=None, edges=None):
    nodes = nodes or [NavNode(id="a", xyz=(0, 0, 0)), NavNode(id="b", xyz=(2, 0, 0))]
    edges = edges or [NavEdge(a=nodes[0].id, b=nodes[1].id)]
    return ScenePrior(
        id="fixture",
        coordinate_frame="room",
        provenance="synthetic bounded geometry fixture",
        scale_source="known metric fixture dimensions",
        synthetic=True,
        nodes=nodes,
        edges=edges,
        surfaces=surfaces or [Surface(id="floor", polygon=[(-5, -5), (5, -5), (5, 5), (-5, 5)], z=0)],
        walls=walls or [],
        intents=[Intent(id="hold", label="Hold", kind="hold"), Intent(id="route", label="Route", kind="route", goal_node=nodes[-1].id)],
    )


def packet(cam, detections):
    return FramePacket(
        frame_id="frame-1",
        clock_id="clock",
        t=1,
        available_t=1,
        camera=cam,
        evidence_source="synthetic_fixture",
        detector="fixture",
        detections=detections,
    )


def test_pinhole_projection_roundtrip_and_polygon_boundary():
    cam = Camera(
        sensor_id="cam",
        coordinate_frame="room",
        calibration_id="cal",
        calibration_provenance="synthetic calibration fixture",
        width=100,
        height=100,
        intrinsics=[[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        camera_to_world=np.eye(4).tolist(),
        available_t=0,
    )
    pixel = project_world(cam, (0, 0, 2))
    assert np.allclose(pixel, [50, 50])
    assert point_in_polygon((0, 0), [(-1, -1), (1, -1), (1, 1), (-1, 1)])
    assert not point_in_polygon((2, 0), [(-1, -1), (1, -1), (1, 1), (-1, 1)])


def test_project_uses_nearest_positive_surface_and_covariance():
    surfaces = [
        Surface(id="lower", polygon=[(-8, -8), (8, -8), (8, 8), (-8, 8)], z=0),
        Surface(id="upper", polygon=[(-2, -2), (2, -2), (2, 2), (-2, 2)], z=1),
    ]
    geometry = WorldGeometry(prior(surfaces=surfaces))
    observations, diagnostics = geometry.project(packet(camera(position=(0, 0, 3)), [Detection(box=(40, 20, 60, 75), confidence=0.1, ground_contact_visible=True)]))
    assert len(observations) == 1
    assert observations[0].surface_id == "upper"
    assert np.isclose(observations[0].xyz[2], 1)
    assert np.asarray(observations[0].covariance_xy).shape == (2, 2)
    assert diagnostics == []


def test_project_rejects_edge_boxes_walls_and_frame_mismatch():
    wall = Wall(id="wall", minimum=(-0.2, -1, 0), maximum=(0.2, 1, 3))
    nav_nodes = [NavNode(id="a", xyz=(0, 3, 0)), NavNode(id="b", xyz=(2, 3, 0))]
    geometry = WorldGeometry(prior(walls=[wall], nodes=nav_nodes))
    cam = camera()
    boxes = [
        Detection(box=(0, 20, 60, 75), confidence=1, ground_contact_visible=True),
        Detection(box=(40, 20, 60, 75), confidence=1, ground_contact_visible=True),
    ]
    observations, diagnostics = geometry.project(packet(cam, boxes))
    assert not observations
    assert diagnostics[0]["reason"] == "truncated_or_edge_box"
    assert diagnostics[1]["reason"] == "occluded_by_wall"
    mismatch = cam.model_copy(update={"coordinate_frame": "other"})
    _, mismatch_diagnostics = geometry.project(packet(mismatch, [boxes[1]]))
    assert mismatch_diagnostics[0]["reason"] == "frame_mismatch"


def test_project_requires_explicit_ground_contact():
    geometry = WorldGeometry(prior())
    _, diagnostics = geometry.project(packet(camera(), [Detection(box=(40, 20, 60, 75), confidence=1)]))
    assert diagnostics[0]["reason"] == "ground_contact_unknown"


def test_wall_line_of_sight_and_multilevel_floor_occlusion():
    wall = Wall(id="wall", minimum=(-0.2, -1, 0), maximum=(0.2, 1, 3))
    nav_nodes = [NavNode(id="a", xyz=(0, 3, 0)), NavNode(id="b", xyz=(2, 3, 0))]
    geometry = WorldGeometry(prior(walls=[wall], nodes=nav_nodes))
    assert not geometry.line_of_sight((0, -2, 2), (0, 2, 2))
    upper = WorldGeometry(prior(surfaces=[Surface(id="low", polygon=[(-4, -4), (4, -4), (4, 4), (-4, 4)], z=0), Surface(id="high", polygon=[(-1, -1), (1, -1), (1, 1), (-1, 1)], z=2)]))
    assert not upper.line_of_sight((0, 0, 3), (0, 0, -1))


def test_vectorized_visibility_matches_scalar_visibility():
    geometry = WorldGeometry(prior())
    cam = camera()
    points = np.asarray([[0, 0, 0.9], [4, 0, 0.9], [0, 0, 3.0]])
    assert np.array_equal(geometry.visible_many(cam, points), [geometry.visible(cam, p) for p in points])
    assert np.array_equal(geometry.line_of_sight_many((0, -2, 1), points), [geometry.line_of_sight((0, -2, 1), p) for p in points])


def test_route_snaps_with_3d_edge_lengths_and_rejects_disconnected_graph():
    nodes = [
        NavNode(id="start", xyz=(0, 0, 0)),
        NavNode(id="riser", xyz=(1, 0, 2)),
        NavNode(id="goal", xyz=(3, 0, 2)),
        NavNode(id="isolated", xyz=(0, 4, 0)),
    ]
    levels = [Surface(id="floor", polygon=[(-5, -5), (5, -5), (5, 5), (-5, 5)], z=0),
              Surface(id="upper", polygon=[(-5, -5), (5, -5), (5, 5), (-5, 5)], z=2)]
    geometry = WorldGeometry(prior(surfaces=levels, nodes=nodes, edges=[NavEdge(a="start", b="riser"), NavEdge(a="riser", b="goal")]))
    route = geometry.route((0.1, 0, 0), "goal")
    assert route is not None
    assert np.allclose(route[0], [0.1, 0, 0])
    assert np.any(np.isclose(route[:, 2], 2))
    assert geometry.route((0, 4, 0), "goal") is None
    assert geometry.route((20, 20, 0), "goal") is None


def test_same_level_navigation_cannot_jump_a_surface_gap():
    surfaces = [Surface(id="left", polygon=[(-4, -1), (-1, -1), (-1, 1), (-4, 1)], z=0),
                Surface(id="right", polygon=[(1, -1), (4, -1), (4, 1), (1, 1)], z=0)]
    nodes = [NavNode(id="a", xyz=(-2, 0, 0)), NavNode(id="b", xyz=(2, 0, 0))]
    with pytest.raises(ValueError, match="passes through a wall|walkable"):
        WorldGeometry(prior(surfaces=surfaces, nodes=nodes, edges=[NavEdge(a="a", b="b")]))


def test_sample_path_piecewise_interpolation_and_clamp():
    path = np.asarray([[0, 0, 0], [3, 0, 0], [3, 0, 4]], dtype=float)
    samples = sample_path(path, np.asarray([-1, 1.5, 5, 10], dtype=float))
    assert np.allclose(samples, [[0, 0, 0], [1.5, 0, 0], [3, 0, 2], [3, 0, 4]])
    grid = sample_path(path, np.asarray([[0, 5], [10, -2]], dtype=float))
    assert grid.shape == (2, 2, 3)


def test_constructor_rejects_duplicate_ids_degenerate_surface_and_blocked_edge():
    base = prior()
    duplicate = base.model_copy(update={"surfaces": [Surface(id="floor", polygon=[(-1, -1), (1, -1), (1, 1)], z=0), Surface(id="floor", polygon=[(-2, -2), (2, -2), (2, 2)], z=1)]})
    with pytest.raises(ValueError, match="Surface IDs"):
        WorldGeometry(duplicate)
    degenerate = base.model_copy(update={"surfaces": [Surface(id="bad", polygon=[(0, 0), (1, 1), (2, 2)], z=0)]})
    with pytest.raises(ValueError, match="degenerate"):
        WorldGeometry(degenerate)
    blocked = prior(walls=[Wall(id="wall", minimum=(0.5, -1, 0), maximum=(1.5, 1, 2))])
    with pytest.raises(ValueError, match="passes through"):
        WorldGeometry(blocked)
