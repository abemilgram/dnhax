from types import SimpleNamespace

import numpy as np
import pytest

from backend.prediction.geometry import WorldGeometry
from backend.prediction.schema import EngineConfig, Observation, ScenePrior
from backend.prediction.branch_motion import BranchTrajectory, generate_trajectories


def prior(*, zigzag=False):
    nodes = [
        {"id": "start", "xyz": [0, 0, 0]},
        {"id": "junction", "xyz": [0, 2, 0] if zigzag else [2, 0, 0]},
        {"id": "goal_a", "xyz": [2, 2, 0] if zigzag else [6, 0, 0]},
        {"id": "goal_b", "xyz": [3, 2, 0] if zigzag else [2, 3, 0]},
    ]
    edges = [
        {"a": "start", "b": "junction"},
        {"a": "junction", "b": "goal_a"},
        {"a": "junction", "b": "goal_b"},
    ]
    walls = []
    if zigzag:
        # The direct start -> goal_a shortcut crosses this box; each authored
        # L-shaped navigation edge goes around it.
        walls = [{"id": "diagonal-block", "minimum": [0.5, 0.5, 0], "maximum": [1.5, 1.5, 3]}]
    return ScenePrior.model_validate({
        "id": "branch-fixture",
        "coordinate_frame": "fixture-m",
        "provenance": "Hand-authored branch trajectory fixture.",
        "scale_source": "Coordinates are explicit fixture meters.",
        "synthetic": True,
        "nodes": nodes,
        "edges": edges,
        "surfaces": [{"id": "floor", "z": 0, "polygon": [[-3, -3], [8, -3], [8, 8], [-3, 8]]}],
        "walls": walls,
        "intents": [
            {"id": "hold", "label": "Hold", "kind": "hold", "prior": 0.2},
            {"id": "route-a", "label": "Route A", "kind": "route", "goal_node": "goal_a", "prior": 0.4},
            {"id": "route-b", "label": "Route B", "kind": "route", "goal_node": "goal_b", "prior": 0.4},
        ],
    })


def track_at(xyz=(0, 0, 0)):
    observation = Observation(
        sensor_id="sensor",
        frame_id="frame-0",
        t=0,
        xyz=xyz,
        covariance_xy=[[1e-8, 0], [0, 1e-8]],
        confidence=0.9,
        surface_id="floor",
    )
    return SimpleNamespace(last_observation=observation, last_seen=0.0)


def cfg(**updates):
    values = dict(
        horizon_s=8,
        particles=8,
        speed_mps=2,
        speed_std_mps=0,
        branch_exploration_probability=0,
        branch_stop_rate_per_s=0,
        branch_reverse_rate_per_s=0,
    )
    values.update(updates)
    if "max_events_per_sample" in values:
        return SimpleNamespace(**values)
    return EngineConfig(**values)


def route_intent(goal="goal_a"):
    return SimpleNamespace(kind="route", goal_node=goal)


def hold_intent():
    return SimpleNamespace(kind="hold", goal_node=None)


class MeanSamplingRng:
    """Keep the start exactly on the observed goal for the time-zero case."""

    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)

    def random(self, *args, **kwargs):
        return self._rng.random(*args, **kwargs)

    def normal(self, loc=0.0, scale=1.0, *args, **kwargs):
        return np.asarray(loc, dtype=float)

    def multivariate_normal(self, mean, cov, *args, **kwargs):
        return np.asarray(mean, dtype=float).copy()

    def exponential(self, *args, **kwargs):
        return self._rng.exponential(*args, **kwargs)

    def uniform(self, *args, **kwargs):
        return self._rng.uniform(*args, **kwargs)

    def choice(self, *args, **kwargs):
        return self._rng.choice(*args, **kwargs)


def test_branch_trajectory_positions_are_piecewise_and_clamped():
    trajectory = BranchTrajectory(
        np.asarray([0.0, 1.0, 2.0]),
        np.asarray([[0, 0, 0], [0, 2, 0], [2, 2, 0]], dtype=float),
        ["moving", "moving", "arrived"],
        2.0,
        2.0,
        0,
    )
    assert trajectory.positions(-1).shape == (3,)
    assert np.allclose(trajectory.positions(-1), [0, 0, 0])
    assert np.allclose(trajectory.positions([0.5, 1.5, 3]), [[0, 1, 0], [1, 2, 0], [2, 2, 0]])


def test_exploration_takes_actual_goal_b_and_absorbs_before_assigned_goal_a():
    geometry = WorldGeometry(prior())
    trajectories = generate_trajectories(
        geometry,
        cfg(branch_exploration_probability=1),
        track_at(),
        route_intent("goal_a"),
        np.random.default_rng(7),
    )
    assert len(trajectories) == 8
    assert any(trajectory.reached_goal_node == "goal_b" for trajectory in trajectories)
    assert any(trajectory.reached_goal_node == "goal_a" for trajectory in trajectories)
    assert all(trajectory.first_arrival_s is None for trajectory in trajectories if trajectory.reached_goal_node == "goal_b")
    for trajectory in trajectories:
        if trajectory.reached_goal_node is not None:
            assert trajectory.modes[-1] == "arrived"
            assert np.allclose(trajectory.positions(trajectory.times[-1] + 100), trajectory.xyz[-1])


def test_starting_on_any_modeled_goal_is_absorbing_at_time_zero():
    geometry = WorldGeometry(prior())
    trajectories = generate_trajectories(
        geometry,
        cfg(),
        track_at((2, 3, 0)),
        route_intent("goal_a"),
        MeanSamplingRng(12),
    )
    assert len(trajectories) == 8
    assert all(trajectory.reached_goal_node == "goal_b" for trajectory in trajectories)
    assert all(trajectory.reached_goal_s == 0 for trajectory in trajectories)
    assert all(trajectory.first_arrival_s is None for trajectory in trajectories)
    assert all(trajectory.modes == ["arrived", "arrived"] for trajectory in trajectories)


def test_hold_is_stationary_and_unavailable_route_is_omitted():
    geometry = WorldGeometry(prior())
    holds = generate_trajectories(geometry, cfg(), track_at(), hold_intent(), np.random.default_rng(4))
    assert len(holds) == 8
    for trajectory in holds:
        assert np.allclose(trajectory.xyz, trajectory.xyz[0])
        assert trajectory.first_arrival_s is None
        assert trajectory.reverse_count == 0

    disconnected_data = prior().model_dump()
    disconnected_data["edges"] = [{"a": "start", "b": "junction"}, {"a": "junction", "b": "goal_b"}]
    disconnected = ScenePrior.model_validate(disconnected_data)
    geometry_disconnected = WorldGeometry(disconnected)
    assert generate_trajectories(geometry_disconnected, cfg(), track_at(), route_intent("goal_a"), np.random.default_rng(4)) == []


def test_reverse_is_real_mid_edge_event_and_graph_corners_are_not_shortcut():
    geometry = WorldGeometry(prior(zigzag=True))
    reverse_config = cfg(branch_reverse_rate_per_s=2.0)
    trajectories = generate_trajectories(geometry, reverse_config, track_at(), route_intent("goal_a"), np.random.default_rng(2))
    assert trajectories
    assert any(trajectory.reverse_count > 0 and "returned" in trajectory.modes for trajectory in trajectories)

    no_events = generate_trajectories(geometry, cfg(), track_at(), route_intent("goal_a"), np.random.default_rng(3))
    assert no_events
    for trajectory in no_events:
        # The authored junction is an exact event breakpoint.  A straight
        # interpolation from start to goal would cross the wall and fail this.
        assert any(np.allclose(point, [0, 2, 0]) for point in trajectory.xyz)
        assert geometry.line_of_sight([0, 0, 0], [0, 2, 0])
        assert geometry.line_of_sight([0, 2, 0], [2, 2, 0])
        assert not geometry.line_of_sight([0, 0, 0], [2, 2, 0])


def test_seeded_generation_is_repeatable_and_event_cap_rejects_pathological_sample():
    geometry = WorldGeometry(prior())
    config = cfg(branch_exploration_probability=.5, branch_stop_rate_per_s=.3, branch_reverse_rate_per_s=.2)
    first = generate_trajectories(geometry, config, track_at(), route_intent("goal_a"), np.random.default_rng(99))
    second = generate_trajectories(geometry, config, track_at(), route_intent("goal_a"), np.random.default_rng(99))
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert np.array_equal(left.times, right.times)
        assert np.array_equal(left.xyz, right.xyz)
        assert left.modes == right.modes
        assert left.reverse_count == right.reverse_count
        assert left.reached_goal_node == right.reached_goal_node

    capped = cfg(branch_stop_rate_per_s=2.0, max_events_per_sample=1)
    assert generate_trajectories(geometry, capped, track_at(), route_intent("goal_a"), np.random.default_rng(13)) == []
