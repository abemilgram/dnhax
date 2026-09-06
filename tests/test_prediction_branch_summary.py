import json
from types import SimpleNamespace

import numpy as np
import pytest

from backend.prediction.branch_summary import summarize_branching


class Branch:
    def __init__(self, points, *, first_arrival_s=None, reached_goal_node=None, reached_goal_s=None, modes=None):
        self.times = np.asarray([0.0, 5.0, 10.0])
        self.xyz = np.asarray(points, dtype=float)
        self.modes = modes or ["walk", "walk", "walk"]
        self.speed = 2.0
        self.first_arrival_s = first_arrival_s
        self.reached_goal_node = reached_goal_node
        self.reached_goal_s = reached_goal_s
        self.reverse_count = 0

    def positions(self, elapsed):
        values = np.asarray(elapsed, dtype=float)
        if values.ndim == 0:
            return np.asarray([np.interp(float(values), self.times, self.xyz[:, axis]) for axis in range(3)])
        return np.column_stack([
            np.interp(values, self.times, self.xyz[:, axis]) for axis in range(3)
        ])


class Geometry:
    def __init__(self, intents):
        self.prior = SimpleNamespace(intents=intents)

    def line_of_sight_many(self, origin, targets):
        return np.ones(len(targets), dtype=bool)


CONFIG = SimpleNamespace(horizon_s=10.0, step_s=0.2)


def intent(intent_id, label, kind="route", goal_node=None):
    return SimpleNamespace(id=intent_id, label=label, kind=kind, goal_node=goal_node)


def branch(x, *, first=None, goal=None, reached=None):
    points = [[x, 0.0, 0.0], [x + 1.0, 0.0, 0.0], [x + 2.0, 0.0, 0.0]]
    return Branch(points, first_arrival_s=first, reached_goal_node=goal, reached_goal_s=reached)


def test_weighted_censored_arrival_summary_and_actual_event_preview():
    geometry = Geometry([intent("route-a", "A", goal_node="goal-a"), intent("hold", "Hold", "hold")])
    trajectories = {
        "route-a": [branch(0, first=2, goal="goal-a", reached=2), branch(1, first=5, goal="goal-a", reached=5), branch(2)],
        "hold": [branch(10)],
    }
    ranking, belief = summarize_branching(
        geometry, CONFIG, trajectories,
        {"route-a": np.array([.2, .3, .5]), "hold": np.array([1.0])},
        {"route-a": .75, "hold": .25}, {"route-a": 0, "hold": 0},
        anchor_t=0.0, now=3.0, player=None,
    )
    route = next(item for item in ranking if item["intent_id"] == "route-a")
    arrival = route["arrival_remaining_s"]
    assert arrival["already_arrived_mass"] == .2
    assert arrival["future_arrival_within_horizon_mass"] == .3
    assert arrival["no_arrival_within_horizon_mass"] == .5
    assert arrival["p10"] == arrival["p50"] == arrival["p90"] == 2.0
    assert route["path_preview_status"] == "actual_event_breakpoints"
    assert [point["dt"] for point in route["path_preview"]] == [0.0, 2.0, 7.0]
    assert route["path_preview"][1]["xyz"] == [3.0, 0.0, 0.0]
    assert belief["snapshots"][0]["mass_accounting"]["total"] == 1.0


def test_multiple_modes_keep_global_mass_and_actual_representatives():
    geometry = Geometry([intent("route-a", "A", goal_node="goal-a"), intent("hold", "Hold", "hold")])
    trajectories = {
        "route-a": [branch(.1, goal="goal-a", reached=4), branch(2.1, goal="goal-b", reached=7)],
        "hold": [branch(5.1)],
    }
    ranking, belief = summarize_branching(
        geometry, CONFIG, trajectories,
        {"route-a": np.array([.4, .6]), "hold": np.array([1.0])},
        {"route-a": .8, "hold": .2}, {"route-a": 0, "hold": 0},
        anchor_t=0.0, now=3.0, player=None,
    )
    snapshot = next(item for item in belief["snapshots"] if item["dt"] == 3.0)
    assert sum(item["probability_model_mass"] for item in snapshot["bins"]) == 1.0
    assert snapshot["unsupported_intent_mass"] == 0.0
    assert snapshot["omitted_mass"] == 0.0
    assert snapshot["mass_accounting"]["total"] == 1.0
    assert any(item["xyz"] != item["voxel_index"] for item in snapshot["bins"])
    route = next(item for item in ranking if item["intent_id"] == "route-a")
    # The representative is one actual high-weight branch, not a mean path.
    assert route["path_preview"][0]["xyz"] == [2.7, 0.0, 0.0]
    exits = {item["goal_node"]: item for item in belief["exit_forecasts"]}
    assert exits["goal-a"]["future_arrival_mass"] == pytest.approx(.32)
    assert exits["goal-b"]["future_arrival_mass"] == pytest.approx(.48)
    assert exits["goal-a"]["conditional_future_quantiles"]["p50"] == 1.0
    assert exits["goal-b"]["conditional_future_quantiles"]["p50"] == 4.0
    assert exits["goal-a"]["no_arrival_here_by_horizon_mass"] == pytest.approx(.68)


def test_unsupported_intent_mass_is_separate_from_no_arrival():
    geometry = Geometry([intent("route-a", "A", goal_node="goal-a"), intent("route-b", "B", goal_node="goal-b")])
    trajectories = {"route-a": [branch(0, first=5, goal="goal-a", reached=5)], "route-b": []}
    ranking, belief = summarize_branching(
        geometry, CONFIG, trajectories,
        {"route-a": np.array([1.0]), "route-b": np.empty(0)},
        {"route-a": .75, "route-b": .25}, {"route-a": 2, "route-b": 64},
        anchor_t=0.0, now=1.0, player=None,
    )
    unsupported = next(item for item in ranking if item["intent_id"] == "route-b")
    assert unsupported["available"] is False
    assert unsupported["arrival_remaining_s"]["status"] == "unavailable"
    for snapshot in belief["snapshots"]:
        assert snapshot["unsupported_intent_mass"] == .25
        assert sum(item["probability_model_mass"] for item in snapshot["bins"]) == .75
        assert snapshot["mass_accounting"]["total"] == 1.0
    exits = {item["goal_node"]: item for item in belief["exit_forecasts"]}
    assert exits["goal-a"]["unsupported_mass"] == .25
    assert exits["goal-a"]["no_arrival_here_by_horizon_mass"] == 0.0
    assert exits["goal-b"]["unsupported_mass"] == .25
    assert exits["goal-b"]["no_arrival_here_by_horizon_mass"] == .75


def test_horizon_expiry_has_no_future_snapshots_and_output_is_json_safe():
    geometry = Geometry([intent("route-a", "A", goal_node="goal-a")])
    trajectories = {"route-a": [branch(0, first=8, goal="goal-a", reached=8)]}
    ranking, belief = summarize_branching(
        geometry, CONFIG, trajectories, {"route-a": np.array([1.0])}, {"route-a": 1.0}, {"route-a": 0},
        anchor_t=0.0, now=10.0, player=None,
    )
    assert ranking[0]["available"] is False
    assert ranking[0]["remaining_horizon_s"] == 0.0
    assert belief["status"] == "expired"
    assert belief["snapshots"] == []
    assert belief["exit_forecasts"] == []
    assert ranking[0]["arrival_remaining_s"] is None
    json.dumps([ranking, belief], allow_nan=False)


def test_representative_replacement_preserves_accumulated_bin_mass():
    geometry = Geometry([intent("hold", "Hold", "hold")])
    trajectories = {"hold": [branch(.1), branch(.2)]}
    _ranking, belief = summarize_branching(
        geometry, CONFIG, trajectories, {"hold": np.array([.25, .75])}, {"hold": 1.0}, {"hold": 0},
        anchor_t=0.0, now=0.0, player=None,
    )
    bins = belief["snapshots"][0]["bins"]
    assert len(bins) == 1
    assert bins[0]["probability_model_mass"] == pytest.approx(1.0)
    assert bins[0]["xyz"] == [0.2, 0.0, 0.0]
