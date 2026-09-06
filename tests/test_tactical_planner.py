from dataclasses import asdict, replace
import json
import math

import numpy as np
import pytest

from backend.tactical import (
    ActorState,
    CueConfig,
    CueReducer,
    EvidenceState,
    IntentKind,
    LifecycleState,
    NavGraph,
    PlanRanking,
    PlannerConfig,
    RiskBand,
    ScoreComponents,
    TacticalMap,
    TacticalPlanner,
    TimedPoint,
    TrackSnapshot,
    TrajectoryCandidate,
    evaluate_route,
    load_map,
)
from backend.tactical.map import AABB


def planner_map(*, covered=True):
    obstacles = (
        (AABB("flank_cover", (-3.0, 0.0, 1.0), (-2.0, 3.0, 9.0)),)
        if covered
        else ()
    )
    return TacticalMap(
        name="planner-test",
        coordinate_system="right-handed:x-east,y-up,z-north",
        obstacles=obstacles,
        nav_nodes=(
            {"id": "start", "xyz": (0.0, 0.0, 0.0)},
            {"id": "cross_mid", "xyz": (0.0, 0.0, 5.0)},
            {"id": "goal", "xyz": (0.0, 0.0, 10.0)},
            {"id": "flank_a", "xyz": (-5.0, 0.0, 0.0)},
            {"id": "flank_b", "xyz": (-5.0, 0.0, 10.0)},
        ),
        nav_edges=(
            {"id": "hold_edge", "from": "start", "to": "cross_mid", "cost": 5.0},
            {"id": "cross_goal", "from": "cross_mid", "to": "goal", "cost": 5.0},
            {"id": "flank_enter", "from": "start", "to": "flank_a", "cost": 5.0},
            {"id": "flank_corridor", "from": "flank_a", "to": "flank_b", "cost": 10.0},
            {"id": "flank_exit", "from": "flank_b", "to": "goal", "cost": 5.0},
        ),
        zones=(
            (
                {
                    "id": "covered_flank",
                    "min": (-6.0, 0.0, -1.0),
                    "max": (-4.0, 3.0, 11.0),
                    "openness": 0.2,
                },
            )
            if covered
            else ()
        ),
        intents=(
            {"id": "hold", "kind": "HOLD", "anchor": "start"},
            {
                "id": "cross",
                "kind": "CROSS",
                "waypoints": ["cross_mid"],
                "goal": "goal",
            },
            {
                "id": "flank",
                "kind": "FLANK",
                "waypoints": ["flank_a", "flank_b"],
                "goal": "goal",
            },
        ),
    )


def observer(*, variance=1e-6):
    covariance = np.diag([variance, variance, variance, variance])
    return TrackSnapshot(
        track_id=1,
        t=0.0,
        last_observed_t=0.0,
        xyz=(0.0, 1.0, 5.0),
        velocity_xz=(0.0, 0.0),
        covariance=tuple(tuple(float(value) for value in row) for row in covariance),
        evidence=EvidenceState.OBSERVED,
        lifecycle=LifecycleState.CONFIRMED,
        hits=5,
        misses=0,
        expected_visible_misses=0,
    )


def score(risk=0.1):
    return ScoreComponents(
        route_length=1.0,
        turn_cost=0.0,
        los_fraction=risk,
        exposure_fraction=risk,
        time_in_open=risk,
        open_fraction=risk,
        uncertainty_risk=0.0,
        risk_std=0.0,
        exposure_cvar90=risk,
        reach_probability=1.0,
        progress=1.0,
        invalid_fraction=0.0,
        risk_score=risk,
    )


def candidate(intent, utility, risk=0.1, valid=True):
    return TrajectoryCandidate(
        intent=intent,
        valid=valid,
        utility=utility,
        score=score(risk),
        route=(TimedPoint(0.0, (0.0, 0.0, 0.0)),),
        reasons=("route_valid",) if valid else ("route_unavailable",),
    )


def ranking(t, *candidates):
    return PlanRanking(t=t, cycle_index=int(t * 10), candidates=candidates)


def test_graph_validation_and_astar_ties_are_deterministic():
    tactical_map = planner_map()
    nodes = tactical_map.nav_nodes + (
        {"id": "tie_a", "xyz": (-1.0, 0.0, 2.0)},
        {"id": "tie_b", "xyz": (1.0, 0.0, 2.0)},
        {"id": "tie_goal", "xyz": (0.0, 0.0, 4.0)},
    )
    edges = tactical_map.nav_edges + (
        {"id": "tie_1", "from": "start", "to": "tie_a", "cost": 2.0},
        {"id": "tie_2", "from": "start", "to": "tie_b", "cost": 2.0},
        {"id": "tie_3", "from": "tie_a", "to": "tie_goal", "cost": 2.0},
        {"id": "tie_4", "from": "tie_b", "to": "tie_goal", "cost": 2.0},
    )
    graph = NavGraph(replace(tactical_map, nav_nodes=nodes, nav_edges=edges))
    assert graph.astar("start", "tie_goal") == ("start", "tie_a", "tie_goal")
    assert graph.nearest_node((0.0, 0.0, 2.0)).node_id == "tie_a"

    duplicate = tactical_map.nav_edges + (
        {"id": "duplicate", "from": "start", "to": "cross_mid", "cost": 5.0},
    )
    with pytest.raises(ValueError, match="duplicate navigation connection"):
        NavGraph(replace(tactical_map, nav_edges=duplicate))
    with pytest.raises(ValueError, match="exactly HOLD, CROSS, and FLANK"):
        NavGraph(replace(tactical_map, intents=tactical_map.intents[:2]))


def test_every_authored_path_and_public_route_stays_on_graph():
    tactical_map = load_map()
    planner = TacticalPlanner(
        tactical_map, PlannerConfig(rollouts_per_intent=2, scenario_seed="routes")
    )
    graph = planner.graph
    actor = ActorState(2.0, (0.0, 0.0, -10.0))
    result = planner.plan(actor)
    assert {item.intent for item in result.candidates} == set(IntentKind)
    for intent in IntentKind:
        node_path = graph.intent_path(intent, actor.xyz)
        assert graph.is_valid_path(node_path)
    segments = [
        (first.xyz, second.xyz)
        for first in graph.nodes
        for second in graph.nodes
        if graph.is_valid_path((first.node_id, second.node_id))
    ]
    for item in result.candidates:
        assert item.valid
        assert all(b.t >= a.t for a, b in zip(item.route, item.route[1:]))
        assert all(
            any(_point_on_segment(point.xyz, start, end) for start, end in segments)
            for point in item.route
        )


def test_obstacle_edges_and_blocked_actor_connectors_invalidate_routes():
    edge_map = replace(
        planner_map(covered=False),
        obstacles=(AABB("cross_block", (-0.5, 0.0, 4.0), (0.5, 2.0, 6.0)),),
    )
    assert edge_map.segment_collides(
        (-2.0, 0.0, 4.0),
        (2.0, 0.0, 4.0),
        horizontal_clearance=0.0,
    )
    ranking = TacticalPlanner(edge_map).plan(ActorState(0.0, (0.0, 0.0, 0.0)))
    cross = next(item for item in ranking.candidates if item.intent is IntentKind.CROSS)
    assert not cross.valid
    assert cross.reasons == ("route_unavailable",)

    connector_map = replace(
        planner_map(covered=False),
        obstacles=(AABB("connector_block", (0.4, 0.0, -0.2), (0.6, 2.0, 0.2)),),
    )
    blocked = TacticalPlanner(connector_map).plan(
        ActorState(0.0, (1.0, 0.0, 0.0))
    )
    assert all(not item.valid for item in blocked.candidates)
    assert all(
        item.reasons == ("actor_connector_blocked",)
        for item in blocked.candidates
    )


def test_actor_connector_is_included_in_route_and_distance():
    planner = TacticalPlanner(
        planner_map(covered=False),
        PlannerConfig(rollouts_per_intent=2, scenario_seed="connector"),
    )
    actor = ActorState(0.0, (0.5, 0.0, 0.5))
    ranking = planner.plan(actor)
    cross = next(item for item in ranking.candidates if item.intent is IntentKind.CROSS)
    assert cross.valid
    assert cross.route[0].xyz == actor.xyz
    assert cross.score.route_length == pytest.approx(10.0 + math.sqrt(0.5))


def test_cover_blocks_los_and_uncertainty_worsens_route_risk():
    tactical_map = planner_map(covered=True)
    route = tuple(
        TimedPoint(float(index), (-5.0, 0.0, float(index)))
        for index in range(10)
    )
    precise = evaluate_route(tactical_map, route, [observer(variance=1e-6)])
    uncertain = evaluate_route(tactical_map, route, [observer(variance=4.0)])
    assert precise.los_fraction == 0.0
    assert precise.open_fraction == pytest.approx(0.2)
    assert precise.time_in_open > 0.0
    assert uncertain.uncertainty_risk > precise.uncertainty_risk
    assert uncertain.risk_std > precise.risk_std
    assert uncertain.risk_score > precise.risk_score


def test_openness_is_independent_of_observers_and_covered_routes_are_lower():
    route = tuple(
        TimedPoint(float(index), (-5.0, 0.0, float(index)))
        for index in range(10)
    )
    open_score = evaluate_route(planner_map(covered=False), route, [])
    covered_score = evaluate_route(planner_map(covered=True), route, [])
    assert open_score.los_fraction == 0.0
    assert open_score.open_fraction == 1.0
    assert open_score.time_in_open == pytest.approx(9.0)
    assert covered_score.time_in_open == pytest.approx(1.8)
    assert covered_score.open_fraction < open_score.open_fraction


def test_reliability_decays_from_last_observation_not_prediction_time():
    stale = replace(observer(), t=5.0, last_observed_t=0.0)
    fresh = replace(stale, last_observed_t=5.0)
    route = (TimedPoint(5.0, (0.0, 0.0, 0.0)),)
    stale_score = evaluate_route(planner_map(covered=False), route, [stale])
    fresh_score = evaluate_route(planner_map(covered=False), route, [fresh])
    assert stale_score.exposure_fraction < fresh_score.exposure_fraction


def test_covered_flank_beats_exposed_cross_and_geometry_can_reverse_it():
    config = PlannerConfig(
        rollouts_per_intent=8,
        scenario_seed="geometry",
        horizon=9.0,
    )
    actor = ActorState(0.0, (0.0, 0.0, 0.0))
    covered = TacticalPlanner(planner_map(covered=True), config).plan(
        actor, [observer()]
    )
    open_map = TacticalPlanner(planner_map(covered=False), config).plan(
        actor, [observer()]
    )
    covered_by_kind = {item.intent: item for item in covered.candidates}
    open_by_kind = {item.intent: item for item in open_map.candidates}
    assert (
        covered_by_kind[IntentKind.FLANK].score.exposure_fraction
        < covered_by_kind[IntentKind.CROSS].score.exposure_fraction
    )
    assert (
        covered_by_kind[IntentKind.FLANK].utility
        > covered_by_kind[IntentKind.CROSS].utility
    )
    assert (
        open_by_kind[IntentKind.CROSS].utility
        > open_by_kind[IntentKind.FLANK].utility
    )


def test_planner_is_repeatable_and_serializes():
    planner = TacticalPlanner(
        planner_map(), PlannerConfig(rollouts_per_intent=4, scenario_seed=1234)
    )
    actor = ActorState(5.0, (0.0, 0.0, 0.0))
    first = planner.plan(actor, [observer(variance=0.25)], cycle_index=7)
    second = planner.plan(actor, [observer(variance=0.25)], cycle_index=7)
    assert first == second
    assert json.dumps(asdict(first), sort_keys=True) == json.dumps(
        asdict(second), sort_keys=True
    )


def test_hysteresis_suppresses_flicker_then_emits_one_sustained_change():
    hold = candidate(IntentKind.HOLD, 1.0)
    flank = candidate(IntentKind.FLANK, 1.2)
    reducer = CueReducer(
        CueConfig(
            switch_margin=0.15,
            consecutive_cycles=2,
            minimum_dwell=0.0,
            emergency_risk_threshold=None,
        )
    )
    initial = reducer.update(ranking(0.0, hold, flank))
    assert initial is not None and initial.intent is IntentKind.HOLD
    assert reducer.update(ranking(1.0, flank, hold)) is None
    assert reducer.update(ranking(2.0, hold, flank)) is None
    assert reducer.update(ranking(3.0, flank, hold)) is None
    changed = reducer.update(ranking(4.0, flank, hold))
    assert changed is not None
    assert changed.intent is IntentKind.FLANK
    assert changed.reasons == ("top_intent_changed",)
    assert reducer.update(ranking(5.0, flank, hold)) is None


def test_identical_cycles_deduplicate_and_status_changes_emit_once():
    hold = candidate(IntentKind.HOLD, 1.0, risk=0.1)
    reducer = CueReducer()
    cues = [
        reducer.update(ranking(float(index), hold), tracking_degraded=False)
        for index in range(20)
    ]
    assert sum(item is not None for item in cues) == 1
    degraded = reducer.update(ranking(20.0, hold), tracking_degraded=True)
    assert degraded is not None
    assert degraded.reasons == ("tracking_degraded",)
    assert reducer.update(ranking(21.0, hold), tracking_degraded=True) is None
    recovered = reducer.update(ranking(22.0, hold), tracking_degraded=False)
    assert recovered is not None
    assert recovered.reasons == ("tracking_recovered",)
    assert recovered.risk_band is RiskBand.LOW


def test_serialized_plans_and_cues_have_no_action_or_targeting_fields():
    planner = TacticalPlanner(
        planner_map(), PlannerConfig(rollouts_per_intent=2, scenario_seed="safe")
    )
    result = planner.plan(
        ActorState(0.0, (0.0, 0.0, 0.0)), [observer()], cycle_index=1
    )
    cue = CueReducer().update(result)
    assert cue is not None
    keys = _all_keys(asdict(result)) | _all_keys(asdict(cue))
    forbidden = {"target", "aim", "fire", "reticle", "action", "control"}
    assert not any(
        forbidden_word in key.lower().split("_")
        for key in keys
        for forbidden_word in forbidden
    )


def _point_on_segment(point, start, end, tolerance=1e-8):
    vector = np.subtract(end, start)
    offset = np.subtract(point, start)
    denominator = float(np.dot(vector, vector))
    if denominator <= tolerance:
        return float(np.linalg.norm(offset)) <= tolerance
    fraction = float(np.dot(offset, vector) / denominator)
    projection = np.asarray(start) + np.clip(fraction, 0.0, 1.0) * vector
    return -tolerance <= fraction <= 1.0 + tolerance and math.dist(
        point, projection
    ) <= tolerance


def _all_keys(value):
    if isinstance(value, dict):
        return set(value) | {
            key
            for nested in value.values()
            for key in _all_keys(nested)
        }
    if isinstance(value, (list, tuple)):
        return {key for nested in value for key in _all_keys(nested)}
    return set()
