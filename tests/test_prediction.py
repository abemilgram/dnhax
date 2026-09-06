import copy
import json
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.prediction.api import router, _sessions
from backend.prediction.engine import PredictionEngine
from backend.prediction.fixture import camera, config, packet, scene
from backend.prediction.replay import replay
from backend.prediction.rollout import CueGate
from backend.prediction.schema import FramePacket, ScenePrior


def engine():
    return PredictionEngine(scene(), clock_id="fixture-clock", evidence_source="synthetic_fixture", config=config())


def test_pixels_only_and_metric_frame_boundaries():
    data = packet(0, "drone", [0, 0, 0]).model_dump()
    data["enemy_xyz"] = [0, 0, 0]
    with pytest.raises(ValidationError):
        FramePacket.model_validate(data)
    e = engine()
    p = packet(0, "drone", [0, 0, 0])
    with pytest.raises(ValueError, match="Coordinate frame"):
        e.ingest(p.model_copy(update={"camera": p.camera.model_copy(update={"coordinate_frame": "radar-pixels"})}), now=0)
    with pytest.raises(ValueError, match="cannot be mixed"):
        PredictionEngine(scene(), clock_id="fixture-clock", evidence_source="video_detector")
    assert e.state()["tracks"] == []


def test_causal_clock_and_duplicate_evidence():
    e = engine()
    p = packet(1, "drone", [0, 0, 0])
    with pytest.raises(ValueError, match="Future"):
        e.ingest(p, now=0)
    state = e.ingest(p, now=1)["state"]
    assert len(state["tracks"]) == 1
    before = copy.deepcopy(state)
    assert e.ingest(p, now=1)["duplicate"]
    assert e.state() == before
    with pytest.raises(ValueError, match="reused"):
        e.ingest(p.model_copy(update={"detector": "different"}), now=1)
    with pytest.raises(ValueError, match="Out-of-order"):
        e.ingest(packet(.5, "player"), now=1)
    with pytest.raises(ValueError, match="Clock mismatch"):
        e.ingest(packet(2, "player").model_copy(update={"clock_id": "other"}), now=2)
    with pytest.raises(ValueError, match="too old"):
        e.ingest(packet(2, "player"), now=4)
    assert e.state() == before


def test_tick_coasts_without_inventing_evidence_and_accepts_transport_lag():
    e = engine()
    s0 = e.ingest(packet(0, "drone", [0, 0, 0]), now=0)["state"]
    s1 = e.tick(1)
    assert s1["tracks"][0]["status"] == "stale"
    assert np.trace(s1["tracks"][0]["covariance_xy"]) > np.trace(s0["tracks"][0]["covariance_xy"])
    assert s1["event_cursor"] == s0["event_cursor"]
    support = lambda state: {h["intent_id"]: h["relative_support"] for h in state["tracks"][0]["hypotheses"]}
    assert support(s0) == support(s1)
    e.ingest(packet(.9, "player"), now=1.1)  # capture behind presentation time is ordinary latency
    assert e.state()["last_evidence"]["capture_to_processing_s"] == pytest.approx(.2)
    assert e.tick(21)["tracks"] == []


def test_no_negative_evidence_without_coverage_or_through_cover():
    e = engine()
    e.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    start = e.banks[next(iter(e.banks))].support.copy()
    s = e.ingest(packet(3.5, "player"), now=3.5)["state"]
    assert e.banks[next(iter(e.banks))].support == start
    assert not next(iter(s["last_evidence"]["negative_evidence"].values()))["applied"]
    # Point away from every target: complete detector coverage is not coverage of the room.
    cam = camera("player", [6, -2, 1.6], [10, -2, 1.6])
    e.ingest(packet(4, "player", complete=True, camera_override=cam), now=4)
    assert e.banks[next(iter(e.banks))].support == start
    data = packet(5, "player").model_dump()
    data["coverage_complete"] = True
    with pytest.raises(ValidationError):
        FramePacket.model_validate(data)


def test_synthetic_replay_reorders_from_visible_miss_and_preserves_alternatives():
    result = replay()
    assert result["synthetic"]
    states = result["snapshots"]
    assert states[0]["tracks"][0]["status"] == "observed"
    assert states[2]["tracks"][0]["status"] == "stale"
    pre = {h["intent_id"]: h["relative_support"] for h in states[3]["tracks"][0]["hypotheses"]}
    post = {h["intent_id"]: h["relative_support"] for h in states[6]["tracks"][0]["hypotheses"]}
    assert post["exit_a"] < pre["exit_a"]
    assert post["exit_b"] > post["exit_a"]
    assert post["hold"] > 0
    assert states[-1]["tracks"][0]["status"] == "observed"
    assert any(event["kind"] == "leading_hypothesis_changed" for event in states[-1]["events"])
    assert json.loads(json.dumps(result, allow_nan=False))


@pytest.mark.parametrize("prediction_model", ["route_bank", "branching"])
def test_api_sessions_tokens_validation_and_deletion(prediction_model):
    app = FastAPI()
    app.include_router(router)
    _sessions.clear()
    client = TestClient(app)
    configured = config().model_dump()
    configured["prediction_model"] = prediction_model
    created = client.post("/api/prediction/sessions", json={"prior": scene().model_dump(),
                          "clock_id": "fixture-clock", "evidence_source": "synthetic_fixture",
                          "config": configured})
    assert created.status_code == 201, created.text
    info = created.json()
    url = f"/api/prediction/sessions/{info['id']}"
    headers = {"X-Prediction-Token": info["token"]}
    assert client.get(url).status_code == 403
    ingest = client.post(url + "/frames", headers=headers, json={"packet": packet(0, "drone", [0, 0, 0]).model_dump(), "now": 0})
    assert ingest.status_code == 200, ingest.text
    assert len(ingest.json()["state"]["tracks"]) == 1
    assert ingest.json()["state"]["prediction_model"] == prediction_model
    assert client.post(url + "/tick", headers=headers, json={"now": 1}).json()["tracks"][0]["status"] == "stale"
    assert client.post(url + "/frames", headers=headers, json={"enemy_xyz": [0, 0, 0], "now": 1}).status_code == 422
    assert client.delete(url, headers=headers).status_code == 204
    assert client.get(url, headers=headers).status_code == 404


def test_display_ticks_do_not_change_gates_banks_or_events():
    e = engine()
    e.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    bank = next(iter(e.banks.values()))
    gate = next(iter(e.gates.values()))
    gate.candidate, gate.since = "exit_b", .5
    events = list(e.events)
    for t in (1, 11, 21):
        e.tick(t)
        assert next(iter(e.banks.values())) is bank
        assert gate.candidate == "exit_b" and gate.since == .5
        assert list(e.events) == events
    new_gate = CueGate(config())
    assert new_gate.update([{"available": True, "intent_id": "x", "label": "X", "rank_score": 1}], 1, allow_change=False) is None
    assert new_gate.leader is None


def test_absence_cooldown_survives_a_matched_observation():
    e = engine()
    e.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    e.ingest(packet(1, "player", complete=True), now=1)
    e.ingest(packet(1.1, "drone", [0, .1, 0]), now=1.1)
    state = e.ingest(packet(1.2, "player", complete=True), now=1.2)["state"]
    result = next(iter(state["last_evidence"]["negative_evidence"].values()))
    assert result["reason"] == "correlated-frame cooldown"
    assert not result["applied"]


def test_cross_floor_reacquisition_requires_connected_geometry_and_travel_time():
    data = scene().model_dump()
    data.update(nodes=[{"id": "low", "xyz": [0, 0, 0]}, {"id": "stairs_low", "xyz": [2, 0, 0]},
                       {"id": "stairs_high", "xyz": [2, 0, 3]}, {"id": "high", "xyz": [4, 0, 3]}],
                edges=[{"a": "low", "b": "stairs_low"}, {"a": "stairs_low", "b": "stairs_high"},
                       {"a": "stairs_high", "b": "high"}], walls=[],
                surfaces=[{"id": "lower", "z": 0, "polygon": [[-2, -2], [6, -2], [6, 2], [-2, 2]]},
                          {"id": "upper", "z": 3, "polygon": [[2, -2], [6, -2], [6, 2], [2, 2]]}],
                intents=[{"id": "up", "label": "upstairs", "kind": "route", "goal_node": "high"},
                         {"id": "hold", "label": "hold", "kind": "hold"}])
    prior = ScenePrior.model_validate(data)
    def run(t, connected=True):
        configured = prior if connected else prior.model_copy(update={"edges": [prior.edges[0], prior.edges[2]]})
        e = PredictionEngine(configured, clock_id="fixture-clock", evidence_source="synthetic_fixture", config=config())
        e.ingest(packet(0, "drone", [0, 0, 0], camera_override=camera("drone", [0, -4, 2], [0, 0, 0])), now=0)
        return e.ingest(packet(t, "drone", [4, 0, 3], camera_override=camera("drone", [4, -4, 6], [4, 0, 3])), now=t)["state"]
    connected = run(3)
    assert len(connected["tracks"]) == 1
    assert connected["tracks"][0]["surface_id"] == "upper"
    assert connected["tracks"][0]["xyz"][2] == 3
    assert len(run(.1)["tracks"]) == 2
    assert len(run(3, connected=False)["tracks"]) == 2
