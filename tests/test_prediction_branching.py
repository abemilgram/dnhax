"""Causal and bounded regression tests for the selectable branching model.

These tests use the existing synthetic scene only.  They intentionally inspect
the documented rank shape and the presence/serializability of ``map_belief``
without depending on an internal occupancy representation.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from pydantic import ValidationError


from backend.prediction.engine import PredictionEngine  # noqa: E402
from backend.prediction.fixture import camera, config, packet, scene  # noqa: E402
from backend.prediction.schema import EngineConfig  # noqa: E402


def engine(model: str = "branching", **overrides: object) -> PredictionEngine:
    values = config().model_dump()
    values.update({
        "prediction_model": model,
        "branch_exploration_probability": 0.08,
        "branch_stop_rate_per_s": 0.12,
        "branch_reverse_rate_per_s": 0.06,
    })
    values.update(overrides)
    return PredictionEngine(
        scene(),
        clock_id="fixture-clock",
        evidence_source="synthetic_fixture",
        config=EngineConfig(**values),
    )


def track(state: dict) -> dict:
    assert len(state["tracks"]) == 1
    return state["tracks"][0]


def support(state: dict) -> dict[str, float]:
    return {item["intent_id"]: item["relative_support"] for item in track(state)["hypotheses"]}


def remaining_horizon(state: dict) -> float:
    values = [float(item["remaining_horizon_s"]) for item in track(state)["hypotheses"]]
    assert values
    return max(values)


@pytest.mark.parametrize("model", ["route_bank", "branching"])
def test_both_models_keep_rank_shape_and_json_safe_state(model: str) -> None:
    state = engine(model).ingest(packet(0, "drone", [0, 0, 0]), now=0)["state"]
    hypotheses = track(state)["hypotheses"]
    assert hypotheses
    assert all({"intent_id", "relative_support", "rank_score", "available"} <= set(item) for item in hypotheses)
    if model == "branching":
        assert isinstance(track(state).get("map_belief"), dict)
    json.dumps(state, allow_nan=False)


@pytest.mark.parametrize("model", ["route_bank", "branching"])
def test_future_evidence_and_duplicate_retries_are_causal_and_idempotent(model: str) -> None:
    prediction = engine(model)
    first_packet = packet(0, "drone", [0, 0, 0])
    before = prediction.ingest(first_packet, now=0)["state"]
    frozen = copy.deepcopy(before)
    with pytest.raises(ValueError, match="Future"):
        prediction.ingest(packet(1, "drone", [0, 0, 0]), now=0.5)
    assert prediction.state() == frozen
    duplicate = prediction.ingest(first_packet, now=0)
    assert duplicate["duplicate"] is True
    assert duplicate["state"] == frozen


def test_empty_frame_without_coverage_does_not_change_branch_support() -> None:
    prediction = engine()
    prediction.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    before = support(prediction.state())
    result = prediction.ingest(packet(1, "player"), now=1)
    assert support(result["state"]) == before
    negative = next(iter(result["state"]["last_evidence"]["negative_evidence"].values()))
    assert negative["applied"] is False


def test_complete_camera_away_from_scene_does_not_reweight_branch_weights() -> None:
    prediction = engine()
    prediction.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    bank = prediction.banks["track-0001"]
    before = {intent_id: weights.copy() for intent_id, weights in bank.weights.items()}
    away = camera("player", position=[0, -100, 1.6], target=[0, -101, 1.6])
    state = prediction.ingest(
        packet(1, "player", complete=True, camera_override=away), now=1
    )["state"]
    negative = next(iter(state["last_evidence"]["negative_evidence"].values()))
    assert negative["applied"] is False
    assert negative["reason"] == "No modeled locations in the verified camera view"
    for intent_id, weights in before.items():
        np.testing.assert_array_equal(bank.weights[intent_id], weights)


def test_empty_coverage_evidence_obeys_source_cooldown() -> None:
    prediction = engine()
    prediction.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    player_view = camera("player", position=[0, -6, 6], target=[0, 1, 0])
    first = prediction.ingest(
        packet(1, "player", complete=True, camera_override=player_view), now=1
    )["state"]
    first_negative = next(iter(first["last_evidence"]["negative_evidence"].values()))
    assert first_negative["applied"] is True
    # A matched positive observation from another source must preserve the
    # per-source cooldown rather than clearing it during bank regeneration.
    prediction.ingest(packet(1.1, "drone", [0, 0.1, 0]), now=1.1)
    before_second = support(prediction.state())
    second = prediction.ingest(
        packet(1.2, "player", complete=True, camera_override=player_view), now=1.2
    )["state"]
    negative = next(iter(second["last_evidence"]["negative_evidence"].values()))
    assert negative["reason"] == "correlated-frame cooldown"
    assert negative["applied"] is False
    assert support(second) == before_second


def test_tick_schedule_does_not_change_branch_weights_or_map_belief() -> None:
    stepped = engine()
    direct = engine()
    initial = packet(0, "drone", [0, 0, 0])
    stepped.ingest(initial, now=0)
    direct.ingest(initial, now=0)
    before_weights = {
        intent_id: weights.copy()
        for intent_id, weights in stepped.banks["track-0001"].weights.items()
    }
    for now in (0.2, 0.4, 0.6, 0.8, 1.0):
        stepped.tick(now)
    for intent_id, weights in before_weights.items():
        np.testing.assert_array_equal(stepped.banks["track-0001"].weights[intent_id], weights)
    stepped_state = stepped.tick(1.5)
    direct_state = direct.tick(1.5)
    assert support(stepped_state) == support(direct_state)
    assert track(stepped_state)["map_belief"] == track(direct_state)["map_belief"]
    assert track(stepped_state)["hypotheses"] == track(direct_state)["hypotheses"]


def test_branch_horizon_expires_from_last_observation_and_refreshes_on_new_observation() -> None:
    prediction = engine(horizon_s=2.0, expire_after_s=5.0)
    prediction.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    at_one = prediction.tick(1.0)
    assert remaining_horizon(at_one) == pytest.approx(1.0, abs=1e-6)
    expired = prediction.tick(2.1)
    assert remaining_horizon(expired) == pytest.approx(0.0, abs=1e-6)
    assert all(item["available"] is False for item in track(expired)["hypotheses"])

    refreshed = prediction.ingest(packet(2.1, "drone", [0, 0, 0]), now=2.1)["state"]
    assert remaining_horizon(refreshed) == pytest.approx(2.0, abs=1e-6)
    assert any(item["available"] for item in track(refreshed)["hypotheses"])
    json.dumps(refreshed, allow_nan=False)


def test_positive_observation_at_exact_horizon_resets_original_priors() -> None:
    prediction = engine(horizon_s=2.0)
    prediction.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    old_bank = prediction.banks["track-0001"]
    prior = old_bank.prior_support.copy()
    # Force a clearly non-prior support vector so the expiry branch is tested
    # directly rather than passing accidentally when likelihoods are uniform.
    intent_ids = list(prior)
    old_bank.support = {
        intent_id: (0.8 if index == 0 else 0.2 / (len(intent_ids) - 1))
        for index, intent_id in enumerate(intent_ids)
    }

    refreshed = prediction.ingest(packet(2.0, "drone", [0, 0, 0]), now=2.0)["state"]
    new_bank = prediction.banks["track-0001"]
    assert new_bank.support == prior
    assert old_bank.last_update["applied"] is False
    assert old_bank.last_update["reason"] == "Old forecast expired; reset behavior support to disclosed priors"
    assert refreshed["tracks"][0]["map_belief"]["last_evidence_update"] == old_bank.last_update


def test_proposal_measurement_uses_ci_and_canonical_fusion_key_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a real tracker fusion first so this test covers the measurement
    # block, then isolate BranchingBank's generator to inspect only its seed
    # input.  Full trajectory order invariance is intentionally out of scope.
    prediction = engine(model="route_bank")
    prediction.ingest(packet(0, "drone", [0, 0, 0]), now=0)
    same_pose_player = camera("player", position=[0, -6, 6], target=[0, 1, 0])
    prediction.ingest(
        packet(0, "player", [0.04, 0, 0], camera_override=same_pose_player), now=0
    )
    source_track = prediction.tracker.tracks["track-0001"]
    from backend.prediction.branching import BranchingBank, _proposal_measurement
    import backend.prediction.branching as branching_module

    proposal, source_keys = _proposal_measurement(source_track)
    assert source_keys == sorted(source_track._fusion_keys)
    np.testing.assert_allclose(proposal.xyz[:2], source_track._fusion_mean)
    np.testing.assert_allclose(proposal.covariance_xy, source_track._fusion_covariance)
    assert not np.allclose(proposal.xyz[:2], source_track.last_observation.xyz[:2])

    draws: list[tuple[str, float, np.ndarray, np.ndarray]] = []

    def fake_generate(geometry, config, proposal_track, intent, rng):
        draws.append((
            intent.id,
            float(rng.random()),
            np.asarray(proposal_track.last_observation.xyz, dtype=float).copy(),
            np.asarray(proposal_track.last_observation.covariance_xy, dtype=float).copy(),
        ))
        return []

    monkeypatch.setattr(branching_module, "generate_trajectories", fake_generate)
    track_a = copy.deepcopy(source_track)
    track_b = copy.deepcopy(source_track)
    track_a._fusion_keys = [("z-camera", "frame-z"), ("a-camera", "frame-a")]
    track_b._fusion_keys = [("a-camera", "frame-a"), ("z-camera", "frame-z")]
    BranchingBank(prediction.geometry, prediction.config, track_a)
    draws_a = draws.copy()
    draws.clear()
    BranchingBank(prediction.geometry, prediction.config, track_b)
    draws_b = draws.copy()

    assert [item[0] for item in draws_a] == [item[0] for item in draws_b]
    for first, second in zip(draws_a, draws_b):
        assert first[1] == second[1]
        np.testing.assert_allclose(first[2][:2], source_track._fusion_mean)
        np.testing.assert_allclose(first[3], source_track._fusion_covariance)
        np.testing.assert_allclose(first[2], second[2])
        np.testing.assert_allclose(first[3], second[3])


def test_same_time_camera_fusion_stays_conservative_and_finite() -> None:
    prediction = engine()
    first_state = prediction.ingest(packet(0, "drone", [0, 0, 0]), now=0)["state"]
    before_support = support(first_state)
    before_covariance = prediction.tracker.tracks["track-0001"].covariance[:2, :2].copy()
    same_pose_player = camera("player", position=[0, -6, 6], target=[0, 1, 0])
    after = prediction.ingest(
        packet(0, "player", [0, 0, 0], camera_override=same_pose_player), now=0
    )["state"]
    after_covariance = prediction.tracker.tracks["track-0001"].covariance[:2, :2]
    assert np.all(np.isfinite(after_covariance))
    assert np.trace(after_covariance) >= 0.45 * np.trace(before_covariance)
    # The second view has the same capture time and does not add independent
    # motion/intent evidence; CI may refine position, but support must not be
    # counted twice.
    assert support(after) == before_support
    values = np.asarray(list(support(after).values()), dtype=float)
    assert np.all(np.isfinite(values))
    assert np.all(values > 0)
    assert float(values.max() - values.min()) < 0.98
    json.dumps(after, allow_nan=False)


def test_config_rejects_unknown_truth_input() -> None:
    with pytest.raises(ValidationError):
        EngineConfig(
            prediction_model="branching",
            scripted_truth_path=["exit_a"],
        )
