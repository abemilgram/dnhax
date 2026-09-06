from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from backend.tactical import (
    CameraIntrinsics,
    CameraPose,
    EvidenceState,
    FixedAerialProjector,
    FixedPoseProvider,
    HomographyCalibration,
    PixelDetection,
    RayGroundCalibration,
    RayGroundProjector,
    TimestampedFrame,
    load_tape,
)
from backend.tactical.service import DEFAULT_TAPE_PATH, TacticalService


class Clock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value


class MemoryPersistence:
    def __init__(self):
        self.sessions = []
        self.cues = []

    def save_session(self, *values):
        self.sessions.append(values)

    def save_cue(self, session_id, revision, cue):
        self.cues.append((session_id, revision, deepcopy(cue)))


def test_golden_tape_validates_and_encodes_required_belief_beats():
    raw_tape = json.loads(Path(DEFAULT_TAPE_PATH).read_text())
    assert all(event["type"] in {"observation", "actor", "feed"} for event in raw_tape["events"])
    assert "ranking" not in json.dumps(raw_tape).lower()
    assert "cue" not in json.dumps(raw_tape).lower()
    tape = load_tape(DEFAULT_TAPE_PATH)
    assert tape.duration < 120.0
    assert tape.final_tick == 60
    assert tape.index.at_tick(0)

    service = TacticalService(tape, clock=lambda: 0.0)
    contact = service.seek(0.5)
    cover_entry = service.seek(1.0)
    covered = service.seek(3.0)
    repeek = service.seek(4.5)
    unchanged_end = service.seek(6.0)

    assert contact["tracks"][0]["evidence"] == EvidenceState.OBSERVED
    assert cover_entry["tracks"] == []
    assert cover_entry["ghosts"][0]["evidence"] == EvidenceState.STALE
    assert _order(contact) != _order(cover_entry)
    assert repeek["tracks"][0]["evidence"] == EvidenceState.OBSERVED
    assert _position_trace(repeek["tracks"][0]) < _position_trace(covered["ghosts"][0])
    assert _order(repeek)[0] != _order(covered)[0]
    assert len(unchanged_end["cue_history"]) == len(repeek["cue_history"])


def test_replay_seek_is_deterministic_and_revisions_are_monotonic():
    clock = Clock()
    service = TacticalService.from_path(clock=clock)
    revisions = [service.current()["revision"]]
    service.start()
    revisions.append(service.current()["revision"])
    clock.value += 1.26
    first = service.current()
    revisions.append(first["revision"])
    paused = service.pause()
    revisions.append(paused["revision"])
    clock.value += 3.0
    assert service.current()["replay"]["position"] == paused["replay"]["position"]
    assert revisions == sorted(revisions)
    assert len(set(revisions)) == len(revisions)

    at_three = service.seek(3.0)
    service.seek(6.0)
    rebuilt = service.seek(3.0)
    for key in ("actor", "feeds", "tracks", "ghosts", "ranking"):
        assert rebuilt[key] == at_three[key]
    assert [_cue_semantics(item) for item in rebuilt["cue_history"]] == [
        _cue_semantics(item) for item in at_three["cue_history"]
    ]

    restarted = service.restart()
    assert restarted["replay"]["position"] == 0.0
    assert restarted["replay"]["playing"] is True
    assert restarted["revision"] > rebuilt["revision"]


def test_rebuild_does_not_duplicate_persisted_cues():
    persistence = MemoryPersistence()
    service = TacticalService.from_path(
        clock=lambda: 0.0,
        persistence=persistence,
        session_id="test-session",
    )
    service.advance_to(6.0)
    sequences = [entry[2]["sequence"] for entry in persistence.cues]
    assert len(sequences) == len(set(sequences))
    first_count = len(persistence.cues)
    service.restart()
    service.advance_to(6.0)
    assert len(persistence.cues) == first_count


def test_malformed_tapes_and_replay_input_are_rejected(tmp_path):
    data = json.loads(Path(DEFAULT_TAPE_PATH).read_text())
    malformed = tmp_path / "malformed.json"
    data["duration"] = 120.0
    malformed.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="under 120"):
        load_tape(malformed)

    data = json.loads(Path(DEFAULT_TAPE_PATH).read_text())
    data["events"][0]["surprise"] = True
    malformed.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="unknown field"):
        load_tape(malformed)

    service = TacticalService.from_path(clock=lambda: 0.0)
    with pytest.raises(ValueError, match="inside"):
        service.seek(6.1)
    with pytest.raises(ValueError, match="10 Hz"):
        service.seek(1.25)


def test_homography_projects_bottom_center_and_propagates_uncertainty():
    frame = TimestampedFrame("aerial", 1.5, 7, 200, 200, b"pixels")
    detection = PixelDetection(
        90,
        80,
        110,
        120,
        0.8,
        covariance_px=((4.0, 0.0), (0.0, 9.0)),
    )
    projector = FixedAerialProjector(
        HomographyCalibration(
            image_to_ground_xz=((0.1, 0.0, -10.0), (0.0, 0.1, -10.0), (0.0, 0.0, 1.0)),
            ground_bounds=((-2.0, 2.0), (-2.0, 4.0)),
            calibration_variance=0.01,
        )
    )
    observation = projector.project(frame, detection)
    assert observation.xyz == pytest.approx((0.0, 0.0, 2.0))
    assert observation.t == frame.t
    assert observation.sequence == frame.sequence
    assert observation.conf == detection.confidence
    assert np.asarray(observation.covariance)[0, 0] > 0.0


def test_ray_projection_accuracy_and_invalid_calibration_rejection():
    frame = TimestampedFrame("moving-future", 2.0, 3, 200, 200, b"pixels")
    detection = PixelDetection(90, 80, 110, 100, 1.0)
    downward = CameraPose(
        rotation=((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        xyz=(0.0, 10.0, 0.0),
    )
    calibration = RayGroundCalibration(
        CameraIntrinsics(100.0, 100.0, 100.0, 100.0),
        ground_bounds=((-1.0, 1.0), (-1.0, 1.0)),
    )
    observation = RayGroundProjector(
        calibration, FixedPoseProvider(downward)
    ).project(frame, detection)
    assert observation.xyz == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)

    parallel = CameraPose(
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        xyz=(0.0, 10.0, 0.0),
    )
    with pytest.raises(ValueError, match="does not intersect"):
        RayGroundProjector(
            calibration, FixedPoseProvider(parallel)
        ).project(frame, detection)
    with pytest.raises(ValueError, match="invertible"):
        HomographyCalibration(
            image_to_ground_xz=((1.0, 0.0, 0.0),) * 3,
        )
    with pytest.raises(ValueError, match="outside"):
        FixedAerialProjector(
            HomographyCalibration(
                image_to_ground_xz=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                ground_bounds=((-1.0, 1.0), (-1.0, 1.0)),
            )
        ).project(frame, detection)


def test_snapshot_has_map_geometry_and_no_forbidden_output_fields():
    state = TacticalService.from_path(clock=lambda: 0.0).advance_to(6.0)
    assert {"obstacles", "nodes", "edges", "zones"} <= set(state["map"])
    forbidden = {"target", "aim", "fire", "reticle", "action", "control"}
    assert not (_all_keys(state) & forbidden)


def _order(state):
    return [item["intent"] for item in state["ranking"]["candidates"]]


def _position_trace(track):
    covariance = track["covariance"]
    return covariance[0][0] + covariance[1][1]


def _cue_semantics(cue):
    return {
        key: value
        for key, value in cue.items()
        if key not in {"revision", "t", "route"}
    }


def _all_keys(value):
    if isinstance(value, dict):
        return {str(key).lower() for key in value} | {
            nested
            for item in value.values()
            for nested in _all_keys(item)
        }
    if isinstance(value, list):
        return {nested for item in value for nested in _all_keys(item)}
    return set()
