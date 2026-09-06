from dataclasses import asdict
import json

import numpy as np
import pytest

from backend.tactical import (
    EvidenceState,
    LifecycleState,
    Observation,
    SensorSpec,
    TacticalMap,
    TacticalTracker,
    TrackerConfig,
    load_map,
)
from backend.tactical.kalman import predict, update
from backend.tactical.map import AABB


SENSOR = SensorSpec("fixed", (0.0, 1.6, -10.0), max_range=100.0)


def empty_map(*obstacles):
    return TacticalMap(
        name="test",
        coordinate_system="right-handed:x-east,y-up,z-north",
        obstacles=tuple(obstacles),
        nav_nodes=(),
        nav_edges=(),
        zones=(),
        intents=(),
    )


def observation(t, x, z=0.0, conf=1.0, sequence=None):
    return Observation(
        t=t,
        sensor_id=SENSOR.sensor_id,
        xyz=(x, 1.0, z),
        conf=conf,
        sequence=int(t * 10) if sequence is None else sequence,
    )


def test_stationary_track_converges_and_serializes_cleanly():
    tracker = TacticalTracker(
        empty_map(),
        [SENSOR],
        TrackerConfig(confirmation_hits=3, acceleration_variance=0.2),
    )
    noise = [0.3, -0.2, 0.1, -0.1, 0.05, -0.04, 0.02, -0.01]
    for index, error in enumerate(noise):
        snapshots = tracker.step(float(index), [observation(index, 2.0 + error)])
    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot.lifecycle is LifecycleState.CONFIRMED
    assert snapshot.evidence is EvidenceState.OBSERVED
    assert snapshot.xyz[0] == pytest.approx(2.0, abs=0.12)
    assert abs(snapshot.velocity_xz[0]) < 0.15
    json.dumps(asdict(snapshot))


def test_constant_velocity_is_recovered():
    tracker = TacticalTracker(
        empty_map(),
        [SENSOR],
        TrackerConfig(
            confirmation_hits=2,
            acceleration_variance=0.05,
            base_measurement_variance=0.04,
        ),
    )
    for index in range(8):
        snapshots = tracker.step(
            float(index), [observation(index, -2.0 + 1.25 * index)]
        )
    assert len(snapshots) == 1
    assert snapshots[0].velocity_xz[0] == pytest.approx(1.25, abs=0.08)
    assert snapshots[0].xyz[0] == pytest.approx(6.75, abs=0.08)


def test_low_confidence_produces_a_weaker_update():
    state = np.zeros(4, dtype=np.float64)
    covariance = np.eye(4, dtype=np.float64)
    strong = observation(0, 4.0, conf=1.0)
    weak = observation(0, 4.0, conf=0.1)
    strong_state, _ = update(state, covariance, strong, 0.25)
    weak_state, _ = update(state, covariance, weak, 0.25)
    assert 0.0 < weak_state[0] < strong_state[0] < 4.0


def test_suspect_jump_does_not_hijack_confirmed_track():
    tracker = TacticalTracker(
        empty_map(),
        [SENSOR],
        TrackerConfig(confirmation_hits=2, base_measurement_variance=0.04),
    )
    tracker.step(0.0, [observation(0, 0.0)])
    tracker.step(1.0, [observation(1, 0.1)])
    snapshots = tracker.step(2.0, [observation(2, 20.0)])
    assert [item.track_id for item in snapshots] == [1, 2]
    assert snapshots[0].evidence is EvidenceState.STALE
    assert snapshots[0].lifecycle is LifecycleState.CONFIRMED
    assert snapshots[1].lifecycle is LifecycleState.TENTATIVE


def test_occluded_dropout_survives_while_visible_miss_expires():
    cover = AABB("cover", (-1.0, 0.0, -5.0), (1.0, 3.0, -4.0))
    config = TrackerConfig(
        confirmation_hits=1,
        visible_miss_limit=2,
        max_coast_seconds=100.0,
    )
    occluded = TacticalTracker(empty_map(cover), [SENSOR], config)
    visible = TacticalTracker(empty_map(), [SENSOR], config)
    first = observation(0, 0.0, z=0.0)
    occluded.step(0.0, [first])
    visible.step(0.0, [first])
    for t in (1.0, 2.0):
        hidden_snapshots = occluded.step(t)
        visible_snapshots = visible.step(t)
    assert len(hidden_snapshots) == 1
    assert hidden_snapshots[0].misses == 2
    assert hidden_snapshots[0].expected_visible_misses == 0
    assert visible_snapshots == ()


def test_equal_tracker_timestamps_are_rejected_without_aging_track():
    tracker = TacticalTracker(
        empty_map(),
        [SENSOR],
        TrackerConfig(
            confirmation_hits=1,
            visible_miss_limit=2,
            max_coast_seconds=100.0,
        ),
    )
    initial = tracker.step(0.0, [observation(0.0, 0.0)])
    with pytest.raises(ValueError, match="strictly increase"):
        tracker.step(0.0)
    assert tracker.snapshots() == initial
    assert tracker.snapshots()[0].misses == 0


def test_crossing_ambiguity_sets_conflicting_evidence():
    tracker = TacticalTracker(
        empty_map(),
        [SENSOR],
        TrackerConfig(confirmation_hits=1, ambiguity_margin=0.5),
    )
    tracker.step(
        0.0,
        [observation(0, -1.0, sequence=1), observation(0, 1.0, sequence=2)],
    )
    snapshots = tracker.step(
        1.0,
        [observation(1, 0.0, sequence=3), observation(1, 0.0, sequence=4)],
    )
    assert len(snapshots) == 2
    assert all(item.evidence is EvidenceState.CONFLICTING for item in snapshots)
    assert all(item.misses == 0 for item in snapshots)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"xyz": (float("nan"), 0.0, 0.0)},
        {"conf": -0.01},
        {"conf": 1.01},
        {"sensor_id": ""},
        {"covariance": ((1.0, 2.0), (2.0, 1.0))},
        {"covariance": ((1.0, 2.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))},
    ],
)
def test_malformed_observations_are_rejected(kwargs):
    values = dict(t=0.0, sensor_id="fixed", xyz=(0.0, 0.0, 0.0), conf=1.0, sequence=0)
    values.update(kwargs)
    with pytest.raises(ValueError):
        Observation(**values)


def test_covariance_grows_during_coast_and_recovers_after_update():
    state = np.zeros(4, dtype=np.float64)
    covariance = np.eye(4, dtype=np.float64) * 0.2
    predicted_state, predicted_covariance = predict(state, covariance, 2.0, 1.0)
    assert np.trace(predicted_covariance) > np.trace(covariance)
    _, recovered_covariance = update(
        predicted_state, predicted_covariance, observation(2, 0.1), 0.1
    )
    assert np.trace(recovered_covariance[:2, :2]) < np.trace(
        predicted_covariance[:2, :2]
    )
    assert np.linalg.eigvalsh(recovered_covariance).min() > 0.0


def test_checked_in_map_validates_and_occludes_deterministically(tmp_path):
    tactical_map = load_map()
    assert tactical_map.name.startswith("Fictional A-Site-Inspired")
    assert tactical_map.coordinate_system == "right-handed:x-east,y-up,z-north"
    assert tactical_map.is_occluded((-4.0, 1.0, 0.0), (4.0, 1.0, 0.0))
    assert tactical_map.ray_occluded((-4.0, 1.0, 0.0), (1.0, 0.0, 0.0), 8.0)
    assert not tactical_map.is_occluded((-4.0, 1.0, -8.0), (4.0, 1.0, -8.0))

    malformed = tmp_path / "bad-map.json"
    malformed.write_text(
        json.dumps(
            {
                "name": "bad",
                "coordinate_system": "left-handed",
                "obstacles": [],
                "nav_nodes": [],
                "nav_edges": [],
                "zones": [],
                "intents": [],
            }
        )
    )
    with pytest.raises(ValueError, match="coordinate_system"):
        load_map(malformed)

    malformed.write_text(
        json.dumps(
            {
                "name": "bad-openness",
                "coordinate_system": "right-handed:x-east,y-up,z-north",
                "obstacles": [],
                "nav_nodes": [],
                "nav_edges": [],
                "zones": [
                    {
                        "id": "zone",
                        "min": [0.0, 0.0, 0.0],
                        "max": [1.0, 1.0, 1.0],
                        "openness": 1.1,
                    }
                ],
                "intents": [],
            }
        )
    )
    with pytest.raises(ValueError, match="openness"):
        load_map(malformed)
