import json
import unittest
import numpy as np
from backend.prediction.schema import EngineConfig, Observation
from backend.prediction.tracking import Tracker

def observation(
    x: float,
    y: float,
    t: float,
    sensor: str = "cam-a",
    frame: str | None = None,
    surface: str = "floor-1",
    variance: float = 0.04,
    confidence: float = 0.95,
    z: float = 0.0,
) -> Observation:
    return Observation(
        sensor_id=sensor,
        frame_id=frame or f"{sensor}-{t}",
        t=t,
        xyz=(x, y, z),
        covariance_xy=[[variance, 0.0], [0.0, variance]],
        confidence=confidence,
        surface_id=surface,
    )


class TrackerTests(unittest.TestCase):
    def config(self, **overrides: object) -> EngineConfig:
        values: dict[str, object] = {
            "stale_after_s": 0.5,
            "expire_after_s": 2.0,
            "association_gate": 9.21,
            "hysteresis_margin": 0.05,
            "max_tracks": 8,
        }
        values.update(overrides)
        return EngineConfig(**values)

    def test_covariance_grows_while_occluded_and_becomes_stale(self) -> None:
        tracker = Tracker(self.config(acceleration_std_mps2=1.5))
        tracker.ingest([observation(0, 0, 0)], 0, "cam-a")
        initial = tracker.tracks["track-0001"].covariance.copy()
        tracker.advance(1.0)
        track = tracker.tracks["track-0001"]
        self.assertGreater(track.covariance[0, 0], initial[0, 0])
        self.assertGreater(track.covariance[2, 2], initial[2, 2])
        self.assertEqual(track.status, "stale")

    def test_identity_reacquires_after_short_occlusion(self) -> None:
        tracker = Tracker(self.config())
        tracker.ingest([observation(2, 3, 0)], 0, "cam-a")
        tracker.advance(0.9)
        result = tracker.ingest([observation(2.05, 3.0, 0.9, frame="reacquire")], 0.9, "cam-a")
        self.assertEqual(result["matched"], ["track-0001"])
        self.assertEqual(list(tracker.tracks), ["track-0001"])
        self.assertEqual(tracker.tracks["track-0001"].status, "observed")

    def test_same_time_cross_view_uses_conservative_ci(self) -> None:
        tracker = Tracker(self.config())
        tracker.ingest([observation(0, 0, 0, "cam-a", "a0", variance=0.04)], 0, "cam-a")
        first = tracker.tracks["track-0001"].covariance[0, 0]
        tracker.ingest([observation(0.04, 0, 0, "cam-b", "b0", variance=0.04)], 0, "cam-b")
        second = tracker.tracks["track-0001"].covariance[0, 0]
        # Sequential independent updates would drive this toward one third of
        # the initial state covariance.  CI keeps it around one half instead.
        self.assertGreater(second, first * 0.45)
        self.assertEqual(len(tracker.tracks), 1)

    def test_association_is_one_to_one_per_frame(self) -> None:
        tracker = Tracker(self.config())
        tracker.ingest(
            [observation(-2, 0, 0, "cam-a", "a0"), observation(2, 0, 0, "cam-a", "a1")],
            0,
            "cam-a",
        )
        result = tracker.ingest(
            [observation(-1.9, 0, 1, "cam-a", "a2"), observation(1.9, 0, 1, "cam-a", "a3")],
            1,
            "cam-a",
        )
        self.assertEqual(set(result["matched"]), {"track-0001", "track-0002"})
        self.assertEqual(len(result["matched"]), 2)
        self.assertEqual(len(tracker.tracks), 2)

    def test_equal_association_is_reported_as_conflict(self) -> None:
        tracker = Tracker(self.config())
        tracker.ingest(
            [observation(-0.5, 0, 0, "cam-a", "a0"), observation(0.5, 0, 0, "cam-a", "a1")],
            0,
            "cam-a",
        )
        result = tracker.ingest([observation(0, 0, 1, "cam-b", "b1")], 1, "cam-b")
        self.assertEqual(result["matched"], [])
        self.assertEqual(result["created"], [])
        self.assertEqual(set(result["conflicts"]), {"track-0001", "track-0002"})
        self.assertEqual(
            {tracker.tracks[track_id].status for track_id in result["conflicts"]},
            {"conflicting"},
        )
        tracker.advance(1.2)
        self.assertEqual(tracker.tracks["track-0001"].status, "stale")
        tracker.ingest([observation(-0.5, 0, 1.3, "cam-b", "clear")], 1.3, "cam-b")
        self.assertEqual(tracker.tracks["track-0001"].status, "observed")

    def test_surfaces_are_separated_unless_physically_close(self) -> None:
        tracker = Tracker(self.config())
        tracker.ingest([observation(0, 0, 0, surface="lower")], 0, "cam-a")
        result = tracker.ingest([observation(0, 0, 1, surface="upper", z=3.0)], 1, "cam-a")
        self.assertEqual(result["matched"], [])
        self.assertEqual(result["created"], ["track-0002"])

    def test_expiry_removes_track(self) -> None:
        tracker = Tracker(self.config(expire_after_s=1.2))
        tracker.ingest([observation(0, 0, 0)], 0, "cam-a")
        tracker.advance(1.21)
        self.assertNotIn("track-0001", tracker.tracks)

    def test_out_of_order_and_invalid_covariance_are_rejected(self) -> None:
        tracker = Tracker(self.config())
        tracker.ingest([observation(0, 0, 1)], 1, "cam-a")
        with self.assertRaises(ValueError):
            tracker.ingest([observation(0, 0, 0, frame="old")], 0, "cam-a")
        bad = observation(0, 0, 2)
        bad = bad.model_copy(update={"covariance_xy": [[1.0, 2.0], [2.0, 1.0]]})
        with self.assertRaises(ValueError):
            tracker.ingest([bad], 2, "cam-a")
        with self.assertRaises(ValueError):
            tracker.advance(float("nan"))
        with self.assertRaises(ValueError):
            tracker.ingest([observation(0, 0, 2, "other", "wrong-sensor")], 2, "cam-a")
        with self.assertRaises(ValueError):
            tracker.ingest([observation(0, 0, 1.5, "cam-a", "wrong-time")], 2, "cam-a")

    def test_snapshot_is_json_serializable(self) -> None:
        tracker = Tracker(self.config())
        tracker.ingest([observation(1.0, 2.0, 0)], 0, "cam-a")
        payload = tracker.tracks["track-0001"].snapshot()
        json.dumps(payload)
        self.assertEqual(payload["xyz"], [1.0, 2.0, 0.0])
        self.assertEqual(len(payload["covariance_xy"]), 2)

    def test_max_track_budget_is_bounded(self) -> None:
        tracker = Tracker(self.config(max_tracks=2))
        tracker.ingest(
            [observation(-10, 0, 0, "cam-a", "a0"), observation(0, 0, 0, "cam-a", "a1"), observation(10, 0, 0, "cam-a", "a2")],
            0,
            "cam-a",
        )
        self.assertEqual(len(tracker.tracks), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
