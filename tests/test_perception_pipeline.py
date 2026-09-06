"""Orchestration tests use mock detections; real accuracy is a separate evaluation."""

import json
import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError
from backend.prediction.fixture import camera, packet, scene, config
from backend.prediction.perception import apply_ground_policy, read_image, run_take
from backend.prediction.perception_schema import RecordedTake


class StubDetector:
    def __init__(self, boxes, diagnostics):
        self.boxes, self.diagnostics = boxes, diagnostics
    def detect(self, image):
        return {"width": image.width, "height": image.height, "detections": self.boxes,
                "diagnostics": self.diagnostics, "model_name": "unit-test-stub", "model_fingerprint": "a"*64,
                "inference_ms": 0.}


def take_data():
    return {"id": "unit-test-only", "clock_id": "fixture-clock", "provenance": "Unit-only mock image and detector; no accuracy evidence.",
            "timing_provenance": "Unit test source timestamps, no real capture.", "clock_uncertainty_s": 0,
            "frames": [{"frame_id": "1", "sensor_id": "drone", "image": "frame.png", "t": 0}]}


@pytest.mark.parametrize("prediction_model", ["route_bank", "branching"])
def test_real_image_without_calibration_retains_detections_but_no_meters(tmp_path, prediction_model):
    Image.new("RGB", (100, 100)).save(tmp_path / "frame.png")
    data = take_data()
    data["config"] = {"prediction_model": prediction_model}
    take = RecordedTake.model_validate(data)
    detector = StubDetector([{"box": [10, 10, 30, 70], "confidence": .9}], [])
    summary = run_take(take, root=tmp_path, detector=detector, output=tmp_path / "out")
    assert summary["detections"] == 1
    assert summary["projected_observations"] == 0
    assert summary["prediction_model"] == prediction_model
    assert set(summary["blocking_reasons"]) == {"metric_map_missing", "camera_calibration_missing"}
    row = json.loads((tmp_path / "out/frames.jsonl").read_text())
    assert row["state"] is None
    assert len(row["image_sha256"]) == 64
    assert not row["detections"][0]["ground_contact_visible"]


def test_unknown_foot_and_invalid_pose_do_not_authorize_projection():
    boxes = [{"box": [10, 10, 40, 80], "confidence": .9, "ground_contact_visible": True}]
    assert not apply_ground_policy(boxes, [], "reject_unknown", 100, 100)[0].ground_contact_visible
    assert not apply_ground_policy(boxes, [], "pose_ankles_provisional_v1", 100, 100)[0].ground_contact_visible
    kp = np.full((17, 2), [25., 40.])
    kp[[15, 16]] = [[18, 75], [33, 77]]
    diagnostic = {"accepted": True, "output_detection_index": 0, "raw_keypoints": kp.tolist(), "raw_keypoint_confidence": [1.]*17}
    assert apply_ground_policy(boxes, [{"accepted": False}, diagnostic], "pose_ankles_provisional_v1", 100, 100)[0].ground_contact_visible
    diagnostic["raw_keypoint_confidence"][15] = .2
    assert not apply_ground_policy(boxes, [diagnostic], "pose_ankles_provisional_v1", 100, 100)[0].ground_contact_visible


@pytest.mark.parametrize("prediction_model", ["route_bank", "branching"])
def test_mock_calibrated_pixels_reach_prediction_and_future_calibration_does_not(tmp_path, prediction_model):
    cam = camera("drone")
    Image.new("RGB", (cam.width, cam.height)).save(tmp_path / "frame.png")
    box = packet(0, "drone", [0, 0, 0]).detections[0].model_dump()
    x1,y1,x2,y2 = box["box"]
    kp = np.full((17, 2), [(x1+x2)/2, (y1+y2)/2])
    kp[[15,16]] = [[x1+3,y2-2], [x2-3,y2-2]]
    diagnostic = {"accepted": True, "output_detection_index": 0, "raw_keypoints": kp.tolist(), "raw_keypoint_confidence": [1.]*17}
    detector = StubDetector([box], [diagnostic])
    data = take_data()
    data.update(prior=scene().model_copy(update={"synthetic": False, "provenance": "Mock unit-test map only; never used as real accuracy evidence."}).model_dump(),
                ground_contact_policy="pose_ankles_provisional_v1", config=config().model_dump(),
                geometry_verified=True, geometry_validation_provenance="Mock unit-test geometry verification, never a real validation certificate.")
    data["frames"][0]["camera"] = cam.model_dump()
    data["config"]["prediction_model"] = prediction_model
    result = run_take(RecordedTake.model_validate(data), root=tmp_path, detector=detector, output=tmp_path / "out")
    assert result["projected_observations"] == 1
    row = json.loads((tmp_path / "out/frames.jsonl").read_text())
    assert len(row["state"]["tracks"]) == 1
    assert row["state"]["prediction_model"] == prediction_model
    if prediction_model == "branching":
        assert row["state"]["tracks"][0]["map_belief"]["model"] == "branching_graph_trajectories_v1"
    assert row["prediction_status"] == "processed_provisional"
    data["frames"][0]["camera"]["available_t"] = 2
    result = run_take(RecordedTake.model_validate(data), root=tmp_path, detector=detector, output=tmp_path / "later")
    assert result["frames_processed_by_predictor"] == 0
    assert "calibration_not_available_at_capture" in result["blocking_reasons"]
    data["frames"][0]["camera"]["available_t"] = 0
    data["geometry_verified"] = False
    result = run_take(RecordedTake.model_validate(data), root=tmp_path, detector=detector, output=tmp_path / "unverified")
    assert result["frames_processed_by_predictor"] == 0
    assert "geometry_accuracy_unverified" in result["blocking_reasons"]
    data["geometry_verified"] = True
    data["frames"][0].update(observation_usable=False, unusable_reason="Excluded transit with camera artifacts.")
    result = run_take(RecordedTake.model_validate(data), root=tmp_path, detector=detector, output=tmp_path / "excluded")
    assert result["frames_processed_by_predictor"] == 0
    assert "frame_excluded_from_observation_evidence" in result["blocking_reasons"]


def test_label_leakage_duplicate_ids_and_file_escape_rejected(tmp_path):
    data = take_data()
    data["enemy_xyz"] = [1, 2, 3]
    with pytest.raises(ValidationError):
        RecordedTake.model_validate(data)
    data = take_data()
    data["frames"] *= 2
    with pytest.raises(ValidationError):
        RecordedTake.model_validate(data)
    data = take_data()
    data["frames"][0]["image"] = "../secret.png"
    with pytest.raises(ValidationError):
        RecordedTake.model_validate(data)
    with pytest.raises(ValueError):
        read_image(tmp_path, "../secret.png")
