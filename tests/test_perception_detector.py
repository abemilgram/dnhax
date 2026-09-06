import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from backend.prediction.schema import Detection
from backend.prediction.detector import PoseDetector, parse_pose_result

class Boxes:
    def __init__(self, boxes: object, confidence: object, classes: object):
        self.xyxy = boxes
        self.conf = confidence
        self.cls = classes


class Keypoints:
    def __init__(self, xy: object, confidence: object):
        self.xy = xy
        self.conf = confidence


class Results:
    def __init__(
        self,
        boxes: object,
        confidence: object,
        classes: object,
        keypoints: object,
        orig_shape: tuple[int, int] = (100, 200),
    ):
        self.boxes = Boxes(boxes, confidence, classes)
        self.keypoints = keypoints
        self.orig_shape = orig_shape
        self.names = {0: "person", 1: "car"}


def pose_result(
    *,
    box: object = [[20.0, 10.0, 80.0, 90.0]],
    confidence: object = [0.9],
    classes: object = [0],
    keypoints: object = "full",
    orig_shape: tuple[int, int] = (100, 200),
) -> Results:
    if keypoints == "full":
        xy = np.zeros((1, 17, 2), dtype=float)
        xy[0, 15] = [35.0, 88.0]
        xy[0, 16] = [65.0, 88.0]
        kp = Keypoints(xy, np.full((1, 17), 0.9, dtype=float))
    else:
        kp = keypoints
    return Results(box, confidence, classes, kp, orig_shape)


class DetectorTests(unittest.TestCase):
    def test_parser_emits_schema_detection_in_original_pixels(self) -> None:
        detections, diagnostics = parse_pose_result(pose_result(), 200, 100)
        self.assertEqual(len(detections), 1)
        Detection(**detections[0])
        self.assertEqual(detections[0]["box"], [20.0, 10.0, 80.0, 90.0])
        self.assertFalse(detections[0]["ground_contact_visible"])
        self.assertEqual(diagnostics[0]["raw_box"], [20.0, 10.0, 80.0, 90.0])
        self.assertEqual(diagnostics[0]["raw_keypoints"][15], [35.0, 88.0])
        self.assertTrue(diagnostics[0]["ankle_support"]["both_ankles_confident"])

    def test_missing_feet_remains_review_candidate_without_authorizing_contact(self) -> None:
        detections, diagnostics = parse_pose_result(
            pose_result(keypoints=None),
            200,
            100,
        )
        self.assertEqual(len(detections), 1)
        self.assertFalse(detections[0]["ground_contact_visible"])
        self.assertIn("keypoints_missing", diagnostics[0]["reasons"])
        self.assertFalse(diagnostics[0]["ankle_support"]["both_ankles_confident"])

    def test_edge_box_is_kept_for_review_and_flagged_truncated(self) -> None:
        detections, diagnostics = parse_pose_result(
            pose_result(box=[[0.0, 10.0, 80.0, 90.0]]),
            200,
            100,
        )
        self.assertEqual(len(detections), 1)
        self.assertIn("truncated_or_edge_box", diagnostics[0]["reasons"])

    def test_nan_box_is_rejected_and_diagnostics_are_json_safe(self) -> None:
        detections, diagnostics = parse_pose_result(
            pose_result(box=[[float("nan"), 10.0, 80.0, 90.0]]),
            200,
            100,
        )
        self.assertEqual(detections, [])
        self.assertIn("non_finite_box", diagnostics[0]["reasons"])
        json.dumps(diagnostics, allow_nan=False)

    def test_wrong_result_dimensions_reject_all_boxes(self) -> None:
        detections, diagnostics = parse_pose_result(
            pose_result(orig_shape=(101, 200)),
            200,
            100,
        )
        self.assertEqual(detections, [])
        self.assertIn("result_shape_mismatch", diagnostics[0]["reasons"])

    def test_empty_result_and_non_person_class_are_safe(self) -> None:
        empty = pose_result(box=np.empty((0, 4)), confidence=np.empty(0), classes=np.empty(0), keypoints=None)
        detections, diagnostics = parse_pose_result(empty, 200, 100)
        self.assertEqual(detections, [])
        self.assertEqual(diagnostics[0]["reason"], "no_detections")
        detections, diagnostics = parse_pose_result(pose_result(classes=[1]), 200, 100)
        self.assertEqual(detections, [])
        self.assertIn("non_person_class", diagnostics[0]["reasons"])

    def test_constructor_requires_explicit_local_weights_and_does_not_import_ultralytics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.pt"
            path.write_bytes(b"fixture-weights")
            detector = PoseDetector(path)
            self.assertIsNone(detector._model)
            self.assertEqual(detector.model_fingerprint, hashlib.sha256(b"fixture-weights").hexdigest())
        with self.assertRaises(FileNotFoundError):
            PoseDetector(Path("/private/tmp/no-such-pose-weights.pt"))

    def test_detect_uses_resident_model_pil_rgb_and_reports_provenance(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.calls: list[tuple[object, dict]] = []

            def __call__(self, image: Image.Image, **kwargs: object) -> list[Results]:
                self.calls.append((image, kwargs))
                return [pose_result()]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pose.pt"
            path.write_bytes(b"weights")
            detector = PoseDetector(path, device="cpu", imgsz=640)
            fake = FakeModel()
            detector._model = fake
            output = detector.detect(Image.new("L", (200, 100)))
            self.assertEqual(output["width"], 200)
            self.assertEqual(output["height"], 100)
            self.assertEqual(len(output["detections"]), 1)
            image, kwargs = fake.calls[0]
            self.assertEqual(image.mode, "RGB")
            self.assertEqual(kwargs, {"device": "cpu", "imgsz": 640, "conf": 0.25, "max_det": 64, "verbose": False})
            self.assertEqual(output["provenance"]["coordinate_system"], "original_image_pixels")
            self.assertFalse(output["provenance"]["ground_contact_visible_default"])
            self.assertIn("no enemy", output["provenance"]["class_semantics"])
            self.assertGreaterEqual(output["inference_ms"], 0.0)
            json.dumps(output, allow_nan=False)

    def test_lazy_ultralytics_loads_once_from_path_only(self) -> None:
        calls: list[str] = []

        class FakeYOLO:
            def __init__(self, path: str) -> None:
                calls.append(path)

            def __call__(self, image: Image.Image, **kwargs: object) -> list[Results]:
                return [pose_result()]

        fake_module = types.SimpleNamespace(YOLO=FakeYOLO)
        old_module = sys.modules.get("ultralytics")
        sys.modules["ultralytics"] = fake_module
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "custom-local.pt"
                path.write_bytes(b"weights")
                detector = PoseDetector(path)
                detector.detect(Image.new("RGB", (200, 100)))
                detector.detect(Image.new("RGB", (200, 100)))
            self.assertEqual(calls, [str(path.resolve())])
        finally:
            if old_module is None:
                sys.modules.pop("ultralytics", None)
            else:
                sys.modules["ultralytics"] = old_module


if __name__ == "__main__":
    unittest.main(verbosity=2)
