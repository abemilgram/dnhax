"""Tests for tiled coordinate mapping, tile clipping, and deterministic NMS."""

from __future__ import annotations

import json
import unittest

from PIL import Image

from backend.prediction.tiled_detector import TiledPoseDetector, _tile_specs  # noqa: E402


def diagnostic(box: list[float], *, keypoint: tuple[float, float] = (10.0, 10.0), output_index: int = 0, contact_eligible: bool | None = None) -> dict:
    keypoints = [[keypoint[0], keypoint[1]] for _ in range(17)]
    output = {
        "detection_index": 0,
        "output_detection_index": output_index,
        "class_id": 0,
        "class_name": "person",
        "accepted": True,
        "reason": "accepted_pose_candidate",
        "reasons": ["ankle_support_available_for_review"],
        "raw_box": box,
        "raw_keypoints": keypoints,
        "raw_keypoint_confidence": [0.9 for _ in range(17)],
        "ankle_support": {
            "left_ankle": {"xy": [keypoint[0], keypoint[1]], "confidence": 0.9},
            "right_ankle": {"xy": [keypoint[0], keypoint[1]], "confidence": 0.9},
        },
    }
    if contact_eligible is not None:
        output["contact_eligible"] = contact_eligible
    return output


def result(detections: list[dict], diagnostics: list[dict] | None = None) -> dict:
    return {
        "width": 200,
        "height": 100,
        "detections": detections,
        "diagnostics": diagnostics if diagnostics is not None else [diagnostic(item["box"]) for item in detections],
        "model_name": "mock-pose",
        "model_fingerprint": "mock-sha256",
        "inference_ms": 1.5,
        "names": ["person"],
    }


class MockDetector:
    model_name = "mock-pose"
    model_fingerprint = "mock-sha256"

    def __init__(self, outputs: list[dict]):
        self.outputs = outputs
        self.images: list[Image.Image] = []

    def detect(self, image: Image.Image) -> dict:
        self.images.append(image)
        return dict(self.outputs[len(self.images) - 1], width=image.width, height=image.height)


class TiledDetectorTests(unittest.TestCase):
    def test_unknown_ankles_keep_person_box_but_veto_contact(self) -> None:
        unknown = diagnostic([5, 5, 20, 30], keypoint=(0, 0))
        outputs = [result([]), result([{"box": [5, 5, 20, 30], "confidence": .9}], [unknown]),
                   result([]), result([]), result([])]
        output = TiledPoseDetector(MockDetector(outputs)).detect(Image.new("RGB", (200, 100)))
        self.assertEqual(len(output["detections"]), 1)
        self.assertFalse(next(d for d in output["diagnostics"] if d.get("kept"))["contact_eligible"])

    def test_fixed_layout_is_full_plus_four_tiles(self) -> None:
        specs = _tile_specs(200, 100, 0.2)
        self.assertEqual([spec["source"] for spec in specs], ["full", "tile-00", "tile-01", "tile-02", "tile-03"])
        self.assertEqual(specs[0]["bounds"], (0, 0, 200, 100))
        self.assertEqual(specs[1]["bounds"], (0, 0, 120, 60))
        self.assertEqual(specs[2]["bounds"], (80, 0, 200, 60))
        self.assertEqual(specs[3]["bounds"], (0, 40, 120, 100))
        self.assertEqual(specs[4]["bounds"], (80, 40, 200, 100))

    def test_maps_tile_boxes_and_suppresses_duplicate_with_confidence_nms(self) -> None:
        outputs = [
            result([{"box": [10, 10, 30, 30], "confidence": 0.6, "label": "person", "ground_contact_visible": True}]),
            result([{"box": [10, 10, 30, 30], "confidence": 0.9, "label": "person", "ground_contact_visible": True}]),
            result([]),
            result([{"box": [5, 5, 20, 20], "confidence": 0.8, "label": "person", "ground_contact_visible": True}]),
            result([]),
        ]
        detector = MockDetector(outputs)
        tiled = TiledPoseDetector(detector)
        output = tiled.detect(Image.new("RGB", (200, 100)))
        self.assertEqual(len(detector.images), 5)
        self.assertEqual([item["box"] for item in output["detections"]], [[10.0, 10.0, 30.0, 30.0], [5.0, 45.0, 20.0, 60.0]])
        self.assertEqual([item["confidence"] for item in output["detections"]], [0.9, 0.8])
        self.assertTrue(any(item.get("reason") == "suppressed_by_nms" for item in output["diagnostics"]))
        self.assertTrue(all(item["ground_contact_visible"] is False for item in output["detections"]))
        accepted = [item for item in output["diagnostics"] if item.get("detector_accepted")]
        self.assertTrue(any(item.get("kept") is True for item in accepted))
        self.assertTrue(any(item.get("reason") == "suppressed_by_nms" for item in accepted))
        kept = [item for item in accepted if item.get("kept") is True]
        self.assertEqual(sorted(item["output_detection_index"] for item in kept), [0, 1])
        self.assertTrue(all(item["output_detection_index"] is None for item in accepted if item.get("kept") is not True))

    def test_output_index_is_explicit_and_contact_veto_is_preserved(self) -> None:
        detections = [
            {"box": [10, 10, 30, 30], "confidence": 0.8, "label": "person"},
            {"box": [130, 10, 150, 30], "confidence": 0.7, "label": "person"},
        ]
        diagnostics = [
            diagnostic(detections[1]["box"], output_index=1, contact_eligible=False),
            diagnostic(detections[0]["box"], output_index=0),
        ]
        outputs = [result(detections, diagnostics), result([]), result([]), result([]), result([])]
        output = TiledPoseDetector(MockDetector(outputs)).detect(Image.new("RGB", (200, 100)))
        self.assertEqual([item["box"] for item in output["detections"]], [detections[0]["box"], detections[1]["box"]])
        accepted = [item for item in output["diagnostics"] if item.get("kept") is True]
        self.assertEqual([item["output_detection_index"] for item in accepted], [0, 1])
        self.assertEqual([item["contact_eligible"] for item in accepted], [True, False])

    def test_unjoined_or_invalid_keypoint_metadata_is_rejected(self) -> None:
        invalid_keypoint = diagnostic([10, 10, 30, 30])
        invalid_keypoint["raw_keypoint_confidence"][3] = 1.5
        unjoined = diagnostic([40, 10, 60, 30])
        unjoined["output_detection_index"] = None
        outputs = [
            result(
                [
                    {"box": [10, 10, 30, 30], "confidence": 0.9, "label": "person"},
                    {"box": [40, 10, 60, 30], "confidence": 0.8, "label": "person"},
                ],
                [invalid_keypoint, unjoined],
            ),
            result([]), result([]), result([]), result([]),
        ]
        output = TiledPoseDetector(MockDetector(outputs)).detect(Image.new("RGB", (200, 100)))
        self.assertEqual(output["detections"], [])
        reasons = [item.get("reason") for item in output["diagnostics"]]
        self.assertIn("invalid_keypoint_confidence", reasons)
        self.assertIn("missing_output_detection_index", reasons)

    def test_tile_boundary_box_or_ankle_is_discarded(self) -> None:
        border_diag = diagnostic([5, 5, 20, 60], keypoint=(10, 60))
        outputs = [
            result([]),
            result([]),
            result([{"box": [5, 5, 20, 60], "confidence": 0.9, "label": "person", "ground_contact_visible": False}], [border_diag]),
            result([]),
            result([]),
        ]
        tiled = TiledPoseDetector(MockDetector(outputs))
        output = tiled.detect(Image.new("RGB", (200, 100)))
        self.assertEqual(output["detections"], [])
        discarded = [item for item in output["diagnostics"] if item.get("reason") == "tile_boundary_truncation"]
        self.assertEqual(len(discarded), 1)
        self.assertFalse(discarded[0]["contact_eligible"])
        self.assertIsNone(discarded[0]["output_detection_index"])
        self.assertEqual(discarded[0]["raw_box"], [85.0, 5.0, 100.0, 60.0])

    def test_mapped_bounds_are_checked_and_provenance_is_explicit(self) -> None:
        outputs = [
            result([{"box": [-2, 5, 20, 20], "confidence": 0.9, "label": "person", "ground_contact_visible": True}]),
            result([]),
            result([]),
            result([]),
            result([]),
        ]
        detector = MockDetector(outputs)
        output = TiledPoseDetector(detector).detect(Image.new("L", (200, 100)))
        self.assertEqual(output["detections"], [])
        self.assertTrue(any(item.get("reason") == "mapped_box_out_of_bounds" for item in output["diagnostics"]))
        self.assertEqual(output["provenance"]["layout"], "full_frame_plus_2x2")
        self.assertEqual(output["provenance"]["overlap"], 0.2)
        self.assertEqual(output["timings"]["base_calls"], 5)
        self.assertGreaterEqual(output["inference_ms"], 0.0)
        self.assertGreaterEqual(output["wall_ms"], 0.0)
        json.dumps(output, allow_nan=False)

    def test_constructor_and_maximum_are_bounded(self) -> None:
        with self.assertRaises(TypeError):
            TiledPoseDetector(None)
        outputs = [result([]) for _ in range(5)]
        detector = MockDetector(outputs)
        output = TiledPoseDetector(detector, max_detections=1).detect(Image.new("RGB", (200, 100)))
        self.assertEqual(len(output["detections"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
