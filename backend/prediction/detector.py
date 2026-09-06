"""Optional Ultralytics pose candidates for the fictional CS2 pixel pipeline.

This module is deliberately a detector adapter, not a projection or tracking
layer.  It emits generic COCO ``person`` candidates in the original PIL image
coordinates.  A pose model cannot establish that a foot is visible against a
known floor, so every emitted Detection keeps ``ground_contact_visible`` set to
False.  A separately disclosed ground-contact policy is required before projection;
model confidence alone is not proof of visible feet.

Ultralytics is imported only when the detector is first used.  The constructor
requires an existing local weights file and never asks Ultralytics to resolve
or download a model by name.
"""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


COCO_PERSON_CLASS_ID = 0
COCO_KEYPOINT_COUNT = 17
LEFT_ANKLE_INDEX = 15
RIGHT_ANKLE_INDEX = 16
PROVENANCE_VERSION = "ultralytics-pose-original-pixels-v1"


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be finite")
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(output):
        raise ValueError(f"{name} must be finite")
    return output


def _validate_probability(value: Any, name: str) -> float:
    output = _finite_float(value, name)
    if not 0.0 <= output <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return output


def _validate_dimensions(width: Any, height: Any) -> tuple[int, int]:
    if isinstance(width, (bool, np.bool_)) or isinstance(height, (bool, np.bool_)):
        raise ValueError("image dimensions must be positive integers")
    try:
        width_f, height_f = float(width), float(height)
        width_i, height_i = int(width_f), int(height_f)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("image dimensions must be positive integers") from exc
    if width_i <= 0 or height_i <= 0 or width_i != width_f or height_i != height_f:
        raise ValueError("image dimensions must be positive integers")
    return width_i, height_i


def _array(value: Any) -> np.ndarray | None:
    """Convert NumPy, Torch, and list-like result values without importing Torch."""

    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)
    except (TypeError, ValueError, RuntimeError):
        return None


def _json_value(value: Any) -> Any:
    """Make raw model output JSON-safe, replacing non-finite numbers with null."""

    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.floating, float)):
        output = float(value)
        return output if math.isfinite(output) else None
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _one_box_array(value: np.ndarray | None) -> tuple[np.ndarray | None, str | None]:
    if value is None:
        return None, None
    if value.size == 0:
        if value.ndim == 2 and value.shape[1] == 4:
            return value.reshape(0, 4), None
        return np.empty((0, 4), dtype=float), None
    if value.ndim == 1 and value.size == 4:
        return value.reshape(1, 4), None
    if value.ndim != 2 or value.shape[1] != 4:
        return None, "invalid_box_shape"
    return value, None


def _flatten_metadata(value: np.ndarray | None, count: int) -> tuple[np.ndarray | None, str | None]:
    if value is None:
        return None, "missing_box_metadata"
    if value.size != count:
        return None, "box_metadata_length_mismatch"
    return value.reshape(-1), None


def _keypoint_arrays(result: Any, count: int) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    keypoints = getattr(result, "keypoints", None)
    if keypoints is None:
        return None, None, "keypoints_missing"
    xy = _array(getattr(keypoints, "xy", None))
    conf = _array(getattr(keypoints, "conf", None))
    if xy is None:
        return None, None, "keypoints_missing"
    if xy.ndim == 2 and xy.shape == (COCO_KEYPOINT_COUNT, 2) and count == 1:
        xy = xy.reshape(1, COCO_KEYPOINT_COUNT, 2)
    if xy.ndim != 3 or xy.shape[0] != count or xy.shape[1:] != (COCO_KEYPOINT_COUNT, 2):
        return None, None, "keypoints_invalid_shape"
    if conf is not None and conf.ndim == 3 and conf.shape[-1] == 1:
        conf = conf[..., 0]
    if conf is None or conf.ndim != 2 or conf.shape != (count, COCO_KEYPOINT_COUNT):
        return xy, None, "keypoint_confidence_missing_or_invalid"
    return xy, conf, None


def _result_shape(result: Any) -> tuple[int, int] | None:
    raw = getattr(result, "orig_shape", None)
    if raw is None:
        return None
    try:
        if len(raw) != 2:
            return None
        height, width = int(raw[0]), int(raw[1])
        if (height, width) != (float(raw[0]), float(raw[1])):
            return None
        return height, width
    except (TypeError, ValueError, OverflowError):
        return None


def _class_name(result: Any, class_id: int) -> str | None:
    names = getattr(result, "names", None)
    if isinstance(names, dict):
        value = names.get(class_id)
    elif isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        value = names[class_id]
    else:
        value = None
    return str(value) if value is not None else None


def _ankle_support(
    xy: np.ndarray | None,
    conf: np.ndarray | None,
    index: int,
    width: int,
    height: int,
    threshold: float,
) -> dict[str, Any]:
    if xy is None or index >= len(xy):
        return {"xy": None, "confidence": None, "usable_for_review": False}
    try:
        point = np.asarray(xy[index], dtype=float)
    except (TypeError, ValueError, OverflowError):
        return {"xy": _json_value(xy[index]), "confidence": None, "usable_for_review": False}
    confidence = None if conf is None or index >= len(conf) else _json_value(conf[index])
    finite = point.shape == (2,) and np.all(np.isfinite(point))
    in_frame = finite and 0.0 <= point[0] <= width and 0.0 <= point[1] <= height
    confident = isinstance(confidence, (int, float)) and math.isfinite(float(confidence)) and threshold <= float(confidence) <= 1
    return {
        "xy": _json_value(point),
        "confidence": confidence,
        "in_frame": bool(in_frame),
        "usable_for_review": bool(finite and in_frame and confident),
    }


def parse_pose_result(
    result: Any,
    width: int,
    height: int,
    confidence: float = 0.25,
    keypoint_confidence: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse one Ultralytics Results object in original image coordinates.

    The return value is ``(detections, diagnostics)``.  Detection dictionaries
    are directly compatible with the project's ``Detection`` model.  Every
    detection is a generic ``person`` candidate and explicitly leaves ground
    contact unapproved for a later reviewed annotation.
    """

    width_i, height_i = _validate_dimensions(width, height)
    confidence_f = _validate_probability(confidence, "confidence")
    keypoint_confidence_f = _validate_probability(keypoint_confidence, "keypoint_confidence")
    if result is None:
        return [], [{"detection_index": None, "accepted": False, "reason": "missing_result", "reasons": ["missing_result"], "raw_box": None, "raw_keypoints": []}]

    boxes = getattr(result, "boxes", None)
    raw_boxes = _array(getattr(boxes, "xyxy", None)) if boxes is not None else None
    box_array, box_error = _one_box_array(raw_boxes)
    raw_shape = _result_shape(result)
    if raw_shape is None:
        shape_reason = "missing_orig_shape"
    elif raw_shape != (height_i, width_i):
        shape_reason = "result_shape_mismatch"
    else:
        shape_reason = None

    if box_array is None:
        reason = "boxes_missing" if raw_boxes is None and box_error is None else (box_error or "invalid_box_shape")
        return [], [{
            "detection_index": None,
            "accepted": False,
            "reason": reason,
            "reasons": [reason],
            "raw_box": _json_value(raw_boxes),
            "raw_keypoints": [],
            "expected_orig_shape": [height_i, width_i],
            "actual_orig_shape": list(raw_shape) if raw_shape else None,
        }]

    count = len(box_array)
    scores, score_error = _flatten_metadata(_array(getattr(boxes, "conf", None)) if boxes is not None else None, count)
    classes, class_error = _flatten_metadata(_array(getattr(boxes, "cls", None)) if boxes is not None else None, count)
    keypoint_xy, keypoint_conf, keypoint_error = _keypoint_arrays(result, count)
    if count == 0:
        return [], [{
            "detection_index": None,
            "accepted": False,
            "reason": shape_reason or "no_detections",
            "reasons": [shape_reason] if shape_reason else ["no_detections"],
            "raw_box": [],
            "raw_boxes": [],
            "raw_keypoints": [],
            "expected_orig_shape": [height_i, width_i],
            "actual_orig_shape": list(raw_shape) if raw_shape else None,
        }]

    detections: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for index, raw_box in enumerate(box_array):
        reasons: list[str] = []
        if shape_reason:
            reasons.append(shape_reason)
        if box_error:
            reasons.append(box_error)
        try:
            box = np.asarray(raw_box, dtype=float)
        except (TypeError, ValueError, OverflowError):
            box = np.asarray(raw_box, dtype=object)
            reasons.append("non_numeric_box")
        try:
            box_finite = box.shape == (4,) and np.all(np.isfinite(box))
        except (TypeError, ValueError):
            box_finite = False
        if not box_finite:
            reasons.append("non_finite_box")
        else:
            left, top, right, bottom = (float(value) for value in box)
            if left >= right or top >= bottom:
                reasons.append("invalid_box_extent")
            elif left < 0 or top < 0 or right > width_i or bottom > height_i:
                reasons.append("box_out_of_frame")
            elif left <= 0 or top <= 0 or right >= width_i or bottom >= height_i:
                # Keep this candidate for review/FramePacket construction;
                # geometry will reject edge contact before ray casting.
                reasons.append("truncated_or_edge_box")

        score = None if scores is None else scores[index]
        try:
            score_finite = score is not None and np.isfinite(float(score))
        except (TypeError, ValueError, OverflowError):
            score_finite = False
        if not score_finite:
            reasons.append(score_error or "non_finite_confidence")
        elif float(score) < confidence_f:
            reasons.append("below_confidence")
        elif float(score) > 1.0 or float(score) < 0.0:
            reasons.append("invalid_confidence")

        class_id_value = None if classes is None else classes[index]
        try:
            class_id_finite = class_id_value is not None and np.isfinite(float(class_id_value))
        except (TypeError, ValueError, OverflowError):
            class_id_finite = False
        class_id: int | None = None
        if not class_id_finite:
            reasons.append(class_error or "non_finite_class")
        else:
            try:
                class_id = int(class_id_value)
            except (TypeError, ValueError, OverflowError):
                reasons.append("invalid_class")
            else:
                if class_id != float(class_id_value):
                    reasons.append("invalid_class")
                elif class_id != COCO_PERSON_CLASS_ID:
                    reasons.append("non_person_class")
                elif _class_name(result, class_id) != "person":
                    reasons.append("unverified_person_class_mapping")

        kp_xy_i = None if keypoint_xy is None else keypoint_xy[index]
        kp_conf_i = None if keypoint_conf is None else keypoint_conf[index]
        left_ankle = _ankle_support(kp_xy_i, kp_conf_i, LEFT_ANKLE_INDEX, width_i, height_i, keypoint_confidence_f)
        right_ankle = _ankle_support(kp_xy_i, kp_conf_i, RIGHT_ANKLE_INDEX, width_i, height_i, keypoint_confidence_f)
        if keypoint_error:
            reasons.append(keypoint_error)
        elif not (left_ankle["usable_for_review"] and right_ankle["usable_for_review"]):
            reasons.append("ankle_support_insufficient")
        else:
            reasons.append("ankle_support_available_for_review")

        raw_keypoints = _json_value(kp_xy_i) if kp_xy_i is not None else []
        raw_keypoint_confidence = _json_value(kp_conf_i) if kp_conf_i is not None else []
        # Pose/keypoint quality is review metadata.  A valid person box stays
        # available as a candidate even when ankles are missing or uncertain;
        # only explicit ground-contact review may authorize projection later.
        accepted_reasons = {
            "truncated_or_edge_box",
            "keypoints_missing",
            "keypoint_confidence_missing_or_invalid",
            "keypoints_invalid_shape",
            "ankle_support_insufficient",
            "ankle_support_available_for_review",
        }
        rejected_reasons = [reason for reason in reasons if reason not in accepted_reasons]
        accepted = not rejected_reasons
        diagnostic = {
            "detection_index": index,
            "output_detection_index": len(detections) if accepted else None,
            "accepted": accepted,
            "reason": "accepted_pose_candidate" if accepted else rejected_reasons[0],
            "reasons": reasons,
            "raw_box": _json_value(box),
            "raw_keypoints": raw_keypoints,
            "raw_keypoint_confidence": raw_keypoint_confidence,
            "class_id": class_id,
            "class_name": _class_name(result, class_id) if class_id is not None else None,
            "ankle_support": {
                "left_ankle": left_ankle,
                "right_ankle": right_ankle,
                "both_ankles_confident": bool(left_ankle["usable_for_review"] and right_ankle["usable_for_review"]),
                "ground_contact_visible": False,
                "review_required": "Pose confidence does not establish visible contact with a calibrated floor",
            },
            "provenance": PROVENANCE_VERSION,
        }
        diagnostics.append(diagnostic)
        if accepted:
            detections.append({
                "box": [float(value) for value in box],
                "confidence": float(score),
                "label": "person",
                "ground_contact_visible": False,
            })

    return detections, diagnostics


class PoseDetector:
    """Resident, optional Ultralytics pose detector using explicit local weights."""

    def __init__(
        self,
        weights: Path,
        device: str = "cpu",
        imgsz: int = 1280,
        confidence: float = 0.25,
        keypoint_confidence: float = 0.5,
    ) -> None:
        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.is_file():
            raise FileNotFoundError(f"Pose weights file does not exist: {self.weights}")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be a non-empty string")
        if isinstance(imgsz, (bool, np.bool_)) or int(imgsz) != imgsz or int(imgsz) <= 0:
            raise ValueError("imgsz must be a positive integer")
        self.device = device
        self.imgsz = int(imgsz)
        self.confidence = _validate_probability(confidence, "confidence")
        self.keypoint_confidence = _validate_probability(keypoint_confidence, "keypoint_confidence")
        self.model_fingerprint = self._fingerprint(self.weights)
        self.model_name = self.weights.stem
        self._model: Any | None = None

    @staticmethod
    def _fingerprint(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        if self._fingerprint(self.weights) != self.model_fingerprint:
            raise ValueError("Weights changed after detector construction; provenance would be invalid")
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Ultralytics is optional; install it to run PoseDetector") from exc
        try:
            # Passing the existing path is intentional: no model name or
            # network-backed checkpoint resolution is permitted here.
            model = YOLO(str(self.weights))
        except Exception as exc:
            raise RuntimeError(f"Could not load local pose weights {self.weights}") from exc
        self._model = model
        model_name = getattr(model, "model_name", None)
        if isinstance(model_name, str) and model_name:
            self.model_name = model_name
        return model

    @staticmethod
    def _first_result(raw: Any) -> Any:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            return raw[0] if raw else None
        # Ultralytics returns a list, but accepting a one-result object keeps
        # this adapter straightforward to mock without changing semantics.
        return raw

    def detect(self, image: Image.Image) -> dict[str, Any]:
        """Run resident pose inference on a PIL image and return JSON data."""

        if not isinstance(image, Image.Image):
            raise TypeError("detect expects a PIL.Image.Image")
        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        width, height = rgb_image.size
        model = self._load_model()
        started = time.perf_counter()
        try:
            raw_results = model(
                rgb_image,
                device=self.device,
                imgsz=self.imgsz,
                conf=self.confidence,
                max_det=64,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError("Pose inference failed") from exc
        result = self._first_result(raw_results)
        detections, diagnostics = parse_pose_result(
            result,
            width,
            height,
            confidence=self.confidence,
            keypoint_confidence=self.keypoint_confidence,
        )
        inference_ms = (time.perf_counter() - started) * 1000.0
        return {
            "width": int(width),
            "height": int(height),
            "detections": detections,
            "diagnostics": diagnostics,
            "model_name": self.model_name,
            "model_fingerprint": self.model_fingerprint,
            "inference_ms": float(inference_ms),
            "provenance": {
                "adapter": PROVENANCE_VERSION,
                "weights_path": str(self.weights),
                "weights_sha256": self.model_fingerprint,
                "coordinate_system": "original_image_pixels",
                "input_mode": "RGB",
                "ground_contact_visible_default": False,
                "ground_contact_policy": "Pose keypoints require a separately reviewed visible-ground-contact annotation",
                "class_semantics": "Generic COCO person candidate; no enemy or game-entity identity",
            },
        }


__all__ = ["PoseDetector", "parse_pose_result"]
