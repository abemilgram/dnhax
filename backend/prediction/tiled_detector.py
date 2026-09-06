"""Fixed overlapping tiled wrapper for the resident pose detector.

The wrapper runs one full-frame inference and four fixed 2x2 tiles.  Tile
coordinates are mapped back into the original image, tile-boundary candidates
are discarded because their feet or boxes may be clipped, and the remaining
generic person candidates are merged with deterministic confidence-sorted NMS.
No tile or full-frame result is allowed to authorize ground contact.
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from PIL import Image


PROVENANCE_VERSION = "fixed-2x2-overlap-tiling-v1"


def _json_value(value: Any) -> Any:
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
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _finite_box(value: Any) -> list[float] | None:
    try:
        box = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError, OverflowError):
        return None
    if box.shape != (4,) or not np.all(np.isfinite(box)):
        return None
    output = [float(item) for item in box]
    if output[0] >= output[2] or output[1] >= output[3]:
        return None
    return output


def _map_point(value: Any, offset_x: int, offset_y: int) -> Any:
    try:
        point = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return _json_value(value)
    if point.shape == (2,) and np.all(np.isfinite(point)):
        return [float(point[0] + offset_x), float(point[1] + offset_y)]
    if point.ndim >= 1 and point.shape[-1:] == (2,):
        return [_map_point(item, offset_x, offset_y) for item in point]
    return _json_value(value)


def _map_diagnostic(diagnostic: Any, offset_x: int, offset_y: int, source: str, tile_bounds: tuple[int, int, int, int] | None) -> dict[str, Any]:
    item = dict(_json_value(diagnostic)) if isinstance(diagnostic, dict) else {
        "accepted": False,
        "reason": "invalid_base_diagnostic",
        "raw_box": None,
        "raw_keypoints": [],
    }
    if isinstance(item.get("raw_box"), list):
        box = item["raw_box"]
        try:
            if len(box) == 4:
                item["raw_box"] = [float(box[0] + offset_x), float(box[1] + offset_y), float(box[2] + offset_x), float(box[3] + offset_y)]
        except (TypeError, ValueError):
            pass
    if "raw_boxes" in item and isinstance(item["raw_boxes"], list):
        item["raw_boxes"] = [_map_point(box, offset_x, offset_y) for box in item["raw_boxes"]]
    if "raw_keypoints" in item:
        item["raw_keypoints"] = _map_point(item["raw_keypoints"], offset_x, offset_y)
    ankle_support = item.get("ankle_support")
    if isinstance(ankle_support, dict):
        for side in ("left_ankle", "right_ankle"):
            support = ankle_support.get(side)
            if isinstance(support, dict) and "xy" in support and support["xy"] is not None:
                support["xy"] = _map_point(support["xy"], offset_x, offset_y)
    item["source"] = source
    item["tile_offset"] = [int(offset_x), int(offset_y)]
    # This is assigned only after NMS has established the final output order.
    # Keeping it explicit prevents callers from accidentally joining against
    # the base detector's raw detection_index.
    item["output_detection_index"] = None
    if tile_bounds is not None:
        item["tile_bounds"] = [int(value) for value in tile_bounds]
    # Keep the original parser decision separately; wrapper decisions can
    # suppress a candidate without losing the base detector's alignment.
    item.setdefault("detector_accepted", bool(item.get("accepted", False)))
    item["ground_contact_visible"] = False
    if "contact_eligible" not in item:
        item["contact_eligible"] = bool(item["detector_accepted"])
    else:
        # An explicit false from the base detector is a veto.  Rejected base
        # diagnostics cannot be eligible even if a compatible adapter supplied
        # an inconsistent true value.
        item["contact_eligible"] = bool(item["contact_eligible"]) and bool(item["detector_accepted"])
    return item


def _names_zero_is_person(result: dict[str, Any]) -> bool | None:
    """Return the explicit class-0 name when the base result exposes names.

    The resident PoseDetector records this same check in each accepted
    diagnostic.  A compatible adapter may also expose ``names`` at the top
    level, so honor it when present and let the diagnostic provide the
    fallback provenance when it is not.
    """

    if "names" not in result:
        return None
    names = result.get("names")
    value: Any = None
    if isinstance(names, dict):
        value = names.get(0, names.get("0"))
    elif isinstance(names, (list, tuple)) and names:
        value = names[0]
    return value == "person"


def _person_provenance_is_valid(result: dict[str, Any], diagnostic: dict[str, Any]) -> bool:
    """Require generic COCO person provenance for an accepted candidate."""

    names_check = _names_zero_is_person(result)
    if names_check is False:
        return False

    class_id = diagnostic.get("class_id")
    if class_id is not None:
        try:
            if int(class_id) != 0 or float(class_id) != 0.0:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    class_name = diagnostic.get("class_name")
    if class_name is not None and class_name != "person":
        return False

    # Parsed PoseDetector diagnostics carry class_id and class_name.  If a
    # result omits top-level names, require those fields rather than trusting
    # an arbitrary detection label supplied by a compatible adapter.
    if names_check is None and (class_id is None or class_name != "person"):
        return False
    return True


def _keypoint_confidence_is_valid(value: Any) -> bool:
    """Validate supplied keypoint confidence metadata as probabilities."""

    if value is None:
        return True
    try:
        confidence = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return False
    if confidence.size == 0:
        return True
    return bool(np.all(np.isfinite(confidence)) and np.all((confidence >= 0.0) & (confidence <= 1.0)))


def _tile_specs(width: int, height: int, overlap: float) -> list[dict[str, Any]]:
    """Return deterministic full-image plus 2x2 tile bounds.

    Each tile spans ``(1 + overlap) / 2`` of the corresponding image axis,
    giving an overlap of ``overlap`` times the image dimension between the two
    neighboring tiles.  The final tile is pinned to the right/bottom edge.
    """

    tile_width = max(1, int(math.ceil((1.0 + overlap) * width / 2.0)))
    tile_height = max(1, int(math.ceil((1.0 + overlap) * height / 2.0)))
    x_starts = [0, max(0, width - tile_width)]
    y_starts = [0, max(0, height - tile_height)]
    specs = [{"source": "full", "bounds": (0, 0, width, height)}]
    tile_index = 0
    for top in y_starts:
        for left in x_starts:
            specs.append({
                "source": f"tile-{tile_index:02d}",
                "bounds": (left, top, min(width, left + tile_width), min(height, top + tile_height)),
            })
            tile_index += 1
    return specs


def _iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    if intersection <= 0:
        return 0.0
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    denominator = area_first + area_second - intersection
    return intersection / denominator if denominator > 0 else 0.0


class TiledPoseDetector:
    """Run a resident detector on a full frame plus fixed overlapping tiles."""

    def __init__(
        self,
        detector: Any | None = None,
        overlap: float = 0.20,
        iou_threshold: float = 0.50,
        max_detections: int = 64,
        *,
        base_detector: Any | None = None,
    ) -> None:
        if detector is not None and base_detector is not None:
            raise ValueError("pass detector or base_detector, not both")
        self.detector = detector if detector is not None else base_detector
        if self.detector is None or not callable(getattr(self.detector, "detect", None)):
            raise TypeError("a resident detector with detect(image) is required")
        try:
            self.overlap = float(overlap)
            self.iou_threshold = float(iou_threshold)
            self.max_detections = int(max_detections)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("tiling parameters are invalid") from exc
        if not math.isfinite(self.overlap) or not 0.0 <= self.overlap < 1.0:
            raise ValueError("overlap must be finite and in [0, 1)")
        if not math.isfinite(self.iou_threshold) or not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be finite and in [0, 1]")
        if isinstance(max_detections, bool) or self.max_detections != max_detections or not 1 <= self.max_detections <= 64:
            raise ValueError("max_detections must be an integer between 1 and 64")

    @staticmethod
    def _source_diagnostics(result: Any, detections: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
        raw = result.get("diagnostics", []) if isinstance(result, dict) else []
        diagnostics = [dict(item) for item in raw if isinstance(item, dict)] if isinstance(raw, (list, tuple)) else []
        accepted_by_output: dict[int, dict[str, Any]] = {}
        for item in diagnostics:
            if not bool(item.get("accepted", False)):
                continue
            output_index = item.get("output_detection_index")
            valid_index = isinstance(output_index, (int, np.integer)) and not isinstance(output_index, bool)
            if not valid_index or int(output_index) < 0 or int(output_index) >= len(detections):
                item.update({
                    "accepted": False,
                    "detector_accepted": True,
                    "reason": "invalid_output_detection_index",
                    "contact_eligible": False,
                    "output_detection_index": None,
                })
                reasons = item.setdefault("reasons", [])
                if isinstance(reasons, list) and "invalid_output_detection_index" not in reasons:
                    reasons.append("invalid_output_detection_index")
                continue
            output_index_i = int(output_index)
            if output_index_i in accepted_by_output:
                item.update({
                    "accepted": False,
                    "detector_accepted": True,
                    "reason": "duplicate_output_detection_index",
                    "contact_eligible": False,
                    "output_detection_index": None,
                })
                reasons = item.setdefault("reasons", [])
                if isinstance(reasons, list) and "duplicate_output_detection_index" not in reasons:
                    reasons.append("duplicate_output_detection_index")
                continue
            accepted_by_output[output_index_i] = item
        return diagnostics, accepted_by_output

    def detect(self, image: Image.Image) -> dict[str, Any]:
        if not isinstance(image, Image.Image):
            raise TypeError("detect expects a PIL.Image.Image")
        rgb_image = image if image.mode == "RGB" else image.convert("RGB")
        width, height = rgb_image.size
        specs = _tile_specs(width, height, self.overlap)
        started = time.perf_counter()
        candidates: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        inference_ms = 0.0
        base_provenance = None
        model_name = getattr(self.detector, "model_name", None)
        model_fingerprint = getattr(self.detector, "model_fingerprint", None)

        for spec_index, spec in enumerate(specs):
            left, top, right, bottom = spec["bounds"]
            source = str(spec["source"])
            source_image = rgb_image if source == "full" else rgb_image.crop((left, top, right, bottom))
            result = self.detector.detect(source_image)
            if not isinstance(result, dict):
                raise ValueError("base detector must return a JSON result dictionary")
            if (result.get("width"), result.get("height")) != source_image.size:
                raise ValueError("Base detector coordinates differ from tile dimensions")
            elapsed = float(result["inference_ms"])
            if not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError("Base detector inference time must be finite and nonnegative")
            inference_ms += elapsed
            if base_provenance is None:
                base_provenance = result.get("provenance", {})
            model_name = result.get("model_name", model_name)
            model_fingerprint = result.get("model_fingerprint", model_fingerprint)
            base_detections = result.get("detections", [])
            if not isinstance(base_detections, list):
                base_detections = []
            base_diagnostics, accepted_diagnostics = self._source_diagnostics(result, base_detections)
            mapped_diagnostics: list[dict[str, Any]] = []
            for diagnostic in base_diagnostics:
                mapped_diagnostics.append(_map_diagnostic(diagnostic, left, top, source, None if source == "full" else (left, top, right, bottom)))
            # Rejected base diagnostics are emitted immediately.  Accepted
            # diagnostics are emitted once alongside their candidate below so
            # their alignment survives mapping, tile filtering, and NMS.
            diagnostics.extend(item for item in mapped_diagnostics if not bool(item.get("accepted", False)))

            for detection_index, raw_detection in enumerate(base_detections):
                if not isinstance(raw_detection, dict):
                    continue
                box = _finite_box(raw_detection.get("box"))
                try:
                    confidence = float(raw_detection.get("confidence"))
                except (TypeError, ValueError, OverflowError):
                    confidence = float("nan")
                mapped_box = None if box is None else [box[0] + left, box[1] + top, box[2] + left, box[3] + top]
                candidate_diag = accepted_diagnostics.get(detection_index)
                if candidate_diag is None:
                    mapped_diag = _map_diagnostic({
                        "accepted": True,
                        "reason": "missing_output_detection_index",
                        "raw_box": box,
                        "raw_keypoints": [],
                    }, left, top, source, None if source == "full" else (left, top, right, bottom))
                    mapped_diag.update({
                        "accepted": False,
                        "detector_accepted": True,
                        "reason": "missing_output_detection_index",
                        "contact_eligible": False,
                        "output_detection_index": None,
                    })
                    diagnostics.append(mapped_diag)
                    continue
                if not _person_provenance_is_valid(result, candidate_diag):
                    mapped_diag = _map_diagnostic(candidate_diag, left, top, source, None if source == "full" else (left, top, right, bottom))
                    mapped_diag.update({
                        "accepted": False,
                        "detector_accepted": True,
                        "reason": "unverified_person_class_mapping",
                        "contact_eligible": False,
                        "output_detection_index": None,
                    })
                    reasons = mapped_diag.setdefault("reasons", [])
                    if isinstance(reasons, list) and "unverified_person_class_mapping" not in reasons:
                        reasons.append("unverified_person_class_mapping")
                    diagnostics.append(mapped_diag)
                    continue
                if not _keypoint_confidence_is_valid(candidate_diag.get("raw_keypoint_confidence")):
                    mapped_diag = _map_diagnostic(candidate_diag, left, top, source, None if source == "full" else (left, top, right, bottom))
                    mapped_diag.update({
                        "accepted": False,
                        "detector_accepted": True,
                        "reason": "invalid_keypoint_confidence",
                        "contact_eligible": False,
                        "output_detection_index": None,
                    })
                    reasons = mapped_diag.setdefault("reasons", [])
                    if isinstance(reasons, list) and "invalid_keypoint_confidence" not in reasons:
                        reasons.append("invalid_keypoint_confidence")
                    diagnostics.append(mapped_diag)
                    continue
                if mapped_box is None or not np.all(np.isfinite(mapped_box)) or mapped_box[0] < 0 or mapped_box[1] < 0 or mapped_box[2] > width or mapped_box[3] > height or mapped_box[0] >= mapped_box[2] or mapped_box[1] >= mapped_box[3]:
                    candidate_diag = _map_diagnostic(candidate_diag, left, top, source, None if source == "full" else (left, top, right, bottom))
                    candidate_diag.update({"accepted": False, "detector_accepted": True, "reason": "mapped_box_out_of_bounds", "contact_eligible": False})
                    diagnostics.append(candidate_diag)
                    continue
                if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
                    candidate_diag = _map_diagnostic(candidate_diag, left, top, source, None if source == "full" else (left, top, right, bottom))
                    candidate_diag.update({"accepted": False, "detector_accepted": True, "reason": "invalid_mapped_confidence", "contact_eligible": False})
                    diagnostics.append(candidate_diag)
                    continue

                local_box = box
                tile_width, tile_height = right - left, bottom - top
                boundary_contact = bool(
                    local_box[0] <= 0.0 or local_box[1] <= 0.0
                    or local_box[2] >= tile_width or local_box[3] >= tile_height
                )
                box_boundary_contact = boundary_contact
                raw_keypoints = candidate_diag.get("raw_keypoints")
                # Only ankle clipping is relevant to ground-contact
                # eligibility; a hand or head on a tile edge is harmless.
                if isinstance(raw_keypoints, list):
                    for ankle_index in (15, 16):
                        if ankle_index >= len(raw_keypoints):
                            continue
                        try:
                            point_array = np.asarray(raw_keypoints[ankle_index], dtype=float)
                        except (TypeError, ValueError):
                            continue
                        if point_array.shape == (2,) and np.all(np.isfinite(point_array)) and (
                            point_array[0] <= 0.0 or point_array[1] <= 0.0
                            or point_array[0] >= tile_width or point_array[1] >= tile_height
                        ):
                            boundary_contact = True
                            break
                mapped_diag = _map_diagnostic(candidate_diag, left, top, source, None if source == "full" else (left, top, right, bottom))
                if box_boundary_contact and source != "full":
                    mapped_diag.update({
                        "accepted": False,
                        "detector_accepted": True,
                        "reason": "tile_boundary_truncation",
                        "contact_eligible": False,
                    })
                    diagnostics.append(mapped_diag)
                    continue
                if boundary_contact:
                    # Full-frame edge candidates may remain useful as generic
                    # boxes, but their contact provenance is not eligible.
                    mapped_diag["contact_eligible"] = False
                mapped_diag["candidate_index"] = len(candidates)
                diagnostics.append(mapped_diag)
                candidates.append({
                    "box": mapped_box,
                    "confidence": confidence,
                    "label": "person",
                    "ground_contact_visible": False,
                    "source": source,
                    "source_order": spec_index,
                    "diagnostic": mapped_diag,
                })

        ordered = sorted(
            candidates,
            key=lambda candidate: (-float(candidate["confidence"]), int(candidate["source_order"]), int(candidate["diagnostic"].get("candidate_index", 0))),
        )
        kept: list[dict[str, Any]] = []
        suppressed: list[dict[str, Any]] = []
        for candidate in ordered:
            if any(_iou(candidate["box"], previous["box"]) >= self.iou_threshold for previous in kept):
                candidate["diagnostic"].update({"kept": False, "accepted": False, "detector_accepted": True, "reason": "suppressed_by_nms", "contact_eligible": False})
                suppressed.append(candidate)
                continue
            if len(kept) >= self.max_detections:
                candidate["diagnostic"].update({"kept": False, "accepted": False, "detector_accepted": True, "reason": "max_detections", "contact_eligible": False})
                suppressed.append(candidate)
                continue
            candidate["diagnostic"]["kept"] = True
            kept.append(candidate)

        # Assign the public join key only after deterministic NMS and the
        # max-detection budget have selected the final output ordering.
        for output_detection_index, candidate in enumerate(kept):
            candidate["diagnostic"]["output_detection_index"] = output_detection_index

        output_detections = [
            {
                "box": [float(value) for value in candidate["box"]],
                "confidence": float(candidate["confidence"]),
                "label": "person",
                "ground_contact_visible": False,
            }
            for candidate in kept
        ]
        wall_ms = (time.perf_counter() - started) * 1000.0
        if not isinstance(model_name, str) or not model_name:
            model_name = str(getattr(self.detector, "model_name", "unknown-pose-model"))
        if model_fingerprint is not None:
            model_fingerprint = str(model_fingerprint)
        return {
            "width": int(width),
            "height": int(height),
            "detections": output_detections,
            "diagnostics": diagnostics,
            "model_name": model_name,
            "model_fingerprint": model_fingerprint,
            "inference_ms": float(inference_ms),
            "wall_ms": float(wall_ms),
            "timings": {
                "inference_ms": float(inference_ms),
                "wall_ms": float(wall_ms),
                "base_calls": len(specs),
            },
            "provenance": {
                "adapter": PROVENANCE_VERSION,
                "base_detector": base_provenance,
                "layout": "full_frame_plus_2x2",
                "overlap": float(self.overlap),
                "iou_threshold": float(self.iou_threshold),
                "max_detections": int(self.max_detections),
                "coordinates": "original_image_pixels",
                "ground_contact_visible_default": False,
                "tile_boundary_policy": "Candidates touching a tile edge are discarded because feet or boxes may be clipped",
                "ground_contact_policy": "No pose or tiling result authorizes visible ground contact; a separately disclosed and evaluated contact policy remains required",
                "class_semantics": "Generic person candidates only; no enemy or game entity data",
            },
        }


TiledDetector = TiledPoseDetector

__all__ = ["TiledPoseDetector", "TiledDetector"]
