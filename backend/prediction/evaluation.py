"""Post-hoc perception evaluation with explicit limits on what is measured.

``evaluate`` consumes completed prediction rows and a separate annotation
artifact.  The annotation is never passed to a detector, tracker, or
predictor.  The required frame join is ``(sensor_id, frame_id)`` plus a
timestamp check; a row without an annotation is reported as uncovered and is
excluded from precision/recall.  An annotated empty frame counts as a true
negative only when the annotation artifact declares ``exhaustive: true``.
The optional per-frame ``exhaustive`` flag must agree with that global flag;
heterogeneous annotation coverage is rejected rather than silently scored as
complete.

Minimum input shape::

    predictions = [{
        "sensor_id": "A", "frame_id": "A-001", "t": 1.0,
        "width": 1280, "height": 720,
        "detections": [{"box": [100, 80, 150, 220],
                        "confidence": 0.9, "label": "person"}],
        "projected_observations": [{
            "detection_index": 0, "xyz": [1, 2, 0],
            "confidence": 0.8, "surface": "floor",
        }],
        "state": {},
    }]

    annotations = {
        "provenance": {"author": "annotator", "method": "frame review"},
        "exhaustive": True,
        "frames": [{
            "sensor_id": "A", "frame_id": "A-001", "t": 1.0,
            "people": [{
                "id": "person-1", "box": [100, 80, 150, 220],
                "ground_contact_visible": True,
                "ground_contact_pixel": [125, 220],
                "xyz": [1, 2, 0],
            }],
        }],
        # XYZ labels are used only when this is explicitly declared after
        # independent surveying.
        "metric_labels_independent": True,
        "metric_units": "meters",
    }

Non-empty ``ignored_regions`` are rejected deliberately until an ignore-region
policy is specified.  This avoids silently turning detections into either
false positives or invisible predictions.  Identity-switch counts are also
reported as unmeasured unless matched detections carry explicit ``track_id``
values and exhaustive annotations provide stable person IDs.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


IOU_THRESHOLD = 0.5
TIMESTAMP_TOLERANCE_S = 1e-6


def _error(message: str) -> ValueError:
    return ValueError(message)


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{label} must be a non-empty string")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise _error(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _error(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise _error(f"{label} must be a finite number")
    return result


def _integer(value: Any, label: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise _error(f"{label} must be an integer")
    result = int(value)
    if result < minimum:
        raise _error(f"{label} must be at least {minimum}")
    return result


def _box(value: Any, label: str, width: int, height: int) -> tuple[float, float, float, float]:
    try:
        values = tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    except TypeError as exc:
        raise _error(f"{label} must contain four finite coordinates") from exc
    if len(values) != 4:
        raise _error(f"{label} must contain four finite coordinates")
    left, top, right, bottom = values
    if left >= right or top >= bottom:
        raise _error(f"{label} must have positive area")
    if left < 0 or top < 0 or right > width or bottom > height:
        raise _error(f"{label} must lie within the calibrated image dimensions")
    return values


def _xyz(value: Any, label: str) -> tuple[float, float, float]:
    try:
        values = tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    except TypeError as exc:
        raise _error(f"{label} must contain three finite coordinates") from exc
    if len(values) != 3:
        raise _error(f"{label} must contain three finite coordinates")
    return values


def _pixel(value: Any, label: str, width: int, height: int) -> tuple[float, float]:
    try:
        values = tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    except TypeError as exc:
        raise _error(f"{label} must contain two finite coordinates") from exc
    if len(values) != 2:
        raise _error(f"{label} must contain two finite coordinates")
    if values[0] < 0 or values[0] > width or values[1] < 0 or values[1] > height:
        raise _error(f"{label} must lie within the calibrated image dimensions")
    return values


def _frame_key(row: dict[str, Any], label: str) -> tuple[str, str]:
    return (_string(row.get("sensor_id"), f"{label}.sensor_id"),
            _string(row.get("frame_id"), f"{label}.frame_id"))


def _confidence(value: Any, label: str) -> float:
    result = _number(value, label)
    if result < 0 or result > 1:
        raise _error(f"{label} must be between 0 and 1")
    return result


def _digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _error(f"{label} must be a non-empty SHA-256 digest string")
    digest = value.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise _error(f"{label} must be a 64-character hexadecimal SHA-256 digest")
    return digest


def _validate_prediction_rows(predictions: Any) -> dict[tuple[str, str], dict[str, Any]]:
    if not isinstance(predictions, list):
        raise _error("predictions must be a list of frame rows")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for frame_index, row in enumerate(predictions):
        if not isinstance(row, dict):
            raise _error(f"predictions[{frame_index}] must be an object")
        key = _frame_key(row, f"predictions[{frame_index}]")
        if key in indexed:
            raise _error(f"Duplicate prediction frame key {key!r}")
        width = _integer(row.get("width"), f"predictions[{frame_index}].width")
        height = _integer(row.get("height"), f"predictions[{frame_index}].height")
        frame = dict(row)
        frame["t"] = _number(row.get("t"), f"predictions[{frame_index}].t")
        if "image_sha256" in row and row["image_sha256"] is not None:
            frame["image_sha256"] = _digest(
                row["image_sha256"], f"predictions[{frame_index}].image_sha256"
            )
        detections = row.get("detections")
        if not isinstance(detections, list):
            raise _error(f"predictions[{frame_index}].detections must be a list")
        normalized_detections = []
        for detection_index, detection in enumerate(detections):
            if not isinstance(detection, dict):
                raise _error(
                    f"predictions[{frame_index}].detections[{detection_index}] must be an object"
                )
            label = detection.get("label", "person")
            if label != "person":
                raise _error(
                    f"predictions[{frame_index}].detections[{detection_index}] has unsupported label {label!r}"
                )
            normalized = dict(detection)
            normalized["box"] = _box(
                detection.get("box"),
                f"predictions[{frame_index}].detections[{detection_index}].box",
                width,
                height,
            )
            normalized["confidence"] = _confidence(
                detection.get("confidence"),
                f"predictions[{frame_index}].detections[{detection_index}].confidence",
            )
            if "ground_contact_visible" in detection and not isinstance(detection["ground_contact_visible"], bool):
                raise _error("Prediction ground_contact_visible must be a boolean")
            if "track_id" in detection:
                track_id = detection["track_id"]
                if isinstance(track_id, bool) or not isinstance(track_id, (str, int, np.integer)):
                    raise _error(
                        f"predictions[{frame_index}].detections[{detection_index}].track_id must be a string or integer"
                    )
                if isinstance(track_id, str) and not track_id.strip():
                    raise _error(
                        f"predictions[{frame_index}].detections[{detection_index}].track_id must be non-empty"
                    )
                normalized["track_id"] = int(track_id) if isinstance(track_id, np.integer) else track_id
            normalized_detections.append(normalized)
        frame["width"], frame["height"] = width, height
        frame["detections"] = normalized_detections
        projections = row.get("projected_observations", [])
        if not isinstance(projections, list):
            raise _error(f"predictions[{frame_index}].projected_observations must be a list")
        normalized_projections = []
        for projection_index, projection in enumerate(projections):
            if not isinstance(projection, dict):
                raise _error(
                    f"predictions[{frame_index}].projected_observations[{projection_index}] must be an object"
                )
            normalized_projection = dict(projection)
            normalized_projection["xyz"] = _xyz(
                projection.get("xyz"),
                f"predictions[{frame_index}].projected_observations[{projection_index}].xyz",
            )
            normalized_projection["confidence"] = _confidence(
                projection.get("confidence"),
                f"predictions[{frame_index}].projected_observations[{projection_index}].confidence",
            )
            surface = projection.get("surface", projection.get("surface_id"))
            normalized_projection["surface"] = _string(
                surface,
                f"predictions[{frame_index}].projected_observations[{projection_index}].surface",
            )
            if "detection_index" in projection:
                detection_index = projection["detection_index"]
                if (
                    isinstance(detection_index, bool)
                    or not isinstance(detection_index, (int, np.integer))
                    or int(detection_index) < 0
                    or int(detection_index) >= len(normalized_detections)
                ):
                    raise _error(
                        f"predictions[{frame_index}].projected_observations[{projection_index}].detection_index is invalid"
                    )
                normalized_projection["detection_index"] = int(detection_index)
            normalized_projections.append(normalized_projection)
        mapped_detection_indices = [
            item["detection_index"]
            for item in normalized_projections
            if "detection_index" in item
        ]
        if len(mapped_detection_indices) != len(set(mapped_detection_indices)):
            raise _error(
                f"predictions[{frame_index}].projected_observations has multiple entries for one detection_index"
            )
        frame["projected_observations"] = normalized_projections
        if "state" in row and row["state"] is not None and not isinstance(row["state"], dict):
            raise _error(f"predictions[{frame_index}].state must be an object when supplied")
        frame["_index"] = frame_index
        indexed[key] = frame
    return indexed


def _validate_annotations(annotations: Any) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    if not isinstance(annotations, dict):
        raise _error("annotations must be an object")
    provenance = annotations.get("provenance")
    if not isinstance(provenance, dict):
        raise _error("annotations.provenance must contain author and method")
    normalized_provenance = {
        "author": _string(provenance.get("author"), "annotations.provenance.author"),
        "method": _string(provenance.get("method"), "annotations.provenance.method"),
    }
    exhaustive = annotations.get("exhaustive")
    if not isinstance(exhaustive, bool):
        raise _error("annotations.exhaustive must be a boolean")
    ignored_regions = annotations.get("ignored_regions", [])
    if not isinstance(ignored_regions, list):
        raise _error("annotations.ignored_regions must be a list")
    if ignored_regions:
        raise _error(
            "Non-empty ignored_regions are not supported until an explicit ignore-region policy is defined"
        )
    frames = annotations.get("frames")
    if not isinstance(frames, list):
        raise _error("annotations.frames must be a list")
    metric_independent = annotations.get("metric_labels_independent", False)
    if not isinstance(metric_independent, bool):
        raise _error("annotations.metric_labels_independent must be a boolean")
    metric_units = annotations.get("metric_units")
    if metric_units is not None and not isinstance(metric_units, str):
        raise _error("annotations.metric_units must be a string when supplied")
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for frame_index, row in enumerate(frames):
        if not isinstance(row, dict):
            raise _error(f"annotations.frames[{frame_index}] must be an object")
        key = _frame_key(row, f"annotations.frames[{frame_index}]")
        if key in indexed:
            raise _error(f"Duplicate annotation frame key {key!r}")
        if "exhaustive" in row:
            if not isinstance(row["exhaustive"], bool):
                raise _error(f"annotations.frames[{frame_index}].exhaustive must be a boolean")
            if row["exhaustive"] != exhaustive:
                raise _error(
                    "Per-frame exhaustive flags must agree with annotations.exhaustive"
                )
        normalized = dict(row)
        normalized["t"] = _number(row.get("t"), f"annotations.frames[{frame_index}].t")
        if "image_sha256" in row and row["image_sha256"] is not None:
            normalized["image_sha256"] = _digest(
                row["image_sha256"], f"annotations.frames[{frame_index}].image_sha256"
            )
        people = row.get("people")
        if not isinstance(people, list):
            raise _error(f"annotations.frames[{frame_index}].people must be a list")
        width = row.get("width")
        height = row.get("height")
        if width is not None:
            width = _integer(width, f"annotations.frames[{frame_index}].width")
        if height is not None:
            height = _integer(height, f"annotations.frames[{frame_index}].height")
        normalized_people = []
        ids = set()
        for person_index, person in enumerate(people):
            if not isinstance(person, dict):
                raise _error(
                    f"annotations.frames[{frame_index}].people[{person_index}] must be an object"
                )
            person_id = _string(
                person.get("id"),
                f"annotations.frames[{frame_index}].people[{person_index}].id",
            )
            if person_id in ids:
                raise _error(
                    f"Duplicate person id {person_id!r} in annotations.frames[{frame_index}]"
                )
            ids.add(person_id)
            normalized_person = dict(person)
            normalized_person["id"] = person_id
            # Box dimensions are checked against prediction dimensions after
            # joining; use optional annotation dimensions here when present.
            if width is not None and height is not None:
                normalized_person["box"] = _box(
                    person.get("box"),
                    f"annotations.frames[{frame_index}].people[{person_index}].box",
                    width,
                    height,
                )
            else:
                try:
                    raw_box = tuple(
                        _number(item, f"annotations.frames[{frame_index}].people[{person_index}].box[{index}]")
                        for index, item in enumerate(person.get("box"))
                    )
                except TypeError as exc:
                    raise _error(
                        f"annotations.frames[{frame_index}].people[{person_index}].box must contain four finite coordinates"
                    ) from exc
                if len(raw_box) != 4 or raw_box[0] >= raw_box[2] or raw_box[1] >= raw_box[3]:
                    raise _error(
                        f"annotations.frames[{frame_index}].people[{person_index}].box must have positive area"
                    )
                normalized_person["box"] = raw_box
            visible = person.get("ground_contact_visible")
            if not isinstance(visible, bool):
                raise _error(
                    f"annotations.frames[{frame_index}].people[{person_index}].ground_contact_visible must be a boolean"
                )
            normalized_person["ground_contact_visible"] = visible
            if "ground_contact_pixel" in person and person["ground_contact_pixel"] is not None:
                if width is not None and height is not None:
                    normalized_person["ground_contact_pixel"] = _pixel(
                        person["ground_contact_pixel"],
                        f"annotations.frames[{frame_index}].people[{person_index}].ground_contact_pixel",
                        width,
                        height,
                    )
                else:
                    try:
                        pixel = tuple(
                            _number(item, f"annotations.frames[{frame_index}].people[{person_index}].ground_contact_pixel[{index}]")
                            for index, item in enumerate(person["ground_contact_pixel"])
                        )
                    except TypeError as exc:
                        raise _error(
                            f"annotations.frames[{frame_index}].people[{person_index}].ground_contact_pixel must contain two finite coordinates"
                        ) from exc
                    if len(pixel) != 2:
                        raise _error(
                            f"annotations.frames[{frame_index}].people[{person_index}].ground_contact_pixel must contain two finite coordinates"
                        )
                    normalized_person["ground_contact_pixel"] = pixel
            if visible and "ground_contact_pixel" not in normalized_person:
                raise _error(
                    f"annotations.frames[{frame_index}].people[{person_index}] needs ground_contact_pixel when visible"
                )
            if "xyz" in person and person["xyz"] is not None:
                normalized_person["xyz"] = _xyz(
                    person["xyz"],
                    f"annotations.frames[{frame_index}].people[{person_index}].xyz",
                )
            if "label_uncertainty_m" in person and person["label_uncertainty_m"] is not None:
                uncertainty = _number(
                    person["label_uncertainty_m"],
                    f"annotations.frames[{frame_index}].people[{person_index}].label_uncertainty_m",
                )
                if uncertainty < 0:
                    raise _error(
                        f"annotations.frames[{frame_index}].people[{person_index}].label_uncertainty_m must be non-negative"
                    )
                normalized_person["label_uncertainty_m"] = uncertainty
            normalized_people.append(normalized_person)
        normalized["people"] = normalized_people
        normalized["width"], normalized["height"] = width, height
        normalized["_index"] = frame_index
        indexed[key] = normalized
    metadata = {
        "provenance": normalized_provenance,
        "exhaustive": exhaustive,
        "metric_labels_independent": metric_independent,
        "metric_units": metric_units,
    }
    return indexed, metadata


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    area_left = (left[2] - left[0]) * (left[3] - left[1])
    area_right = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (area_left + area_right - intersection)


def _one_to_one_matches(
    detections: list[dict[str, Any]], people: list[dict[str, Any]]
) -> list[tuple[int, int, float]]:
    """Return a deterministic maximum-cardinality IoU-threshold matching."""

    adjacency: list[list[tuple[float, int]]] = []
    for detection in detections:
        options = [
            (_iou(detection["box"], person["box"]), person_index)
            for person_index, person in enumerate(people)
        ]
        adjacency.append(sorted(
            [(score, index) for score, index in options if score >= IOU_THRESHOLD],
            key=lambda item: (-item[0], item[1]),
        ))
    matched_detection_for_person = [-1] * len(people)

    def augment(detection_index: int, visited: set[int]) -> bool:
        for score, person_index in adjacency[detection_index]:
            if person_index in visited:
                continue
            visited.add(person_index)
            previous = matched_detection_for_person[person_index]
            if previous == -1 or augment(previous, visited):
                matched_detection_for_person[person_index] = detection_index
                return True
        return False

    for detection_index in range(len(detections)):
        augment(detection_index, set())
    matches = []
    for person_index, detection_index in enumerate(matched_detection_for_person):
        if detection_index >= 0:
            matches.append((detection_index, person_index, _iou(
                detections[detection_index]["box"], people[person_index]["box"]
            )))
    return sorted(matches, key=lambda item: (item[0], item[1]))


def _summary(
    values: list[float],
    units: str,
    *,
    measurement: str | None = None,
    proxy: bool | None = None,
) -> dict[str, Any]:
    if not values:
        result = {
            "status": "unmeasured",
            "count": 0,
            "units": units,
            "mean": None,
            "median": None,
            "p90": None,
            "reason": "No valid paired labels were available",
        }
        if measurement is not None:
            result["measurement"] = measurement
        if proxy is not None:
            result["proxy"] = proxy
        return result
    array = np.asarray(values, dtype=float)
    result = {
        "status": "measured",
        "count": int(array.size),
        "units": units,
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
    }
    if measurement is not None:
        result["measurement"] = measurement
    if proxy is not None:
        result["proxy"] = proxy
    return result


def _box_metrics(
    tp: int,
    fp: int,
    fn: int,
    *,
    exhaustive: bool,
    evaluated_frames: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "iou_threshold": IOU_THRESHOLD,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "status": "measured" if exhaustive and evaluated_frames else "unmeasured",
        "precision": None,
        "recall": None,
        "f1": None,
    }
    if not evaluated_frames:
        result["reason"] = "No annotated prediction frames were evaluated"
        return result
    if not exhaustive:
        result["reason"] = "Annotations are not exhaustive; unlabeled people make FP/FN rates incomplete"
        return result
    if tp + fp:
        result["precision"] = tp / (tp + fp)
    if tp + fn:
        result["recall"] = tp / (tp + fn)
    if result["precision"] is not None and result["recall"] is not None and result["precision"] + result["recall"]:
        result["f1"] = 2 * result["precision"] * result["recall"] / (result["precision"] + result["recall"])
    if tp + fp == 0 and tp + fn == 0:
        result["reason"] = "No labeled people or detections were present in evaluated frames"
    elif result["precision"] is None:
        result["reason"] = "Precision is undefined because no detections were emitted"
    elif result["recall"] is None:
        result["reason"] = "Recall is undefined because no labeled people were present"
    return result


def _new_group() -> dict[str, int]:
    return {
        "prediction_frames": 0,
        "annotation_frames": 0,
        "evaluated_frames": 0,
        "unannotated_prediction_frames": 0,
        "annotation_frames_without_prediction": 0,
        "detections": 0,
        "annotated_people": 0,
        "true_positives": 0,
        "false_positives": 0,
        "false_negatives": 0,
    }


def _identity_metrics(
    evaluated: list[tuple[dict[str, Any], dict[str, Any], list[tuple[int, int, float]]]],
    *,
    exhaustive: bool,
) -> dict[str, Any]:
    base = {
        "status": "unmeasured",
        "identity_switches": None,
        "observations": 0,
        "reason": "Identity switches require exhaustive stable person IDs and explicit prediction track_id values",
    }
    if not exhaustive:
        return base
    assignments: list[tuple[str, str, float, int, Any]] = []
    for prediction, annotation, matches in evaluated:
        for detection_index, person_index, _ in matches:
            detection = prediction["detections"][detection_index]
            if "track_id" not in detection:
                return base
            assignments.append((
                prediction["sensor_id"],
                annotation["people"][person_index]["id"],
                prediction["t"],
                prediction["_index"],
                detection["track_id"],
            ))
    if not assignments:
        return {
            **base,
            "reason": "No matched detections with explicit track IDs were available",
        }
    by_frame: dict[tuple[str, int], list[Any]] = defaultdict(list)
    # One predicted track must not claim two ground-truth people in one frame.
    for sensor, person, _t, frame_index, track_id in assignments:
        by_frame[(sensor, frame_index)].append((person, track_id))
    if any(len({track for _person, track in pairs}) != len(pairs) for pairs in by_frame.values()):
        return {
            **base,
            "reason": "A prediction track_id was assigned to multiple people in one frame",
        }
    frame_order: dict[str, list[int]] = defaultdict(list)
    for prediction, annotation, _matches in evaluated:
        frame_order[prediction["sensor_id"]].append(prediction["_index"])
    prediction_by_index = {
        prediction["_index"]: prediction for prediction, _annotation, _matches in evaluated
    }
    for sensor in frame_order:
        frame_order[sensor] = sorted(
            frame_order[sensor],
            key=lambda index: (prediction_by_index[index]["t"], index),
        )
    ordered_index: dict[tuple[str, int], int] = {
        (sensor, frame_index): order
        for sensor, indices in frame_order.items()
        for order, frame_index in enumerate(indices)
    }
    per_person: dict[tuple[str, str], list[tuple[int, Any]]] = defaultdict(list)
    for sensor, person, _t, frame_index, track_id in assignments:
        per_person[(sensor, person)].append((ordered_index[(sensor, frame_index)], track_id))
    switches = 0
    observations = 0
    for sequence in per_person.values():
        sequence.sort(key=lambda item: item[0])
        observations += len(sequence)
        for (previous_order, previous_track), (current_order, current_track) in zip(sequence, sequence[1:]):
            # A missed/unannotated frame resets continuity; it is not an ID switch.
            if current_order == previous_order + 1 and current_track != previous_track:
                switches += 1
    return {
        "status": "measured",
        "identity_switches": switches,
        "observations": observations,
        "definition": "track_id changes across consecutive evaluated frames for the same sensor/person; gaps reset continuity",
    }


def evaluate(predictions: list[dict], annotations: dict) -> dict:
    """Evaluate completed predictions against a separate annotation artifact.

    The fixed IoU threshold and timestamp tolerance are included in the
    result.  No pass/fail gate is applied; task-specific acceptance belongs to
    the caller after reviewing coverage and provenance.
    """

    predicted = _validate_prediction_rows(predictions)
    annotated, metadata = _validate_annotations(annotations)
    prediction_keys, annotation_keys = set(predicted), set(annotated)
    evaluated_keys = prediction_keys & annotation_keys

    evaluated: list[tuple[dict[str, Any], dict[str, Any], list[tuple[int, int, float]]]] = []
    tp = fp = fn = 0
    foot_errors: list[float] = []
    contact_counts = {"true_positive": 0, "false_positive": 0, "true_negative": 0, "false_negative": 0}
    contact_unmatched = 0
    metric_errors: list[float] = []
    metric_uncertainties: list[float] = []
    group_counts: dict[str, dict[str, int]] = defaultdict(_new_group)
    timestamp_mismatches = []
    image_digest_compared = 0
    image_digest_unverified = 0
    for key in sorted(evaluated_keys):
        prediction = predicted[key]
        annotation = annotated[key]
        prediction_digest = prediction.get("image_sha256")
        annotation_digest = annotation.get("image_sha256")
        if prediction_digest is not None and annotation_digest is not None:
            image_digest_compared += 1
            if prediction_digest != annotation_digest:
                raise _error(f"Image SHA-256 digest does not match for frame {key!r}")
        else:
            image_digest_unverified += 1
        if not np.isclose(prediction["t"], annotation["t"], rtol=0, atol=TIMESTAMP_TOLERANCE_S):
            timestamp_mismatches.append({
                "sensor_id": key[0],
                "frame_id": key[1],
                "prediction_t": prediction["t"],
                "annotation_t": annotation["t"],
            })
            continue
        if annotation["width"] is not None and annotation["width"] != prediction["width"]:
            raise _error(f"Annotation width does not match prediction for frame {key!r}")
        if annotation["height"] is not None and annotation["height"] != prediction["height"]:
            raise _error(f"Annotation height does not match prediction for frame {key!r}")
        # When annotation dimensions were omitted, validate boxes against the
        # prediction's calibrated frame now that the join is known.
        for person_index, person in enumerate(annotation["people"]):
            _box(
                person["box"],
                f"annotations frame {key!r} person {person_index} box",
                prediction["width"],
                prediction["height"],
            )
            if "ground_contact_pixel" in person:
                _pixel(
                    person["ground_contact_pixel"],
                    f"annotations frame {key!r} person {person_index} ground_contact_pixel",
                    prediction["width"],
                    prediction["height"],
                )
        matches = _one_to_one_matches(prediction["detections"], annotation["people"])
        evaluated.append((prediction, annotation, matches))
        tp += len(matches)
        fp += len(prediction["detections"]) - len(matches)
        fn += len(annotation["people"]) - len(matches)
        group = group_counts[key[0]]
        group["evaluated_frames"] += 1
        group["detections"] += len(prediction["detections"])
        group["annotated_people"] += len(annotation["people"])
        group["true_positives"] += len(matches)
        group["false_positives"] += len(prediction["detections"]) - len(matches)
        group["false_negatives"] += len(annotation["people"]) - len(matches)
        matched_indices = {item[0] for item in matches}
        contact_unmatched += sum(d.get("ground_contact_visible") is True and i not in matched_indices
                                 for i, d in enumerate(prediction["detections"]))
        for detection_index, person_index, _iou_value in matches:
            person = annotation["people"][person_index]
            detection = prediction["detections"][detection_index]
            if "ground_contact_visible" in detection:
                predicted_contact = detection["ground_contact_visible"]
                actual_contact = person["ground_contact_visible"]
                category = ("true_" if predicted_contact == actual_contact else "false_") + ("positive" if predicted_contact else "negative")
                contact_counts[category] += 1
            if person["ground_contact_visible"] and "ground_contact_pixel" in person:
                box = detection["box"]
                foot = np.array([(box[0] + box[2]) / 2, box[3]], dtype=float)
                foot_errors.append(float(np.linalg.norm(foot - np.asarray(person["ground_contact_pixel"]))))
            if metadata["metric_labels_independent"] and metadata["metric_units"] == "meters":
                projection_by_detection = {
                    item["detection_index"]: item
                    for item in prediction["projected_observations"]
                    if "detection_index" in item
                }
                projection = projection_by_detection.get(detection_index)
                if projection is not None and "xyz" in person:
                    metric_errors.append(float(np.linalg.norm(
                        np.asarray(projection["xyz"]) - np.asarray(person["xyz"])
                    )))
                    if "label_uncertainty_m" in person:
                        metric_uncertainties.append(person["label_uncertainty_m"])

    if timestamp_mismatches:
        raise _error(
            "Prediction and annotation timestamps differ beyond the declared tolerance: "
            + repr(timestamp_mismatches)
        )

    for key, prediction in predicted.items():
        group_counts[key[0]]["prediction_frames"] += 1
        if key not in annotation_keys:
            group_counts[key[0]]["unannotated_prediction_frames"] += 1
    for key, annotation in annotated.items():
        group_counts[key[0]]["annotation_frames"] += 1
        if key not in prediction_keys:
            group_counts[key[0]]["annotation_frames_without_prediction"] += 1

    metric_result = _summary(metric_errors, "meters")
    if not metadata["metric_labels_independent"]:
        metric_result = {
            "status": "unmeasured",
            "count": 0,
            "units": None,
            "mean": None,
            "median": None,
            "p90": None,
            "reason": "Independent metric XYZ labels were not declared",
        }
    elif metadata["metric_units"] != "meters":
        metric_result = {
            "status": "unmeasured",
            "count": 0,
            "units": None,
            "mean": None,
            "median": None,
            "p90": None,
            "reason": "Metric labels require an explicit metric_units='meters' declaration",
        }
    elif not metric_errors:
        metric_result["reason"] = (
            "No matched projected observation with an independently labeled XYZ was available"
        )
    if metric_uncertainties:
        metric_result["paired_label_uncertainty_mean_m"] = float(np.mean(metric_uncertainties))

    coverage = {
        "prediction_frames": len(predicted),
        "annotation_frames": len(annotated),
        "evaluated_frames": len(evaluated),
        "unannotated_prediction_frames": len(prediction_keys - annotation_keys),
        "annotation_frames_without_prediction": len(annotation_keys - prediction_keys),
        "annotated_people": sum(len(frame["people"]) for frame in annotated.values()),
        "evaluated_people": sum(len(frame["people"]) for _prediction, frame, _matches in evaluated),
        "exhaustive": metadata["exhaustive"],
        "image_sha256_compared_frames": image_digest_compared,
        "image_sha256_unverified_frames": image_digest_unverified,
        "groups": {key: value for key, value in sorted(group_counts.items())},
    }
    return {
        "thresholds": {
            "box_iou": IOU_THRESHOLD,
            "timestamp_tolerance_s": TIMESTAMP_TOLERANCE_S,
        },
        "annotation_provenance": metadata["provenance"],
        "coverage": coverage,
        "box_metrics": _box_metrics(
            tp,
            fp,
            fn,
            exhaustive=metadata["exhaustive"],
            evaluated_frames=len(evaluated),
        ),
        "pixel_foot_error": _summary(
            foot_errors,
            "pixels",
            measurement="bbox_bottom_center_proxy",
            proxy=True,
        ),
        "metric_position_error": metric_result,
        "ground_contact_gate": {
            "status": "measured" if sum(contact_counts.values()) else "unmeasured",
            "scope": "Visibility classification on IoU-matched people; not metric projection accuracy",
            "matched_people_with_predictions": sum(contact_counts.values()),
            **contact_counts,
            "unmatched_detections_marked_visible": contact_unmatched,
            "unsafe_projection_candidates": contact_counts["false_positive"] + contact_unmatched,
        },
        "identity_switches": _identity_metrics(evaluated, exhaustive=metadata["exhaustive"]),
        "ignored_regions": {
            "status": "unsupported",
            "count": 0,
            "reason": "Non-empty ignored_regions are rejected by this evaluator",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("predictions", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    with args.predictions.open() as stream:
        predictions = [json.loads(line) for line in stream if line.strip()] if args.predictions.suffix == ".jsonl" else json.load(stream)
    with args.annotations.open() as stream:
        annotations = json.load(stream)
    result = json.dumps(evaluate(predictions, annotations), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["evaluate"]
