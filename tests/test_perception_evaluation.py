import pytest
from backend.prediction.evaluation import evaluate

def prediction(
    frame_id="f1",
    *,
    t=1.0,
    boxes=((10, 10, 30, 50),),
    projected=False,
    track_ids=None,
    image_sha256=None,
):
    detections = []
    for index, box in enumerate(boxes):
        item = {"box": list(box), "confidence": 0.9, "label": "person"}
        if track_ids is not None:
            item["track_id"] = track_ids[index]
        detections.append(item)
    row = {
        "sensor_id": "A",
        "frame_id": frame_id,
        "t": t,
        "width": 100,
        "height": 100,
        "detections": detections,
    }
    if image_sha256 is not None:
        row["image_sha256"] = image_sha256
    if projected:
        row["projected_observations"] = [
            {
                "detection_index": index,
                "xyz": [float(index), 2.0, 0.0],
                "confidence": 0.8,
                "surface": "floor",
            }
            for index in range(len(detections))
        ]
    return row


def person(
    person_id="p1",
    *,
    box=(10, 10, 30, 50),
    visible=True,
    pixel=(20, 50),
    xyz=None,
):
    item = {
        "id": person_id,
        "box": list(box),
        "ground_contact_visible": visible,
    }
    if pixel is not None:
        item["ground_contact_pixel"] = list(pixel)
    if xyz is not None:
        item["xyz"] = list(xyz)
    return item


def annotations(frames, *, exhaustive=True, independent=False, units=None):
    result = {
        "provenance": {"author": "annotator-1", "method": "frame-by-frame review"},
        "exhaustive": exhaustive,
        "frames": frames,
    }
    if independent:
        result["metric_labels_independent"] = True
    if units is not None:
        result["metric_units"] = units
    return result


def test_ground_contact_gate_reports_hallucinated_feet_and_false_person_candidates():
    row = prediction(boxes=((10, 10, 30, 50), (60, 10, 80, 50)))
    for detection in row["detections"]:
        detection["ground_contact_visible"] = True
    result = evaluate([row], annotations([{"sensor_id": "A", "frame_id": "f1", "t": 1.,
                                          "people": [person(visible=False, pixel=None)]}]))
    gate = result["ground_contact_gate"]
    assert gate["false_positive"] == 1
    assert gate["unmatched_detections_marked_visible"] == 1
    assert gate["unsafe_projection_candidates"] == 2


def test_empty_annotated_frame_counts_false_positive():
    result = evaluate(
        [prediction()],
        annotations([{"sensor_id": "A", "frame_id": "f1", "t": 1.0, "people": []}]),
    )
    metrics = result["box_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (0, 1, 0)
    assert metrics["precision"] == 0
    assert metrics["recall"] is None


def test_missing_ground_truth_person_counts_false_negative():
    result = evaluate(
        [prediction(boxes=[])],
        annotations([{"sensor_id": "A", "frame_id": "f1", "t": 1.0, "people": [person()]}]),
    )
    metrics = result["box_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (0, 0, 1)
    assert metrics["precision"] is None
    assert metrics["recall"] == 0


def test_unannotated_frames_are_ignored_and_reported_as_uncovered():
    result = evaluate(
        [prediction(boxes=[]), prediction("f2", t=2.0)],
        annotations([{"sensor_id": "A", "frame_id": "f1", "t": 1.0, "people": []}]),
    )
    assert result["box_metrics"]["false_positives"] == 0
    assert result["coverage"]["unannotated_prediction_frames"] == 1
    assert result["coverage"]["evaluated_frames"] == 1


def test_duplicate_detections_are_matched_one_to_one():
    result = evaluate(
        [prediction(boxes=((10, 10, 30, 50), (10, 10, 30, 50)))],
        annotations([{"sensor_id": "A", "frame_id": "f1", "t": 1.0, "people": [person()]}]),
    )
    metrics = result["box_metrics"]
    assert (metrics["true_positives"], metrics["false_positives"], metrics["false_negatives"]) == (1, 1, 0)


def test_same_frame_id_with_timestamp_mismatch_is_rejected():
    with pytest.raises(ValueError, match="timestamps"):
        evaluate(
            [prediction(t=1.0)],
            annotations([{"sensor_id": "A", "frame_id": "f1", "t": 1.1, "people": []}]),
        )


def test_pixel_foot_error_requires_visible_ground_contact_label():
    result = evaluate(
        [prediction(boxes=((10, 10, 30, 50), (40, 10, 60, 50)))],
        annotations([
            {
                "sensor_id": "A",
                "frame_id": "f1",
                "t": 1.0,
                "people": [
                    person("p1", pixel=(20, 50)),
                    person("p2", box=(40, 10, 60, 50), visible=False, pixel=None),
                ],
            }
        ]),
    )
    assert result["pixel_foot_error"]["status"] == "measured"
    assert result["pixel_foot_error"]["count"] == 1
    assert result["pixel_foot_error"]["mean"] == 0


def test_metric_error_requires_explicit_independent_meter_labels():
    base_without_xyz = {
        "sensor_id": "A",
        "frame_id": "f1",
        "t": 1.0,
        "people": [person()],
    }
    without_labels = evaluate([prediction(projected=True)], annotations([base_without_xyz]))
    assert without_labels["metric_position_error"]["status"] == "unmeasured"
    base = {
        "sensor_id": "A",
        "frame_id": "f1",
        "t": 1.0,
        "people": [person(xyz=[0, 2, 0])],
    }
    with_labels = evaluate([prediction(projected=True)], annotations([base]))
    assert with_labels["metric_position_error"]["status"] == "unmeasured"
    with_independent_labels = evaluate(
        [prediction(projected=True)],
        annotations([base], independent=True, units="meters"),
    )
    assert with_independent_labels["metric_position_error"]["status"] == "measured"
    assert with_independent_labels["metric_position_error"]["count"] == 1
    assert with_independent_labels["metric_position_error"]["mean"] == 0


def test_identity_switches_are_unmeasured_without_explicit_track_ids():
    result = evaluate(
        [prediction()],
        annotations([{"sensor_id": "A", "frame_id": "f1", "t": 1.0, "people": [person()]}]),
    )
    assert result["identity_switches"]["status"] == "unmeasured"
    assert result["identity_switches"]["identity_switches"] is None


def test_nonempty_ignored_regions_are_explicitly_rejected():
    data = annotations([{"sensor_id": "A", "frame_id": "f1", "t": 1.0, "people": []}])
    data["ignored_regions"] = [{"box": [0, 0, 10, 10]}]
    with pytest.raises(ValueError, match="ignored_regions"):
        evaluate([prediction()], data)


def test_no_annotated_labels_does_not_claim_precision_or_recall():
    result = evaluate([prediction(boxes=[])], annotations([]))
    assert result["coverage"]["evaluated_frames"] == 0
    assert result["box_metrics"]["precision"] is None
    assert result["box_metrics"]["recall"] is None
    assert result["box_metrics"]["status"] == "unmeasured"


def test_identity_order_uses_capture_time_even_when_input_is_reversed():
    first = prediction("z", t=1.0, track_ids=["track-a"])
    gap = prediction("a", t=2.0, boxes=[])  # The labeled person is missed.
    last = prediction("m", t=3.0, track_ids=["track-b"])
    frames = [
        {"sensor_id": "A", "frame_id": "z", "t": 1.0, "people": [person()]},
        {"sensor_id": "A", "frame_id": "a", "t": 2.0, "people": [person()]},
        {"sensor_id": "A", "frame_id": "m", "t": 3.0, "people": [person()]},
    ]
    result = evaluate([last, gap, first], annotations(frames))
    assert result["identity_switches"]["status"] == "measured"
    assert result["identity_switches"]["identity_switches"] == 0


def test_image_digest_mismatch_is_rejected():
    digest_a, digest_b = "a" * 64, "b" * 64
    frame = {"sensor_id": "A", "frame_id": "f1", "t": 1.0, "people": []}
    frame["image_sha256"] = digest_b
    with pytest.raises(ValueError, match="digest"):
        evaluate(
            [prediction(image_sha256=digest_a)],
            annotations([frame]),
        )


def test_per_frame_exhaustive_must_agree_with_global_annotation_policy():
    frame = {"sensor_id": "A", "frame_id": "f1", "t": 1.0, "people": [], "exhaustive": False}
    with pytest.raises(ValueError, match="exhaustive"):
        evaluate([prediction(boxes=[])], annotations([frame], exhaustive=True))
