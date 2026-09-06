
import pytest

from backend.prediction.hlae_alignment import align_cam_frames
from backend.prediction.hlae_camera import VERIFIED_SOURCE_TAG, parse_cam

def cam_result(*, count=2, sensor_id="sensor-a", source_tag=VERIFIED_SOURCE_TAG, allow_unverified=False):
    rows = [f"{100 + index} {index} 0 0 0 0 0 90" for index in range(count)]
    text = "\n".join([
        "advancedfx Cam",
        "version 2",
        "channels time xPosition yPosition zPosition xRotation yRotation zRotation fov",
        "DATA",
        *rows,
    ]) + "\n"
    return parse_cam(
        text,
        origin_game_time=100,
        units_to_meters=1,
        origin_xyz=[0, 0, 0],
        width=640,
        height=480,
        available_t=0,
        calibration_id="cal-hlae",
        calibration_provenance="independent camera calibration",
        coordinate_frame="source-room-m",
        sensor_id=sensor_id,
        source_tag=source_tag,
        allow_unverified=allow_unverified,
    )


def media(*, sensor_id="sensor-a", frames=None, source_frame_count=2, native_capture_available=False):
    if frames is None:
        frames = [
            {"frame_id": "sensor-a-f0", "sensor_id": sensor_id, "image": "sensor-a/frame-000.jpg", "original_frame_index": 0, "t": 4.25},
            {"frame_id": "sensor-a-f1", "sensor_id": sensor_id, "image": "sensor-a/frame-001.jpg", "original_frame_index": 1, "t": 4.50},
        ]
    return {
        "sensor_id": sensor_id,
        "source_frame_count": source_frame_count,
        "native_capture_available": native_capture_available,
        "frames": frames,
    }


def test_verified_mapping_uses_native_game_time_and_audits_original_video_time():
    result = align_cam_frames(
        cam_result(),
        media(),
        row_to_frame=[0, 1],
        mapping_provenance="manifest row order verified against extraction indices",
        mapping_verified=True,
        image_sensor_id="sensor-a",
    )
    assert result.usable is True
    assert [frame["t"] for frame in result] == pytest.approx([0, 1])
    assert [frame["cam_row_index"] for frame in result] == [0, 1]
    assert [frame["native_source_time"] for frame in result] == pytest.approx([100, 101])
    assert [frame["video_t"] for frame in result] == pytest.approx([4.25, 4.50])
    assert [pair["original_video_t"] for pair in result.audit["pairs"]] == pytest.approx([4.25, 4.50])
    assert result.audit["synchronization_verified"] is False
    assert result.audit["metric_scale_proven"] is False
    assert [frame["camera"].available_t for frame in result] == pytest.approx([0, 0])


def test_native_capture_telemetry_updates_available_t_without_using_video_pts():
    result = align_cam_frames(
        cam_result(),
        media(native_capture_available=True),
        row_to_frame={0: 0, 1: 1},
        mapping_provenance="native capture manifest independently checked",
        mapping_verified=True,
        image_sensor_id="sensor-a",
    )
    assert [frame["camera"].available_t for frame in result] == pytest.approx([0, 1])
    assert [frame["video_t"] for frame in result] == pytest.approx([4.25, 4.50])
    assert result.audit["camera_available_t_policy"].startswith("updated to normalized native game t")


def test_unverified_or_missing_mapping_returns_candidate_with_no_usable_frames():
    result = align_cam_frames(
        cam_result(),
        media(),
        row_to_frame=[0, 1],
        mapping_provenance="operator supplied mapping pending review",
        mapping_verified=False,
        image_sensor_id="sensor-a",
    )
    assert result.frames == []
    assert result.audit["status"] == "candidate"
    assert result.audit["synchronization_verified"] is False
    assert "caller has not verified" in result.audit["candidate_reasons"][-1]

    no_mapping = align_cam_frames(cam_result(), media(), image_sensor_id="sensor-a")
    assert no_mapping.frames == []
    assert "row-to-frame mapping was not supplied" in no_mapping.audit["candidate_reasons"]


@pytest.mark.parametrize(
    ("cam_count", "bad_media", "mapping", "pattern"),
    [
        (3, media(frames=[{"frame_id": "a", "sensor_id": "sensor-a", "original_frame_index": 0, "t": 1.0}], source_frame_count=2), [0, 1], "counts differ"),
        (2, media(frames=[{"frame_id": "a", "sensor_id": "sensor-a", "original_frame_index": 2, "t": 1.0}, {"frame_id": "b", "sensor_id": "sensor-a", "original_frame_index": 1, "t": 2.0}], source_frame_count=2), [0, 1], "out of range"),
        (2, media(frames=[{"frame_id": "a", "sensor_id": "sensor-a", "original_frame_index": 0, "t": 1.0}, {"frame_id": "b", "sensor_id": "sensor-a", "original_frame_index": 0, "t": 2.0}], source_frame_count=2), [0, 1], "duplicate"),
    ],
)
def test_invalid_counts_and_indices_are_refused(cam_count, bad_media, mapping, pattern):
    with pytest.raises(ValueError, match=pattern):
        align_cam_frames(
            cam_result(count=cam_count),
            bad_media,
            row_to_frame=mapping,
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )


def test_mapping_must_cover_every_native_row_and_video_time_is_required():
    with pytest.raises(ValueError, match="out of range"):
        align_cam_frames(
            cam_result(),
            media(),
            row_to_frame=[0, 2],
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )

    with pytest.raises(ValueError, match="mapping keys"):
        align_cam_frames(
            cam_result(),
            media(),
            row_to_frame={"0": 0, "1": 1},
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )


def test_full_native_mapping_can_return_only_sampled_media_frames():
    sampled = media(
        source_frame_count=30,
        frames=[
            {"frame_id": "sensor-a-f0", "sensor_id": "sensor-a", "image": "sensor-a/frame-000.jpg", "original_frame_index": 0, "t": 12.0},
            {"frame_id": "sensor-a-f15", "sensor_id": "sensor-a", "image": "sensor-a/frame-015.jpg", "original_frame_index": 15, "t": 12.5},
        ],
    )
    result = align_cam_frames(
        cam_result(count=30),
        sampled,
        row_to_frame=list(range(30)),
        mapping_provenance="full sidecar row order verified against native source indices",
        mapping_verified=True,
        image_sensor_id="sensor-a",
    )
    assert len(result) == 2
    assert [frame["original_frame_index"] for frame in result] == [0, 15]
    assert [frame["t"] for frame in result] == pytest.approx([0, 15])
    assert [frame["cam_row_index"] for frame in result] == [0, 15]
    assert [pair["row_index"] for pair in result.audit["pairs"]] == [0, 15]
    assert result.audit["camera_row_count"] == 30
    assert result.audit["extracted_frame_count"] == 2


def test_mapping_values_must_be_in_native_source_range():
    with pytest.raises(ValueError, match="out of range"):
        align_cam_frames(
            cam_result(),
            media(),
            row_to_frame=[0, 2],
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )


def test_mapping_keys_must_be_integer_and_cover_all_native_rows():
    with pytest.raises(ValueError, match="mapping keys"):
        align_cam_frames(
            cam_result(),
            media(),
            row_to_frame={"0": 0, "1": 1},
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )


def test_missing_video_time_is_refused():
    missing_time = media(frames=[
        {"frame_id": "sensor-a-f0", "sensor_id": "sensor-a", "original_frame_index": 0, "t": None},
        {"frame_id": "sensor-a-f1", "sensor_id": "sensor-a", "original_frame_index": 1, "t": 4.5},
    ])
    with pytest.raises(ValueError, match="native video time"):
        align_cam_frames(
            cam_result(),
            missing_time,
            row_to_frame=[0, 1],
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )


def test_disabled_source_and_wrong_sensor_are_refused():
    disabled = cam_result(source_tag="unverified-build", allow_unverified=True)
    with pytest.raises(ValueError, match="disabled"):
        align_cam_frames(
            disabled,
            media(),
            row_to_frame=[0, 1],
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )
    with pytest.raises(ValueError, match="Image sensor"):
        align_cam_frames(
            cam_result(),
            media(sensor_id="other-sensor"),
            row_to_frame=[0, 1],
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )


def test_native_capture_flag_does_not_accept_negative_normalized_time():
    negative = cam_result()
    negative[0]["t"] = -0.5
    with pytest.raises(ValueError, match="nonnegative normalized game time"):
        align_cam_frames(
            negative,
            media(native_capture_available=True),
            row_to_frame=[0, 1],
            mapping_provenance="independently checked source index mapping",
            mapping_verified=True,
            image_sensor_id="sensor-a",
        )
