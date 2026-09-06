import json

import numpy as np
import pytest

from backend.prediction.hlae_camera import VERIFIED_SOURCE_TAG, fov_to_intrinsics, parse_cam, parse_cam_document, source_angles_to_c2w

def cam_text(*rows, version=2, scale_fov=None):
    header = [
        "advancedfx Cam",
        f"version {version}",
    ]
    if scale_fov is not None:
        header.append(f"scaleFov {scale_fov}")
    header.extend(["channels time xPosition yPosition zPosition xRotation yRotation zRotation fov", "DATA"])
    return "\n".join(header + [" ".join(str(value) for value in row) for row in rows]) + "\n"


def parse(text, **extra):
    values = {
        "origin_game_time": 100.0,
        "units_to_meters": 0.0254,
        "origin_xyz": [10.0, 20.0, 30.0],
        "width": 1920,
        "height": 1080,
        "available_t": 100.0,
        "calibration_id": "cal-hlae",
        "calibration_provenance": "tagged checkerboard camera calibration",
        "coordinate_frame": "source-room-m",
        "source_tag": VERIFIED_SOURCE_TAG,
    }
    values.update(extra)
    return parse_cam(text, **values)


def test_verified_format_timestamp_units_pose_and_fov_matrix():
    rows = parse(cam_text((2, 1, 2, 3, 0, 0, 0, 90)))
    assert len(rows) == 1
    row = rows[0]
    # The importer normalizes the Source clock by subtraction and preserves
    # the original timestamp separately; no row-index/MP4 alignment is made.
    assert row["t"] == pytest.approx(-98)
    assert row["source_time"] == pytest.approx(2)
    assert np.allclose(row["position_m"], [10.0254, 20.0508, 30.0762])
    assert row["enabled"] is True
    assert row["source_tag_status"] == "verified"
    k = np.asarray(row["intrinsics"])
    expected = fov_to_intrinsics(90, 1920, 1080)
    assert np.allclose(k, expected)
    assert np.allclose(row["camera"].camera_to_world, row["camera_to_world"])


def test_source_cardinal_angles_have_expected_opencv_basis_and_no_reflection():
    zero = source_angles_to_c2w(0, 0, 0)
    assert np.allclose(zero, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    yaw_left = source_angles_to_c2w(0, 0, 90)
    assert np.allclose(yaw_left[:, 2], [0, 1, 0])
    for angles in [(0, 0, 0), (20, -30, 45), (90, 15, -120), (-180, 89, 180)]:
        rotation = source_angles_to_c2w(*angles)
        assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
        assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_verified_tag_is_required_and_unverified_is_candidate_disabled():
    text = cam_text((0, 0, 0, 0, 0, 0, 0, 90))
    with pytest.raises(ValueError, match="Unverified HLAE source tag"):
        parse_cam(
            text,
            origin_game_time=0,
            units_to_meters=1,
            origin_xyz=[0, 0, 0],
            width=1280,
            height=720,
            available_t=0,
            calibration_id="cal",
            calibration_provenance="provided external calibration",
            coordinate_frame="room",
        )
    result = parse(text, source_tag="unknown-build", allow_unverified=True)
    assert result.metadata["status"] == "candidate"
    assert result.enabled is False
    assert result[0]["enabled"] is False


def test_header_and_rows_are_strict_and_not_silently_dropped():
    with pytest.raises(ValueError, match="channel order"):
        parse(cam_text((0, 0, 0, 0, 0, 0, 0, 90)).replace("zRotation fov", "fov zRotation"))
    with pytest.raises(ValueError, match="expected 8"):
        parse(cam_text((0, 0, 0, 0, 0, 0, 0)))
    with pytest.raises(ValueError, match="nondecreasing"):
        parse(cam_text((2, 0, 0, 0, 0, 0, 0, 90), (1, 0, 0, 0, 0, 0, 0, 90)))


def test_explicit_none_scaling_is_converted_and_dimensions_are_required():
    text = cam_text((0, 0, 0, 0, 0, 0, 0, 90), scale_fov="none")
    with pytest.raises(ValueError, match="scaleFov none"):
        parse(text)
    result = parse(text, allow_unverified=True)
    expected_fov = 2 * np.rad2deg(np.arctan((1920 / 1080) / (4 / 3) * np.tan(np.deg2rad(90) / 2)))
    assert result[0]["fov_degrees"] == pytest.approx(expected_fov)
    assert result.metadata["fov_scaling"] == "none"
    assert result.metadata["status"] == "candidate"
    assert result.enabled is False
    assert "legacy scaleFov none conversion is not enabled" in result.metadata["candidate_reasons"]


def test_document_is_json_friendly_and_retains_camera_fields():
    document = parse_cam_document(
        cam_text((2, 1, 2, 3, 0, 0, 0, 90)),
        origin_game_time=100.0,
        units_to_meters=0.0254,
        origin_xyz=[10.0, 20.0, 30.0],
        width=1920,
        height=1080,
        available_t=100.0,
        calibration_id="cal-hlae",
        calibration_provenance="tagged checkerboard camera calibration",
        coordinate_frame="source-room-m",
        source_tag=VERIFIED_SOURCE_TAG,
    )
    encoded = json.dumps(document)
    assert encoded
    assert isinstance(document["frames"][0]["camera"], dict)
    assert document["frames"][0]["camera"]["sensor_id"] == "hlae-source2"
