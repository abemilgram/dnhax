import numpy as np
import pytest
from backend.prediction.adapter import camera_from_live_scene

def _scene(*, segment="segment-1", scene_created=100.0, cloud_transform=None):
    center = np.array([1.0, 2.0, 3.0])
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    world_to_camera = np.column_stack((rotation, -rotation @ center))
    pose = np.eye(4)
    pose[:3, :3] = rotation.T
    pose[:3, 3] = center
    intrinsics = [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]]
    frame = {
        "id": 7,
        "source": "A",
        "epoch": "epoch-a",
        "seq": 4,
        "captured": 10.0,
        "received": 11.0,
    }
    sample = {
        "frame_id": 7,
        "source": "A",
        "epoch": "epoch-a",
        "seq": 4,
        "captured": 10.0,
        "received": 11.0,
        "position": center.tolist(),
        "camera_to_world": pose.tolist(),
    }
    cloud = {
        "source": "A",
        "transform": cloud_transform or np.eye(4).tolist(),
        "cameras": [
            {
                "frame": "/api/artifacts/scenes/s/A/frames/7.jpg",
                "t": 10.0,
                "intrinsics": intrinsics,
                "world_to_camera": world_to_camera.tolist(),
            }
        ],
    }
    return {
        "id": "scene-1",
        "created": scene_created,
        "clouds": [cloud],
        "live": {
            "segment": segment,
            "frame_ids": [7, 8],
            "frames": [
                frame,
                {
                    "id": 8,
                    "source": "B",
                    "epoch": "epoch-b",
                    "seq": 1,
                    "captured": 12.0,
                    "received": 13.0,
                },
            ],
            "camera_locations": {
                "coordinate_system": "segment_world",
                "units": "arbitrary",
                "samples": [sample],
            },
        },
    }


def _call(scene, **overrides):
    args = {
        "frame_id": 7,
        "sensor_id": "A",
        "coordinate_frame": "metric_xyground_zup",
        "similarity": np.eye(4).tolist(),
        "calibration_id": "rig-v1",
        "calibration_provenance": "Measured rig scale and axes",
        "expected_segment": "segment-1",
        "width": 640,
        "height": 480,
        "available_t": 100.0,
    }
    args.update(overrides)
    return camera_from_live_scene(scene, **args)


def test_metric_similarity_scales_position_and_rotates_axes():
    angle = 0.31
    q = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    similarity = np.eye(4)
    similarity[:3, :3] = 2.5 * q
    similarity[:3, 3] = [10.0, 20.0, 30.0]
    camera = _call(_scene(), similarity=similarity.tolist())
    expected_position = 2.5 * q @ np.array([1.0, 2.0, 3.0]) + similarity[:3, 3]
    np.testing.assert_allclose(np.array(camera.camera_to_world)[:3, 3], expected_position)
    np.testing.assert_allclose(
        np.array(camera.camera_to_world)[:3, :3],
        q @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]),
    )
    np.testing.assert_allclose(
        np.array(camera.camera_to_world)[:3, :3].T
        @ np.array(camera.camera_to_world)[:3, :3],
        np.eye(3),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda s: s["live"].update(segment="other"), "segment"),
        (lambda s: s["live"].update(frame_ids=[8]), "frame_id"),
        (
            lambda s: s["clouds"][0]["cameras"][0].update(
                frame="/api/artifacts/scenes/s/A/frames/8.jpg"
            ),
            "frame",
        ),
    ],
)
def test_wrong_segment_or_unknown_frame_fails(mutator, message):
    scene = _scene()
    mutator(scene)
    with pytest.raises(ValueError, match=message):
        _call(scene)


def test_nonpositive_and_anisotropic_scale_fail():
    for linear in (
        np.diag([-1.0, 1.0, 1.0]),
        np.diag([2.0, 1.0, 1.0]),
    ):
        similarity = np.eye(4)
        similarity[:3, :3] = linear
        with pytest.raises(ValueError, match="similarity"):
            _call(_scene(), similarity=similarity.tolist())


def test_too_early_calibration_and_nonidentity_cloud_transform_fail():
    with pytest.raises(ValueError, match="watermark"):
        _call(_scene(), available_t=99.0)
    transform = np.eye(4)
    transform[0, 3] = 1.0
    scene = _scene(cloud_transform=transform.tolist())
    with pytest.raises(ValueError, match="cloud.transform"):
        _call(scene)


def test_source_clock_origin_normalizes_scene_and_batch_times():
    scene = _scene(scene_created=1_700_000_100.0)
    camera = _call(
        scene,
        source_clock_origin_unix=1_700_000_000.0,
        available_t=100.0,
    )
    assert camera.available_t == 100.0


def test_duplicate_matching_location_or_cloud_camera_is_rejected():
    scene = _scene()
    scene["live"]["camera_locations"]["samples"].append(
        dict(scene["live"]["camera_locations"]["samples"][0])
    )
    with pytest.raises(ValueError, match="Multiple camera locations"):
        _call(scene)

    scene = _scene()
    scene["clouds"][0]["cameras"].append(dict(scene["clouds"][0]["cameras"][0]))
    with pytest.raises(ValueError, match="Multiple cloud cameras"):
        _call(scene)
