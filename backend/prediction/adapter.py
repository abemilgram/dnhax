"""Small, explicit adapter from a published ``macoslive`` scene to ``Camera``.

The live exporter stores camera poses in the reconstruction's
``segment_world`` coordinates.  This module deliberately keeps that context
visible while requiring the caller to provide the external metric similarity
that maps that segment into the prediction contract's X/Y-ground, Z-up frame.

This adapter is intentionally read-only.  A scene is accepted only when the
selected frame can be joined unambiguously through all three records that
``macoslive`` publishes: ``live.camera_locations``, ``live.frames`` and the
source cloud's camera export.  A non-identity cloud transform is rejected
until camera extrinsics and point-cloud transforms have a single supported
composition contract.
"""

from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit

import numpy as np

from backend.prediction.schema import Camera


_IDENTITY_ATOL = 1e-8
_POSE_ATOL = 1e-5
_SIMILARITY_RTOL = 1e-5


def _fail(message: str) -> ValueError:
    return ValueError(message)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise _fail(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{label} must be a finite number") from exc
    if not np.isfinite(result):
        raise _fail(f"{label} must be a finite number")
    return result


def _frame_number(value: Any, label: str) -> int:
    """Require the integer frame-ID representation used by ``live``."""

    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise _fail(f"{label} must be an integer frame ID")
    return int(value)


def _array(value: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise _fail(f"{label} must be a finite {shape} matrix") from exc
    if result.shape != shape or not np.isfinite(result).all():
        raise _fail(f"{label} must be a finite {shape} matrix")
    return result


def _is_rigid_pose(pose: np.ndarray, label: str) -> None:
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=_POSE_ATOL):
        raise _fail(f"{label} must be homogeneous")
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_POSE_ATOL) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=_POSE_ATOL
    ):
        raise _fail(f"{label} must contain a proper rigid rotation")


def _cloud_transform(cloud: dict[str, Any], index: int) -> None:
    value = cloud.get("transform", np.eye(4).tolist())
    transform = _array(value, (4, 4), f"cloud[{index}].transform")
    if not np.allclose(transform, np.eye(4), atol=_IDENTITY_ATOL):
        raise _fail(
            "Non-identity cloud.transform is unsupported; compose cloud and "
            "camera coordinates explicitly before metric calibration"
        )


def _normalize_frame_id_from_path(frame: Any, label: str) -> tuple[str, int]:
    if not isinstance(frame, str) or not frame:
        raise _fail(f"{label}.frame must be a non-empty artifact URL")
    path = unquote(urlsplit(frame).path)
    parts = PurePosixPath(path).parts
    if len(parts) < 3 or parts[-2] != "frames":
        raise _fail(f"{label}.frame must identify a source/frames/<frame_id>.jpg asset")
    source = parts[-3]
    filename = parts[-1]
    if not filename.lower().endswith(".jpg"):
        raise _fail(f"{label}.frame must use the exported .jpg frame filename")
    stem = filename[:-4]
    if not stem.isdecimal():
        raise _fail(f"{label}.frame has an invalid integer frame filename")
    return source, int(stem)


def _same_time(left: Any, right: Any, label: str) -> None:
    a = _finite_number(left, label)
    b = _finite_number(right, label)
    if not np.isclose(a, b, rtol=0, atol=1e-6):
        raise _fail(f"{label} does not match the selected frame metadata")


def _validate_source_and_frame_metadata(
    live: dict[str, Any], sensor_id: str, frame_id: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = live.get("frames")
    if not isinstance(frames, list) or not frames:
        raise _fail("live.frames is required to prove batch timing and frame identity")

    matching = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise _fail(f"live.frames[{index}] must be an object")
        current_id = _frame_number(frame.get("id"), f"live.frames[{index}].id")
        source = frame.get("source")
        if not isinstance(source, str) or not source:
            raise _fail(f"live.frames[{index}].source must be a non-empty string")
        # Every published batch frame contributes to the availability
        # watermark, so malformed timing anywhere in the batch is unsafe.
        _finite_number(frame.get("captured"), f"live.frames[{index}].captured")
        _finite_number(frame.get("received"), f"live.frames[{index}].received")
        if current_id == frame_id and source == sensor_id:
            matching.append(frame)
    if len(matching) == 0:
        raise _fail(
            f"No live.frames entry matches source={sensor_id!r}, frame_id={frame_id}"
        )
    if len(matching) > 1:
        raise _fail(
            f"Multiple live.frames entries match source={sensor_id!r}, frame_id={frame_id}"
        )

    frame_ids = live.get("frame_ids")
    if not isinstance(frame_ids, list):
        raise _fail("live.frame_ids is required to constrain the published batch")
    batch_ids = [_frame_number(value, "live.frame_ids entry") for value in frame_ids]
    if batch_ids.count(frame_id) != 1:
        raise _fail(
            f"frame_id={frame_id} is not uniquely present in the published batch"
        )
    return matching[0], frames


def _find_location_sample(
    live: dict[str, Any], frame: dict[str, Any], sensor_id: str, frame_id: int
) -> dict[str, Any]:
    locations = live.get("camera_locations")
    if not isinstance(locations, dict):
        raise _fail("live.camera_locations is required")
    if locations.get("coordinate_system") != "segment_world":
        raise _fail("live.camera_locations must use the macoslive segment_world coordinate system")
    if locations.get("units") != "arbitrary":
        raise _fail(
            "live.camera_locations units must remain explicitly arbitrary until metric similarity is supplied"
        )
    samples = locations.get("samples")
    if not isinstance(samples, list):
        raise _fail("live.camera_locations.samples is required")
    matching = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise _fail(f"live.camera_locations.samples[{index}] must be an object")
        current_id = _frame_number(
            sample.get("frame_id"), f"live.camera_locations.samples[{index}].frame_id"
        )
        source = sample.get("source")
        if not isinstance(source, str) or not source:
            raise _fail(f"live.camera_locations.samples[{index}].source must be a non-empty string")
        if current_id == frame_id and source == sensor_id:
            matching.append(sample)
    if len(matching) == 0:
        raise _fail(
            f"No camera location matches source={sensor_id!r}, frame_id={frame_id}"
        )
    if len(matching) > 1:
        raise _fail(
            f"Multiple camera locations match source={sensor_id!r}, frame_id={frame_id}"
        )
    sample = matching[0]
    for key in ("epoch", "seq"):
        if key in frame and key in sample and frame[key] != sample[key]:
            raise _fail(f"camera location {key} does not match live frame metadata")
    for key in ("captured", "received"):
        if key in frame and key in sample:
            _same_time(sample[key], frame[key], f"camera location {key}")

    pose = _array(sample.get("camera_to_world"), (4, 4), "camera location camera_to_world")
    _is_rigid_pose(pose, "camera location camera_to_world")
    position = _array(sample.get("position"), (3,), "camera location position")
    if not np.allclose(position, pose[:3, 3], atol=_POSE_ATOL):
        raise _fail("camera location position disagrees with camera_to_world translation")
    return sample


def _find_cloud_camera(
    clouds: list[Any], sensor_id: str, frame_id: int
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for cloud_index, cloud in enumerate(clouds):
        if not isinstance(cloud, dict):
            raise _fail(f"clouds[{cloud_index}] must be an object")
        _cloud_transform(cloud, cloud_index)
        source = cloud.get("source")
        if not isinstance(source, str) or not source:
            raise _fail(f"clouds[{cloud_index}].source must be a non-empty string")
        cameras = cloud.get("cameras")
        if not isinstance(cameras, list):
            raise _fail(f"clouds[{cloud_index}].cameras is required")
        for camera_index, camera in enumerate(cameras):
            if not isinstance(camera, dict):
                raise _fail(f"clouds[{cloud_index}].cameras[{camera_index}] must be an object")
            path_source, camera_frame_id = _normalize_frame_id_from_path(
                camera.get("frame"), f"clouds[{cloud_index}].cameras[{camera_index}]"
            )
            if path_source != source:
                raise _fail(
                    f"clouds[{cloud_index}].cameras[{camera_index}] frame source does not match its cloud"
                )
            if path_source == sensor_id and camera_frame_id == frame_id:
                candidates.append(camera)
    if len(candidates) == 0:
        raise _fail(
            f"No cloud camera filename matches source={sensor_id!r}, frame_id={frame_id}"
        )
    if len(candidates) > 1:
        raise _fail(
            f"Multiple cloud cameras match source={sensor_id!r}, frame_id={frame_id}"
        )
    return candidates[0]


def _validate_camera_pose(
    sample: dict[str, Any], cloud_camera: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    sample_pose = _array(sample.get("camera_to_world"), (4, 4), "camera location camera_to_world")
    world_to_camera = _array(
        cloud_camera.get("world_to_camera"), (3, 4), "cloud camera world_to_camera"
    )
    rotation = world_to_camera[:, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=_POSE_ATOL) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=_POSE_ATOL
    ):
        raise _fail("cloud camera world_to_camera must contain a proper rigid rotation")
    derived_pose = np.eye(4)
    derived_pose[:3, :3] = rotation.T
    derived_pose[:3, 3] = -rotation.T @ world_to_camera[:, 3]
    if not np.allclose(sample_pose, derived_pose, rtol=0, atol=_POSE_ATOL):
        raise _fail(
            "camera location pose disagrees with the matching cloud camera world_to_camera"
        )
    return sample_pose, world_to_camera


def _similarity_parts(value: Any) -> tuple[float, np.ndarray, np.ndarray]:
    matrix = _array(value, (4, 4), "similarity")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=_POSE_ATOL):
        raise _fail("similarity must be homogeneous")
    linear = matrix[:3, :3]
    determinant = float(np.linalg.det(linear))
    if determinant <= 0:
        raise _fail("similarity must have a positive isotropic scale and no reflection")
    scale = float(np.cbrt(determinant))
    if scale <= 0 or not np.isfinite(scale):
        raise _fail("similarity must have a positive isotropic scale")
    if not np.allclose(
        linear.T @ linear,
        (scale * scale) * np.eye(3),
        rtol=_SIMILARITY_RTOL,
        atol=1e-8,
    ):
        raise _fail("similarity must have one positive isotropic scale across all axes")
    rotation = linear / scale
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=_POSE_ATOL) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=_POSE_ATOL
    ):
        raise _fail("similarity rotation must be a proper orthonormal rotation")
    return scale, rotation, matrix[:3, 3]


def _availability_watermark(
    scene: dict[str, Any], frames: list[dict[str, Any]], samples: list[dict[str, Any]], origin: float
) -> float:
    created = _finite_number(scene.get("created"), "scene.created")
    watermark = created - origin
    for index, frame in enumerate(frames):
        for key in ("captured", "received"):
            watermark = max(
                watermark,
                _finite_number(frame[key], f"live.frames[{index}].{key}") - origin,
            )
    # Include location timestamps too.  They are expected to mirror frames,
    # and this conservative check prevents a malformed/future location from
    # being exposed through a past availability time.
    for index, sample in enumerate(samples):
        for key in ("captured", "received"):
            if key in sample:
                watermark = max(
                    watermark,
                    _finite_number(
                        sample[key], f"live.camera_locations.samples[{index}].{key}"
                    )
                    - origin,
                )
    return watermark


def camera_from_live_scene(
    scene: dict,
    *,
    frame_id: int,
    sensor_id: str,
    coordinate_frame: str,
    similarity: list[list[float]],
    calibration_id: str,
    calibration_provenance: str,
    expected_segment: int | str,
    width: int,
    height: int,
    available_t: float,
    position_std_m: float = 0.25,
    source_clock_origin_unix: float = 0.0,
) -> Camera:
    """Build a metric ``Camera`` from one published macoslive frame.

    ``available_t`` and the scene's event timestamps use one clock after
    normalization.  The normalization is ``raw - source_clock_origin_unix``;
    pass zero when the supplied timestamps are already Unix timestamps.  A
    source-relative clock must provide the explicit Unix origin used for that
    source.  The selected batch is considered knowable only when its scene
    publication time and every frame's capture/receipt time are at or before
    ``available_t``.

    ``similarity`` maps the arbitrary ``segment_world`` reconstruction into
    the caller's explicitly named metric ``coordinate_frame``.  Its linear
    part must be ``scale * proper_rotation``.  Scale is applied to the camera
    position only; camera orientation remains orthonormal.
    """

    if not isinstance(scene, dict):
        raise _fail("scene must be an object")
    if isinstance(sensor_id, str) is False or not sensor_id.strip():
        raise _fail("sensor_id must be a non-empty string")
    if isinstance(coordinate_frame, str) is False or not coordinate_frame.strip():
        raise _fail("coordinate_frame must be a non-empty string")
    if isinstance(expected_segment, bool) or not isinstance(expected_segment, (int, str)):
        raise _fail("expected_segment must be an integer or string")
    normalized_frame_id = _frame_number(frame_id, "frame_id")
    origin = _finite_number(source_clock_origin_unix, "source_clock_origin_unix")
    availability = _finite_number(available_t, "available_t")
    if availability < 0:
        raise _fail("available_t must be non-negative in the normalized source clock")

    live = scene.get("live")
    if not isinstance(live, dict):
        raise _fail("scene.live is required for a macoslive camera adapter")
    actual_segment = live.get("segment")
    if isinstance(actual_segment, bool) or not isinstance(actual_segment, (int, str)):
        raise _fail("live.segment must be an integer or string")
    if actual_segment != expected_segment:
        raise _fail(
            f"Published segment {actual_segment!r} does not match expected_segment {expected_segment!r}"
        )

    frame, all_frames = _validate_source_and_frame_metadata(
        live, sensor_id, normalized_frame_id
    )
    sample = _find_location_sample(live, frame, sensor_id, normalized_frame_id)
    all_samples = live["camera_locations"]["samples"]
    cloud_list = scene.get("clouds")
    if not isinstance(cloud_list, list) or not cloud_list:
        raise _fail("scene.clouds is required")
    cloud_camera = _find_cloud_camera(cloud_list, sensor_id, normalized_frame_id)

    if "t" in cloud_camera:
        _same_time(cloud_camera["t"], frame["captured"], "cloud camera t")
    sample_pose, _ = _validate_camera_pose(sample, cloud_camera)
    intrinsics = _array(cloud_camera.get("intrinsics"), (3, 3), "cloud camera intrinsics")

    scale, alignment_rotation, translation = _similarity_parts(similarity)
    watermark = _availability_watermark(scene, all_frames, all_samples, origin)
    if availability < watermark:
        raise _fail(
            "available_t predates the completed batch watermark "
            f"({watermark:g}) after source-clock normalization"
        )

    metric_pose = np.eye(4)
    metric_pose[:3, :3] = alignment_rotation @ sample_pose[:3, :3]
    metric_pose[:3, 3] = scale * (alignment_rotation @ sample_pose[:3, 3]) + translation

    # Camera performs the final schema-level validation, including dimensions,
    # finite calibration values, and rigid pose invariants.
    return Camera(
        sensor_id=sensor_id,
        coordinate_frame=coordinate_frame,
        calibration_id=calibration_id,
        calibration_provenance=calibration_provenance,
        width=width,
        height=height,
        intrinsics=intrinsics.tolist(),
        camera_to_world=metric_pose.tolist(),
        position_std_m=position_std_m,
        available_t=availability,
    )


__all__ = ["camera_from_live_scene"]
