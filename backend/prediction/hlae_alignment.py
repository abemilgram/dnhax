"""Explicit alignment of parsed HLAE camera rows with extracted images.

This module joins two already-created records:

* ``cam_result`` is a :class:`CamImportResult` (or an equivalent sequence)
  whose rows carry normalized native game time ``t`` and a typed ``camera``.
* ``media`` is extracted-image metadata with a ``frames`` list.  Each image
  frame must carry its original source frame index and its original video PTS
  as ``original_frame_index`` and ``t``.

The caller supplies a row-order-to-``original_frame_index`` mapping and must
explicitly verify that mapping.  The function never infers an offset from
video PTS, frame rate, filename, or row position.  A verified mapping makes
the paired frames usable for downstream processing, but it does not certify
camera/video synchronization.  The original video PTS is retained in each
output row and in the audit table.

``camera.available_t`` is updated only when the caller or media record marks
the image as having native capture-available telemetry.  In that case the
value is the nonnegative, origin-subtracted native game time from the CAM row;
video PTS is never used to reconstruct availability or future data.

No metric scale is estimated here.  The camera importer records the supplied
Source2-to-meter scale and its provenance; this join preserves that audit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np


def _record(value: Any, name: str) -> Mapping[str, Any]:
    """Normalize dict-like metadata and Pydantic/dataclass records."""

    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    if hasattr(value, "__dict__"):
        dumped = vars(value)
        if isinstance(dumped, Mapping):
            return dumped
    raise TypeError(f"{name} must be a metadata mapping or record")


def _as_mapping(value: Any, name: str) -> Mapping[str, Any]:
    return _record(value, name)


def _finite_number(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _required_frame_list(media: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = media.get("frames")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not raw:
        raise ValueError("Extracted media metadata needs a nonempty frames list")
    result: list[Mapping[str, Any]] = []
    for number, frame in enumerate(raw):
        result.append(_record(frame, f"Media frame {number}"))
    return result


def _source_frame_count(media: Mapping[str, Any]) -> int:
    value = media.get("source_frame_count")
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
        raise ValueError("source_frame_count must be a positive integer")
    return int(value)


def _original_indices(media_frames: list[Mapping[str, Any]], source_count: int) -> list[int]:
    indices: list[int] = []
    for number, frame in enumerate(media_frames):
        value = frame.get("original_frame_index")
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Media frame {number} lacks an integer original_frame_index")
        index = int(value)
        if not 0 <= index < source_count:
            raise ValueError(f"Media frame {number} original_frame_index is out of range")
        indices.append(index)
    if len(indices) != len(set(indices)):
        raise ValueError("Extracted media contains duplicate original_frame_index values")
    return indices


def _cam_frames_and_metadata(cam_result: Any) -> tuple[list[Mapping[str, Any]], Mapping[str, Any], bool]:
    if isinstance(cam_result, Mapping):
        raw_frames = cam_result.get("frames")
        metadata = cam_result.get("metadata", {})
        enabled = bool(cam_result.get("enabled", metadata.get("enabled", False))) if isinstance(metadata, Mapping) else bool(cam_result.get("enabled", False))
    else:
        try:
            raw_frames = list(cam_result)
        except TypeError as exc:
            raise TypeError("cam_result must be a parsed camera result") from exc
        metadata = getattr(cam_result, "metadata", {})
        enabled = bool(getattr(cam_result, "enabled", metadata.get("enabled", False) if isinstance(metadata, Mapping) else False))
    if not isinstance(raw_frames, Sequence) or isinstance(raw_frames, (str, bytes)) or not raw_frames:
        raise ValueError("Parsed CAM result needs a nonempty frames list")
    if not isinstance(metadata, Mapping):
        raise TypeError("Parsed CAM metadata must be a mapping")
    frames: list[Mapping[str, Any]] = []
    for number, frame in enumerate(raw_frames):
        frames.append(_record(frame, f"Parsed CAM frame {number}"))
    return frames, metadata, enabled


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _validated_mapping(mapping: Any, row_count: int, source_count: int) -> list[int]:
    if isinstance(mapping, Mapping):
        expected = set(range(row_count))
        if any(isinstance(key, bool) or not isinstance(key, (int, np.integer)) for key in mapping):
            raise ValueError("row_to_frame mapping keys must be integer row indices")
        try:
            keys = {int(key) for key in mapping}
        except (TypeError, ValueError) as exc:
            raise ValueError("row_to_frame mapping keys must be row integers") from exc
        if keys != expected or len(keys) != len(mapping):
            raise ValueError("row_to_frame mapping must contain every camera row exactly once")
        ordered = [mapping[index] for index in range(row_count)]
    else:
        if isinstance(mapping, (str, bytes)) or not isinstance(mapping, Sequence):
            raise TypeError("row_to_frame must be a sequence or row-index mapping")
        if len(mapping) != row_count:
            raise ValueError("Camera row and mapping counts differ")
        ordered = list(mapping)
    result: list[int] = []
    for row_number, value in enumerate(ordered):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"Mapping entry {row_number} must be an integer original_frame_index")
        index = int(value)
        if not 0 <= index < source_count:
            raise ValueError(f"Mapping entry {row_number} is out of range")
        result.append(index)
    if len(result) != len(set(result)):
        raise ValueError("row_to_frame mapping contains duplicate original_frame_index values")
    # A full native sidecar has one row per source frame.  Length + uniqueness
    # + bounds proves that the mapping is a permutation of every source index,
    # without allocating a potentially huge set(range(source_count)).
    if len(result) != source_count or len(result) != row_count:
        raise ValueError("Mapping does not cover every native camera row/source index")
    return result


def _camera_with_available_t(camera: Any, available_t: float) -> Any:
    """Rebuild a camera through its public model/dataclass representation."""

    value = _finite_number(available_t, "normalized capture time")
    if value < 0:
        raise ValueError("Native capture availability requires nonnegative normalized game time")
    if isinstance(camera, Mapping):
        result = dict(camera)
        result["available_t"] = value
        return result
    model_dump = getattr(camera, "model_dump", None)
    if callable(model_dump):
        data = dict(model_dump())
        data["available_t"] = value
        try:
            return type(camera)(**data)
        except TypeError:
            copy = getattr(camera, "model_copy", None)
            if callable(copy):
                return copy(update={"available_t": value})
            raise
    try:
        return replace(camera, available_t=value)
    except TypeError as exc:
        raise TypeError("Camera must expose model_dump/model_copy or be a dataclass") from exc


@dataclass(frozen=True)
class AlignmentResult:
    """Aligned image/camera rows and an explicit audit record.

    ``frames`` is empty whenever mapping evidence is insufficient.  A result
    with usable frames still has ``audit['synchronization_verified'] == False``
    because row/frame correspondence alone does not prove clock alignment.
    """

    frames: list[dict[str, Any]]
    audit: dict[str, Any]

    @property
    def usable_frames(self) -> list[dict[str, Any]]:
        return self.frames

    @property
    def usable(self) -> bool:
        return bool(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, item: Any) -> Any:
        return self.frames[item]

    def __iter__(self):
        return iter(self.frames)


def align_cam_frames(
    cam_result: Any,
    media: Mapping[str, Any],
    *,
    row_to_frame: Sequence[int] | Mapping[int, int] | None = None,
    mapping_provenance: str | None = None,
    mapping_verified: bool = False,
    image_sensor_id: str | None = None,
    native_capture_available: bool | None = None,
) -> AlignmentResult:
    """Join CAM rows to extracted media frames using an explicit mapping.

    ``row_to_frame[i]`` is the *original source* index for CAM row ``i``.  The
    mapping is a one-to-one permutation of the complete native CAM sidecar;
    extracted media may be a sampled subset of those indices.  The
    ``mapping_verified`` flag is a caller assertion about that mapping;
    it is never inferred from matching counts or timestamps.  If it is false,
    the function returns a candidate audit with no usable frames.

    ``media['frames'][j]['t']`` is the original video time and is preserved as
    ``video_t``/``original_video_t``.  The output frame ``t`` comes only from
    the parsed CAM row's origin-subtracted native game time; the original CAM
    row index and source timestamp are retained as ``cam_row_index`` and
    ``native_source_time``.  No offset is estimated and no synchronization
    certificate is emitted.
    """

    media_map = _as_mapping(media, "media")
    cam_frames, cam_metadata, source_enabled = _cam_frames_and_metadata(cam_result)
    if not source_enabled:
        raise ValueError("Cannot align disabled or unverified HLAE camera source format")

    media_frames = _required_frame_list(media_map)
    source_count = _source_frame_count(media_map)
    media_indices = _original_indices(media_frames, source_count)
    row_count = len(cam_frames)
    if row_count != source_count:
        raise ValueError("CAM rows and source_frame_count counts differ")

    media_sensor = media_map.get("sensor_id")
    if media_sensor is not None and not isinstance(media_sensor, str):
        raise ValueError("Media sensor_id must be a string")
    frame_sensors: list[str] = []
    for number, frame in enumerate(media_frames):
        sensor = frame.get("sensor_id", media_sensor)
        if not isinstance(sensor, str) or not sensor:
            raise ValueError(f"Media frame {number} has no image sensor_id")
        frame_sensors.append(sensor)
    if media_sensor is not None and any(sensor != media_sensor for sensor in frame_sensors):
        raise ValueError("Media frame sensor differs from media sensor_id")
    expected_sensor = image_sensor_id if image_sensor_id is not None else media_sensor
    if expected_sensor is None:
        if any(sensor != frame_sensors[0] for sensor in frame_sensors[1:]):
            raise ValueError("Extracted media frames contain more than one image sensor")
        expected_sensor = frame_sensors[0]
    if expected_sensor is not None and (not isinstance(expected_sensor, str) or not expected_sensor):
        raise ValueError("image_sensor_id must be a nonempty string")
    if expected_sensor is not None and any(sensor != expected_sensor for sensor in frame_sensors):
        raise ValueError("Image sensor does not match requested camera sensor")

    cameras: list[Any] = []
    for number, row in enumerate(cam_frames):
        camera = row.get("camera")
        if camera is None:
            raise ValueError(f"CAM row {number} has no camera model")
        camera_sensor = _field(camera, "sensor_id")
        if expected_sensor is not None and camera_sensor != expected_sensor:
            raise ValueError("Image sensor does not match parsed camera sensor")
        if row.get("enabled", True) is not True:
            raise ValueError(f"CAM row {number} is disabled")
        # ``t`` is the shared normalized native game time. Do not substitute
        # video PTS when a parser omitted it.
        _finite_number(row.get("t"), f"CAM row {number} native time")
        if row.get("source_time") is None:
            raise ValueError(f"CAM row {number} lacks native source time")
        _finite_number(row.get("source_time"), f"CAM row {number} source time")
        cameras.append(camera)

    # Native image PTS is required even when no synchronization claim is made;
    # it is retained in the audit so later review cannot lose source timing.
    video_times: list[float] = []
    for number, frame in enumerate(media_frames):
        if "t" not in frame or frame.get("t") is None:
            raise ValueError(f"Media frame {number} lacks native video time")
        video_times.append(_finite_number(frame.get("t"), f"Media frame {number} native video time"))

    mapping_reasons: list[str] = []
    if row_to_frame is None:
        mapping: list[int] | None = None
        mapping_reasons.append("row-to-frame mapping was not supplied")
    else:
        mapping = _validated_mapping(row_to_frame, row_count, source_count)

    provenance = mapping_provenance.strip() if isinstance(mapping_provenance, str) else ""
    if not provenance:
        mapping_reasons.append("mapping provenance is missing")
    if not mapping_verified:
        mapping_reasons.append("caller has not verified row-to-frame correspondence")
    if mapping_verified and not provenance:
        raise ValueError("Verified row-to-frame mapping requires explicit provenance")
    if mapping_verified and mapping is None:
        raise ValueError("mapping_verified=True requires a row-to-frame mapping")

    native_default = media_map.get("native_capture_available", False) if native_capture_available is None else native_capture_available
    if not isinstance(native_default, (bool, np.bool_)):
        raise ValueError("native_capture_available must be boolean")
    native_default = bool(native_default)

    origin_game_time = cam_metadata.get("origin_game_time")
    time_normalization = cam_metadata.get("time_normalization")
    if origin_game_time is None or not isinstance(time_normalization, str) or "source_time - origin_game_time" not in time_normalization:
        raise ValueError("CAM metadata lacks the required origin-subtracted native time provenance")

    pairs: list[dict[str, Any]] = []
    aligned: list[dict[str, Any]] = []
    if mapping is not None:
        row_by_original_index = {original_index: row_number for row_number, original_index in enumerate(mapping)}
        for media_position, (media_frame, original_index) in enumerate(zip(media_frames, media_indices)):
            row_number = row_by_original_index[original_index]
            row = cam_frames[row_number]
            camera = cameras[row_number]
            native_t = _finite_number(row["t"], f"CAM row {row_number} native time")
            video_t = video_times[media_position]
            frame_native_flag = media_frame.get("native_capture_available", native_default)
            if not isinstance(frame_native_flag, (bool, np.bool_)):
                raise ValueError(f"Media frame {media_position} native_capture_available must be boolean")
            frame_native_flag = bool(frame_native_flag)
            output_camera = _camera_with_available_t(camera, native_t) if frame_native_flag else camera
            output = dict(media_frame)
            output.update({
                "t": native_t,
                "video_t": video_t,
                "original_video_t": video_t,
                "original_frame_index": original_index,
                "cam_row_index": row_number,
                "native_source_time": _finite_number(row["source_time"], f"CAM row {row_number} source time"),
                "native_game_t": native_t,
                "camera": output_camera,
                "native_capture_available": frame_native_flag,
                "camera_available_t_from_native_capture": frame_native_flag,
            })
            aligned.append(output)
            pairs.append({
                "row_index": row_number,
                "media_position": media_position,
                "original_frame_index": original_index,
                "source_time": _finite_number(row.get("source_time"), f"CAM row {row_number} source time") if row.get("source_time") is not None else None,
                "native_game_t": native_t,
                "original_video_t": video_t,
                "frame_id": media_frame.get("frame_id"),
                "native_capture_available": frame_native_flag,
            })

    usable = bool(mapping_verified and mapping is not None)
    if not usable:
        aligned = []
    audit = {
        "status": "usable_mapping_no_sync_certificate" if usable else "candidate",
        "usable": usable,
        "camera_row_count": row_count,
        "extracted_frame_count": len(media_frames),
        "source_frame_count": source_count,
        "mapping_count": 0 if mapping is None else len(mapping),
        "mapping_verified": bool(mapping_verified),
        "mapping_coverage": "all native CAM rows and source indices",
        "mapping_provenance": provenance or None,
        "candidate_reasons": mapping_reasons,
        "synchronization_verified": False,
        "synchronization_status": "not_certified; row/frame mapping does not establish a clock offset",
        "time_basis": "output frame t is parsed origin-subtracted native game time; original_video_t is retained for audit",
        "origin_game_time": _finite_number(origin_game_time, "origin_game_time"),
        "time_normalization": time_normalization,
        "camera_available_t_policy": "updated to normalized native game t only for native capture-available telemetry; never reconstructed from video PTS",
        "metric_scale_proven": False,
        "metric_scale_status": "not assessed by alignment",
        "scale_provenance": cam_metadata.get("scale_provenance"),
        "source_tag_status": cam_metadata.get("status"),
        "source_format_enabled": source_enabled,
        "image_sensor_id": expected_sensor,
        "pairs": pairs if usable else [],
    }
    return AlignmentResult(aligned, audit)


def align_camera_rows_to_frames(*args: Any, **kwargs: Any) -> AlignmentResult:
    """Alias for :func:`align_cam_frames` with an integration-friendly name."""

    return align_cam_frames(*args, **kwargs)


__all__ = ["AlignmentResult", "align_cam_frames", "align_camera_rows_to_frames"]
