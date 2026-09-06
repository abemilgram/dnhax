"""Verified HLAE Source2 ``.cam`` import and camera conversion.

The parser is intentionally text-only and strict.  The format and angle/FOV
semantics below are taken from AdvancedFX tag ``v2.191.1`` (release commit
``b976368``):

* ``shared/CamIO.cpp`` writes ``advancedfx Cam``, ``version 2``, the eight
  named channels, ``DATA``, then eight whitespace-separated values.
* ``AfxHookSource2/main.cpp`` maps Source view pitch/yaw/roll to file
  ``yRotation``/``zRotation``/``xRotation`` on export.
* ``shared/AfxMath.cpp``'s ``MakeVectors`` defines the Source forward/right/up
  basis, and ``main.cpp`` reports the Source2 default FOV scaling as
  ``AlienSwarm``.

References:
https://github.com/advancedfx/advancedfx/blob/v2.191.1/shared/CamIO.cpp
https://github.com/advancedfx/advancedfx/blob/v2.191.1/shared/AfxMath.cpp
https://github.com/advancedfx/advancedfx/blob/v2.191.1/shared/FovScaling.cpp
https://github.com/advancedfx/advancedfx/blob/v2.191.1/AfxHookSource2/main.cpp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .schema import Camera

VERIFIED_SOURCE_TAG = "v2.191.1"
VERIFIED_SOURCE_COMMIT = "b976368"
SOURCE_CAMIO_URL = "https://github.com/advancedfx/advancedfx/blob/v2.191.1/shared/CamIO.cpp"
SOURCE_ANGLE_MATH_URL = "https://github.com/advancedfx/advancedfx/blob/v2.191.1/shared/AfxMath.cpp"
SOURCE2_MAIN_URL = "https://github.com/advancedfx/advancedfx/blob/v2.191.1/AfxHookSource2/main.cpp"
SOURCE_FOV_SCALING_URL = "https://github.com/advancedfx/advancedfx/blob/v2.191.1/shared/FovScaling.cpp"
CHANNELS = (
    "time",
    "xPosition",
    "yPosition",
    "zPosition",
    "xRotation",
    "yRotation",
    "zRotation",
    "fov",
)


def _finite_vec(value: Any, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite {size}-vector") from exc
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite {size}-vector")
    return result


def _canonical_source_tag(source_tag: str | None) -> str | None:
    if source_tag is None:
        return None
    candidate = str(source_tag).strip()
    if candidate == VERIFIED_SOURCE_TAG or candidate == VERIFIED_SOURCE_TAG[1:]:
        return VERIFIED_SOURCE_TAG
    return candidate or None


def _alien_swarm_fov_scaling(width: float, height: float, fov_degrees: float) -> float:
    """The Source/HLAE AlienSwarm aspect conversion from FovScaling.cpp."""

    ratio = (float(width) / float(height)) / (4.0 / 3.0)
    half_angle = np.deg2rad(float(fov_degrees)) * 0.5
    return float(np.rad2deg(2.0 * np.arctan(ratio * np.tan(half_angle))))


def fov_to_intrinsics(fov_degrees: float, width: int, height: int, *, fov_axis: str = "horizontal") -> np.ndarray:
    """Convert current Source2 exported FOV to a square-pixel OpenCV K.

    HLAE's current Source2 export uses the ``AlienSwarm``-scaled FOV, which is
    the horizontal image FOV after aspect conversion.  ``fov_axis`` is kept
    explicit to prevent silently interpreting an external vertical-FOV value.
    The default is the verified HLAE convention; ``vertical`` is available for
    callers with independently documented non-HLAE input.
    """

    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)) or width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive integers")
    fov = float(fov_degrees)
    if not np.isfinite(fov) or not 0.0 < fov < 180.0:
        raise ValueError("FOV must be finite and between 0 and 180 degrees")
    if fov_axis not in {"horizontal", "vertical"}:
        raise ValueError("fov_axis must be horizontal or vertical")
    focal = (float(width) if fov_axis == "horizontal" else float(height)) / (2.0 * np.tan(np.deg2rad(fov) * 0.5))
    if not np.isfinite(focal) or focal <= 0:
        raise ValueError("FOV produced an invalid focal length")
    # Source and the prediction schema use full-frame pixel coordinates with
    # the principal point at the image center.
    return np.asarray([[focal, 0.0, width * 0.5], [0.0, focal, height * 0.5], [0.0, 0.0, 1.0]], dtype=float)


def source_angles_to_c2w(roll_degrees: float, pitch_degrees: float, yaw_degrees: float) -> np.ndarray:
    """Convert Source roll/pitch/yaw to a proper OpenCV camera-to-world R.

    Source2's ``MakeVectors`` uses forward/right/up vectors with Source angles
    ``(roll, pitch, yaw)`` corresponding to file x/y/z rotation columns. An
    OpenCV camera basis is right/down/forward, hence the columns are
    ``[source_right, -source_up, source_forward]``.
    """

    roll, pitch, yaw = np.deg2rad([float(roll_degrees), float(pitch_degrees), float(yaw_degrees)])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    cr, cp, cy = np.cos([roll, pitch, yaw])
    forward = np.asarray([cp * cy, cp * sy, -sp], dtype=float)
    right = np.asarray([
        -sr * sp * cy + cr * sy,
        -sr * sp * sy - cr * cy,
        -sr * cp,
    ], dtype=float)
    up = np.asarray([
        cr * sp * cy + sr * sy,
        cr * sp * sy - sr * cy,
        cr * cp,
    ], dtype=float)
    rotation = np.column_stack((right, -up, forward))
    if not np.isfinite(rotation).all() or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError("Source angles produced a non-rigid or reflected camera pose")
    return rotation


@dataclass(frozen=True)
class CamImportResult(list):
    """List-compatible parsed frames with source-verification metadata.

    The list payload contains dictionaries with both a typed ``camera`` and
    JSON-friendly matrix/FOV fields. Metadata is available as ``.metadata``;
    ``parse_cam_document`` exposes the same information as a plain dict.
    """

    frames: list[dict[str, Any]]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        list.__init__(self, self.frames)

    @property
    def enabled(self) -> bool:
        return bool(self.metadata.get("enabled", False))


def _parse_header(lines: list[str]) -> tuple[int, tuple[str, ...], str | None, int]:
    if not lines or lines[0].lstrip("\ufeff").strip() != "advancedfx Cam":
        raise ValueError("Not an AdvancedFX Cam file")
    version: int | None = None
    channels: tuple[str, ...] | None = None
    scale_fov: str | None = None
    data_index: int | None = None
    for index, raw in enumerate(lines[1:], start=1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        verb = fields[0]
        if verb == "DATA":
            data_index = index
            break
        if verb == "version":
            if len(fields) != 2:
                raise ValueError("Malformed version header")
            try:
                version = int(fields[1])
            except ValueError as exc:
                raise ValueError("Malformed version header") from exc
        elif verb == "channels":
            channels = tuple(fields[1:])
        elif verb == "scaleFov":
            if len(fields) != 2 or fields[1] not in {"none", "alienSwarm"}:
                raise ValueError("Unsupported scaleFov header")
            scale_fov = fields[1]
        else:
            raise ValueError(f"Unknown AdvancedFX Cam header field: {verb}")
    if data_index is None:
        raise ValueError("Missing DATA header")
    if version != 2:
        raise ValueError("Unsupported AdvancedFX Cam version")
    if channels != CHANNELS:
        raise ValueError("Unexpected AdvancedFX Cam channel order")
    return version, channels, scale_fov, data_index


def _parse_rows(lines: list[str], data_index: int) -> list[np.ndarray]:
    parsed: list[np.ndarray] = []
    for line_number, raw in enumerate(lines[data_index + 1 :], start=data_index + 2):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != len(CHANNELS):
            raise ValueError(f"Line {line_number}: expected {len(CHANNELS)} numeric camera columns")
        try:
            row = np.asarray([float(field) for field in fields], dtype=float)
        except ValueError as exc:
            raise ValueError(f"Line {line_number}: camera columns must be numeric") from exc
        if not np.isfinite(row).all():
            raise ValueError(f"Line {line_number}: camera columns must be finite")
        parsed.append(row)
    if not parsed:
        raise ValueError("AdvancedFX Cam contains no camera rows")
    times = np.asarray([row[0] for row in parsed])
    if np.any(np.diff(times) < 0):
        raise ValueError("Camera rows must be nondecreasing in source time")
    return parsed


def _metadata(*, source_tag: str | None, allow_unverified: bool, scale_fov_header: str | None, origin_game_time: float, units_to_meters: float, origin_xyz: np.ndarray, width: int, height: int) -> dict[str, Any]:
    canonical = _canonical_source_tag(source_tag)
    verified = canonical == VERIFIED_SOURCE_TAG
    unsupported_scaling = scale_fov_header == "none"
    if unsupported_scaling and not allow_unverified:
        raise ValueError("Legacy scaleFov none is not enabled for verified Source2 import")
    if not verified and not allow_unverified:
        claimed = "missing" if canonical is None else repr(canonical)
        raise ValueError(f"Unverified HLAE source tag {claimed}; pass the verified tag {VERIFIED_SOURCE_TAG!r} before enabling import")
    enabled = bool(verified and not unsupported_scaling)
    candidate_reasons: list[str] = []
    if not verified:
        candidate_reasons.append("source tag is not the verified Source2 tag")
    if unsupported_scaling:
        candidate_reasons.append("legacy scaleFov none conversion is not enabled")
    return {
        "status": "verified" if enabled else "candidate",
        "enabled": enabled,
        "candidate_reasons": candidate_reasons,
        "source_tag": canonical,
        "verified_source_tag": VERIFIED_SOURCE_TAG,
        "verified_source_commit": VERIFIED_SOURCE_COMMIT,
        "source_urls": {"camio": SOURCE_CAMIO_URL, "angle_math": SOURCE_ANGLE_MATH_URL, "fov_scaling": SOURCE_FOV_SCALING_URL, "source2_main": SOURCE2_MAIN_URL},
        "format": "AdvancedFX Cam version 2 channels time xPosition yPosition zPosition xRotation yRotation zRotation fov",
        "fov_scaling": scale_fov_header or "source2_default_alienSwarm",
        "fov_scaling_supported_for_enable": not unsupported_scaling,
        "fov_axis": "horizontal",
        "origin_game_time": float(origin_game_time),
        "time_normalization": "source_time - origin_game_time; source timestamps retained; no row-index/MP4 alignment",
        "units_to_meters": float(units_to_meters),
        "origin_xyz_m": origin_xyz.tolist(),
        "dimensions_px": [int(width), int(height)],
        "source_units": "Source2 game units; metric conversion is caller supplied",
        "scale_provenance": "caller supplied; importer does not independently establish Source2-to-meter scale",
    }


def parse_cam(
    text: str,
    *,
    origin_game_time: float,
    units_to_meters: float,
    origin_xyz: Iterable[float],
    width: int,
    height: int,
    available_t: float,
    calibration_id: str,
    calibration_provenance: str,
    coordinate_frame: str,
    sensor_id: str = "hlae-source2",
    source_tag: str | None = None,
    allow_unverified: bool = False,
) -> CamImportResult:
    """Parse verified AdvancedFX Cam text into typed camera frame records.

    ``t`` is ``source_time - origin_game_time``. Positions are converted as
    ``origin_xyz + units_to_meters * [xPosition,yPosition,zPosition]``.
    ``units_to_meters``, origin, dimensions, availability, and calibration
    provenance are required to prevent silent unit or clock assumptions.
    Unverified source tags can be parsed only with ``allow_unverified=True``;
    those frames are marked ``enabled=False`` and remain candidates.
    """

    if not isinstance(text, str):
        raise TypeError("CAM input must be text")
    try:
        origin_time = float(origin_game_time)
        scale = float(units_to_meters)
        available = float(available_t)
    except (TypeError, ValueError) as exc:
        raise ValueError("origin_game_time, units_to_meters, and available_t must be numeric") from exc
    if not np.isfinite([origin_time, scale, available]).all() or scale <= 0 or available < 0:
        raise ValueError("Invalid time, positive unit scale, or availability")
    origin = _finite_vec(origin_xyz, 3, "origin_xyz")
    if not isinstance(width, (int, np.integer)) or not isinstance(height, (int, np.integer)) or width < 16 or height < 16:
        raise ValueError("Image dimensions must be integers >= 16")
    if not isinstance(coordinate_frame, str) or not coordinate_frame:
        raise ValueError("coordinate_frame is required")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise ValueError("calibration_id is required")
    if not isinstance(calibration_provenance, str) or len(calibration_provenance) < 8:
        raise ValueError("calibration_provenance must document the calibration source")
    lines = text.splitlines()
    version, _, scale_fov_header, data_index = _parse_header(lines)
    rows = _parse_rows(lines, data_index)
    metadata = _metadata(
        source_tag=source_tag,
        allow_unverified=allow_unverified,
        scale_fov_header=scale_fov_header,
        origin_game_time=origin_time,
        units_to_meters=scale,
        origin_xyz=origin,
        width=width,
        height=height,
    )
    frames: list[dict[str, Any]] = []
    for row in rows:
        source_time, x, y, z, roll, pitch, yaw, exported_fov = (float(value) for value in row)
        position = origin + scale * np.asarray([x, y, z], dtype=float)
        rotation = source_angles_to_c2w(roll, pitch, yaw)
        # v2.191.1 CamExport uses Auto_FovScaling, whose Source2 default is
        # AlienSwarm. The file therefore carries the horizontal image FOV.
        if scale_fov_header == "none":
            # Legacy ``none`` files carry the unscaled Source FOV. Source2's
            # current view still applies the default aspect conversion to the
            # rendered image; make that conversion explicit here.
            image_fov = _alien_swarm_fov_scaling(width, height, exported_fov)
        else:
            image_fov = exported_fov
        intrinsics = fov_to_intrinsics(image_fov, width, height)
        pose = np.eye(4, dtype=float)
        pose[:3, :3] = rotation
        pose[:3, 3] = position
        camera = Camera(
            sensor_id=sensor_id,
            coordinate_frame=coordinate_frame,
            calibration_id=calibration_id,
            calibration_provenance=calibration_provenance,
            width=int(width),
            height=int(height),
            intrinsics=intrinsics.tolist(),
            camera_to_world=pose.tolist(),
            available_t=available,
        )
        frames.append({
            "t": source_time - origin_time,
            "source_time": source_time,
            "camera": camera,
            "camera_to_world": pose.tolist(),
            "intrinsics": intrinsics.tolist(),
            "position_m": position.tolist(),
            "fov_degrees": float(image_fov),
            "exported_fov_degrees": exported_fov,
            "source_tag_status": metadata["status"],
            "enabled": metadata["enabled"],
            "version": version,
        })
    return CamImportResult(frames, metadata)


def parse_cam_document(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Return a JSON-friendly document wrapper around :func:`parse_cam`."""

    result = parse_cam(*args, **kwargs)
    frames = []
    for frame in result:
        public = dict(frame)
        camera = public.get("camera")
        if hasattr(camera, "model_dump"):
            public["camera"] = camera.model_dump()
        frames.append(public)
    return {"metadata": result.metadata, "frames": frames, "enabled": result.enabled}


__all__ = [
    "CHANNELS",
    "CamImportResult",
    "VERIFIED_SOURCE_COMMIT",
    "VERIFIED_SOURCE_TAG",
    "fov_to_intrinsics",
    "parse_cam",
    "parse_cam_document",
    "source_angles_to_c2w",
]
