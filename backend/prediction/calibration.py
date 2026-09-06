"""Fixed-pose camera calibration assessment against independent landmarks.

This module evaluates a supplied metric world and a supplied camera pose. It
does not manufacture a camera fit from the same observations it reports. The
``fit``/``holdout`` roles describe an external calibration split; in this
module both are evaluated against the supplied pose and holdouts are never
used to modify it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema import Camera


_PIXEL_THRESHOLDS = {"median_px": 3.0, "p95_px": 8.0, "max_px": 15.0}
_MIN_FIT = 4
_MIN_HOLDOUT = 2
_MIN_TOTAL = 6
_MIN_LANDMARK_SEPARATION_M = 0.25
_MIN_WORLD_SPAN_M = 1.0
_MIN_IMAGE_SPAN_PX = 10.0


def _finite_vector(value: Any, size: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.shape != (size,) or not np.isfinite(array).all():
        return None
    return array


def _camera_arrays(camera: Camera) -> tuple[np.ndarray, np.ndarray, str | None]:
    try:
        intrinsics = np.asarray(camera.intrinsics, dtype=float)
        pose = np.asarray(camera.camera_to_world, dtype=float)
    except (AttributeError, TypeError, ValueError):
        return np.empty((0, 0)), np.empty((0, 0)), "camera_matrices_unreadable"
    if intrinsics.shape != (3, 3) or pose.shape != (4, 4):
        return intrinsics, pose, "camera_matrix_shape"
    if not np.isfinite(intrinsics).all() or not np.isfinite(pose).all():
        return intrinsics, pose, "camera_matrix_nonfinite"
    rotation = pose[:3, :3]
    if (
        not np.allclose(intrinsics[2], [0, 0, 1])
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or abs(float(np.linalg.det(intrinsics))) < 1e-8
    ):
        return intrinsics, pose, "invalid_intrinsics"
    if (
        not np.allclose(pose[3], [0, 0, 0, 1])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(float(np.linalg.det(rotation)), 1, atol=1e-5)
    ):
        return intrinsics, pose, "invalid_camera_pose"
    try:
        width, height = int(camera.width), int(camera.height)
    except (AttributeError, TypeError, ValueError):
        return intrinsics, pose, "invalid_image_dimensions"
    if width <= 0 or height <= 0:
        return intrinsics, pose, "invalid_image_dimensions"
    return intrinsics, pose, None


def _project(intrinsics: np.ndarray, pose: np.ndarray, world: np.ndarray) -> tuple[np.ndarray | None, str | None]:
    camera_xyz = pose[:3, :3].T @ (world - pose[:3, 3])
    if not np.isfinite(camera_xyz).all():
        return None, "projection_nonfinite"
    if float(camera_xyz[2]) <= 1e-8:
        return None, "behind_camera"
    homogeneous = intrinsics @ camera_xyz
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[2])) <= 1e-12:
        return None, "projection_nonfinite"
    return np.asarray(homogeneous[:2] / homogeneous[2], dtype=float), None


def _pixel_from_check(check: dict[str, Any]) -> tuple[np.ndarray | None, str | None, str | None]:
    has_map = "map_pixel" in check
    has_image = "image_pixel" in check
    if has_map and has_image:
        return None, None, "ambiguous_pixel_field"
    if not has_map and not has_image:
        return None, None, "missing_pixel_field"
    field = "map_pixel" if has_map else "image_pixel"
    pixel = _finite_vector(check[field], 2)
    if pixel is None:
        return None, field, "invalid_pixel"
    return pixel, field, None


def _row_base(check: Any, index: int) -> dict[str, Any]:
    if isinstance(check, dict):
        identifier = check.get("id", f"check-{index}")
        role = check.get("role")
    else:
        identifier = f"check-{index}"
        role = None
    if not isinstance(identifier, str) or not identifier:
        identifier = str(identifier)
    return {"index": index, "id": identifier, "role": role}


def _metric_summary(rows: list[dict[str, Any]], role: str | None = None) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if (role is None or row.get("role") == role)
        and row.get("residual_px") is not None
        and np.isfinite(float(row["residual_px"]))
    ]
    errors = np.asarray([float(row["residual_px"]) for row in selected], dtype=float)
    if not len(errors):
        return {"count": 0, "median_px": None, "p95_px": None, "max_px": None}
    return {
        "count": int(len(errors)),
        "median_px": float(np.median(errors)),
        "p95_px": float(np.percentile(errors, 95)),
        "max_px": float(np.max(errors)),
    }


def _distribution(rows: list[dict[str, Any]], camera: Camera) -> dict[str, Any]:
    points = [row["xyz_array"] for row in rows if row.get("xyz_array") is not None]
    observed = [row["observed_array"] for row in rows if row.get("observed_array") is not None]
    if points:
        world = np.asarray(points, dtype=float)
        centered = world - world.mean(axis=0)
        rank = int(np.linalg.matrix_rank(centered, tol=1e-8)) if len(world) > 1 else 0
        span = np.ptp(world, axis=0)
        world_span = float(np.linalg.norm(span))
        if len(world) > 1:
            pairwise = world[:, None, :] - world[None, :, :]
            pairwise_distances = np.linalg.norm(pairwise, axis=2)
            pairwise_distances[pairwise_distances <= 1e-9] = np.inf
            minimum_separation = float(np.min(pairwise_distances))
        else:
            minimum_separation = None
    else:
        rank, world_span, minimum_separation = 0, 0.0, None
        span = np.zeros(3, dtype=float)
    if observed:
        image_span = np.ptp(np.asarray(observed, dtype=float), axis=0)
        image_span_px = float(np.linalg.norm(image_span))
    else:
        image_span_px = 0.0
    enough = (
        len(points) >= _MIN_TOTAL
        and rank >= 2
        and minimum_separation is not None
        and minimum_separation >= _MIN_LANDMARK_SEPARATION_M
        and world_span >= _MIN_WORLD_SPAN_M
        and image_span_px >= _MIN_IMAGE_SPAN_PX
    )
    return {
        "count": int(len(points)),
        "world_rank": rank,
        "world_span_m": world_span,
        "world_axis_span_m": [float(x) for x in span],
        "minimum_separation_m": minimum_separation,
        "observed_image_span_px": image_span_px,
        "adequate": bool(enough),
        "requirements": {
            "count": _MIN_TOTAL,
            "world_rank": 2,
            "minimum_separation_m": _MIN_LANDMARK_SEPARATION_M,
            "world_span_m": _MIN_WORLD_SPAN_M,
            "observed_image_span_px": _MIN_IMAGE_SPAN_PX,
        },
    }


def _empty_report(camera: Any, errors: list[dict[str, Any]], rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    rows = rows or []
    try:
        calibration_id = camera.calibration_id
        provenance = camera.calibration_provenance
    except AttributeError:
        calibration_id, provenance = None, None
    return {
        "status": "invalid",
        "accepted": False,
        "pose_source": "supplied_camera",
        "fit_method": "none_fixed_pose",
        "holdout_used_for_fit": False,
        "calibration_id": calibration_id,
        "calibration_provenance": provenance,
        "coordinate_convention": "world X/Y ground, Z up; camera OpenCV right/down/forward",
        "counts": {"supplied": len(rows), "fit": 0, "holdout": 0, "projected": 0, "behind_camera": 0, "out_of_image": 0, "errors": len(errors)},
        "fit": _metric_summary([]),
        "holdout": _metric_summary([]),
        "metrics": {"all": _metric_summary([]), "thresholds": dict(_PIXEL_THRESHOLDS)},
        "spatial_distribution": {"count": 0, "world_rank": 0, "adequate": False},
        "check_results": rows,
        "checks": rows,
        "errors": errors,
        "metric_scale": {
            "units": "meters",
            "status": "supplied_not_independently_proven",
            "independently_proven": False,
            "basis": "caller-supplied metric xyz anchors",
            "pixel_fit_does_not_validate_metric_scale": True,
        },
    }


def assess_calibration(camera: Camera, checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Assess a supplied camera against independent static landmark checks.

    ``map_pixel`` and ``image_pixel`` are accepted aliases for calibrated
    full-frame image coordinates ``[u, v]`` in pixels. The result records which
    spelling each row used as ``pixel_field`` and preserves every supplied row.
    """

    if not isinstance(checks, list):
        return _empty_report(camera, [{"index": None, "reason": "checks_must_be_list"}])
    intrinsics, pose, camera_error = _camera_arrays(camera)
    if camera_error is not None:
        rows = []
        errors = []
        for index, check in enumerate(checks):
            row = _row_base(check, index)
            row.update({"status": "invalid", "reason": camera_error})
            rows.append(row)
            errors.append({"index": index, "id": row["id"], "reason": camera_error})
        if not rows:
            errors.append({"index": None, "reason": camera_error})
        return _empty_report(camera, errors, rows)
    width, height = int(camera.width), int(camera.height)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    for index, raw_check in enumerate(checks):
        row = _row_base(raw_check, index)
        if not isinstance(raw_check, dict):
            row.update({"status": "invalid", "reason": "check_must_be_object"})
            errors.append({"index": index, "id": row["id"], "reason": "check_must_be_object"})
            rows.append(row)
            continue
        check = raw_check
        identifier = row["id"]
        duplicate = identifier in identifiers
        identifiers.add(identifier)
        xyz = _finite_vector(check.get("xyz"), 3)
        pixel, pixel_field, pixel_error = _pixel_from_check(check)
        role = check.get("role")
        if role not in {"fit", "holdout"}:
            row.update({"status": "invalid", "reason": "invalid_role"})
            errors.append({"index": index, "id": identifier, "reason": "invalid_role"})
            rows.append(row)
            continue
        row["role"] = role
        if duplicate:
            row.update({"status": "invalid", "reason": "duplicate_id"})
            errors.append({"index": index, "id": identifier, "reason": "duplicate_id"})
            rows.append(row)
            continue
        if xyz is None:
            row.update({"status": "invalid", "reason": "invalid_xyz", "pixel_field": pixel_field})
            errors.append({"index": index, "id": identifier, "reason": "invalid_xyz"})
            rows.append(row)
            continue
        if pixel_error is not None or pixel is None:
            row.update({"status": "invalid", "reason": pixel_error, "xyz": xyz.tolist(), "pixel_field": pixel_field})
            errors.append({"index": index, "id": identifier, "reason": pixel_error})
            rows.append(row)
            continue

        row.update({
            "xyz": xyz.tolist(),
            "xyz_array": xyz,
            "observed_pixel": pixel.tolist(),
            "observed_array": pixel,
            "pixel_field": pixel_field,
        })
        projected, projection_error = _project(intrinsics, pose, xyz)
        if projection_error is not None:
            row.update({"status": projection_error, "projected_pixel": None, "residual_px": None})
            errors.append({"index": index, "id": identifier, "reason": projection_error})
            rows.append(row)
            continue
        assert projected is not None
        residual = float(np.linalg.norm(projected - pixel))
        predicted_in_image = bool(0.0 <= projected[0] < width and 0.0 <= projected[1] < height)
        observed_in_image = bool(0.0 <= pixel[0] < width and 0.0 <= pixel[1] < height)
        out_of_image = not predicted_in_image or not observed_in_image
        reason = "out_of_image" if out_of_image else ("projection_error" if residual > _PIXEL_THRESHOLDS["max_px"] else "accepted")
        row.update({
            "status": reason,
            "projected_pixel": projected.tolist(),
            "residual_px": residual,
            "predicted_in_image": predicted_in_image,
            "observed_in_image": observed_in_image,
        })
        if reason != "accepted":
            errors.append({"index": index, "id": identifier, "reason": reason, "residual_px": residual})
        rows.append(row)

    fit_rows = [row for row in rows if row.get("role") == "fit" and row.get("xyz_array") is not None]
    holdout_rows = [row for row in rows if row.get("role") == "holdout" and row.get("xyz_array") is not None]
    projected_rows = [row for row in rows if row.get("projected_pixel") is not None]
    behind_count = sum(row.get("status") == "behind_camera" for row in rows)
    out_count = sum(row.get("status") == "out_of_image" for row in rows)
    fit_metrics = _metric_summary(rows, "fit")
    holdout_metrics = _metric_summary(rows, "holdout")
    all_metrics = _metric_summary(rows)
    distribution = _distribution(rows, camera)
    counts = {
        "supplied": len(checks),
        "fit": len(fit_rows),
        "holdout": len(holdout_rows),
        "projected": len(projected_rows),
        "behind_camera": int(behind_count),
        "out_of_image": int(out_count),
        "errors": len(errors),
    }
    enough_checks = counts["fit"] >= _MIN_FIT and counts["holdout"] >= _MIN_HOLDOUT and counts["projected"] >= _MIN_TOTAL
    quality_ok = (
        all_metrics["count"] > 0
        and all_metrics["median_px"] <= _PIXEL_THRESHOLDS["median_px"]
        and all_metrics["p95_px"] <= _PIXEL_THRESHOLDS["p95_px"]
        and all_metrics["max_px"] <= _PIXEL_THRESHOLDS["max_px"]
        and counts["behind_camera"] == 0
        and counts["out_of_image"] == 0
        and not errors
    )
    adequate = enough_checks and bool(distribution["adequate"])
    if not adequate:
        status = "insufficient"
    elif not quality_ok:
        status = "mismatch"
    else:
        status = "accepted"
    # Internal numpy helper values are removed from the JSON-facing rows.
    public_rows = []
    for row in rows:
        public = {key: value for key, value in row.items() if key not in {"xyz_array", "observed_array"}}
        public_rows.append(public)
    return {
        "status": status,
        "accepted": status == "accepted",
        "pose_source": "supplied_camera",
        "fit_method": "none_fixed_pose",
        "holdout_used_for_fit": False,
        "calibration_id": str(camera.calibration_id),
        "calibration_provenance": str(camera.calibration_provenance),
        "coordinate_convention": "world X/Y ground, Z up; camera OpenCV right/down/forward",
        "counts": counts,
        "fit": fit_metrics,
        "holdout": holdout_metrics,
        "metrics": {"all": all_metrics, "thresholds": dict(_PIXEL_THRESHOLDS)},
        "spatial_distribution": distribution,
        "check_results": public_rows,
        "checks": public_rows,
        "errors": errors,
        "metric_scale": {
            "units": "meters",
            "status": "supplied_not_independently_proven",
            "independently_proven": False,
            "basis": "caller-supplied metric xyz anchors",
            "pixel_fit_does_not_validate_metric_scale": True,
        },
    }


__all__ = ["assess_calibration"]
