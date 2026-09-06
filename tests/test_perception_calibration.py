import numpy as np
from backend.prediction.schema import Camera
from backend.prediction.calibration import assess_calibration

def camera(position=(0, 0, 0)):
    pose = np.eye(4)
    pose[:3, 3] = position
    return Camera(
        sensor_id="static",
        coordinate_frame="room",
        calibration_id="fixture-calibration",
        calibration_provenance="independent checkerboard calibration fixture",
        width=100,
        height=100,
        intrinsics=[[100, 0, 50], [0, 100, 50], [0, 0, 1]],
        camera_to_world=pose.tolist(),
        available_t=0,
    )


def checks_for(cam, *, pixel_key="image_pixel"):
    points = np.asarray([
        [-1, -1, 5], [1, -1, 5], [1, 1, 5], [-1, 1, 5],
        [-2, 0, 6], [2, 0, 6],
    ], dtype=float)
    rows = []
    for index, xyz in enumerate(points):
        camera_xyz = np.asarray(xyz) - np.asarray(cam.camera_to_world)[:3, 3]
        projected = np.asarray(cam.intrinsics) @ camera_xyz
        pixel = (projected[:2] / projected[2]).tolist()
        rows.append({"id": f"landmark-{index}", "xyz": xyz.tolist(), pixel_key: pixel,
                     "role": "fit" if index < 4 else "holdout"})
    return rows


def test_perfect_supplied_pose_reports_fit_and_holdout_metrics():
    report = assess_calibration(camera(), checks_for(camera()))
    assert report["status"] == "accepted"
    assert report["accepted"]
    assert report["counts"]["fit"] == 4
    assert report["counts"]["holdout"] == 2
    assert report["metrics"]["all"]["max_px"] == 0
    assert report["holdout_used_for_fit"] is False
    assert report["metric_scale"]["independently_proven"] is False
    assert all(row["status"] == "accepted" for row in report["check_results"])


def test_map_pixel_alias_is_explicitly_recorded():
    report = assess_calibration(camera(), checks_for(camera(), pixel_key="map_pixel"))
    assert report["status"] == "accepted"
    assert {row["pixel_field"] for row in report["check_results"]} == {"map_pixel"}


def test_wrong_pose_and_metric_scale_mismatch_are_not_accepted():
    checks = checks_for(camera())
    wrong_pose = assess_calibration(camera(position=(1, 0, 0)), checks)
    wrong_scale = assess_calibration(camera(position=(0, 0, 1)), checks)
    assert wrong_pose["status"] == "mismatch"
    assert wrong_scale["status"] == "mismatch"
    assert wrong_pose["metrics"]["all"]["max_px"] > 15


def test_behind_camera_out_of_image_and_invalid_rows_are_reported():
    checks = checks_for(camera())
    checks.extend([
        {"id": "behind", "xyz": [0, 0, -2], "image_pixel": [50, 50], "role": "holdout"},
        {"id": "outside", "xyz": [100, 0, 5], "image_pixel": [50, 50], "role": "holdout"},
        {"id": "bad", "xyz": [0, 0, 5], "role": "holdout"},
    ])
    report = assess_calibration(camera(), checks)
    statuses = {row["id"]: row["status"] for row in report["check_results"]}
    assert statuses["behind"] == "behind_camera"
    assert statuses["outside"] == "out_of_image"
    assert statuses["bad"] == "invalid"
    assert report["counts"]["behind_camera"] == 1
    assert report["counts"]["out_of_image"] == 1
    assert report["counts"]["errors"] >= 3


def test_degenerate_distribution_is_insufficient_even_with_zero_pixel_error():
    checks = []
    for index in range(6):
        checks.append({"id": f"same-{index}", "xyz": [0, 0, 5], "image_pixel": [50, 50],
                       "role": "fit" if index < 4 else "holdout"})
    report = assess_calibration(camera(), checks)
    assert report["status"] == "insufficient"
    assert report["spatial_distribution"]["world_rank"] == 0
    assert report["spatial_distribution"]["adequate"] is False


def test_invalid_and_duplicate_checks_are_not_silently_dropped():
    checks = checks_for(camera())
    checks.append(dict(checks[0]))
    checks.append({"id": "unknown-role", "xyz": [0, 0, 5], "image_pixel": [50, 50], "role": "guess"})
    checks.append({"id": "ambiguous", "xyz": [0, 0, 5], "image_pixel": [50, 50], "map_pixel": [50, 50], "role": "fit"})
    report = assess_calibration(camera(), checks)
    assert len(report["check_results"]) == len(checks)
    reasons = {row["reason"] for row in report["check_results"] if row["status"] == "invalid"}
    assert {"duplicate_id", "invalid_role", "ambiguous_pixel_field"} <= reasons
