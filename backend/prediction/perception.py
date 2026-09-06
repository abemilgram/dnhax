"""Actual decoded pixels → resident detector → projection → belief → predictions.

Evaluation labels are deliberately absent from this module's input contract.
Recorded processing time is separate from the shared scene timeline.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw

from .engine import PredictionEngine
from .perception_schema import RecordedTake
from .schema import Detection, FramePacket


def read_image(root: Path, relative: str) -> tuple[Image.Image, str]:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Image is missing or escapes the take directory")
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Input image exceeds 32 MB")
    data = path.read_bytes()
    import io
    with Image.open(io.BytesIO(data)) as loaded:
        if loaded.width > 8192 or loaded.height > 8192 or loaded.width * loaded.height > 20_000_000:
            raise ValueError("Input image exceeds the pixel budget")
        loaded.load()
        rgb = loaded.convert("RGB")
    return rgb, hashlib.sha256(data).hexdigest()


def apply_ground_policy(detections: list[dict], diagnostics: list[dict], policy: str, width: int, height: int) -> list[Detection]:
    """Conservative optional pose rule; it is explicitly an unvalidated estimate."""
    result = []
    accepted_diagnostics = {}
    for diagnostic in diagnostics:
        if diagnostic.get("accepted") and isinstance(diagnostic.get("output_detection_index"), int):
            key = diagnostic["output_detection_index"]
            if key in accepted_diagnostics:
                raise ValueError("Duplicate diagnostic output detection index")
            accepted_diagnostics[key] = diagnostic
    for index, value in enumerate(detections):
        detection = Detection.model_validate(value)
        visible = False
        if any(v < 0 for v in detection.box[:2]) or detection.box[2] > width or detection.box[3] > height:
            raise ValueError("Detector boxes must be inside the original decoded image")
        diagnostic = accepted_diagnostics.get(index, {})
        if policy == "pose_ankles_provisional_v1" and diagnostic.get("contact_eligible", True):
            points = np.asarray(diagnostic.get("raw_keypoints", []), dtype=float)
            confidence = np.asarray(diagnostic.get("raw_keypoint_confidence", []), dtype=float)
            x1, y1, x2, y2 = detection.box
            if (points.shape == (17, 2) and confidence.shape == (17,) and np.isfinite(points).all()
                    and np.isfinite(confidence).all() and np.all((confidence >= 0) & (confidence <= 1))):
                ankles = points[[15, 16]]
                visible = bool(y2-y1 >= 40 and x1 > 2 and y1 > 2 and x2 < width-2 and y2 < height-2
                               and np.all(confidence[[15, 16]] >= .8)
                               and np.all(confidence[[11, 12]] >= .7)
                               and np.all(ankles[:, 0] >= x1) and np.all(ankles[:, 0] <= x2)
                               and np.all(ankles[:, 1] >= y1 + .70*(y2-y1))
                               and np.all(ankles[:, 1] <= y2))
        result.append(detection.model_copy(update={"ground_contact_visible": visible}))
    return result


def readiness(take: RecordedTake, frame) -> list[str]:
    reasons = []
    if not frame.observation_usable:
        reasons.append("frame_excluded_from_observation_evidence")
    if take.prior is None:
        reasons.append("metric_map_missing")
    if frame.camera is None:
        reasons.append("camera_calibration_missing")
    if frame.camera is not None and not take.geometry_verified:
        reasons.append("geometry_accuracy_unverified")
    if len({f.sensor_id for f in take.frames}) > 1 and not take.synchronization_verified:
        reasons.append("cross_camera_synchronization_unverified")
    if take.clock_uncertainty_s > take.config.max_clock_uncertainty_s:
        reasons.append("clock_uncertainty_exceeds_budget")
    return reasons


def draw_diagnostics(image: Image.Image, detections: list[Detection], destination: Path):
    """Measured box overlay for inspection; never modify an inference input."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for detection in detections:
        x1, y1, x2, y2 = detection.box
        color = "lime" if detection.ground_contact_visible else "orange"
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
        draw.text((x1, max(0, y1-15)), f"person {detection.confidence:.2f}; ground {'estimated' if detection.ground_contact_visible else 'unknown'}", fill=color)
    annotated.save(destination)


def run_take(take: RecordedTake, *, root: Path, detector, output: Path) -> dict:
    """Offline causal replay: compute latency is measured, never disguised as live latency."""
    output.mkdir(parents=True, exist_ok=True)
    engine = None
    if take.prior is not None:
        engine = PredictionEngine(take.prior, clock_id=take.clock_id, evidence_source="video_detector", config=take.config)
    rows, latencies = [], []
    for index, frame in enumerate(sorted(take.frames, key=lambda f: (f.t, f.sensor_id, f.frame_id))):
        started = time.perf_counter()
        image, digest = read_image(root, frame.image)
        detection = detector.detect(image)
        if (detection["width"], detection["height"]) != image.size:
            raise ValueError("Detector coordinates do not match decoded image dimensions")
        boxes = apply_ground_policy(detection["detections"], detection.get("diagnostics", []), take.ground_contact_policy, *image.size)
        reasons = readiness(take, frame)
        row = {"sensor_id": frame.sensor_id, "frame_id": frame.frame_id, "t": frame.t,
               "width": image.width, "height": image.height, "image": frame.image,
               "observation_usable": frame.observation_usable, "unusable_reason": frame.unusable_reason,
               "image_sha256": digest, "detections": [d.model_dump() for d in boxes],
               "diagnostics": detection.get("diagnostics", []),
               "model": {"name": detection.get("model_name"), "sha256": detection.get("model_fingerprint"),
                         "provenance": detection.get("provenance", {})},
               "detector_inference_ms": detection["inference_ms"],
               "ground_contact_policy": take.ground_contact_policy,
               "prediction_status": "not_ready" if reasons else "ready", "blocking_reasons": reasons,
               "projected_observations": [], "state": None}
        if not reasons:
            if image.size != (frame.camera.width, frame.camera.height):
                raise ValueError("Calibration resolution differs from decoded input; explicit resizing/calibration required")
            if frame.camera.available_t > frame.t:
                # This first replay intentionally refuses retroactive geometry. No buffering
                # or smoothing with a future reconstructed batch is silently performed.
                row["prediction_status"] = "not_ready"
                row["blocking_reasons"].append("calibration_not_available_at_capture")
            else:
                packet = FramePacket(frame_id=frame.frame_id, clock_id=take.clock_id, t=frame.t,
                                     available_t=frame.t, clock_uncertainty_s=take.clock_uncertainty_s,
                                     camera=frame.camera, evidence_source="video_detector",
                                     detector=f"{detection.get('model_name', 'unknown')}:{detection.get('model_fingerprint', 'unverified')}", detections=boxes,
                                     coverage_complete=False)
                observations, rejections = engine.geometry.project(packet)
                row["projected_observations"] = [o.model_dump() for o in observations]
                row["projection_rejections"] = rejections
                row["state"] = engine.ingest(packet, now=frame.t)["state"]
                row["prediction_status"] = "processed_provisional" if take.ground_contact_policy != "reject_unknown" else "processed"
        overlay = output / f"frame-{index:06d}.jpg"
        draw_diagnostics(image, boxes, overlay)
        row["overlay"] = overlay.name
        row["wall_processing_ms"] = (time.perf_counter() - started) * 1000
        latencies.append(row["wall_processing_ms"])
        rows.append(row)
    summary = {"take_id": take.id, "mode": "recorded_replay", "provenance": take.provenance,
               "prediction_model": take.config.prediction_model,
               "timing_provenance": take.timing_provenance, "synchronization_verified": take.synchronization_verified,
               "geometry_verified": take.geometry_verified,
               "geometry_validation_provenance": take.geometry_validation_provenance,
               "ground_contact_policy": take.ground_contact_policy, "frames": len(rows),
               "detections": sum(len(row["detections"]) for row in rows),
               "projected_observations": sum(len(row["projected_observations"]) for row in rows),
               "frames_processed_by_predictor": sum(row["state"] is not None for row in rows),
               "blocking_reasons": sorted({reason for row in rows for reason in row["blocking_reasons"]}),
               "processing_ms": {"p50": float(np.percentile(latencies, 50)), "p95": float(np.percentile(latencies, 95))},
               "accuracy": "unmeasured until a separate evaluation with independent annotations",
               "timing_scope": "Offline decode + detection + optional prediction + diagnostic write. Source clock is replayed, not live latency.",
               "prediction_clock_assumption": "Capture-time replay; detector processing delay is measured separately and is not simulated in belief timestamps.",
               "negative_evidence": "disabled until detector visibility/miss assumptions are evaluated",
               "rows": "frames.jsonl"}
    (output / "frames.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--take", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--tiled", action="store_true", help="Full frame plus fixed 2x2 overlapping crops with NMS")
    parser.add_argument("--prediction-model", choices=["route_bank", "branching"],
                        help="Override take.config.prediction_model; geometry and synchronization gates still apply")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from .detector import PoseDetector
    take = RecordedTake.model_validate_json(args.take.read_text())
    if args.prediction_model is not None:
        data = take.model_dump()
        data["config"]["prediction_model"] = args.prediction_model
        take = RecordedTake.model_validate(data)
    detector = PoseDetector(args.weights, device=args.device, imgsz=args.imgsz)
    if args.tiled:
        from .tiled_detector import TiledPoseDetector
        detector = TiledPoseDetector(detector)
    result = run_take(take, root=args.take.parent, detector=detector, output=args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
