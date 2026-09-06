"""Joint A+B reconstruction from distributed video keyframes, without registration."""

import json
import os
import shutil
import subprocess
import time

import numpy as np

from .reconstruct import compute_device, infer_images, export_cloud


def frames_per_source():
    try:
        count = int(os.environ.get("SIMV1_JOINT_FRAMES_PER_SOURCE", "4"))
    except ValueError as exc:
        raise ValueError("SIMV1_JOINT_FRAMES_PER_SOURCE must be an integer from 2 to 8.") from exc
    if not 2 <= count <= 8:
        raise ValueError("SIMV1_JOINT_FRAMES_PER_SOURCE must be an integer from 2 to 8.")
    return count


def sample_candidates(capture, folder, count):
    import cv2

    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and FFprobe are required for joint video reconstruction.")
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", capture["path"],
    ], capture_output=True, text=True, timeout=30)
    try:
        duration = float(json.loads(probe.stdout)["format"]["duration"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Could not read source {capture['source']} video duration.") from exc
    if probe.returncode or not np.isfinite(duration) or duration < 0.5:
        raise RuntimeError(f"Source {capture['source']} needs at least 0.5 seconds of usable video.")
    folder.mkdir(parents=True, exist_ok=True)
    candidates = []
    sift = cv2.SIFT_create(nfeatures=1200)
    # Seek across the whole clip, leaving a margin before its final frame.
    for index, timestamp in enumerate(np.linspace(0, duration - min(0.1, duration / 10), count)):
        path = folder / f"{index:04d}.jpg"
        result = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{timestamp:.6f}",
            "-i", capture["path"], "-frames:v", "1", "-vf",
            "scale=960:960:force_original_aspect_ratio=decrease", str(path),
        ], capture_output=True, text=True, timeout=60)
        if result.returncode:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        image = cv2.resize(image, (round(image.shape[1] * 518 / max(image.shape)),
                                   round(image.shape[0] * 518 / max(image.shape))))
        _, descriptors = sift.detectAndCompute(image, None)
        candidates.append({"path": path, "t": float(timestamp),
                           "sharpness": float(cv2.Laplacian(image, cv2.CV_64F).var()),
                           "descriptors": descriptors})
    if len(candidates) < 2:
        raise RuntimeError(f"Could not decode enough frames from source {capture['source']}.")
    return candidates


def reciprocal_matches(a, b):
    """Appearance cue for choosing a shared anchor, not geometric verification."""
    if a is None or b is None or min(len(a), len(b)) < 2:
        return 0
    import cv2

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def ratio_matches(first, second):
        return {(m.queryIdx, m.trainIdx) for pair in matcher.knnMatch(first, second, k=2)
                if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance}
    forward = ratio_matches(a, b)
    backward = {(j, i) for i, j in ratio_matches(b, a)}
    return len(forward & backward)


def select_frames(candidates, count, anchor):
    """Keep the anchor plus sharp candidates spread over the rest of the clip."""
    chosen = [anchor]
    duration = max(c["t"] for c in candidates) or 1
    max_sharpness = max(c["sharpness"] for c in candidates) or 1
    while len(chosen) < min(count, len(candidates)):
        options = [i for i in range(len(candidates)) if i not in chosen]
        chosen.append(max(options, key=lambda i:
            min(abs(candidates[i]["t"] - candidates[j]["t"]) for j in chosen) / duration
            * (0.25 + 0.75 * np.sqrt(candidates[i]["sharpness"] / max_sharpness))))
    return [candidates[chosen[0]]] + sorted([candidates[i] for i in chosen[1:]], key=lambda c: c["t"])


def reconstruct_joint(captures, folder, progress):
    started = time.perf_counter()
    count = frames_per_source()
    device = compute_device()
    candidates = []
    for capture in captures:
        progress(f"Selecting keyframes across source {capture['source']}")
        candidates.append(sample_candidates(capture, folder / capture["source"] / "frames", max(12, count * 3)))
    progress("Looking for shared visual features between A and B")
    matches, anchor_a, anchor_b = max(
        (reciprocal_matches(a["descriptors"], b["descriptors"]), i, j)
        for i, a in enumerate(candidates[0]) for j, b in enumerate(candidates[1])
    )
    selected = [select_frames(candidates[0], count, anchor_a),
                select_frames(candidates[1], count, anchor_b)]
    paths = [frame["path"] for frames in selected for frame in frames]
    # All A and B frames occupy the same sequence dimension, not separate batches.
    prediction = infer_images(paths, device, progress)
    clouds, offset = [], 0
    for capture, frames in zip(captures, selected):
        progress(f"Exporting source {capture['source']} in shared coordinates")
        indices = list(range(offset, offset + len(frames)))
        clouds.append(export_cloud(folder / capture["source"], capture, frames,
                                   prediction, indices, device, reconstruction_method="joint_vggt"))
        offset += len(frames)
    return clouds, {
        "method": "joint_vggt",
        "model": prediction["model"],
        "model_variant": prediction["model_variant"],
        "frames_per_source": [len(frames) for frames in selected],
        "anchor_times": [frames[0]["t"] for frames in selected],
        "anchor_reciprocal_matches": matches,
        "overlap_verified": False,
        "quality_note": ("Few shared visual features found. Overlap may be insufficient; inspect both sources."
                         if matches < 20 else "Shared visual features found; geometric alignment has not been independently validated."),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "compute_device": device,
    }
