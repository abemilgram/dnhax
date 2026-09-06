"""Sample a recorded video while retaining decoded frame index and source PTS.

Time offsets are supplied explicitly. Sampling does not verify camera synchronization.
"""

import argparse
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import subprocess


def _selection_expression(indices: list[int]) -> str:
    # FFmpeg's expression parser has a nesting limit. A flat 100+ term sum
    # recurses too deeply; balance the tree so 2000 frame choices remain safe.
    if len(indices) == 1:
        return f"eq(n\\,{indices[0]})"
    middle = len(indices) // 2
    return "(" + _selection_expression(indices[:middle]) + "+" + _selection_expression(indices[middle:]) + ")"


def extract_video(video: Path, output: Path, *, sensor_id: str, fps: float,
                  time_offset_s: float, max_frames: int = 2000) -> dict:
    if not sensor_id or not sensor_id.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Sensor ID must contain only letters, digits, dash or underscore")
    if not math.isfinite(fps) or not 0 < fps <= 30 or not math.isfinite(time_offset_s):
        raise ValueError("Sampling rate must be 0–30 fps and offset finite")
    if not 1 <= max_frames <= 2000 or not video.is_file():
        raise ValueError("Use an existing video and at most 2000 extracted frames per source")
    metadata = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                               "-show_entries", "frame=best_effort_timestamp_time:stream=time_base", "-of", "json", str(video)],
                              check=True, capture_output=True, text=True, timeout=90)
    document = json.loads(metadata.stdout)
    raw = document["frames"]
    time_base = float(Fraction(document["streams"][0]["time_base"]))
    if not math.isfinite(time_base) or time_base <= 0:
        raise ValueError("Video stream needs a positive timestamp time base")
    # Respect timestamp quantization (e.g. 33ms/67ms alternating in Matroska).
    tolerance = max(time_base / 2, 1e-6)
    selected = []
    previous = -float("inf")
    next_t = None
    for index, frame in enumerate(raw):
        if "best_effort_timestamp_time" not in frame:
            raise ValueError("Video frame lacks a presentation timestamp")
        t = float(frame["best_effort_timestamp_time"])
        if not math.isfinite(t) or t <= previous:
            raise ValueError("Video presentation timestamps must increase strictly")
        previous = t
        if next_t is None or t + tolerance >= next_t:
            if len(selected) >= max_frames:
                raise ValueError("Sample exceeds frame budget; trim video or reduce fps")
            normalized = t + time_offset_s
            if not math.isfinite(normalized) or normalized < 0:
                raise ValueError("Time offset creates an invalid source timestamp")
            if selected and normalized <= selected[-1][1] + time_offset_s:
                raise ValueError("Time offset loses source timestamp precision")
            selected.append((index, t))
            next_t = t + 1 / fps if next_t is None else next_t + 1 / fps
            if next_t <= t + tolerance:
                next_t += (math.floor((t + tolerance - next_t) * fps) + 1) / fps
    if not selected:
        raise ValueError("Video has no timestamped frames")
    output.mkdir(parents=True, exist_ok=True)
    frames_dir = output / sensor_id
    frames_dir.mkdir(exist_ok=True)
    if any(frames_dir.iterdir()):
        raise ValueError("Use an empty sensor output directory to preserve prior extraction")
    expression = _selection_expression([index for index, _ in selected])
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(video), "-map", "0:v:0",
                    "-vf", "select=" + expression, "-fps_mode", "vfr", "-q:v", "2", "-start_number", "0",
                    str(frames_dir / "frame-%06d.jpg")], check=True, timeout=180)
    files = sorted(frames_dir.glob("frame-*.jpg"))
    if len(files) != len(selected):
        raise ValueError("Decoded image count does not match the selected source frame indices")
    digest = hashlib.sha256()
    with video.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    rows = [{"frame_id": f"{sensor_id}-f{index:07d}", "sensor_id": sensor_id,
             "original_frame_index": index, "t": t + time_offset_s,
             "image": str(file.relative_to(output))} for (index, t), file in zip(selected, files)]
    return {"video_name": video.name, "video_sha256": digest.hexdigest(), "sensor_id": sensor_id,
            "sampling_fps_requested": fps, "time_offset_s": time_offset_s,
            "timestamp_quantization_s": time_base, "sampling_tolerance_s": tolerance,
            "source_frame_count": len(raw), "synchronization_verified": False, "frames": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sensor", required=True)
    parser.add_argument("--fps", type=float, default=5)
    parser.add_argument("--time-offset", type=float, required=True)
    args = parser.parse_args()
    result = extract_video(args.video, args.output, sensor_id=args.sensor, fps=args.fps, time_offset_s=args.time_offset)
    destination = args.output / (args.sensor + "-frames.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"metadata": str(destination), "frames": len(result["frames"]), "synchronization_verified": False}))


if __name__ == "__main__":
    main()
