"""Verify decoded frame selection preserves source PTS and image order."""

import shutil
import subprocess
import pytest
from PIL import Image
from backend.prediction.media import extract_video


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe required")
def test_frame_selection_keeps_native_indices_pts_and_pixels(tmp_path):
    inputs = tmp_path / "input"
    inputs.mkdir()
    colors = [(220, 10, 10), (10, 220, 10), (10, 10, 220), (220, 220, 10), (10, 220, 220)]
    for index, color in enumerate(colors):
        Image.new("RGB", (64, 48), color).save(inputs / f"{index:03d}.png")
    video = tmp_path / "test.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-framerate", "5", "-i", str(inputs / "%03d.png"),
                    "-c:v", "ffv1", str(video)], check=True, timeout=20)
    result = extract_video(video, tmp_path / "sample", sensor_id="drone", fps=2.5, time_offset_s=7)
    assert result["source_frame_count"] == 5
    assert not result["synchronization_verified"]
    assert len(result["video_sha256"]) == 64
    assert [f["original_frame_index"] for f in result["frames"]] == [0, 2, 4]
    assert [f["t"] for f in result["frames"]] == pytest.approx([7, 7.4, 7.8])
    for row, index in zip(result["frames"], [0, 2, 4]):
        with Image.open(tmp_path / "sample" / row["image"]) as image:
            assert max(abs(a-b) for a, b in zip(image.getpixel((32, 24)), colors[index])) <= 4
    with pytest.raises(ValueError, match="empty sensor"):
        extract_video(video, tmp_path / "sample", sensor_id="drone", fps=2.5, time_offset_s=7)
    with pytest.raises(ValueError, match="frame budget"):
        extract_video(video, tmp_path / "too-many", sensor_id="drone", fps=5, time_offset_s=0, max_frames=2)


def test_invalid_rate_and_sensor_rejected_before_decode(tmp_path):
    for sensor, rate in [("../escape", 5), ("drone", float("nan")), ("drone", 100)]:
        with pytest.raises(ValueError):
            extract_video(tmp_path / "missing.mp4", tmp_path / "out", sensor_id=sensor, fps=rate, time_offset_s=0)


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg/ffprobe required")
def test_more_than_100_frame_choices_and_fractional_rate_do_not_drift(tmp_path):
    video = tmp_path / "long.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=32x32:rate=30:duration=4",
                    "-c:v", "ffv1", str(video)], check=True, timeout=20)
    result = extract_video(video, tmp_path / "all", sensor_id="A", fps=30, time_offset_s=0)
    assert len(result["frames"]) == 120
    sampled = extract_video(video, tmp_path / "sample", sensor_id="A", fps=4, time_offset_s=0)
    assert len(sampled["frames"]) == 16
    assert [f["original_frame_index"] for f in sampled["frames"][:5]] == [0, 8, 15, 23, 30]
