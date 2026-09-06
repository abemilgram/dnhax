"""Model inference and export in the coordinate system shared by each input sequence."""

import json
import shutil
import subprocess
import numpy as np
from . import store
from .geometry import depth_edge_mask
from .runtime import select_device, frame_limit
from .models import model_config


def compute_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Install PyTorch and VGGT. See MACOS.md or README.md."
        ) from exc
    return select_device(torch)


def infer_images(images, device, progress):
    from .model_runtime import runtime

    return runtime.infer(images, device, progress)


def export_cloud(folder, capture, frames, prediction, indices, device, **extra):
    """Split provenance only: never recenter, rescale or refit a source's geometry."""
    confidence = prediction["confidence"][indices]
    rgb = prediction["rgb"][indices]
    ex = prediction.get("extrinsics")
    ex = ex[indices] if ex is not None else None
    ins = prediction.get("intrinsics")
    ins = ins[indices] if ins is not None else None
    depth = prediction.get("depth")
    depth = depth[indices] if depth is not None else None
    world_points = prediction.get("world_points")
    world_points = world_points[indices] if world_points is not None else None
    unmapped = set(prediction.get("unmapped_frames", []))
    if confidence.ndim == 4:
        confidence = confidence[..., 0]
    points, colors, cameras = [], [], []
    for i, original_index in enumerate(indices):
        if original_index in unmapped:
            continue
        if world_points is not None:
            world = world_points[i]
            valid = np.isfinite(world).all(-1) & np.isfinite(confidence[i])
            if ex is not None:
                camera_points = (
                    world @ ex[i, :3, :3].T + ex[i, :3, 3]
                )
                camera_depth = camera_points[..., 2]
                valid &= (camera_depth > 0) & ~depth_edge_mask(camera_depth)
        else:
            d = depth[i]
            h, w = d.shape
            y, x = np.mgrid[:h, :w]
            rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(ins[i]).T
            cam = rays * d[..., None]
            world = (cam - ex[i, :3, 3]) @ ex[i, :3, :3]
            valid = np.isfinite(world).all(-1) & (d > 0) & np.isfinite(confidence[i])
            valid &= ~depth_edge_mask(d)
        valid &= confidence[i] > 1e-5
        if valid.any():
            valid &= confidence[i] >= np.percentile(confidence[i][valid], 20)
        points.append(world[valid])
        colors.append(
            (np.clip(rgb[i].transpose(1, 2, 0), 0, 1) * 255).astype("uint8")[valid]
        )
        camera = {
            "frame": store.artifact_url(frames[i]["path"]),
            "t": frames[i]["t"],
        }
        if ins is not None:
            camera["intrinsics"] = ins[i].tolist()
        if ex is not None:
            camera["world_to_camera"] = ex[i].tolist()
        cameras.append(camera)
    if not points:
        raise RuntimeError(
            f"No mapped geometry was reconstructed for source {capture['source']}."
        )
    points, colors = np.concatenate(points), np.concatenate(colors)
    if not len(points):
        raise RuntimeError(
            f"No valid geometry was reconstructed for source {capture['source']}."
        )
    folder.mkdir(parents=True, exist_ok=True)
    geometry = {"confidence": confidence}
    for key, value in (
        ("depth", depth),
        ("world_points", world_points),
        ("extrinsics", ex),
        ("intrinsics", ins),
    ):
        if value is not None:
            geometry[key] = value
    np.savez_compressed(folder / "geometry.npz", **geometry)
    from .worker import cloud_record

    limit = extra.pop("point_limit", 1000000)
    if len(points) > limit:
        keep = np.linspace(0, len(points) - 1, limit).astype(int)
        points, colors = points[keep], colors[keep]
    record = cloud_record(
        folder,
        points,
        colors,
        capture["source"],
        capture_id=capture["id"],
        cameras=cameras,
        compute_device=device,
        frame_count=len(frames),
        precision=prediction.get("precision", "float32"),
        model=prediction["model"],
        model_variant=prediction["model_variant"],
        **extra,
    )
    (folder / "cloud.json").write_text(json.dumps(record, allow_nan=False))
    return record


def reconstruct(capture, progress):
    device = compute_device()
    max_frames = frame_limit(device)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required on the processing machine.")
    config = model_config()
    capture_folder = store.ROOT / "reconstructions" / capture["id"]
    # Immutable model-specific artifacts keep previously published scenes intact.
    folder = capture_folder / config["key"]
    frames = folder / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    if (folder / "cloud.json").exists():
        record = json.loads((folder / "cloud.json").read_text())
        (capture_folder / "cloud.json").write_text(json.dumps(record, allow_nan=False))
        return record
    progress(
        f"Extracting {'all' if max_frames is None else f'up to {max_frames}'} "
        f"1 fps frames for {device.upper()}"
    )
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-i",
        capture["path"],
        "-vf",
        "fps=1,vflip,scale=960:960:force_original_aspect_ratio=decrease",
    ]
    if max_frames is not None:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(frames / "%04d.jpg"))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=180 if max_frames is not None else 3600,
    )
    if result.returncode:
        raise RuntimeError(
            "FFmpeg could not decode this capture: " + result.stderr[-400:]
        )
    images = sorted(frames.glob("*.jpg"))
    if max_frames is not None:
        images = images[:max_frames]
    if len(images) < 2:
        raise RuntimeError(
            "Capture needs at least two usable frames. Submit a longer walkthrough."
        )
    prediction = infer_images(images, device, progress)
    progress("Exporting geometry and source cameras")
    record = export_cloud(
        folder,
        capture,
        [{"path": p, "t": float(i)} for i, p in enumerate(images)],
        prediction,
        list(range(len(images))),
        device,
    )
    (capture_folder / "cloud.json").write_text(json.dumps(record, allow_nan=False))
    return record
