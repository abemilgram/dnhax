"""VGGT inference and export in the coordinate system shared by each input sequence."""

import json
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from . import store
from .geometry import depth_edge_mask
from .runtime import select_device, frame_limit, predict_geometry, release_memory


def compute_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and VGGT. See MACOS.md or README.md.") from exc
    return select_device(torch)


def infer_images(images, device, progress):
    """One sequence / one inference call, including frames from two captures."""
    try:
        import torch
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    except ImportError as exc:
        raise RuntimeError(
            "Install PyTorch and the official vggt package. See MACOS.md or README.md."
        ) from exc
    checkpoint = Path(os.environ.get(
        "VGGT_CHECKPOINT",
        str(Path(__file__).resolve().parents[1] / "models" / "vggt-1b" / "model.safetensors"),
    ))
    if not checkpoint.is_file():
        raise RuntimeError(
            "Download the public VGGT-1B checkpoint with scripts/download_vggt.py or set VGGT_CHECKPOINT."
        )
    model = inputs = predictions = extrinsics = intrinsics = state = None
    try:
        progress("Loading VGGT-1B weights")
        model = VGGT(enable_point=False, enable_track=False).eval()
        state = (load_file(str(checkpoint), device="cpu")
                 if checkpoint.suffix == ".safetensors"
                 else torch.load(str(checkpoint), map_location="cpu", weights_only=True))
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys
                      if not key.startswith(("point_head.", "track_head."))]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "VGGT checkpoint does not match the pinned model code: "
                f"{len(incompatible.missing_keys)} missing and {len(unexpected)} unexpected keys."
            )
        state = None
        model = model.to(device=device, dtype=torch.float32)
        inputs = load_and_preprocess_images([str(p) for p in images], mode="crop").to(
            device=device, dtype=torch.float32
        )
        progress(f"Predicting {len(images)} frames together on {device.upper()}")
        with torch.inference_mode():
            predictions = predict_geometry(model, inputs, device)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"], predictions["images"].shape[-2:]
            )
        return {
            "depth": predictions["depth"].detach().float().cpu().numpy()[0, ..., 0],
            "confidence": predictions["depth_conf"].detach().float().cpu().numpy()[0],
            "extrinsics": extrinsics.detach().float().cpu().numpy()[0],
            "intrinsics": intrinsics.detach().float().cpu().numpy()[0],
            "rgb": predictions["images"].detach().float().cpu().numpy()[0],
        }
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise RuntimeError(
                f"{device.upper()} ran out of memory processing {len(images)} frames. "
                "Reduce SIMV1_JOINT_FRAMES_PER_SOURCE for joint jobs or SIMV1_MAX_FRAMES "
                "for single captures, restart the worker, and retry. No automatic CPU retry was performed."
            ) from exc
        raise
    finally:
        del predictions, inputs, model, extrinsics, intrinsics, state
        release_memory(torch, device)


def export_cloud(folder, capture, frames, prediction, indices, device, **extra):
    """Split provenance only: never recenter, rescale or refit a source's geometry."""
    depth = prediction["depth"][indices]
    confidence = prediction["confidence"][indices]
    ex = prediction["extrinsics"][indices]
    ins = prediction["intrinsics"][indices]
    rgb = prediction["rgb"][indices]
    if confidence.ndim == 4:
        confidence = confidence[..., 0]
    points, colors, cameras = [], [], []
    for i, d in enumerate(depth):
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
        colors.append((np.clip(rgb[i].transpose(1, 2, 0), 0, 1) * 255).astype("uint8")[valid])
        cameras.append({
            "frame": store.artifact_url(frames[i]["path"]),
            "t": frames[i]["t"],
            "intrinsics": ins[i].tolist(),
            "world_to_camera": ex[i].tolist(),
        })
    points, colors = np.concatenate(points), np.concatenate(colors)
    if not len(points):
        raise RuntimeError(f"No valid geometry was reconstructed for source {capture['source']}.")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "geometry.npz", depth=depth, confidence=confidence,
                        extrinsics=ex, intrinsics=ins)
    from .worker import cloud_record
    record = cloud_record(folder, points, colors, capture["source"],
                          capture_id=capture["id"], cameras=cameras,
                          compute_device=device, frame_count=len(frames),
                          precision="float32", model="facebook/VGGT-1B", **extra)
    (folder / "cloud.json").write_text(json.dumps(record, allow_nan=False))
    return record


def reconstruct(capture, progress):
    device = compute_device()
    max_frames = frame_limit(device)
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required on the processing machine.")
    folder = store.ROOT / "reconstructions" / capture["id"]
    frames = folder / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    if (folder / "cloud.json").exists():
        return json.loads((folder / "cloud.json").read_text())
    progress(f"Extracting up to {max_frames} frames for {device.upper()}")
    result = subprocess.run([
        "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", capture["path"],
        "-vf", "fps=1,scale=960:960:force_original_aspect_ratio=decrease",
        "-frames:v", str(max_frames), str(frames / "%04d.jpg"),
    ], capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError("FFmpeg could not decode this capture: " + result.stderr[-400:])
    images = sorted(frames.glob("*.jpg"))[:max_frames]
    if len(images) < 2:
        raise RuntimeError("Capture needs at least two usable frames. Submit a longer walkthrough.")
    prediction = infer_images(images, device, progress)
    progress("Exporting geometry and source cameras")
    return export_cloud(folder, capture,
                        [{"path": p, "t": float(i)} for i, p in enumerate(images)],
                        prediction, list(range(len(images))), device)
