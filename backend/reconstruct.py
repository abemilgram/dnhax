"""VGGT inference and export in the coordinate system shared by each input sequence."""

import json
import shutil
import subprocess
import numpy as np
from . import store
from .geometry import depth_edge_mask
from .runtime import select_device, frame_limit, predict_geometry, release_memory
from .models import model_config


def compute_device():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Install PyTorch and VGGT. See MACOS.md or README.md.") from exc
    return select_device(torch)


def infer_images(images, device, progress):
    """One sequence / one inference call, including frames from two captures."""
    config = model_config()
    omega = config["key"] == "vggt_omega"
    checkpoint = config["checkpoint"]
    if not checkpoint.is_file():
        raise RuntimeError(
            f"Missing {config['model']} checkpoint: {checkpoint}. Run {config['download']}."
        )
    try:
        import torch
        from safetensors.torch import load_file
        if omega:
            from vggt_omega.models import VGGTOmega as Model
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            from vggt_omega.utils.pose_enc import encoding_to_camera as decode_camera
        else:
            from vggt.models.vggt import VGGT as Model
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri as decode_camera
    except ImportError as exc:
        raise RuntimeError(
            f"Install PyTorch and the official {config['key']} package. See MACOS.md."
        ) from exc
    model = inputs = predictions = extrinsics = intrinsics = state = None
    try:
        progress(f"Loading {config['model']} {config['variant']} weights")
        model = (Model() if omega else Model(enable_point=False, enable_track=False)).eval()
        state = (load_file(str(checkpoint), device="cpu")
                 if checkpoint.suffix == ".safetensors"
                 else torch.load(str(checkpoint), map_location="cpu", weights_only=True))
        incompatible = model.load_state_dict(state, strict=omega)
        unexpected = [key for key in incompatible.unexpected_keys
                      if not key.startswith(("point_head.", "track_head."))]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError(
                "VGGT checkpoint does not match the pinned model code: "
                f"{len(incompatible.missing_keys)} missing and {len(unexpected)} unexpected keys."
            )
        state = None
        model = model.to(device=device, dtype=torch.float32)
        preprocessing = {"image_resolution": 512} if omega else {"mode": "crop"}
        inputs = load_and_preprocess_images([str(p) for p in images], **preprocessing).to(
            device=device, dtype=torch.float32
        )
        progress(f"Predicting {len(images)} frames together on {device.upper()}")
        with torch.inference_mode():
            predictions = predict_geometry(model, inputs, device, config["key"])
            extrinsics, intrinsics = decode_camera(
                predictions["pose_enc"], predictions["images"].shape[-2:]
            )
        return {
            "model": config["model"], "model_variant": config["variant"],
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
                          precision="float32", model=prediction["model"],
                          model_variant=prediction["model_variant"], **extra)
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
    record = export_cloud(folder, capture,
                        [{"path": p, "t": float(i)} for i, p in enumerate(images)],
                        prediction, list(range(len(images))), device)
    (capture_folder / "cloud.json").write_text(json.dumps(record, allow_nan=False))
    return record
