"""Optional CUDA adapter. Never substitutes fixture geometry for real captures."""

import json
import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from . import store
from .geometry import depth_edge_mask


def reconstruct(capture, progress):
    try:
        import torch
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import load_and_preprocess_images
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ImportError as exc:
        raise RuntimeError(
            "Install CUDA PyTorch and the official vggt-omega package in this worker environment. See README.md."
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run the worker on the NVIDIA laptop; sample scenes remain available on CPU."
        )
    checkpoint = Path(os.environ.get("VGGT_OMEGA_CHECKPOINT", ""))
    if not checkpoint.is_file():
        raise RuntimeError(
            "Set VGGT_OMEGA_CHECKPOINT to your approved VGGT-Ω 1B 512 checkpoint."
        )
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required on the processing machine.")
    folder = store.ROOT / "reconstructions" / capture["id"]
    frames = folder / "frames"
    frames.mkdir(parents=True, exist_ok=True)
    if (folder / "cloud.json").exists():
        return json.loads((folder / "cloud.json").read_text())
    # Bounded submitted-clip processing. This is not a streaming reconstruction loop.
    progress("Extracting up to 24 frames from the submitted clip")
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            capture["path"],
            "-vf",
            "fps=1,scale=960:960:force_original_aspect_ratio=decrease",
            "-frames:v",
            "24",
            str(frames / "%04d.jpg"),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(
            "FFmpeg could not decode this capture: " + result.stderr[-400:]
        )
    images = sorted(frames.glob("*.jpg"))
    if len(images) < 2:
        raise RuntimeError(
            "Capture needs at least two usable frames. Submit a longer walkthrough."
        )
    progress(f"Reconstructing {len(images)} frames on CUDA")
    model = None
    predictions = None
    inputs = None
    try:
        model = VGGTOmega().to("cuda").eval()
        model.load_state_dict(
            torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        )
        inputs = load_and_preprocess_images(
            [str(p) for p in images], image_resolution=512
        ).to("cuda")
        with torch.inference_mode():
            predictions = model(inputs)
        extrinsics, intrinsics = encoding_to_camera(
            predictions["pose_enc"], predictions["images"].shape[-2:]
        )
        depth = predictions["depth"].detach().float().cpu().numpy().squeeze(0)
        confidence = predictions["depth_conf"].detach().float().cpu().numpy().squeeze(0)
        ex = extrinsics.detach().float().cpu().numpy().squeeze(0)
        ins = intrinsics.detach().float().cpu().numpy().squeeze(0)
        rgb = predictions["images"].detach().float().cpu().numpy().squeeze(0)
        if depth.ndim == 4:
            depth = depth[..., 0]
        if confidence.ndim == 4:
            confidence = confidence[..., 0]
        points = []
        colors = []
        camera_records = []
        for i, d in enumerate(depth):
            h, w = d.shape
            y, x = np.mgrid[:h, :w]
            rays = np.stack([x, y, np.ones_like(x)], -1) @ np.linalg.inv(ins[i]).T
            cam = rays * d[..., None]
            rot = ex[i, :3, :3]
            translation = ex[i, :3, 3]
            world = (cam - translation) @ rot
            valid = np.isfinite(world).all(-1) & (d > 0) & np.isfinite(confidence[i])
            valid &= ~depth_edge_mask(d)
            valid &= confidence[i] > 1e-5
            if valid.any():
                valid &= confidence[i] >= np.percentile(confidence[i][valid], 20)
            points.append(world[valid])
            colors.append(
                (np.clip(rgb[i].transpose(1, 2, 0), 0, 1) * 255).astype("uint8")[valid]
            )
            camera_records.append(
                {
                    "frame": store.artifact_url(images[i]),
                    "t": float(i),
                    "intrinsics": ins[i].tolist(),
                    "world_to_camera": ex[i].tolist(),
                }
            )
        points = np.concatenate(points)
        colors = np.concatenate(colors)
        if not len(points):
            raise RuntimeError("No valid geometry was reconstructed.")
        np.savez_compressed(
            folder / "geometry.npz",
            depth=depth,
            confidence=confidence,
            extrinsics=ex,
            intrinsics=ins,
        )
        from .worker import cloud_record

        progress("Exporting geometry and source cameras")
        record = cloud_record(
            folder,
            points,
            colors,
            capture["source"],
            capture_id=capture["id"],
            cameras=camera_records,
        )
        (folder / "cloud.json").write_text(json.dumps(record, allow_nan=False))
        return record
    finally:
        del predictions, inputs, model
        torch.cuda.empty_cache()
