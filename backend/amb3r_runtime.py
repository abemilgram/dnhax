"""In-process adapter for the pinned AMB3R model and AMB3R-SfM pipeline."""

import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from .runtime import release_memory


AMB3R_ROOT = Path(os.environ.get("AMB3R_ROOT", "/opt/amb3r"))
SFM_CONFIG = Path(__file__).with_name("amb3r_sfm.yaml")
TARGET_SIZE = (518, 392)


def _load_image(path):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        image = ImageOps.fit(
            image,
            TARGET_SIZE,
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        pixels = np.asarray(image, dtype=np.float32) / 127.5 - 1.0
    return np.transpose(pixels, (2, 0, 1))


def _world_to_camera(poses):
    return np.linalg.inv(poses)[:, :3].astype(np.float32)


def _cpu_array(tensor):
    return tensor.detach().float().cpu().numpy()


class Amb3rRuntime:
    def __init__(self):
        self.model = self.key = self.device = None
        self.loads = 0

    @staticmethod
    def fingerprint(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def unload(self):
        self.model = self.key = None
        if self.device:
            import torch

            release_memory(torch, self.device)
        self.device = None

    def load(self, config, device, progress):
        if device != "cuda":
            raise RuntimeError(
                "AMB3R requires CUDA. Set SIMV1_MODEL=vggt for macOS or CPU use."
            )
        checkpoint = config["checkpoint"]
        if not checkpoint.is_file():
            raise RuntimeError(
                f"Missing {config['model']} checkpoint: {checkpoint}. "
                f"Run {config['download']}."
            )
        fingerprint = self.fingerprint(checkpoint)
        key = (str(checkpoint.resolve()), fingerprint, device, "bfloat16")
        if self.model is not None and self.key == key:
            return 0.0

        self.unload()
        if not AMB3R_ROOT.is_dir():
            raise RuntimeError(
                f"AMB3R source is missing at {AMB3R_ROOT}. Use the AMB3R Docker image."
            )
        thirdparty = str(AMB3R_ROOT / "thirdparty")
        root = str(AMB3R_ROOT)
        for entry in (root, thirdparty):
            if entry in sys.path:
                sys.path.remove(entry)
        # AMB3R needs its modified VGGT fork, not the public vggt package.
        sys.path[:0] = [thirdparty, root]

        progress("Loading AMB3R weights; retaining them for subsequent batches")
        started = time.perf_counter()
        try:
            from amb3r.model import AMB3R
        except ImportError as exc:
            raise RuntimeError(
                "AMB3R dependencies are incomplete. Rebuild the AMB3R Docker image."
            ) from exc
        model = AMB3R(device=device, precision="bf16")
        # Upstream's loader is required because the released file is a raw state dict.
        model.load_weights(str(checkpoint), data_type="bf16")
        self.model = model.eval().to(device)
        self.key = key
        self.device = device
        self.loads += 1
        return time.perf_counter() - started

    def infer(self, images, config, device, progress):
        import torch

        load_seconds = self.load(config, device, progress)
        started = time.perf_counter()
        inputs = torch.from_numpy(np.stack([_load_image(path) for path in images]))
        inputs = inputs.unsqueeze(0)
        progress(f"Reconstructing {len(images)} frames together with AMB3R")

        memory = prediction = None
        try:
            if len(images) >= 5:
                from sfm.pipeline import AMB3R_SfM

                pipeline = AMB3R_SfM(self.model, cfg_path=str(SFM_CONFIG))
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    memory = pipeline.run(inputs)
                points = _cpu_array(memory.pts)
                confidence = _cpu_array(memory.conf)
                poses = _cpu_array(memory.poses)
                unmapped = sorted(memory.unmapped_frames)
                mode = "unordered_sfm"
                intrinsics = None
                depth = None
            else:
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    prediction = self.model(
                        {"images": inputs.to(device=device, dtype=torch.float32)}
                    )[-1]
                points = _cpu_array(prediction["world_points"][0])
                raw_confidence = _cpu_array(prediction["world_points_conf"][0])
                confidence = (raw_confidence - 1.0) / np.maximum(
                    raw_confidence, 1e-6
                )
                poses = _cpu_array(prediction["pose"][0])
                intrinsics = _cpu_array(prediction["intrinsic"][0])
                depth = _cpu_array(prediction["depth"][0, ..., 0])
                unmapped = []
                mode = "joint_forward"

            result = {
                "model": config["model"],
                "model_key": "amb3r",
                "model_variant": config["variant"],
                "checkpoint_sha256": self.key[1],
                "preprocessing": {
                    "width": TARGET_SIZE[0],
                    "height": TARGET_SIZE[1],
                    "fit": "center_crop",
                    "range": [-1, 1],
                    "mode": mode,
                },
                "world_points": points,
                "confidence": confidence,
                "poses_c2w": poses,
                "extrinsics": _world_to_camera(poses),
                "rgb": ((inputs[0].numpy() + 1.0) / 2.0).astype(np.float32),
                "unmapped_frames": unmapped,
                "precision": "bfloat16",
            }
            if intrinsics is not None:
                result["intrinsics"] = intrinsics
            if depth is not None:
                result["depth"] = depth
            result["timings"] = {
                "model_load_seconds": load_seconds,
                "prediction_seconds": time.perf_counter() - started,
            }
            return result
        finally:
            del memory, prediction, inputs
            release_memory(torch, device)
