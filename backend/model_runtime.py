"""Worker-owned resident model, with checkpoint identity and stage timings."""

import hashlib
import time
from .models import model_config
from .runtime import predict_geometry, release_memory


class ModelRuntime:
    def __init__(self):
        self.model = self.key = self.device = None
        self.loads = 0
        self.fingerprints = {}

    def fingerprint(self, path):
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in self.fingerprints:
            with path.open("rb") as stream:
                self.fingerprints = {
                    key: hashlib.file_digest(stream, "sha256").hexdigest()
                }
        return self.fingerprints[key]

    def unload(self):
        self.model = self.key = None
        if self.device:
            import torch

            release_memory(torch, self.device)

    def load(self, device, progress):
        import torch
        from safetensors.torch import load_file

        config = model_config()
        checkpoint = config["checkpoint"]
        if not checkpoint.is_file():
            raise RuntimeError(
                f"Missing {config['model']} checkpoint: {checkpoint}. Run {config['download']}."
            )
        key = (config["key"], self.fingerprint(checkpoint), device, "float32")
        if self.model is not None and key == self.key:
            return 0.0
        self.unload()
        start = time.perf_counter()
        progress(
            f"Loading {config['model']} weights; retaining them for subsequent batches"
        )
        omega = config["key"] == "vggt_omega"
        if omega:
            from vggt_omega.models import VGGTOmega as Model
            from vggt_omega.utils.load_fn import load_and_preprocess_images
            from vggt_omega.utils.pose_enc import encoding_to_camera as decode
        else:
            from vggt.models.vggt import VGGT as Model
            from vggt.utils.load_fn import load_and_preprocess_images
            from vggt.utils.pose_enc import pose_encoding_to_extri_intri as decode
        model = (
            Model() if omega else Model(enable_point=False, enable_track=False)
        ).eval()
        state = (
            load_file(str(checkpoint), device="cpu")
            if checkpoint.suffix == ".safetensors"
            else torch.load(str(checkpoint), map_location="cpu", weights_only=True)
        )
        incompatible = model.load_state_dict(state, strict=omega)
        unexpected = [
            k
            for k in incompatible.unexpected_keys
            if not k.startswith(("point_head.", "track_head."))
        ]
        if incompatible.missing_keys or unexpected:
            raise RuntimeError("Checkpoint does not match the pinned model code.")
        del state
        self.device = device
        self.model = model.to(device=device, dtype=torch.float32)
        self.preprocess, self.decode, self.config, self.key = (
            load_and_preprocess_images,
            decode,
            config,
            key,
        )
        self.loads += 1
        return time.perf_counter() - start

    def infer(self, images, device, progress):
        import torch

        load_seconds = self.load(device, progress)
        start = time.perf_counter()
        inputs = predictions = ex = ins = None
        try:
            options = (
                {"image_resolution": 512}
                if self.config["key"] == "vggt_omega"
                else {"mode": "crop"}
            )
            inputs = self.preprocess([str(p) for p in images], **options).to(
                device=device, dtype=torch.float32
            )
            progress(f"Predicting {len(images)} frames together on {device.upper()}")
            with torch.inference_mode():
                predictions = predict_geometry(
                    self.model, inputs, device, self.config["key"]
                )
                ex, ins = self.decode(
                    predictions["pose_enc"], predictions["images"].shape[-2:]
                )
            result = {
                "model": self.config["model"],
                "model_variant": self.config["variant"],
                "checkpoint_sha256": self.key[1],
                "preprocessing": options,
                "depth": predictions["depth"].detach().float().cpu().numpy()[0, ..., 0],
                "confidence": predictions["depth_conf"]
                .detach()
                .float()
                .cpu()
                .numpy()[0],
                "extrinsics": ex.detach().float().cpu().numpy()[0],
                "intrinsics": ins.detach().float().cpu().numpy()[0],
                "rgb": predictions["images"].detach().float().cpu().numpy()[0],
            }
            result["timings"] = {
                "model_load_seconds": load_seconds,
                "prediction_seconds": time.perf_counter() - start,
            }
            return result
        finally:
            del inputs, predictions, ex, ins
            release_memory(torch, device)


runtime = ModelRuntime()
