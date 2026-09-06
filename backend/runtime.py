"""Device policy for batch inference; imports Torch only when requested."""

import gc
import os


def select_device(torch, requested=None):
    requested = requested or os.environ.get("SIMV1_DEVICE", "auto")
    if requested not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("SIMV1_DEVICE must be auto, cuda, mps, or cpu.")
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if requested == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise RuntimeError(
            "CUDA is unavailable. Choose SIMV1_DEVICE=mps on Apple Silicon or cpu for a slow compatibility run."
        )
    if requested == "mps" and not mps:
        raise RuntimeError(
            "Apple MPS is unavailable to this Python process. Use native arm64 Python and an MPS-enabled PyTorch in a normal macOS terminal, or explicitly select cpu."
        )
    return requested


_UNLIMITED = {"", "0", "inf", "infinite", "unlimited", "none"}


def parse_optional_frames(raw, name, minimum=2):
    """Return a frame count, or None when the caller asked for no cap."""
    text = str(raw).strip().lower()
    if text in _UNLIMITED:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be 0 (unlimited) or an integer of at least {minimum}."
        ) from exc
    if value < minimum:
        raise ValueError(
            f"{name} must be 0 (unlimited) or an integer of at least {minimum}."
        )
    return value


def frame_limit(device):
    del device
    return parse_optional_frames(
        os.environ.get("SIMV1_MAX_FRAMES", "0"),
        "SIMV1_MAX_FRAMES",
    )


def device_label(torch):
    device = select_device(torch)
    if device == "cuda":
        return f"CUDA / {torch.cuda.get_device_name(0)}"
    if device == "mps":
        return "Apple GPU / MPS / unified memory"
    return "CPU / slow inference; sample and alignment available"


def release_memory(torch, device):
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def predict_geometry(model, images, device, model_key="vggt"):
    """Use upstream CUDA forward unchanged; dispatch VGGT heads without CUDA on Mac.

    Targets the official revisions pinned in requirements-vggt*.txt.
    Omega uses its dense head and direct camera encoding; public VGGT uses
    its depth head and the last iterative camera prediction.
    MPS/CPU use float32 throughout rather than CUDA-specific mixed precision.
    No global Torch patches or edits to the installed upstream package.
    """
    if device == "cuda":
        return model(images)
    batch = images.unsqueeze(0) if images.ndim == 4 else images
    if model.training:
        raise ValueError("Reconstruction requires model.eval().")
    features, patch_start_idx = model.aggregator(batch)
    if not features or features[-1] is None:
        raise RuntimeError("VGGT returned no final features; check the installed revision.")
    result = {"images": batch}
    if model_key == "vggt_omega":
        if model.camera_head is None or model.dense_head is None:
            raise RuntimeError("Omega requires camera and depth heads.")
        result["pose_enc"] = model.camera_head(features, patch_token_start=patch_start_idx)
        result["depth"], result["depth_conf"] = model.dense_head(
            features, images=batch, patch_token_start=patch_start_idx
        )
        return result
    if model.camera_head is None or model.depth_head is None:
        raise RuntimeError("The checkpoint must enable camera and depth heads.")
    result["pose_enc"] = model.camera_head(features)[-1]
    result["depth"], result["depth_conf"] = model.depth_head(
        features, images=batch, patch_start_idx=patch_start_idx
    )
    return result
