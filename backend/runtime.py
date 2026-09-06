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
