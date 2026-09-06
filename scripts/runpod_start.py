"""Prepare persistent RunPod storage, download selected weights, and serve."""

import hashlib
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

from backend.models import model_config


MODEL_RELEASES = {
    "vggt": {
        "repository": "facebook/VGGT-1B",
        "filename": "model.safetensors",
        "revision": "860abec7937da0a4c03c41d3c269c366e82abdf9",
        "sha256": None,
    },
    "vggt_omega": {
        "repository": "facebook/VGGT-Omega",
        "filename": "vggt_omega_1b_512.pt",
        "revision": "55d1f4b2ce41fd0925887362a4a5049ce94e0be1",
        "sha256": "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934",
    },
}


def verify(path: Path, expected: str | None) -> None:
    if expected is None:
        return
    with path.open("rb") as checkpoint:
        actual = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if actual != expected:
        path.unlink(missing_ok=True)
        raise RuntimeError(f"Checkpoint checksum does not match: {path}")


def ensure_checkpoint() -> None:
    config = model_config()
    target = config["checkpoint"]
    release = MODEL_RELEASES[config["key"]]
    if target.is_file():
        verify(target, release["sha256"])
        print(f"Using checkpoint {target}", flush=True)
        return
    if os.environ.get("SIMV1_DOWNLOAD_MODEL", "1") != "1":
        raise RuntimeError(f"Missing checkpoint and download is disabled: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    print(
        f"Downloading {config['model']} to persistent storage at {target}",
        flush=True,
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=release["repository"],
            filename=release["filename"],
            revision=release["revision"],
            local_dir=target.parent,
        )
    )
    if downloaded.resolve() != target.resolve():
        temporary = target.with_name(f".{target.name}.download")
        shutil.copyfile(downloaded, temporary)
        temporary.replace(target)
    verify(target, release["sha256"])


def verify_cuda() -> None:
    if os.environ.get("SIMV1_DEVICE", "cuda") != "cuda":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("SIMV1_DEVICE=cuda, but CUDA is unavailable in this container.")
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    print(
        f"CUDA ready: {name}; capability {capability}; "
        f"PyTorch {torch.__version__}; CUDA {torch.version.cuda}",
        flush=True,
    )


def main() -> None:
    Path(os.environ.get("SIMV1_DATA", "/workspace/simv1-data")).mkdir(
        parents=True, exist_ok=True
    )
    verify_cuda()
    ensure_checkpoint()
    port = int(os.environ.get("PORT", "8000"))
    os.execv(
        sys.executable,
        [
            sys.executable,
            "scripts/serve.py",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
    )


if __name__ == "__main__":
    main()
