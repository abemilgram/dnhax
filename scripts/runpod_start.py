"""Prepare persistent AMB3R storage and serve, failing on any model error."""

import os
from pathlib import Path
import sys

from backend.models import model_config


def ensure_checkpoint() -> None:
    config = model_config()
    target = config["checkpoint"]
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location(
        "download_amb3r", Path(__file__).with_name("download_amb3r.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the AMB3R checkpoint downloader.")
    downloader = module_from_spec(spec)
    spec.loader.exec_module(downloader)
    if target.is_file():
        downloader.validate(target)
        print(f"Using checkpoint {target}", flush=True)
        return
    if os.environ.get("SIMV1_DOWNLOAD_MODEL", "1") != "1":
        raise RuntimeError(f"Missing checkpoint and download is disabled: {target}")
    print(
        f"Downloading {config['model']} to persistent storage at {target}",
        flush=True,
    )
    downloader.download(target)


def verify_cuda() -> None:
    if os.environ.get("SIMV1_DEVICE", "cuda") != "cuda":
        raise RuntimeError("The AMB3R image requires SIMV1_DEVICE=cuda.")
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
