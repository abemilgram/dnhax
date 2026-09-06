"""Check the selected backend with a small allocation; no model weights loaded."""

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.runtime import select_device, frame_limit, release_memory
from backend.models import model_config


def main():
    report = {"python": sys.version.split()[0], "architecture": platform.machine()}
    try:
        import torch

        report["torch"] = torch.__version__
        device = select_device(torch)
        report.update(device=device, max_frames=frame_limit(device))
        if device != "cuda":
            raise RuntimeError(
                "AMB3R reconstruction requires CUDA; CPU and MPS are unsupported."
            )
        sample = torch.ones((16, 16), device=device)
        result = (sample @ sample).cpu()
        assert result[0, 0].item() == 16
        del sample, result
        release_memory(torch, device)
        report["device_smoke_test"] = "passed"
        config = model_config()
        report["model"] = config["model"]
        report["model_variant"] = config["variant"]
        amb3r_root = Path(os.environ.get("AMB3R_ROOT", "/opt/amb3r"))
        report["backend_installed"] = (
            (amb3r_root / "amb3r/model.py").is_file()
            and all(
                importlib.util.find_spec(module) is not None
                for module in ("spconv", "torch_scatter", "pytorch3d")
            )
        )
        checkpoint = config["checkpoint"]
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_present"] = checkpoint.is_file()
        print(json.dumps(report, indent=2))
        return 0 if report["backend_installed"] and report["checkpoint_present"] else 2
    except Exception as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
