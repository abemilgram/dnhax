"""Check the selected backend with a small allocation; no model weights loaded."""

import importlib.util
import json
import os
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.runtime import select_device, frame_limit, release_memory


def main():
    report = {"python": sys.version.split()[0], "architecture": platform.machine()}
    try:
        import torch

        report["torch"] = torch.__version__
        device = select_device(torch)
        report.update(device=device, max_frames=frame_limit(device))
        sample = torch.ones((16, 16), device=device)
        result = (sample @ sample).cpu()
        assert result[0, 0].item() == 16
        del sample, result
        release_memory(torch, device)
        report["device_smoke_test"] = "passed"
        report["vggt_installed"] = importlib.util.find_spec("vggt") is not None
        checkpoint = Path(
            os.environ.get(
                "VGGT_CHECKPOINT",
                str(
                    Path(__file__).resolve().parents[1]
                    / "models"
                    / "vggt-1b"
                    / "model.safetensors"
                ),
            )
        )
        report["checkpoint"] = str(checkpoint)
        report["checkpoint_present"] = checkpoint.is_file()
        print(json.dumps(report, indent=2))
        return 0 if report["vggt_installed"] and report["checkpoint_present"] else 2
    except Exception as exc:
        report["error"] = str(exc)
        print(json.dumps(report, indent=2))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
