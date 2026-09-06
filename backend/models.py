"""Select an installed model explicitly, or prefer approved Omega weights in auto mode."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_config():
    local = ROOT / "models/vggt-omega/vggt_omega_1b_512.pt"
    downloaded = Path.home() / "Downloads/vggt_omega_1b_512.pt"
    default_omega = local if local.is_file() or not downloaded.is_file() else downloaded
    omega = Path(
        os.environ.get("VGGT_OMEGA_CHECKPOINT", str(default_omega))
    ).expanduser()
    selected = os.environ.get("SIMV1_MODEL", "auto")
    if selected == "auto":
        selected = (
            "vggt_omega"
            if "VGGT_OMEGA_CHECKPOINT" in os.environ or omega.is_file()
            else "vggt"
        )
    if selected == "vggt_omega":
        return {
            "key": selected,
            "model": "facebook/VGGT-Omega",
            "variant": "1B-512",
            "checkpoint": omega,
            "download": "scripts/download_vggt_omega.py",
        }
    if selected == "vggt":
        return {
            "key": selected,
            "model": "facebook/VGGT-1B",
            "variant": "1B",
            "checkpoint": Path(
                os.environ.get(
                    "VGGT_CHECKPOINT", str(ROOT / "models/vggt-1b/model.safetensors")
                )
            ).expanduser(),
            "download": "scripts/download_vggt.py",
        }
    raise ValueError("SIMV1_MODEL must be auto, vggt, or vggt_omega.")
