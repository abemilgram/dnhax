"""AMB3R-only reconstruction configuration.

The reconstruction path deliberately has no model fallback: configuration,
dependency, checkpoint, or CUDA failures must stop the job loudly.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def model_config():
    selected = os.environ.get("SIMV1_MODEL", "amb3r")
    if selected != "amb3r":
        raise ValueError(
            "Only SIMV1_MODEL=amb3r is supported; fallback models are disabled."
        )
    amb3r = Path(
        os.environ.get(
            "AMB3R_CHECKPOINT", str(ROOT / "models/amb3r/amb3r.pt")
        )
    ).expanduser()
    return {
        "key": selected,
        "model": "AMB3R-SfM",
        "variant": "92c4081",
        "checkpoint": amb3r,
        "download": "scripts/download_amb3r.py",
    }
