"""Download the official AMB3R checkpoint into persistent model storage."""

import os
from pathlib import Path


DRIVE_ID = "14x0WW2rUE_he2hUEouP6ywSRnlJDeLel"
MIN_CHECKPOINT_BYTES = 3_000_000_000
ROOT = Path(__file__).resolve().parents[1]


def checkpoint_path():
    return Path(
        os.environ.get(
            "AMB3R_CHECKPOINT", str(ROOT / "models/amb3r/amb3r.pt")
        )
    ).expanduser()


def validate(path):
    if not path.is_file() or path.stat().st_size < MIN_CHECKPOINT_BYTES:
        raise RuntimeError(
            f"AMB3R download is incomplete or is not a checkpoint: {path}"
        )


def download(target=None):
    import gdown

    target = Path(target) if target else checkpoint_path()
    if target.is_file():
        validate(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.download")
    result = gdown.download(
        id=DRIVE_ID,
        output=str(temporary),
        quiet=False,
        resume=True,
    )
    if not result:
        raise RuntimeError("Google Drive did not return the AMB3R checkpoint.")
    validate(temporary)
    temporary.replace(target)
    return target


if __name__ == "__main__":
    print(download())
