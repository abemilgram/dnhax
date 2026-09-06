"""Download approved VGGT-Omega 1B-512 weights using local Hugging Face login."""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
REVISION = "55d1f4b2ce41fd0925887362a4a5049ce94e0be1"
SHA256 = "c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"


def main():
    path = Path(hf_hub_download(
        repo_id="facebook/VGGT-Omega", filename="vggt_omega_1b_512.pt",
        revision=REVISION, local_dir=ROOT / "models/vggt-omega",
    ))
    with path.open("rb") as checkpoint:
        actual = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if actual != SHA256:
        raise RuntimeError("Checkpoint checksum does not match the official 1B-512 release.")
    print(f"Verified: {path.resolve()}")


if __name__ == "__main__":
    main()
