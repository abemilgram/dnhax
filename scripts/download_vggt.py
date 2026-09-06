"""Historical VGGT checkpoint utility; the runtime no longer selects VGGT."""

from pathlib import Path

from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "facebook/VGGT-1B"
REVISION = "860abec7937da0a4c03c41d3c269c366e82abdf9"
DESTINATION = ROOT / "models" / "vggt-1b"


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=REPOSITORY,
        filename="model.safetensors",
        revision=REVISION,
        local_dir=DESTINATION,
    )
    print(Path(path).resolve())


if __name__ == "__main__":
    main()
