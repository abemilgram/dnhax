"""Cold/warm benchmark. No scenes are published; result files persist per run."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.model_runtime import runtime
from backend.reconstruct import compute_device
from backend.joint import sample_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-a", required=True)
    parser.add_argument("--capture-b", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", choices=[4, 8], default=[4])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = compute_device()
    candidates = [
        sample_candidates({"source": s, "path": p}, args.output / s, 4)
        for s, p in zip(("A", "B"), (args.capture_a, args.capture_b))
    ]
    results = []
    try:
        for budget in args.budgets:
            for repeat in range(2):
                images = [
                    f["path"] for frames in candidates for f in frames[: budget // 2]
                ]
                started = time.perf_counter()
                prediction = runtime.infer(
                    images, device, lambda s: print(s, flush=True)
                )
                result = {
                    "frames": len(images),
                    "repeat": repeat,
                    "device": device,
                    "model_loads": runtime.loads,
                    "elapsed_seconds": time.perf_counter() - started,
                    "checkpoint_sha256": prediction["checkpoint_sha256"],
                    **prediction["timings"],
                }
                if device == "mps":
                    import torch

                    result["allocated_bytes"] = torch.mps.current_allocated_memory()
                    result["driver_bytes"] = torch.mps.driver_allocated_memory()
                results.append(result)
                print(json.dumps(result), flush=True)
                (args.output / "results.json").write_text(json.dumps(results, indent=2))
                del prediction
    finally:
        runtime.unload()


if __name__ == "__main__":
    main()
