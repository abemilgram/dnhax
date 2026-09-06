"""Run `python -m backend.prediction.replay --output work/prediction-replay.json`."""

import argparse
import json
from pathlib import Path
import time
import numpy as np

from .engine import PredictionEngine
from .fixture import config, replay_packets, scene


def replay(prediction_model="route_bank"):
    configured = config().model_copy(update={"prediction_model": prediction_model})
    # Validate overrides through the same contract as the API and recorded runner.
    configured = type(configured).model_validate(configured.model_dump())
    engine = PredictionEngine(scene(), clock_id="fixture-clock", evidence_source="synthetic_fixture", config=configured)
    snapshots, timings = [], []
    for packet in replay_packets():
        start = time.perf_counter()
        snapshots.append(engine.ingest(packet, now=packet.available_t)["state"])
        timings.append((time.perf_counter() - start) * 1000)
    return {"synthetic": True, "prediction_model": prediction_model,
            "description": "Synthetic pixel-box replay; not CS2 video, detector validation, or cloud benchmark.",
            "timing_scope": "This process, projection + belief + simulation + serialization preparation; excludes video and detector.",
            "ingest_ms": {"p50": float(np.percentile(timings, 50)), "p95": float(np.percentile(timings, 95)), "max": max(timings)},
            "snapshots": snapshots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("work/prediction-replay.json"))
    parser.add_argument("--prediction-model", choices=["route_bank", "branching"], default="route_bank")
    args = parser.parse_args()
    result = replay(args.prediction_model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "synthetic": True, "prediction_model": args.prediction_model,
                      "ingest_ms": result["ingest_ms"],
                      "final_tracks": len(result["snapshots"][-1]["tracks"]),
                      "events": result["snapshots"][-1]["events"]}, indent=2))


if __name__ == "__main__":
    main()
