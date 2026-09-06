"""Runpod queue entrypoint for the existing single-workspace processor."""

import json

import runpod

from backend import store
from backend.worker import process
from backend.worker_lock import acquire


def handler(job):
    data = job.get("input")
    if not isinstance(data, dict):
        raise ValueError("input must be an object")
    kind = data.get("kind")
    if kind not in {"sample", "reconstruct", "joint", "pair", "register"}:
        raise ValueError("kind must be sample, reconstruct, joint, pair, or register")
    payload = data.get("payload", {})
    if not isinstance(payload, dict):
        raise ValueError("payload must be an object")
    # Keep the same singleton guarantee as the local worker.
    lock = acquire()
    try:
        with store.connect() as db:
            before = {row["id"] for row in db.execute("SELECT id FROM scenes")}
        identity = store.enqueue(kind, payload)
        try:
            process({"id": identity, "kind": kind, "payload": payload})
        except Exception as exc:
            store.update(identity, "Processing failed", "failed", str(exc))
            raise
        with store.connect() as db:
            scenes = [
                json.loads(row["manifest"])
                for row in db.execute("SELECT id, manifest FROM scenes ORDER BY created")
                if row["id"] not in before
            ]
        return {"job_id": identity, "status": "completed", "scenes": scenes}
    finally:
        lock.close()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
