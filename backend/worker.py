import json
import os
import signal
import threading
import time
import numpy as np
from . import store
from .geometry import register, save_cloud, transform


def cloud_record(folder, points, colors, source, **extra):
    count = save_cloud(folder, points, colors)
    return {
        "source": source,
        "count": count,
        "points": store.artifact_url(folder / "points.bin"),
        "colors": store.artifact_url(folder / "colors.bin"),
        "ply": store.artifact_url(folder / "cloud.ply"),
        "transform": np.eye(4).tolist(),
        **extra,
    }


def sample_scene():
    rng = np.random.default_rng(12)
    # Explicitly synthetic room surfaces, used only as an alignment fixture.
    floor = np.column_stack(
        [rng.uniform(-3, 3, 8000), np.zeros(8000), rng.uniform(-2, 2, 8000)]
    )
    back = np.column_stack(
        [rng.uniform(-3, 3, 6000), rng.uniform(0, 2.8, 6000), np.full(6000, -2)]
    )
    wall = np.column_stack(
        [np.full(4000, -3), rng.uniform(0, 2.8, 4000), rng.uniform(-2, 2, 4000)]
    )
    table = np.column_stack(
        [
            rng.uniform(-1.2, 1.2, 2500),
            np.full(2500, 0.85),
            rng.uniform(-0.6, 0.6, 2500),
        ]
    )
    world = np.concatenate([floor, back, wall, table])
    a = world[world[:, 0] < 1.5]
    b = world[world[:, 0] > -2]
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * 1.18
    matrix[:3, 3] = [1.4, 0.3, -0.7]
    b = transform(b, np.linalg.inv(matrix))
    identity = store.uid()
    folder = store.ROOT / "scenes" / identity
    clouds = [
        cloud_record(folder / "A", a, np.tile([81, 196, 168], (len(a), 1)), "A"),
        cloud_record(folder / "B", b, np.tile([226, 161, 109], (len(b), 1)), "B"),
    ]
    landmarks = world[rng.choice(len(world), 40, replace=False)]
    source = transform(landmarks, np.linalg.inv(matrix)) + rng.normal(
        0, 0.004, landmarks.shape
    )
    fitted, diagnostics = register(source, landmarks)
    clouds[1]["transform"] = fitted.tolist()
    return {
        "id": identity,
        "created": time.time(),
        "sample": True,
        "title": "Sample / room geometry",
        "clouds": clouds,
        "diagnostics": diagnostics,
        "scale_source": "None — arbitrary reconstruction units",
        "landmarks": {
            "source_points": source.tolist(),
            "target_points": landmarks.tolist(),
        },
        "provenance": "Synthetic room fixture. No video reconstruction.",
    }


def process(job):
    identity = job["id"]
    payload = job["payload"]
    kind = job["kind"]
    if kind == "live":
        from .live import process as process_live

        process_live(job)
        return
    if kind == "sample":
        store.update(identity, "Generating labeled sample geometry")
        store.publish(sample_scene())
    elif kind == "reconstruct":
        from .reconstruct import reconstruct

        store.update(identity, "Checking compute device and checkpoint")
        with store.connect() as db:
            capture = dict(
                db.execute(
                    "SELECT * FROM captures WHERE id=?", (payload["capture_id"],)
                ).fetchone()
            )
        cloud = reconstruct(capture, lambda stage: store.update(identity, stage))
        scene = {
            "id": store.uid(),
            "created": time.time(),
            "sample": False,
            "title": f"Capture {capture['source']}",
            "clouds": [cloud],
            "diagnostics": None,
            "scale_source": "None — arbitrary reconstruction units",
            "provenance": f"{cloud['model']} independent video reconstruction",
        }
        store.publish(scene)
    elif kind == "joint":
        from .joint import reconstruct_joint

        with store.connect() as db:
            captures = [
                dict(
                    db.execute(
                        "SELECT * FROM captures WHERE id=?", (payload[key],)
                    ).fetchone()
                )
                for key in ("capture_a", "capture_b")
            ]
        scene_id = store.uid()
        clouds, reconstruction = reconstruct_joint(
            captures,
            store.ROOT / "scenes" / scene_id,
            lambda stage: store.update(identity, stage),
        )
        store.publish(
            {
                "id": scene_id,
                "created": time.time(),
                "sample": False,
                "title": "Joint A + B",
                "clouds": clouds,
                "diagnostics": None,
                "reconstruction": reconstruction,
                "scale_source": "None — arbitrary reconstruction units",
                "provenance": f"A and B reconstructed together by {reconstruction['model']} in one shared coordinate system. Alignment quality is unverified.",
            }
        )
    elif kind == "pair":
        store.update(
            identity, "Combining independent reconstructions for landmark alignment"
        )
        clouds = []
        for key, source in [("capture_a", "A"), ("capture_b", "B")]:
            folder = store.ROOT / "reconstructions" / payload[key]
            cloud = json.loads((folder / "cloud.json").read_text())
            cloud["source"] = source
            clouds.append(cloud)
        store.publish(
            {
                "id": store.uid(),
                "created": time.time(),
                "sample": False,
                "title": "Room / independent captures",
                "clouds": clouds,
                "diagnostics": None,
                "scale_source": "None — arbitrary reconstruction units",
                "provenance": "Two independent reconstructions. Supply landmark pairs to register.",
            }
        )
    elif kind == "register":
        store.update(identity, "Fitting transform and validating held-out landmarks")
        with store.connect() as db:
            scene = json.loads(
                db.execute(
                    "SELECT manifest FROM scenes WHERE id=?", (payload["scene_id"],)
                ).fetchone()["manifest"]
            )
        matrix, diagnostics = register(
            payload["source_points"], payload["target_points"], payload["threshold"]
        )
        scene.pop("live", None)
        scene.update(id=store.uid(), created=time.time(), diagnostics=diagnostics)
        scene["clouds"][1]["transform"] = matrix.tolist()
        scene["landmarks"] = {
            "source_points": payload["source_points"],
            "target_points": payload["target_points"],
        }
        store.publish(scene)
    else:
        raise ValueError("Unknown job type.")
    store.update(identity, "Completed", "completed")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    from .worker_lock import acquire
    from . import live

    lock = acquire()
    device = "CPU / sample and alignment available"
    try:
        import torch

        from .runtime import device_label

        device = device_label(torch)
    except ImportError:
        pass
    except (ValueError, RuntimeError) as exc:
        device = f"Compute configuration error: {exc}"

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    stop = threading.Event()

    def beat():
        while not stop.is_set():
            with store.connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO heartbeat VALUES(1,?,?)",
                    (time.time(), device),
                )
            stop.wait(3)

    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    # A single worker is supported. Recover jobs abandoned by its previous process.
    with store.connect() as db:
        db.execute(
            "UPDATE jobs SET status='failed',stage='Interrupted',error='Worker stopped before completion. Submit the job again.' WHERE status='running'"
        )
    live.recover()
    last_live = False
    retention_hours = float(os.environ.get("SIMV1_RETENTION_HOURS", "0"))
    last_retention = 0.0
    print("simv1 worker:", device, flush=True)
    try:
        while True:
            if retention_hours > 0 and time.time() - last_retention >= 3600:
                removed = store.cleanup_old(retention_hours)
                if any(removed.values()):
                    print(f"Retention cleanup: {removed}", flush=True)
                last_retention = time.time()
            live.cleanup()
            job = store.claim() if last_live else None
            if not job:
                live.schedule()
                job = store.claim(prefer_live=True)
            if job:
                watchdog = None
                if job["kind"] == "live":
                    def expire(batch_id=job["payload"]["batch_id"]):
                        live.fail(
                            batch_id,
                            "Inference watchdog expired. Restart the worker and resume.",
                        )
                        os._exit(70)

                    watchdog = threading.Timer(
                        float(os.environ.get("SIMV1_LIVE_JOB_TIMEOUT", "900")), expire
                    )
                    watchdog.daemon = True
                    watchdog.start()
                try:
                    process(job)
                except Exception as exc:
                    store.update(job["id"], "Processing failed", "failed", str(exc))
                    if job["kind"] == "live":
                        live.fail(job["payload"]["batch_id"], str(exc))
                    print("Job failed:", exc, flush=True)
                finally:
                    if watchdog:
                        watchdog.cancel()
                    last_live = job["kind"] == "live"
            if args.once:
                break
            if not job:
                stop.wait(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Parent and terminal may both signal shutdown; finish cleanup once.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        from .model_runtime import runtime

        runtime.unload()
        stop.set()
        thread.join(timeout=5)
        with store.connect() as db:
            db.execute("DELETE FROM heartbeat WHERE id=1")
        lock.close()


if __name__ == "__main__":
    main()
