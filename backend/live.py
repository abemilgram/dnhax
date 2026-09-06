"""Single-worker live scheduler, publication fences and bounded retention."""

import json
import shutil
import time
from pathlib import Path
import numpy as np
from . import store
from .models import model_config


def select_frames(rows, budget, previous_ids):
    selected = []
    sources = sorted({r["source"] for r in rows})
    count = budget // max(len(sources), 1)
    for source in sources:
        candidates = sorted(
            [r for r in rows if r["source"] == source and r["sharpness"] >= 4],
            key=lambda r: (r["seq"], r["id"]),
        )
        if len(candidates) < 2:
            continue
        retained = [r for r in candidates if r["id"] in previous_ids][
            -max(1, count // 2) :
        ]
        chosen = retained[:]
        for frame in reversed(candidates):
            if len(chosen) >= count:
                break
            if any(
                r["id"] == frame["id"]
                or r["digest"] == frame["digest"]
                or sum(a != b for a, b in zip(r["signature"], frame["signature"])) < 10
                for r in chosen
            ):
                continue
            chosen.append(frame)
        if len(chosen) >= 2:
            selected.extend(chosen)
    return selected


def schedule():
    now = time.time()
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        db.execute(
            "UPDATE live_sessions SET status='stopping',stop_deadline=? WHERE status='open' AND created<?",
            (now, now - 600),
        )
        row = db.execute(
            "SELECT * FROM live_sessions WHERE status IN ('open','stopping') LIMIT 1"
        ).fetchone()
        if not row:
            return
        identity = row["id"]
        if db.execute(
            "SELECT 1 FROM live_batches WHERE session_id=? AND status IN ('queued','running')",
            (identity,),
        ).fetchone():
            return
        if row["status"] == "stopping" and now < row["stop_deadline"]:
            return
        if row["processing"] == "paused_error":
            if row["status"] == "stopping":
                db.execute(
                    "UPDATE live_sessions SET status='completed',processing='stopped' WHERE id=?",
                    (identity,),
                )
            return
        previous = db.execute(
            "SELECT frames FROM live_batches WHERE session_id=? AND status='completed' ORDER BY number DESC LIMIT 1",
            (identity,),
        ).fetchone()
        previous_ids = (
            [r["id"] for r in json.loads(previous["frames"])] if previous else []
        )
        frames = []
        for src in db.execute(
            "SELECT * FROM live_sources WHERE session_id=?", (identity,)
        ).fetchall():
            if not src["last_received"] or (
                row["status"] == "open" and now - src["last_received"] > 10
            ):
                continue
            frames.extend(
                dict(r)
                for r in db.execute(
                    "SELECT * FROM live_frames WHERE session_id=? AND source=? AND epoch=? ORDER BY seq DESC LIMIT 60",
                    (identity, src["source"], src["epoch"]),
                )
            )
        chosen = select_frames(
            frames, json.loads(row["config"])["frame_budget"], previous_ids
        )
        if len(chosen) < 2 or not any(r["id"] > row["consumed"] for r in chosen):
            db.execute(
                "UPDATE live_sessions SET processing=?,status=? WHERE id=?",
                (
                    "stopped" if row["status"] == "stopping" else "waiting",
                    "completed" if row["status"] == "stopping" else row["status"],
                    identity,
                ),
            )
            return
        number = db.execute(
            "SELECT COALESCE(MAX(number),0)+1 FROM live_batches WHERE session_id=?",
            (identity,),
        ).fetchone()[0]
        batch, job = store.uid(), store.uid()
        db.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,NULL)",
            (
                job,
                "live",
                "queued",
                "Fresh live batch ready",
                now,
                now,
                json.dumps({"batch_id": batch}),
            ),
        )
        db.execute(
            "INSERT INTO live_batches(id,session_id,number,job_id,status,frames,watermark,generation,created) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                batch,
                identity,
                number,
                job,
                "queued",
                json.dumps(chosen),
                max(r["id"] for r in frames),
                row["generation"],
                now,
            ),
        )
        db.execute(
            "UPDATE live_sessions SET processing='queued' WHERE id=?", (identity,)
        )


def fail(batch_id, message):
    with store.connect() as db:
        row = db.execute(
            "SELECT * FROM live_batches WHERE id=?", (batch_id,)
        ).fetchone()
        if not row:
            return
        db.execute(
            "UPDATE live_batches SET status='failed',error=? WHERE id=?",
            (message, batch_id),
        )
        db.execute(
            "UPDATE live_sessions SET processing='paused_error',error=? WHERE id=? AND status IN ('open','stopping')",
            (message, row["session_id"]),
        )
        db.execute(
            "UPDATE jobs SET status='failed',stage='Live processing failed',error=?,updated=? WHERE id=?",
            (message, time.time(), row["job_id"]),
        )


def recover():
    with store.connect() as db:
        for row in db.execute(
            "SELECT * FROM live_batches WHERE status IN ('queued','running')"
        ).fetchall():
            db.execute(
                "UPDATE live_batches SET status='failed',error='Worker interrupted; fresh input will be selected.' WHERE id=?",
                (row["id"],),
            )
            db.execute(
                "UPDATE jobs SET status='failed',stage='Interrupted',error='Worker interrupted' WHERE id=?",
                (row["job_id"],),
            )
        db.execute(
            "UPDATE live_sessions SET processing='waiting' WHERE status IN ('open','stopping') AND processing!='paused_error'"
        )


def process(job):
    from .reconstruct import compute_device, infer_images, export_cloud
    from .live_geometry import continuity, apply_similarity

    started = time.perf_counter()
    with store.connect() as db:
        batch = dict(
            db.execute(
                "SELECT * FROM live_batches WHERE id=?", (job["payload"]["batch_id"],)
            ).fetchone()
        )
        session = dict(
            db.execute(
                "SELECT * FROM live_sessions WHERE id=?", (batch["session_id"],)
            ).fetchone()
        )
        if batch["status"] == "completed":
            return
        if (
            session["status"] == "cancelled"
            or session["generation"] != batch["generation"]
        ):
            db.execute(
                "UPDATE live_batches SET status='cancelled' WHERE id=?", (batch["id"],)
            )
            db.execute(
                "UPDATE jobs SET status='cancelled',stage='Cancelled' WHERE id=?",
                (job["id"],),
            )
            return
        db.execute(
            "UPDATE live_batches SET status='running',attempts=attempts+1 WHERE id=?",
            (batch["id"],),
        )
        db.execute(
            "UPDATE live_sessions SET processing='running' WHERE id=?", (session["id"],)
        )
    frames = json.loads(batch["frames"])
    pinned, config = json.loads(session["config"]), model_config()
    checkpoint = config["checkpoint"]
    if (
        config["key"] != pinned["model"]
        or str(checkpoint.resolve()) != pinned["checkpoint"]
    ):
        raise RuntimeError("Live model configuration changed. Start a new session.")
    if checkpoint.is_file() and pinned["checkpoint_size"] is not None:
        stat = checkpoint.stat()
        if (stat.st_size, stat.st_mtime_ns) != (
            pinned["checkpoint_size"],
            pinned["checkpoint_mtime"],
        ):
            raise RuntimeError("Checkpoint changed during this session.")
    device = compute_device()
    progress = lambda stage: store.update(job["id"], stage)
    try:
        prediction = infer_images([Path(f["path"]) for f in frames], device, progress)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or len(frames) <= 4:
            raise
        from .model_runtime import runtime

        runtime.unload()
        frames = [
            f
            for source in ("A", "B")
            for f in [r for r in frames if r["source"] == source][-2:]
        ]
        with store.connect() as db:
            db.execute(
                "UPDATE live_batches SET attempts=attempts+1,frames=? WHERE id=?",
                (json.dumps(frames), batch["id"]),
            )
        prediction = infer_images([Path(f["path"]) for f in frames], device, progress)
    scene_id = store.uid()
    folder = store.ROOT / "scenes" / scene_id
    folder.mkdir(parents=True)
    ids = [f["id"] for f in frames]
    segment = scene_id
    matrix = np.eye(4)
    detail = {"status": "new_segment", "reason": "First preview in this segment."}
    if session["latest_scene"]:
        old_folder = store.ROOT / "scenes" / session["latest_scene"]
        if (old_folder / "manifest.json").exists() and (
            old_folder / "continuity.npz"
        ).exists():
            old_manifest = json.loads((old_folder / "manifest.json").read_text())
            with np.load(old_folder / "continuity.npz") as old:
                matrix, detail = continuity(
                    prediction, ids, old, old_manifest["live"]["frame_ids"]
                )
            if detail["status"] == "accepted":
                segment = old_manifest["live"]["segment"]
                apply_similarity(prediction, matrix)
    np.savez_compressed(
        folder / "continuity.npz",
        **{
            k: prediction[k]
            for k in (
                "depth",
                "world_points",
                "confidence",
                "extrinsics",
                "intrinsics",
            )
            if k in prediction
        },
    )
    clouds = []
    for source in ("A", "B"):
        indices = [i for i, f in enumerate(frames) if f["source"] == source]
        if not indices:
            continue
        exported = []
        for i in indices:
            path = folder / source / "frames" / f"{frames[i]['id']}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(frames[i]["path"], path)
            exported.append({"path": path, "t": frames[i]["captured"]})
        progress(f"Exporting live source {source}")
        clouds.append(
            export_cloud(
                folder / source,
                {"id": session["id"], "source": source},
                exported,
                prediction,
                indices,
                device,
                point_limit=200000,
                reconstruction_method=(
                    "live_amb3r"
                    if prediction.get("model_key") == "amb3r"
                    else "live_vggt"
                ),
            )
        )
    manifest = {
        "id": scene_id,
        "created": time.time(),
        "sample": False,
        "title": f"Live batch {batch['number']}",
        "clouds": clouds,
        "diagnostics": None,
        "scale_source": "None — arbitrary reconstruction units",
        "provenance": "Live rolling-window prediction. Cross-source alignment and physical accuracy are unverified.",
        "live": {
            "session_id": session["id"],
            "batch": batch["number"],
            "segment": segment,
            "frame_ids": ids,
            "continuity": detail,
            "transform": matrix.tolist(),
            "model": prediction["model"],
            "checkpoint_sha256": prediction.get("checkpoint_sha256"),
            "preprocessing": prediction.get("preprocessing"),
            "compute_device": device,
            "frames": [
                {
                    k: f[k]
                    for k in (
                        "id",
                        "source",
                        "epoch",
                        "seq",
                        "captured",
                        "received",
                        "digest",
                    )
                }
                for f in frames
            ],
            "elapsed_seconds": time.perf_counter() - started,
            "timings": prediction.get("timings", {}),
            "preview_points_per_source": 200000,
        },
    }
    encoded = json.dumps(manifest, allow_nan=False)
    temporary = folder / "manifest.tmp"
    temporary.write_text(encoded)
    temporary.replace(folder / "manifest.json")
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        current = db.execute(
            "SELECT * FROM live_sessions WHERE id=?", (session["id"],)
        ).fetchone()
        if db.execute(
            "SELECT scene_id FROM live_batches WHERE id=?", (batch["id"],)
        ).fetchone()["scene_id"]:
            return
        if (
            current["status"] == "cancelled"
            or current["generation"] != batch["generation"]
        ):
            db.execute(
                "UPDATE live_batches SET status='cancelled' WHERE id=?", (batch["id"],)
            )
            db.execute(
                "UPDATE jobs SET status='cancelled',stage='Cancelled',updated=? WHERE id=?",
                (time.time(), job["id"]),
            )
            return
        db.execute(
            "INSERT INTO scenes VALUES(?,?,?)", (scene_id, manifest["created"], encoded)
        )
        db.execute(
            "UPDATE live_batches SET status='completed',scene_id=? WHERE id=?",
            (scene_id, batch["id"]),
        )
        db.execute(
            "UPDATE jobs SET status='completed',stage='Completed',updated=? WHERE id=?",
            (time.time(), job["id"]),
        )
        db.execute(
            "UPDATE live_sessions SET latest_scene=?,latest_batch=?,consumed=?,processing='waiting',error=NULL WHERE id=? AND latest_batch<?",
            (
                scene_id,
                batch["number"],
                batch["watermark"],
                session["id"],
                batch["number"],
            ),
        )


def cleanup():
    files, folders = [], []
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        for session in db.execute(
            "SELECT id,latest_scene FROM live_sessions"
        ).fetchall():
            identity = session["id"]
            protected = set()
            batches = db.execute(
                "SELECT * FROM live_batches WHERE session_id=? ORDER BY number DESC",
                (identity,),
            ).fetchall()
            for batch in batches:
                if (
                    batch["status"] in ("queued", "running")
                    or batch["scene_id"] == session["latest_scene"]
                ):
                    protected.update(f["id"] for f in json.loads(batch["frames"]))
            for source in ("A", "B"):
                frames = db.execute(
                    "SELECT id,path FROM live_frames WHERE session_id=? AND source=? ORDER BY id DESC",
                    (identity, source),
                ).fetchall()
                for frame in frames[60:]:
                    if frame["id"] not in protected:
                        db.execute("DELETE FROM live_frames WHERE id=?", (frame["id"],))
                        files.append(Path(frame["path"]))
            completed = [b for b in batches if b["scene_id"]]
            for batch in completed[5:]:
                scene_id = batch["scene_id"]
                if db.execute(
                    "SELECT 1 FROM live_pins WHERE scene_id=?", (scene_id,)
                ).fetchone():
                    continue
                if db.execute(
                    "SELECT 1 FROM scenes WHERE id!=? AND manifest LIKE ? LIMIT 1",
                    (scene_id, f"%/scenes/{scene_id}/%"),
                ).fetchone():
                    continue
                db.execute(
                    "UPDATE live_batches SET scene_id=NULL WHERE id=?", (batch["id"],)
                )
                db.execute("DELETE FROM scenes WHERE id=?", (scene_id,))
                folders.append(store.ROOT / "scenes" / scene_id)
    for path in files:
        path.unlink(missing_ok=True)
    for folder in folders:
        shutil.rmtree(folder, ignore_errors=True)

    # Clean crash/cancellation leftovers only after an age grace period.
    # The single worker invokes cleanup between jobs, never during inference.
    with store.connect() as db:
        known = {r[0] for r in db.execute("SELECT path FROM live_frames")}
        published = {r[0] for r in db.execute("SELECT id FROM scenes")}
    cutoff = time.time() - 60
    for path in (store.ROOT / "live").glob("*/*"):
        if path.is_file() and str(path) not in known and path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
    for folder in (store.ROOT / "scenes").glob("*"):
        # Only newly generated live folders carry continuity.npz.
        if (
            folder.name not in published
            and (folder / "continuity.npz").exists()
            and folder.stat().st_mtime < cutoff
        ):
            shutil.rmtree(folder, ignore_errors=True)
