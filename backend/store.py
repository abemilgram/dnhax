"""Single-workspace persistence. Artifacts are immutable after publication."""

import json
from contextlib import contextmanager
import os
import shutil
import sqlite3
import time
import uuid
from pathlib import Path

ROOT = Path(
    os.environ.get("SIMV1_DATA", Path(__file__).resolve().parents[1] / "data")
).resolve()


@contextmanager
def connect():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / "workspace.sqlite", timeout=20)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS captures(id TEXT PRIMARY KEY, source TEXT NOT NULL, name TEXT NOT NULL, created REAL NOT NULL, path TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, payload TEXT NOT NULL, error TEXT);
    CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status,created);
    CREATE TABLE IF NOT EXISTS scenes(id TEXT PRIMARY KEY, created REAL NOT NULL, manifest TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS heartbeat(id INTEGER PRIMARY KEY CHECK(id=1), updated REAL NOT NULL, device TEXT NOT NULL);
    """)
    from .live_schema import migrate

    migrate(db)
    from .tactical.migrations import migrate as migrate_tactical

    migrate_tactical(db)
    try:
        with db:
            yield db
    finally:
        db.close()


def uid():
    return uuid.uuid4().hex


def enqueue(kind, payload):
    job = uid()
    with connect() as db:
        db.execute(
            "INSERT INTO jobs VALUES(?,?,?,?,?,?,?,NULL)",
            (
                job,
                kind,
                "queued",
                "Waiting for worker",
                time.time(),
                time.time(),
                json.dumps(payload),
            ),
        )
    return job


def update(job, stage, status="running", error=None):
    with connect() as db:
        db.execute(
            "UPDATE jobs SET stage=?,status=?,updated=?,error=? WHERE id=?",
            (stage, status, time.time(), error, job),
        )


def claim(prefer_live=False):
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY (kind='live') "
            + ("DESC" if prefer_live else "ASC")
            + ", created LIMIT 1"
        ).fetchone()
        if row:
            db.execute(
                "UPDATE jobs SET status='running',updated=? WHERE id=?",
                (time.time(), row["id"]),
            )
            return {**dict(row), "payload": json.loads(row["payload"])}


def publish(manifest):
    folder = ROOT / "scenes" / manifest["id"]
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "manifest.json"
    temporary = folder / "manifest.tmp"
    temporary.write_text(json.dumps(manifest, allow_nan=False))
    temporary.replace(target)
    with connect() as db:
        db.execute(
            "INSERT INTO scenes VALUES(?,?,?)",
            (
                manifest["id"],
                manifest["created"],
                json.dumps(manifest, allow_nan=False),
            ),
        )


def artifact_url(path):
    return "/api/artifacts/" + str(Path(path).relative_to(ROOT)).replace(os.sep, "/")


def _delete_scene_rows(db, scene_ids):
    if not scene_ids:
        return
    marks = ",".join("?" for _ in scene_ids)
    db.execute(f"DELETE FROM live_pins WHERE scene_id IN ({marks})", scene_ids)
    db.execute(
        f"UPDATE live_batches SET scene_id=NULL WHERE scene_id IN ({marks})",
        scene_ids,
    )
    db.execute(f"DELETE FROM scenes WHERE id IN ({marks})", scene_ids)


def delete_capture(identity):
    """Delete one submitted capture and the ordinary artifacts derived from it."""
    with connect() as db:
        capture = db.execute(
            "SELECT id FROM captures WHERE id=?", (identity,)
        ).fetchone()
        if not capture:
            return None
        scene_ids = []
        for row in db.execute("SELECT id,manifest FROM scenes").fetchall():
            manifest = json.loads(row["manifest"])
            if any(
                cloud.get("capture_id") == identity
                for cloud in manifest.get("clouds", [])
            ):
                scene_ids.append(row["id"])
        _delete_scene_rows(db, scene_ids)
        db.execute(
            """
            DELETE FROM jobs
            WHERE json_extract(payload,'$.capture_id')=?
               OR json_extract(payload,'$.capture_a')=?
               OR json_extract(payload,'$.capture_b')=?
            """,
            (identity, identity, identity),
        )
        if scene_ids:
            marks = ",".join("?" for _ in scene_ids)
            db.execute(
                f"DELETE FROM jobs WHERE json_extract(payload,'$.scene_id') IN ({marks})",
                scene_ids,
            )
        db.execute("DELETE FROM captures WHERE id=?", (identity,))
    shutil.rmtree(ROOT / "captures" / identity, ignore_errors=True)
    shutil.rmtree(ROOT / "reconstructions" / identity, ignore_errors=True)
    for scene_id in scene_ids:
        shutil.rmtree(ROOT / "scenes" / scene_id, ignore_errors=True)
    return {"capture": identity, "scenes": len(scene_ids)}


def reset_workspace():
    """Clear iteration data while retaining the database, lock, and model volume."""
    with connect() as db:
        db.execute("DELETE FROM tactical_cues")
        db.execute("DELETE FROM tactical_sessions")
        db.execute("DELETE FROM live_pins")
        db.execute("DELETE FROM live_batches")
        db.execute("DELETE FROM live_frames")
        db.execute("DELETE FROM live_sources")
        db.execute("DELETE FROM live_sessions")
        db.execute("DELETE FROM scenes")
        db.execute("DELETE FROM jobs")
        db.execute("DELETE FROM captures")
    for name in ("captures", "reconstructions", "scenes", "live"):
        shutil.rmtree(ROOT / name, ignore_errors=True)


def cleanup_old(hours):
    """Remove completed ordinary workspace data older than the given age."""
    cutoff = time.time() - float(hours) * 3600
    with connect() as db:
        captures = [
            row["id"]
            for row in db.execute(
                """
                SELECT id FROM captures
                WHERE created<?
                  AND NOT EXISTS (
                    SELECT 1 FROM jobs
                    WHERE status IN ('queued','running')
                      AND (
                        json_extract(payload,'$.capture_id')=captures.id
                        OR json_extract(payload,'$.capture_a')=captures.id
                        OR json_extract(payload,'$.capture_b')=captures.id
                      )
                  )
                """,
                (cutoff,),
            ).fetchall()
        ]
    deleted_scenes = 0
    for identity in captures:
        result = delete_capture(identity)
        if result:
            deleted_scenes += result["scenes"]

    with connect() as db:
        old_scenes = [
            row["id"]
            for row in db.execute(
                """
                SELECT id FROM scenes
                WHERE created<?
                  AND json_extract(manifest,'$.live') IS NULL
                  AND NOT EXISTS (
                    SELECT 1 FROM jobs
                    WHERE status IN ('queued','running')
                      AND json_extract(payload,'$.scene_id')=scenes.id
                  )
                """,
                (cutoff,),
            ).fetchall()
        ]
        _delete_scene_rows(db, old_scenes)
        result = db.execute(
            """
            DELETE FROM jobs
            WHERE kind!='live' AND status IN ('completed','failed') AND updated<?
            """,
            (cutoff,),
        )
        deleted_jobs = result.rowcount
    for scene_id in old_scenes:
        shutil.rmtree(ROOT / "scenes" / scene_id, ignore_errors=True)
    return {
        "captures": len(captures),
        "scenes": deleted_scenes + len(old_scenes),
        "jobs": deleted_jobs,
    }
