"""Single-workspace persistence. Artifacts are immutable after publication."""

import json
from contextlib import contextmanager
import os
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
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript("""
    CREATE TABLE IF NOT EXISTS captures(id TEXT PRIMARY KEY, source TEXT NOT NULL, name TEXT NOT NULL, created REAL NOT NULL, path TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL, stage TEXT NOT NULL, created REAL NOT NULL, updated REAL NOT NULL, payload TEXT NOT NULL, error TEXT);
    CREATE INDEX IF NOT EXISTS jobs_status_created ON jobs(status,created);
    CREATE TABLE IF NOT EXISTS scenes(id TEXT PRIMARY KEY, created REAL NOT NULL, manifest TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS heartbeat(id INTEGER PRIMARY KEY CHECK(id=1), updated REAL NOT NULL, device TEXT NOT NULL);
    """)
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


def claim():
    with connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute(
            "SELECT * FROM jobs WHERE status='queued' ORDER BY created LIMIT 1"
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
