"""Additive, transactional migration for live processing."""

SQL = """
CREATE TABLE IF NOT EXISTS schema_versions(version INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS live_sessions(
 id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, token TEXT NOT NULL,
 status TEXT NOT NULL, processing TEXT NOT NULL, created REAL NOT NULL,
 stop_deadline REAL, config TEXT NOT NULL, latest_scene TEXT, latest_batch INTEGER DEFAULT 0,
 error TEXT, generation INTEGER DEFAULT 0, consumed INTEGER DEFAULT 0);
CREATE UNIQUE INDEX IF NOT EXISTS one_live_session ON live_sessions((1)) WHERE status IN ('open','stopping');
CREATE TABLE IF NOT EXISTS live_sources(
 session_id TEXT NOT NULL REFERENCES live_sessions(id), source TEXT NOT NULL,
 epoch TEXT NOT NULL, token TEXT NOT NULL, lease REAL NOT NULL, status TEXT NOT NULL,
 last_received REAL, final_seq INTEGER, skipped INTEGER DEFAULT 0,
 PRIMARY KEY(session_id,source));
CREATE TABLE IF NOT EXISTS live_frames(
 id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL REFERENCES live_sessions(id),
 source TEXT NOT NULL, epoch TEXT NOT NULL, seq INTEGER NOT NULL, digest TEXT NOT NULL,
 path TEXT NOT NULL, captured REAL NOT NULL, received REAL NOT NULL, sharpness REAL NOT NULL,
 signature TEXT NOT NULL, width INTEGER NOT NULL, height INTEGER NOT NULL,
 UNIQUE(session_id,source,epoch,seq));
CREATE INDEX IF NOT EXISTS live_frame_source ON live_frames(session_id,source,id);
CREATE TABLE IF NOT EXISTS live_batches(
 id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES live_sessions(id),
 number INTEGER NOT NULL, job_id TEXT NOT NULL UNIQUE REFERENCES jobs(id), status TEXT NOT NULL,
 frames TEXT NOT NULL, watermark INTEGER NOT NULL, generation INTEGER NOT NULL,
 scene_id TEXT, created REAL NOT NULL, error TEXT, attempts INTEGER DEFAULT 0,
 UNIQUE(session_id,number));
CREATE TABLE IF NOT EXISTS live_pins(scene_id TEXT PRIMARY KEY REFERENCES scenes(id));
INSERT OR IGNORE INTO schema_versions VALUES(1);
"""


def migrate(db):
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE name='schema_versions'"
    ).fetchone()
    if (
        exists
        and db.execute("SELECT 1 FROM schema_versions WHERE version=1").fetchone()
    ):
        return
    db.executescript("BEGIN IMMEDIATE;\n" + SQL + "\nCOMMIT;")
