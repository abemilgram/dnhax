"""Additive tactical metadata and cue-history persistence."""

import json
import time
from typing import Callable, ContextManager


SQL = """
CREATE TABLE IF NOT EXISTS tactical_sessions(
 id TEXT PRIMARY KEY,
 tape_id TEXT NOT NULL,
 status TEXT NOT NULL CHECK(status IN ('playing','paused')),
 replay_time REAL NOT NULL,
 revision INTEGER NOT NULL,
 created REAL NOT NULL,
 updated REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tactical_cues(
 session_id TEXT NOT NULL,
 sequence INTEGER NOT NULL,
 revision INTEGER NOT NULL,
 emitted REAL NOT NULL,
 payload TEXT NOT NULL,
 PRIMARY KEY(session_id,sequence)
);
CREATE INDEX IF NOT EXISTS tactical_cues_emitted
 ON tactical_cues(session_id,emitted);
INSERT OR IGNORE INTO schema_versions VALUES(2);
"""


def migrate(db) -> None:
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE name='schema_versions'"
    ).fetchone()
    if (
        exists
        and db.execute("SELECT 1 FROM schema_versions WHERE version=2").fetchone()
    ):
        return
    db.executescript("BEGIN IMMEDIATE;\n" + SQL + "\nCOMMIT;")


class SqliteTacticalPersistence:
    """Small adapter that resolves ``store.ROOT`` on every write."""

    def __init__(
        self,
        connect: Callable[[], ContextManager],
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._connect = connect
        self._clock = clock

    def load_revision(self, session_id: str) -> int:
        with self._connect() as db:
            row = db.execute(
                "SELECT revision FROM tactical_sessions WHERE id=?",
                (session_id,),
            ).fetchone()
        return max(0, int(row["revision"])) if row is not None else 0

    def save_session(
        self,
        session_id: str,
        tape_id: str,
        status: str,
        replay_time: float,
        revision: int,
    ) -> None:
        now = float(self._clock())
        with self._connect() as db:
            db.execute(
                """
                INSERT INTO tactical_sessions(
                  id,tape_id,status,replay_time,revision,created,updated
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                  tape_id=excluded.tape_id,
                  status=excluded.status,
                  replay_time=excluded.replay_time,
                  revision=excluded.revision,
                  updated=excluded.updated
                """,
                (
                    session_id,
                    tape_id,
                    status,
                    replay_time,
                    revision,
                    now,
                    now,
                ),
            )

    def save_cue(
        self,
        session_id: str,
        revision: int,
        cue: dict[str, object],
    ) -> None:
        payload = json.dumps(
            cue,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        with self._connect() as db:
            db.execute(
                """
                INSERT OR IGNORE INTO tactical_cues(
                  session_id,sequence,revision,emitted,payload
                ) VALUES(?,?,?,?,?)
                """,
                (
                    session_id,
                    int(cue["sequence"]),
                    revision,
                    float(self._clock()),
                    payload,
                ),
            )
