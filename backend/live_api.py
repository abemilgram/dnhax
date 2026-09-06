"""HTTP live sessions and bounded independent JPEG ingestion."""

import hashlib
import io
import json
import os
import secrets
import shutil
import time
from typing import Literal
import numpy as np
from fastapi import APIRouter, Header, HTTPException, Request, Response
from pydantic import BaseModel, Field
from . import store
from .models import model_config

router = APIRouter(prefix="/api/live")
MAX_BYTES = 2 * 1024 * 1024


def session(db, identity):
    row = db.execute("SELECT * FROM live_sessions WHERE id=?", (identity,)).fetchone()
    if not row:
        raise HTTPException(404, "Live session not found.")
    return row


def authorize(expected, supplied):
    if not supplied or not secrets.compare_digest(expected, supplied):
        raise HTTPException(409, "This browser no longer owns this session or source.")


def source_row(db, identity, source, token):
    row = db.execute(
        "SELECT * FROM live_sources WHERE session_id=? AND source=?", (identity, source)
    ).fetchone()
    if not row:
        raise HTTPException(404, "Source not found.")
    authorize(row["token"], token)
    return row


def ingress_open(row):
    now = time.time()
    return (row["status"] == "open" and now < row["created"] + 600) or (
        row["status"] == "stopping" and now < (row["stop_deadline"] or 0)
    )


class Create(BaseModel):
    request_key: str = Field(min_length=8, max_length=100)
    frame_budget: Literal[4, 8] = 4


@router.post("/sessions", status_code=201)
def create(body: Create):
    config = model_config()
    path = config["checkpoint"]
    stat = path.stat() if path.is_file() else None
    pinned = {
        "model": config["key"],
        "checkpoint": str(path.resolve()),
        "frame_budget": body.frame_budget,
        "checkpoint_size": stat.st_size if stat else None,
        "checkpoint_mtime": stat.st_mtime_ns if stat else None,
    }
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        old = db.execute(
            "SELECT id,token FROM live_sessions WHERE request_key=?",
            (body.request_key,),
        ).fetchone()
        if old:
            return dict(old)
        if db.execute(
            "SELECT 1 FROM live_sessions WHERE status IN ('open','stopping')"
        ).fetchone():
            raise HTTPException(
                409, "A live session already exists. Join it or stop it first."
            )
        identity, token = store.uid(), secrets.token_urlsafe(32)
        db.execute(
            "INSERT INTO live_sessions(id,request_key,token,status,processing,created,config) VALUES(?,?,?,?,?,?,?)",
            (
                identity,
                body.request_key,
                token,
                "open",
                "waiting",
                time.time(),
                json.dumps(pinned),
            ),
        )
    return {"id": identity, "token": token}


@router.get("/sessions")
def list_sessions():
    with store.connect() as db:
        return [
            dict(r)
            for r in db.execute(
                "SELECT id,status,processing,created,latest_scene,error FROM live_sessions ORDER BY created DESC LIMIT 20"
            )
        ]


@router.get("/sessions/{identity}")
def state(identity: str, request: Request):
    with store.connect() as db:
        row = dict(session(db, identity))
        row.pop("token")
        row.pop("request_key")
        row["config"] = json.loads(row["config"])
        row["config"].pop("checkpoint", None)
        row["server_time"] = int(time.time())
        row["sources"] = [
            dict(r)
            for r in db.execute(
                "SELECT source,epoch,status,last_received,lease,final_seq,skipped FROM live_sources WHERE session_id=?",
                (identity,),
            )
        ]
        for src in row["sources"]:
            src["stale"] = (
                not src["last_received"] or time.time() - src["last_received"] > 10
            )
            src["accepted"] = [
                r[0]
                for r in db.execute(
                    "SELECT seq FROM live_frames WHERE session_id=? AND source=? AND epoch=? ORDER BY seq",
                    (identity, src["source"], src["epoch"]),
                )
            ]
        row["batches"] = [
            dict(r)
            for r in db.execute(
                "SELECT number,status,scene_id,created,error,attempts FROM live_batches WHERE session_id=? ORDER BY number DESC LIMIT 10",
                (identity,),
            )
        ]
        row["scene"] = None
        if row["latest_scene"]:
            found = db.execute(
                "SELECT manifest FROM scenes WHERE id=?", (row["latest_scene"],)
            ).fetchone()
            if found:
                row["scene"] = json.loads(found[0])
    encoded = json.dumps(row)
    etag = '"' + hashlib.sha256(encoded.encode()).hexdigest() + '"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag})
    return Response(encoded, media_type="application/json", headers={"ETag": etag})


class Claim(BaseModel):
    takeover: bool = False


@router.post("/sessions/{identity}/sources/{source}/claim")
def claim(identity: str, source: Literal["A", "B"], body: Claim):
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if not ingress_open(session(db, identity)):
            raise HTTPException(409, "Session has stopped.")
        old = db.execute(
            "SELECT * FROM live_sources WHERE session_id=? AND source=?",
            (identity, source),
        ).fetchone()
        if (
            old
            and old["status"] == "active"
            and old["lease"] > time.time()
            and not body.takeover
        ):
            raise HTTPException(
                409,
                "Source is in use. Select explicit takeover to replace its producer.",
            )
        token, epoch = secrets.token_urlsafe(32), store.uid()
        db.execute(
            "INSERT INTO live_sources(session_id,source,epoch,token,lease,status) VALUES(?,?,?,?,?,?) ON CONFLICT(session_id,source) DO UPDATE SET epoch=excluded.epoch,token=excluded.token,lease=excluded.lease,status=excluded.status,last_received=NULL,final_seq=NULL,skipped=0",
            (identity, source, epoch, token, time.time() + 20, "active"),
        )
    return {"token": token, "epoch": epoch, "sample_interval_ms": 1000}


@router.post("/sessions/{identity}/sources/{source}/heartbeat")
def heartbeat(
    identity: str, source: Literal["A", "B"], x_live_token: str = Header(default="")
):
    with store.connect() as db:
        row = session(db, identity)
        source_row(db, identity, source, x_live_token)
        db.execute(
            "UPDATE live_sources SET lease=? WHERE session_id=? AND source=?",
            (time.time() + 20, identity, source),
        )
    return {"status": row["status"], "server_time": time.time()}


@router.put("/sessions/{identity}/sources/{source}/frames/{epoch}/{seq}")
async def frame(
    identity: str,
    source: Literal["A", "B"],
    epoch: str,
    seq: int,
    request: Request,
    x_live_token: str = Header(default=""),
    x_capture_time: float = Header(default=0),
):
    if (
        seq < 0
        or seq > 1000000
        or not np.isfinite(x_capture_time)
        or x_capture_time < 0
    ):
        raise HTTPException(422, "Invalid sequence or media timestamp.")
    if request.headers.get("content-type", "").split(";")[0] != "image/jpeg":
        raise HTTPException(415, "Send an independent JPEG image.")

    def validate(db):
        row = session(db, identity)
        src = source_row(db, identity, source, x_live_token)
        if src["epoch"] != epoch or src["status"] != "active" or not ingress_open(row):
            raise HTTPException(409, "Capture stopped or ownership changed.")

    with store.connect() as db:
        validate(db)
    data = bytearray()
    async for chunk in request.stream():
        if len(data) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "Frame exceeds 2 MiB.")
        data.extend(chunk)
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if (
                image.format != "JPEG"
                or width * height > 2000000
                or min(width, height) < 16
            ):
                raise HTTPException(
                    422,
                    "JPEG must be at least 16px on each edge and at most 2 megapixels.",
                )
            image.load()
            gray = np.asarray(image.convert("L").resize((32, 32)), dtype=float)
            sharpness = float(np.diff(gray, axis=0).var() + np.diff(gray, axis=1).var())
            signature = "".join("1" if v > gray.mean() else "0" for v in gray.ravel())
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(422, "JPEG could not be decoded.") from exc
    digest = hashlib.sha256(data).hexdigest()
    directory = store.ROOT / "live" / identity
    directory.mkdir(parents=True, exist_ok=True)
    with store.connect() as db:
        scene_ids = [
            r[0]
            for r in db.execute(
                "SELECT scene_id FROM live_batches WHERE session_id=? AND scene_id IS NOT NULL",
                (identity,),
            )
        ]
    folders = [directory] + [store.ROOT / "scenes" / scene_id for scene_id in scene_ids]
    total = 0
    for folder in folders:
        for asset in folder.rglob("*"):
            try:
                if asset.is_file():
                    total += asset.stat().st_size
            except FileNotFoundError:
                pass
    if total + len(data) > 2 * 1024**3:
        raise HTTPException(
            429,
            "Session reached its 2 GiB storage quota. Stop capture or start a new session.",
            headers={"Retry-After": "5"},
        )
    if shutil.disk_usage(store.ROOT).free < 5 * 1024**3:
        raise HTTPException(
            429, "Storage reserve reached.", headers={"Retry-After": "5"}
        )
    path = directory / (store.uid() + ".jpg")
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    try:
        with store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            validate(db)
            old = db.execute(
                "SELECT id,digest FROM live_frames WHERE session_id=? AND source=? AND epoch=? AND seq=?",
                (identity, source, epoch, seq),
            ).fetchone()
            if old:
                if old["digest"] != digest:
                    raise HTTPException(
                        409, "Sequence already contains different bytes."
                    )
                return {
                    "id": old["id"],
                    "seq": seq,
                    "duplicate": True,
                    "sample_interval_ms": 1000,
                }
            last = db.execute(
                "SELECT MAX(seq) AS seq, MAX(received) AS received FROM live_frames WHERE session_id=? AND source=? AND epoch=?",
                (identity, source, epoch),
            ).fetchone()
            if last["seq"] is not None and seq < last["seq"] - 60:
                raise HTTPException(
                    410, "Frame is older than the retained input window."
                )
            if last["received"] and time.time() - last["received"] < 0.2:
                raise HTTPException(
                    429,
                    "Capture is faster than the ingress limit.",
                    headers={"Retry-After": "1"},
                )
            cursor = db.execute(
                "INSERT INTO live_frames(session_id,source,epoch,seq,digest,path,captured,received,sharpness,signature,width,height) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    identity,
                    source,
                    epoch,
                    seq,
                    digest,
                    str(path),
                    x_capture_time,
                    time.time(),
                    sharpness,
                    signature,
                    width,
                    height,
                ),
            )
            db.execute(
                "UPDATE live_sources SET last_received=?,lease=? WHERE session_id=? AND source=?",
                (time.time(), time.time() + 20, identity, source),
            )
            result = {"id": cursor.lastrowid, "seq": seq, "sample_interval_ms": 1000}
        path = None
        return result
    finally:
        if path:
            path.unlink(missing_ok=True)


class SourceStop(BaseModel):
    final_seq: int = Field(ge=-1, le=1000000)
    skipped: int = Field(default=0, ge=0)


@router.post("/sessions/{identity}/sources/{source}/stop")
def stop_source(
    identity: str,
    source: Literal["A", "B"],
    body: SourceStop,
    x_live_token: str = Header(default=""),
):
    with store.connect() as db:
        source_row(db, identity, source, x_live_token)
        db.execute(
            "UPDATE live_sources SET status='stopped',final_seq=?,skipped=? WHERE session_id=? AND source=?",
            (body.final_seq, body.skipped, identity, source),
        )
    return {"status": "stopped"}


@router.post("/sessions/{identity}/stop")
def stop(identity: str, x_live_token: str = Header(default="")):
    with store.connect() as db:
        authorize(session(db, identity)["token"], x_live_token)
        db.execute(
            "UPDATE live_sessions SET status='stopping',stop_deadline=? WHERE id=? AND status='open'",
            (time.time() + 5, identity),
        )
    return {"status": "stopping"}


@router.post("/sessions/{identity}/cancel")
def cancel(identity: str, x_live_token: str = Header(default="")):
    with store.connect() as db:
        authorize(session(db, identity)["token"], x_live_token)
        db.execute(
            "UPDATE live_sessions SET status='cancelled',processing='stopped',generation=generation+1 WHERE id=? AND status IN ('open','stopping')",
            (identity,),
        )
    return {"status": "cancelled"}


@router.post("/sessions/{identity}/resume")
def resume(identity: str, x_live_token: str = Header(default="")):
    with store.connect() as db:
        authorize(session(db, identity)["token"], x_live_token)
        db.execute(
            "UPDATE live_sessions SET processing='waiting',error=NULL WHERE id=? AND status='open'",
            (identity,),
        )
    return {"status": "waiting"}


@router.get("/sessions/{identity}/scenes")
def scenes(identity: str, cursor: int = 1000000000):
    with store.connect() as db:
        session(db, identity)
        rows = db.execute(
            "SELECT b.number,s.manifest FROM live_batches b JOIN scenes s ON s.id=b.scene_id WHERE b.session_id=? AND b.number<? ORDER BY b.number DESC LIMIT 20",
            (identity, cursor),
        ).fetchall()
    return {
        "scenes": [json.loads(r["manifest"]) for r in rows],
        "cursor": rows[-1]["number"] if rows else None,
    }


@router.post("/sessions/{identity}/scenes/{scene_id}/pin")
def pin(identity: str, scene_id: str, x_live_token: str = Header(default="")):
    with store.connect() as db:
        authorize(session(db, identity)["token"], x_live_token)
        if not db.execute(
            "SELECT 1 FROM live_batches WHERE session_id=? AND scene_id=?",
            (identity, scene_id),
        ).fetchone():
            raise HTTPException(404, "Published scene not found in this session.")
        db.execute("INSERT OR IGNORE INTO live_pins VALUES(?)", (scene_id,))
    return {"pinned": True}
