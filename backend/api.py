import json
import time
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from . import store

app = FastAPI(title="simv1 room reconstruction", version="0.1.0")


@app.get("/ping")
def ping():
    return {"status": "ok"}


@app.get("/api/state")
def state():
    with store.connect() as db:
        heartbeat = db.execute("SELECT * FROM heartbeat WHERE id=1").fetchone()
        captures = [
            {**dict(r), "url": store.artifact_url(r["path"])}
            for r in db.execute("SELECT * FROM captures ORDER BY created DESC LIMIT 40")
        ]
        for capture in captures:
            capture.pop("path")
        jobs = [
            {k: v for k, v in dict(r).items() if k != "payload"}
            for r in db.execute("SELECT * FROM jobs ORDER BY created DESC LIMIT 20")
        ]
        scenes = [
            json.loads(r["manifest"])
            for r in db.execute(
                "SELECT manifest FROM scenes WHERE json_extract(manifest, '$.live') IS NULL ORDER BY created DESC LIMIT 10"
            )
        ]
    return {
        "captures": captures,
        "jobs": jobs,
        "scenes": scenes,
        "worker": {
            "online": bool(heartbeat and time.time() - heartbeat["updated"] < 20),
            "device": heartbeat["device"] if heartbeat else "Not started",
        },
        "mode": "submitted captures",
    }


@app.post("/api/captures", status_code=201)
async def upload(source: Literal["A", "B"] = Form(...), file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp4", ".mov", ".webm", ".mkv"}:
        raise HTTPException(415, "Use MP4, MOV, WebM, or MKV video.")
    identity = store.uid()
    directory = store.ROOT / "captures" / identity
    directory.mkdir(parents=True)
    path = directory / ("video" + suffix)
    size = 0
    try:
        with path.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > 512 * 1024 * 1024:
                    raise HTTPException(413, "Capture exceeds the 512 MB limit.")
                out.write(chunk)
        if size == 0:
            raise HTTPException(400, "The capture is empty.")
        with store.connect() as db:
            db.execute(
                "INSERT INTO captures VALUES(?,?,?,?,?)",
                (
                    identity,
                    source,
                    Path(file.filename or "capture").name,
                    time.time(),
                    str(path),
                ),
            )
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"id": identity, "source": source}


@app.post("/api/captures/{identity}/delete")
def delete_capture(identity: str):
    result = store.delete_capture(identity)
    if result is None:
        raise HTTPException(404, "Capture not found.")
    return result


class CleanupRequest(BaseModel):
    hours: float = Field(default=24, gt=0, le=8760, allow_inf_nan=False)


@app.post("/api/cleanup")
def cleanup(request: CleanupRequest):
    return store.cleanup_old(request.hours)


@app.post("/api/reset")
def reset():
    store.reset_workspace()
    return {"reset": True}


class CaptureJob(BaseModel):
    capture_id: str


@app.post("/api/reconstruct", status_code=202)
def reconstruct(request: CaptureJob):
    with store.connect() as db:
        if not db.execute(
            "SELECT id FROM captures WHERE id=?", (request.capture_id,)
        ).fetchone():
            raise HTTPException(404, "Capture not found.")
        existing = db.execute(
            "SELECT id FROM jobs WHERE kind='reconstruct' AND status IN ('queued','running') AND json_extract(payload,'$.capture_id')=?",
            (request.capture_id,),
        ).fetchone()
    if existing:
        return {"job_id": existing["id"]}
    return {"job_id": store.enqueue("reconstruct", request.model_dump())}


@app.post("/api/sample", status_code=202)
def sample():
    return {"job_id": store.enqueue("sample", {})}


class Registration(BaseModel):
    scene_id: str
    source_points: list[list[float]] = Field(min_length=8, max_length=2000)
    target_points: list[list[float]] = Field(min_length=8, max_length=2000)
    threshold: float = Field(default=0.08, gt=0, le=10, allow_inf_nan=False)


@app.post("/api/register", status_code=202)
def registration(request: Registration):
    from .geometry import fit_similarity

    try:
        fit_similarity(request.source_points, request.target_points)
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with store.connect() as db:
        row = db.execute(
            "SELECT manifest FROM scenes WHERE id=?", (request.scene_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Scene not found.")
        if len(json.loads(row["manifest"])["clouds"]) != 2:
            raise HTTPException(409, "Two clouds are required.")
    return {"job_id": store.enqueue("register", request.model_dump())}


class PairRequest(BaseModel):
    capture_a: str
    capture_b: str


@app.post("/api/joint", status_code=202)
def joint(request: PairRequest):
    if request.capture_a == request.capture_b:
        raise HTTPException(422, "Choose two different captures.")
    with store.connect() as db:
        for identity, source in [(request.capture_a, "A"), (request.capture_b, "B")]:
            capture = db.execute(
                "SELECT source FROM captures WHERE id=?", (identity,)
            ).fetchone()
            if not capture:
                raise HTTPException(404, f"Source {source} capture not found.")
            if capture["source"] != source:
                raise HTTPException(422, f"Choose a source {source} capture.")
        existing = db.execute(
            "SELECT id FROM jobs WHERE kind='joint' AND status IN ('queued','running') "
            "AND json_extract(payload,'$.capture_a')=? AND json_extract(payload,'$.capture_b')=?",
            (request.capture_a, request.capture_b),
        ).fetchone()
    if existing:
        return {"job_id": existing["id"]}
    return {"job_id": store.enqueue("joint", request.model_dump())}


@app.post("/api/pair", status_code=202)
def pair(request: PairRequest):
    for identity in [request.capture_a, request.capture_b]:
        if not identity.isalnum() or len(identity) != 32:
            raise HTTPException(422, "Invalid capture ID.")
        if not (store.ROOT / "reconstructions" / identity / "cloud.json").exists():
            raise HTTPException(409, "Reconstruct both captures first.")
    if request.capture_a == request.capture_b:
        raise HTTPException(422, "Choose two different captures.")
    return {"job_id": store.enqueue("pair", request.model_dump())}


@app.get("/api/artifacts/{relative:path}")
def artifact(relative: str):
    path = (store.ROOT / relative).resolve()
    allowed = {"captures", "reconstructions", "scenes"}
    if (
        not path.is_relative_to(store.ROOT)
        or Path(relative).parts[0] not in allowed
        or path.suffix
        not in {".bin", ".ply", ".json", ".jpg", ".mp4", ".webm", ".mov", ".mkv"}
        or not path.is_file()
    ):
        raise HTTPException(404, "Artifact not found.")
    parts = Path(relative).parts
    if parts[0] == 'scenes':
        with store.connect() as db:
            if len(parts) < 3 or not db.execute('SELECT 1 FROM scenes WHERE id=?', (parts[1],)).fetchone():
                raise HTTPException(404, 'Scene has not been published.')
    return FileResponse(path)


from .live_api import router as live_router

app.include_router(live_router)

# Production export: one origin serves interface and API on port 8000.
web = Path(__file__).resolve().parents[1] / "dist" / "client"
if web.is_dir():
    app.mount("/", StaticFiles(directory=web, html=True), name="web")
