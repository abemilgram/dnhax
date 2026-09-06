"""FastAPI transport for deterministic tactical replay."""

import asyncio
import json
import time

from fastapi import APIRouter, Header, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .. import store
from .migrations import SqliteTacticalPersistence
from .service import TacticalService


router = APIRouter(prefix="/api/tactical", tags=["tactical"])
_services: dict[str, TacticalService] = {}


def _service() -> TacticalService:
    """Keep one hot replay per monkeypatchable workspace root."""

    key = str(store.ROOT)
    current = _services.get(key)
    if current is None:
        current = TacticalService.from_path(
            persistence=SqliteTacticalPersistence(store.connect),
            session_id="golden-a-site",
        )
        _services[key] = current
    return current


def _checked(operation):
    try:
        return operation()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/state")
def state():
    return _service().current()


@router.post("/start")
def start():
    return _service().start()


@router.post("/pause")
def pause():
    return _service().pause()


@router.post("/restart")
def restart():
    return _service().restart()


class SeekRequest(BaseModel):
    position: float = Field(ge=0.0, lt=120.0, allow_inf_nan=False)


@router.post("/seek")
def seek(body: SeekRequest):
    return _checked(lambda: _service().seek(body.position))


def _last_revision(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        revision = int(value.split(":", 1)[0])
    except ValueError as exc:
        raise HTTPException(400, "Last-Event-ID must start with a revision") from exc
    if revision < 0:
        raise HTTPException(400, "Last-Event-ID revision must be non-negative")
    return revision


def format_sse(event: str, revision: int, data: dict[str, object]) -> str:
    encoded = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return f"id: {revision}\nevent: {event}\ndata: {encoded}\n\n"


@router.get("/events")
async def events(
    request: Request,
    since: int = Query(default=0, ge=0),
    once: bool = Query(default=False),
    last_event_id: str | None = Header(default=None),
):
    header_revision = _last_revision(last_event_id)
    cursor = max(since, header_revision or 0)

    async def stream():
        nonlocal cursor
        last_heartbeat = time.monotonic()
        try:
            while True:
                pending = _service().events_since(cursor)
                if pending:
                    for event, revision, payload in pending:
                        yield format_sse(event, revision, payload)
                        cursor = max(cursor, revision)
                    if once:
                        return
                elif once:
                    yield ": heartbeat\n\n"
                    return
                if await request.is_disconnected():
                    return
                now = time.monotonic()
                if now - last_heartbeat >= 15.0:
                    yield ": heartbeat\n\n"
                    last_heartbeat = now
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
