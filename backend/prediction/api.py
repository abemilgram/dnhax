"""In-process live predictor. Run one persistent worker; sessions are not serverless jobs."""

from dataclasses import dataclass, field
import hmac
import secrets
import threading
import time
from typing import Annotated, Literal

from fastapi import APIRouter, Header, HTTPException
from pydantic import Field
from .engine import PredictionEngine
from .schema import Contract, EngineConfig, FramePacket, ScenePrior

router = APIRouter(prefix="/api/prediction", tags=["prediction"])


class CreateSession(Contract):
    prior: ScenePrior
    clock_id: str = Field(min_length=1, max_length=100)
    evidence_source: Literal["video_detector", "synthetic_fixture"]
    config: EngineConfig = Field(default_factory=EngineConfig)


class Ingest(Contract):
    packet: FramePacket
    now: float = Field(ge=0, allow_inf_nan=False)


class Tick(Contract):
    now: float = Field(ge=0, allow_inf_nan=False)


@dataclass
class Session:
    engine: PredictionEngine
    token: str
    touched: float = field(default_factory=time.monotonic)
    lock: threading.RLock = field(default_factory=threading.RLock)


_sessions: dict[str, Session] = {}
_lock = threading.RLock()
Token = Annotated[str | None, Header(alias="X-Prediction-Token")]


def session(identity: str, token: str | None) -> Session:
    with _lock:
        result = _sessions.get(identity)
        if result is None or time.monotonic() - result.touched > 1800:
            _sessions.pop(identity, None)
            raise HTTPException(404, "Prediction session missing or idle for more than 30 minutes")
        if token is None or not hmac.compare_digest(result.token, token):
            raise HTTPException(403, "Prediction session token required")
        result.touched = time.monotonic()
        return result


@router.post("/sessions", status_code=201)
def create(request: CreateSession):
    try:
        engine = PredictionEngine(request.prior, clock_id=request.clock_id,
                                  evidence_source=request.evidence_source, config=request.config)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    with _lock:
        for identity, item in list(_sessions.items()):
            if time.monotonic() - item.touched > 1800:
                del _sessions[identity]
        if len(_sessions) >= 8:
            raise HTTPException(503, "Prediction session limit reached")
        identity, token = secrets.token_hex(16), secrets.token_urlsafe(32)
        _sessions[identity] = Session(engine, token)
    return {"id": identity, "token": token, "state": engine.state()}


@router.post("/sessions/{identity}/frames")
def ingest(identity: str, request: Ingest, token: Token = None):
    item = session(identity, token)
    with item.lock:
        try:
            return item.engine.ingest(request.packet, now=request.now)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@router.post("/sessions/{identity}/tick")
def tick(identity: str, request: Tick, token: Token = None):
    item = session(identity, token)
    with item.lock:
        try:
            return item.engine.tick(request.now)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc


@router.get("/sessions/{identity}")
def state(identity: str, token: Token = None):
    item = session(identity, token)
    with item.lock:
        return item.engine.state()


@router.delete("/sessions/{identity}", status_code=204)
def remove(identity: str, token: Token = None):
    session(identity, token)
    with _lock:
        _sessions.pop(identity, None)
