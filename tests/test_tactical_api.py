import json

from fastapi.testclient import TestClient

from backend import store
from backend.api import app
from backend.tactical import api as tactical_api


def client_for(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    tactical_api._services.clear()
    return TestClient(app)


def test_tactical_state_and_controls_contract(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch) as client:
        initial = client.get("/api/tactical/state")
        assert initial.status_code == 200
        state = initial.json()
        assert state["schema_version"] == 1
        assert {"obstacles", "nodes", "edges", "zones"} <= set(state["map"])

        started = client.post("/api/tactical/start").json()
        paused = client.post("/api/tactical/pause").json()
        sought = client.post("/api/tactical/seek", json={"position": 3.0}).json()
        restarted = client.post("/api/tactical/restart").json()
        revisions = [
            state["revision"],
            started["revision"],
            paused["revision"],
            sought["revision"],
            restarted["revision"],
        ]
        assert revisions == sorted(revisions)
        assert len(set(revisions)) == len(revisions)
        assert paused["replay"]["playing"] is False
        assert sought["replay"]["position"] == 3.0
        assert restarted["replay"] == {
            "id": "golden-a-site-v1",
            "duration": 6.0,
            "position": 0.0,
            "playing": True,
        }


def test_tactical_api_rejects_malformed_seek_and_event_cursor(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch) as client:
        assert client.post("/api/tactical/seek", json={"position": 7.0}).status_code == 422
        assert client.post("/api/tactical/seek", json={"position": 1.25}).status_code == 422
        assert client.post("/api/tactical/seek", json={}).status_code == 422
        assert (
            client.get(
                "/api/tactical/events?once=true",
                headers={"Last-Event-ID": "not-a-revision"},
            ).status_code
            == 400
        )


def test_sse_framing_snapshot_cues_heartbeat_and_resume(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch) as client:
        response = client.get("/api/tactical/events?once=true")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        assert "event: cue\n" in response.text
        assert "event: snapshot\n" in response.text
        assert "\ndata: {" in response.text
        assert response.text.endswith("\n\n")
        for block in response.text.strip().split("\n\n"):
            data_line = next(line for line in block.splitlines() if line.startswith("data: "))
            json.loads(data_line.removeprefix("data: "))

        revision = client.get("/api/tactical/state").json()["revision"]
        resumed = client.get(
            "/api/tactical/events?once=true",
            headers={"Last-Event-ID": str(revision)},
        )
        assert resumed.text == ": heartbeat\n\n"
        assert (
            client.get(f"/api/tactical/events?once=true&since={revision}").text
            == ": heartbeat\n\n"
        )


def test_cues_are_not_duplicated_in_sqlite_after_rebuild(tmp_path, monkeypatch):
    with client_for(tmp_path, monkeypatch) as client:
        client.post("/api/tactical/seek", json={"position": 6.0})
        with store.connect() as db:
            first = db.execute(
                "SELECT COUNT(*) FROM tactical_cues WHERE session_id='golden-a-site'"
            ).fetchone()[0]
            session = db.execute(
                "SELECT tape_id,status FROM tactical_sessions WHERE id='golden-a-site'"
            ).fetchone()
        assert first > 1
        assert dict(session) == {
            "tape_id": "golden-a-site-v1",
            "status": "paused",
        }

        client.post("/api/tactical/restart")
        client.post("/api/tactical/seek", json={"position": 6.0})
        with store.connect() as db:
            second = db.execute(
                "SELECT COUNT(*) FROM tactical_cues WHERE session_id='golden-a-site'"
            ).fetchone()[0]
        assert second == first


def test_workspace_reset_recreates_tactical_service_on_reset_database(
    tmp_path, monkeypatch
):
    with client_for(tmp_path, monkeypatch) as client:
        client.post("/api/tactical/seek", json={"position": 3.0})
        key = str(store.ROOT)
        previous = tactical_api._services[key]

        response = client.post("/api/reset")
        assert response.status_code == 200
        assert key not in tactical_api._services

        fresh_state = client.get("/api/tactical/state").json()
        assert tactical_api._services[key] is not previous
        assert fresh_state["replay"]["position"] == 0.0
        with store.connect() as db:
            session = db.execute(
                "SELECT replay_time,revision FROM tactical_sessions "
                "WHERE id='golden-a-site'"
            ).fetchone()
        assert dict(session) == {
            "replay_time": 0.0,
            "revision": fresh_state["revision"],
        }


def test_tactical_router_is_registered_before_static_mount():
    paths = [getattr(route, "path", "") for route in app.routes]
    assert "/api/tactical/state" in paths
    if "/" in paths:
        assert paths.index("/api/tactical/state") < paths.index("/")
