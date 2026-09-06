import io
import json
import time
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend import live, store, reconstruct
from backend.api import app
from backend.worker import process
from backend.live_geometry import apply_similarity, continuity, world_pixels


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    from backend import live_api
    from types import SimpleNamespace

    monkeypatch.setattr(
        live_api.shutil, "disk_usage", lambda path: SimpleNamespace(free=20 * 1024**3)
    )
    monkeypatch.setenv("SIMV1_MODEL", "vggt")
    client = TestClient(app)
    session = client.post(
        "/api/live/sessions", json={"request_key": "test-session"}
    ).json()
    owners = {
        s: client.post(
            f"/api/live/sessions/{session['id']}/sources/{s}/claim", json={}
        ).json()
        for s in ("A", "B")
    }
    monkeypatch.setattr(reconstruct, "compute_device", lambda: "cpu")
    monkeypatch.setattr(reconstruct, "infer_images", fake_prediction)
    return client, session, owners


def fake_prediction(images, *args):
    n = len(images)
    ex = np.tile(np.eye(4)[:3], (n, 1, 1))
    ex[:, 0, 3] = np.arange(n) * 0.1
    return {
        "model": "fixture-model",
        "model_variant": "test",
        "depth": np.ones((n, 32, 32)),
        "confidence": np.ones((n, 32, 32)),
        "extrinsics": ex,
        "intrinsics": np.tile(np.eye(3), (n, 1, 1)),
        "rgb": np.ones((n, 3, 32, 32)) * 0.5,
    }


def jpeg(seed):
    buffer = io.BytesIO()
    Image.fromarray(
        np.random.default_rng(seed).integers(0, 255, (64, 64, 3), dtype=np.uint8)
    ).save(buffer, format="JPEG")
    return buffer.getvalue()


def upload(rig, source, seq, body=None):
    client, session, owners = rig
    with store.connect() as db:
        db.execute("UPDATE live_frames SET received=?", (time.time() - 1,))
    return client.put(
        f"/api/live/sessions/{session['id']}/sources/{source}/frames/{owners[source]['epoch']}/{seq}",
        content=body or jpeg(seq + (100 if source == "B" else 0)),
        headers={
            "Content-Type": "image/jpeg",
            "X-Live-Token": owners[source]["token"],
            "X-Capture-Time": str(seq),
        },
    )


def seed(rig, count=2):
    for s in ("A", "B"):
        for n in range(count):
            assert upload(rig, s, n).status_code == 200


def state(rig):
    return rig[0].get("/api/live/sessions/" + rig[1]["id"]).json()


def test_ingress_idempotency_validation_and_ownership(rig):
    client, session, owners = rig
    first = upload(rig, "A", 1)
    assert first.status_code == 200
    assert upload(rig, "A", 1).json()["id"] == first.json()["id"]
    assert upload(rig, "A", 1, jpeg(55)).status_code == 409
    assert upload(rig, "A", 2, b"broken").status_code == 422
    assert upload(rig, "A", 2, b"x" * (2 * 1024 * 1024 + 1)).status_code == 413
    assert upload(rig, "A", 0).status_code == 200
    assert state(rig)["sources"][0]["accepted"] == [0, 1]
    url = f"/api/live/sessions/{session['id']}/sources/A/claim"
    assert client.post(url, json={}).status_code == 409
    assert client.post(url, json={"takeover": True}).status_code == 200
    assert upload(rig, "A", 2).status_code == 409
    assert (
        client.post(
            "/api/live/sessions", json={"request_key": "another-session"}
        ).status_code
        == 409
    )
    assert (
        client.post("/api/live/sessions", json={"request_key": "test-session"}).json()
        == session
    )


def test_batch_publication_is_atomic_and_retry_is_idempotent(rig):
    seed(rig)
    live.schedule()
    job = store.claim()
    process(job)
    result = state(rig)
    assert result["latest_batch"] == 1 and result["scene"]["live"]["batch"] == 1
    assert [c["source"] for c in result["scene"]["clouds"]] == ["A", "B"]
    for cloud in result["scene"]["clouds"]:
        assert len(rig[0].get(cloud["points"]).content) == cloud["count"] * 12
        assert rig[0].get(cloud["cameras"][0]["frame"]).status_code == 200
    process(job)
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0] == 1
        assert (
            db.execute("SELECT status FROM jobs WHERE id=?", (job["id"],)).fetchone()[0]
            == "completed"
        )


def test_queue_stays_bounded_and_next_batch_uses_freshest_frames(rig):
    seed(rig)
    live.schedule()
    first = store.claim()
    for n in range(2, 12):
        assert upload(rig, "A", n).status_code == 200
        live.schedule()
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM live_batches").fetchone()[0] == 1
    process(first)
    live.schedule()
    second = store.claim()
    with store.connect() as db:
        frames = json.loads(
            db.execute(
                "SELECT frames FROM live_batches WHERE id=?",
                (second["payload"]["batch_id"],),
            ).fetchone()[0]
        )
    assert max(f["seq"] for f in frames if f["source"] == "A") == 11
    assert len(frames) <= 4


def test_stop_drains_at_most_one_tail_batch(rig):
    seed(rig)
    live.schedule()
    process(store.claim())
    upload(rig, "A", 2)
    client, s, _ = rig
    headers = {"X-Live-Token": s["token"]}
    assert (
        client.post(
            f"/api/live/sessions/{s['id']}/stop", json={}, headers=headers
        ).status_code
        == 200
    )
    live.schedule()
    assert store.claim() is None
    with store.connect() as db:
        db.execute("UPDATE live_sessions SET stop_deadline=?", (time.time() - 1,))
    assert upload(rig, "A", 3).status_code == 409
    live.schedule()
    process(store.claim())
    live.schedule()
    assert state(rig)["status"] == "completed"
    assert state(rig)["latest_batch"] == 2
    assert store.claim() is None


def test_cancel_during_inference_cannot_publish(rig, monkeypatch):
    seed(rig)
    live.schedule()
    client, s, _ = rig

    def cancel_then_predict(images, *args):
        client.post(
            f"/api/live/sessions/{s['id']}/cancel",
            json={},
            headers={"X-Live-Token": s["token"]},
        )
        return fake_prediction(images)

    monkeypatch.setattr(reconstruct, "infer_images", cancel_then_predict)
    process(store.claim())
    assert state(rig)["status"] == "cancelled" and state(rig)["scene"] is None
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0] == 0


def test_stale_source_does_not_block_fresh_source(rig):
    seed(rig, 4)
    with store.connect() as db:
        db.execute(
            "UPDATE live_sources SET last_received=? WHERE source='B'",
            (time.time() - 30,),
        )
    live.schedule()
    process(store.claim())
    assert [c["source"] for c in state(rig)["scene"]["clouds"]] == ["A"]


def test_worker_recovery_replaces_interrupted_work(rig):
    seed(rig)
    live.schedule()
    job = store.claim()
    live.recover()
    live.schedule()
    replacement = store.claim()
    assert replacement["id"] != job["id"]
    process(replacement)
    assert state(rig)["latest_batch"] == 2


def test_failure_keeps_last_good_preview(rig):
    seed(rig)
    live.schedule()
    process(store.claim())
    old = state(rig)["latest_scene"]
    upload(rig, "A", 2)
    live.schedule()
    job = store.claim()
    live.fail(job["payload"]["batch_id"], "Test OOM")
    assert state(rig)["latest_scene"] == old
    assert state(rig)["processing"] == "paused_error"
    live.schedule()
    assert store.claim() is None


def test_retention_preserves_pinned_and_latest_artifacts(rig):
    seed(rig)
    client, s, _ = rig
    live.schedule()
    process(store.claim())
    first = state(rig)["scene"]
    client.post(
        f"/api/live/sessions/{s['id']}/scenes/{first['id']}/pin",
        json={},
        headers={"X-Live-Token": s["token"]},
    )
    for seq in range(2, 9):
        upload(rig, "A", seq)
        live.schedule()
        process(store.claim())
        live.cleanup()
    assert client.get(first["clouds"][0]["points"]).status_code == 200
    assert client.get(state(rig)["scene"]["clouds"][0]["points"]).status_code == 200
    with store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM scenes").fetchone()[0] == 6


def test_continuity_transforms_cameras_and_geometry_together():
    prior = fake_prediction(["a", "b"])
    current = fake_prediction(["a", "b"])
    matrix = np.eye(4)
    matrix[:3, :3] *= 1.1
    matrix[:3, 3] = [0.1, 0.2, 0.3]
    apply_similarity(prior, matrix)
    fitted, detail = continuity(current, [1, 2], prior, [1, 2])
    assert detail["status"] == "accepted"
    np.testing.assert_allclose(fitted, matrix, atol=1e-8)
    apply_similarity(current, fitted)
    np.testing.assert_allclose(
        world_pixels(current, 0)[0], world_pixels(prior, 0)[0], atol=1e-8
    )
    assert continuity(current, [3, 4], prior, [1, 2])[1]["status"] == "new_segment"


def test_low_disk_backpressure_is_explicit(rig, monkeypatch):
    from backend import live_api
    from types import SimpleNamespace

    monkeypatch.setattr(
        live_api.shutil, "disk_usage", lambda path: SimpleNamespace(free=0)
    )
    response = upload(rig, "A", 0)
    assert response.status_code == 429
    assert "Storage reserve" in response.json()["detail"]


def test_cancelled_artifacts_are_not_public(rig, monkeypatch):
    seed(rig); live.schedule(); client,s,_=rig
    def cancelled(images,*args):
        client.post(f"/api/live/sessions/{s['id']}/cancel", json={}, headers={'X-Live-Token':s['token']})
        return fake_prediction(images)
    monkeypatch.setattr(reconstruct, 'infer_images', cancelled)
    process(store.claim())
    for manifest in (store.ROOT/'scenes').glob('*/manifest.json'):
        assert client.get('/api/artifacts/scenes/'+manifest.parent.name+'/manifest.json').status_code==404


def test_existing_workspace_state_does_not_auto_follow_live(rig):
    seed(rig); live.schedule(); process(store.claim())
    assert state(rig)['scene'] is not None
    assert rig[0].get('/api/state').json()['scenes']==[]
