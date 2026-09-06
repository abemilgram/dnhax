import numpy as np
import pytest
import time
from fastapi.testclient import TestClient
from backend import store
from backend.api import app
from backend.geometry import fit_similarity, register, transform
from backend.worker import process


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    return TestClient(app)


def test_similarity_recovers_scale_rotation_translation():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(100, 3))
    m = np.eye(4)
    m[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]) * 1.7
    m[:3, 3] = [4, -2, 8]
    np.testing.assert_allclose(fit_similarity(a, transform(a, m)), m, atol=1e-10)


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(100, 3))
    b = a * 1.3 + [2, 4, -1]
    b[:20] += 20
    matrix, diagnostics = register(a, b, 0.03)
    np.testing.assert_allclose(transform(a[30:], matrix), b[30:], atol=1e-8)
    assert diagnostics["inlier_ratio"] < 1
    assert diagnostics["heldout_count"] == 20


def test_degenerate_landmarks_rejected():
    a = np.column_stack([np.arange(10), np.zeros(10), np.zeros(10)])
    with pytest.raises(ValueError, match="single line"):
        fit_similarity(a, a)


def test_sample_end_to_end_and_binary_artifacts(client):
    response = client.post("/api/sample")
    assert response.status_code == 202
    job = store.claim()
    process(job)
    state = client.get("/api/state").json()
    scene = state["scenes"][0]
    assert scene["sample"] is True and scene["diagnostics"]["status"] == "accepted"
    assert state["jobs"][0]["status"] == "completed"
    for cloud in scene["clouds"]:
        response = client.get(cloud["points"])
        assert response.status_code == 200
        assert len(response.content) == cloud["count"] * 12
        assert client.get(cloud["ply"]).content.startswith(
            b"ply\nformat binary_little_endian"
        )
    assert (
        client.get(f"/api/artifacts/scenes/{scene['id']}/manifest.json").json() == scene
    )


def test_capture_validation_and_missing_reconstruction(client):
    assert (
        client.post(
            "/api/captures",
            data={"source": "C"},
            files={"file": ("video.mp4", b"test", "video/mp4")},
        ).status_code
        == 422
    )
    assert (
        client.post(
            "/api/captures", data={"source": "A"}, files={"file": ("file.exe", b"test")}
        ).status_code
        == 415
    )
    assert (
        client.post(
            "/api/captures", data={"source": "A"}, files={"file": ("video.mp4", b"")}
        ).status_code
        == 400
    )
    capture = client.post(
        "/api/captures",
        data={"source": "A"},
        files={"file": ("../../video.mp4", b"fake-video", "video/mp4")},
    ).json()
    state = client.get("/api/state").json()
    assert state["captures"][0]["name"] == "video.mp4"
    assert "path" not in state["captures"][0]
    assert (
        client.post("/api/reconstruct", json={"capture_id": "missing"}).status_code
        == 404
    )
    first = client.post("/api/reconstruct", json={"capture_id": capture["id"]}).json()
    assert (
        first
        == client.post("/api/reconstruct", json={"capture_id": capture["id"]}).json()
    )
    assert not state["scenes"]


def test_capture_delete_removes_derived_data(client):
    capture = client.post(
        "/api/captures",
        data={"source": "A"},
        files={"file": ("video.mp4", b"fake-video", "video/mp4")},
    ).json()
    reconstruction = store.ROOT / "reconstructions" / capture["id"]
    reconstruction.mkdir(parents=True)
    (reconstruction / "cloud.json").write_text("{}")
    scene_id = store.uid()
    store.publish(
        {
            "id": scene_id,
            "created": 1,
            "sample": False,
            "clouds": [{"capture_id": capture["id"]}],
        }
    )
    client.post("/api/reconstruct", json={"capture_id": capture["id"]})

    response = client.post(f"/api/captures/{capture['id']}/delete")
    assert response.status_code == 200
    assert response.json()["scenes"] == 1
    state = client.get("/api/state").json()
    assert state["captures"] == []
    assert state["scenes"] == []
    assert state["jobs"] == []
    assert not reconstruction.exists()
    assert not (store.ROOT / "scenes" / scene_id).exists()


def test_cleanup_and_reset_iteration_data(client):
    capture = client.post(
        "/api/captures",
        data={"source": "A"},
        files={"file": ("old.mp4", b"old-video", "video/mp4")},
    ).json()
    with store.connect() as db:
        db.execute(
            "UPDATE captures SET created=? WHERE id=?",
            (time.time() - 48 * 3600, capture["id"]),
        )
    response = client.post("/api/cleanup", json={"hours": 24})
    assert response.status_code == 200
    assert response.json()["captures"] == 1

    client.post(
        "/api/captures",
        data={"source": "B"},
        files={"file": ("new.mp4", b"new-video", "video/mp4")},
    )
    client.post("/api/sample")
    models = store.ROOT / "models"
    models.mkdir()
    (models / "keep.txt").write_text("weights stay outside workspace cleanup")
    assert client.post("/api/reset").json() == {"reset": True}
    state = client.get("/api/state").json()
    assert state["captures"] == []
    assert state["jobs"] == []
    assert state["scenes"] == []
    assert (models / "keep.txt").is_file()


def test_artifact_traversal_and_database_are_not_exposed(client):
    client.get("/api/state")
    assert client.get("/api/artifacts/workspace.sqlite").status_code == 404
    assert client.get("/api/artifacts/%2e%2e/requirements.txt").status_code == 404
    assert client.get("/api/artifacts/captures/missing.mp4").status_code == 404


def test_registration_publishes_new_version_preserves_original(client):
    client.post("/api/sample")
    process(store.claim())
    original = client.get("/api/state").json()["scenes"][0]
    response = client.post(
        "/api/register", json={"scene_id": original["id"], **original["landmarks"]}
    )
    assert response.status_code == 202
    process(store.claim())
    scenes = client.get("/api/state").json()["scenes"]
    assert len(scenes) == 2
    assert scenes[1] == original and scenes[0]["id"] != original["id"]
    assert scenes[0]["diagnostics"]["heldout_error"] < 0.08


def test_claim_is_exclusive(client):
    client.post("/api/sample")
    assert store.claim() is not None
    assert store.claim() is None


def test_no_fake_reconstruction_on_missing_model(client, monkeypatch):
    # Failure is explicit and never creates a scene.
    from backend import reconstruct

    def missing(*args):
        raise RuntimeError("Checkpoint not configured")

    monkeypatch.setattr(reconstruct, "reconstruct", missing)
    capture = client.post(
        "/api/captures", data={"source": "A"}, files={"file": ("video.mp4", b"test")}
    ).json()
    client.post("/api/reconstruct", json={"capture_id": capture["id"]})
    with pytest.raises(RuntimeError, match="Checkpoint"):
        process(store.claim())
    assert client.get("/api/state").json()["scenes"] == []


def test_depth_edges_remove_boundary_not_flat_surface():
    from backend.geometry import depth_edge_mask

    depth = np.ones((8, 8))
    depth[:, 4:] = 2
    mask = depth_edge_mask(depth)
    assert mask[:, 3:5].all()
    assert not mask[:, :3].any()
    assert not mask[:, 5:].any()


def test_dense_export_keeps_color_correspondence(tmp_path):
    from backend.geometry import save_cloud

    points = np.arange(300001 * 3, dtype=np.float32).reshape(-1, 3)
    colors = (points % 256).astype("uint8")
    count = save_cloud(tmp_path, points, colors)
    assert count == 300001
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "points.bin", dtype="<f4").reshape(-1, 3), points
    )
    np.testing.assert_array_equal(
        np.fromfile(tmp_path / "colors.bin", dtype="uint8").reshape(-1, 3), colors
    )
