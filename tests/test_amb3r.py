import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import store
from backend.amb3r_runtime import Amb3rRuntime, TARGET_SIZE, _load_image, _world_to_camera
from backend.api import app
from backend.reconstruct import export_cloud


def prediction(unmapped=None):
    y, x = np.mgrid[:4, :5]
    first = np.stack(
        [x * 0.001, y * 0.001, np.ones_like(x)], axis=-1
    ).astype(np.float32)
    second = first + np.array([10, 0, 0], dtype=np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    poses[1, 0, 3] = 10
    return {
        "model": "AMB3R-SfM",
        "model_key": "amb3r",
        "model_variant": "92c4081",
        "precision": "bfloat16",
        "world_points": np.stack([first, second]),
        "confidence": np.ones((2, 4, 5), dtype=np.float32),
        "extrinsics": np.linalg.inv(poses)[:, :3],
        "rgb": np.ones((2, 3, 4, 5), dtype=np.float32),
        "unmapped_frames": unmapped or [],
    }


def test_amb3r_export_uses_world_points_without_intrinsics(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    frames = []
    for index in range(2):
        path = tmp_path / "frames" / f"{index}.jpg"
        path.parent.mkdir(exist_ok=True)
        path.touch()
        frames.append({"path": path, "t": float(index)})
    record = export_cloud(
        tmp_path / "cloud",
        {"id": "capture", "source": "A"},
        frames,
        prediction(),
        [0, 1],
        "cuda",
        reconstruction_method="joint_amb3r",
    )
    points = np.fromfile(tmp_path / "cloud/points.bin", dtype="<f4").reshape(-1, 3)
    assert record["model"] == "AMB3R-SfM"
    assert record["precision"] == "bfloat16"
    assert record["count"] == 40
    np.testing.assert_allclose(points[:20], prediction()["world_points"][0].reshape(-1, 3))
    np.testing.assert_allclose(points[20:], prediction()["world_points"][1].reshape(-1, 3))
    assert "intrinsics" not in record["cameras"][0]
    assert record["cameras"][1]["world_to_camera"][0][3] == -10


def test_amb3r_export_rejects_fully_unmapped_source(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    path = tmp_path / "frame.jpg"
    path.touch()
    with pytest.raises(RuntimeError, match="No mapped geometry"):
        export_cloud(
            tmp_path / "cloud",
            {"id": "capture", "source": "B"},
            [{"path": path, "t": 0.0}],
            prediction(unmapped=[1]),
            [1],
            "cuda",
        )


def test_legacy_depth_artifact_export_still_unprojects_depth(tmp_path, monkeypatch):
    """Keep old depth-based scene artifacts readable after the runtime cutover."""
    monkeypatch.setattr(store, "ROOT", tmp_path)
    path = tmp_path / "frame.jpg"
    path.touch()
    depth = np.ones((4, 4), dtype=np.float32)
    prediction = {
        "model": "facebook/VGGT-1B",
        "model_variant": "1B",
        "depth": depth[None],
        "confidence": np.ones((1, 4, 4), dtype=np.float32),
        "extrinsics": np.eye(4, dtype=np.float32)[None, :3],
        "intrinsics": np.eye(3, dtype=np.float32)[None],
        "rgb": np.ones((1, 3, 4, 4), dtype=np.float32),
    }
    record = export_cloud(
        tmp_path / "cloud",
        {"id": "capture", "source": "A"},
        [{"path": path, "t": 0.0}],
        prediction,
        [0],
        "cpu",
    )
    points = np.fromfile(tmp_path / "cloud/points.bin", dtype="<f4").reshape(-1, 3)
    assert record["model"] == "facebook/VGGT-1B"
    assert record["count"] == 16
    assert points[0, 2] == pytest.approx(1.0)


def test_amb3r_preprocess_is_center_cropped_and_normalized(tmp_path):
    path = tmp_path / "frame.jpg"
    Image.new("RGB", (800, 400), (255, 0, 0)).save(path)
    image = _load_image(path)
    assert image.shape == (3, TARGET_SIZE[1], TARGET_SIZE[0])
    assert image.min() >= -1 and image.max() <= 1


def test_amb3r_pose_inverse_is_world_to_camera():
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    poses[1, 0, 3] = 4
    extrinsics = _world_to_camera(poses)
    assert extrinsics.shape == (2, 3, 4)
    assert extrinsics[1, 0, 3] == pytest.approx(-4)


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_amb3r_runtime_rejects_non_cuda(device):
    with pytest.raises(RuntimeError, match="AMB3R requires CUDA"):
        Amb3rRuntime().load(
            {"checkpoint": None, "model": "AMB3R-SfM"},
            device,
            lambda stage: None,
        )


def test_amb3r_runtime_rejects_missing_checkpoint(tmp_path):
    with pytest.raises(RuntimeError, match="Missing AMB3R-SfM checkpoint"):
        Amb3rRuntime().load(
            {
                "checkpoint": tmp_path / "missing.pt",
                "model": "AMB3R-SfM",
                "download": "scripts/download_amb3r.py",
            },
            "cuda",
            lambda stage: None,
        )


def test_ping_is_available_for_amb3r_image_health_checks():
    assert TestClient(app).get("/ping").json() == {"status": "ok"}
