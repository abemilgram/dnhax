import json
import pytest
from backend import models, reconstruct, store


def test_model_selection_and_missing_explicit_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ROOT", tmp_path)
    monkeypatch.delenv("VGGT_OMEGA_CHECKPOINT", raising=False)
    monkeypatch.setenv("SIMV1_MODEL", "auto")
    assert models.model_config()["key"] == "vggt"
    omega = tmp_path / "models/vggt-omega/vggt_omega_1b_512.pt"
    omega.parent.mkdir(parents=True)
    omega.touch()
    assert models.model_config()["key"] == "vggt_omega"
    monkeypatch.setenv("SIMV1_MODEL", "vggt")
    assert models.model_config()["key"] == "vggt"
    monkeypatch.setenv("SIMV1_MODEL", "auto")
    monkeypatch.setenv("VGGT_OMEGA_CHECKPOINT", str(tmp_path / "missing.pt"))
    assert models.model_config()["key"] == "vggt_omega"
    with pytest.raises(RuntimeError, match="Missing facebook/VGGT-Omega"):
        reconstruct.infer_images([], "cpu", lambda stage: None)
    monkeypatch.setenv("SIMV1_MODEL", "bad")
    with pytest.raises(ValueError, match="SIMV1_MODEL"):
        models.model_config()


def test_omega_never_reuses_public_capture_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    monkeypatch.setenv("SIMV1_MODEL", "vggt_omega")
    monkeypatch.setattr(reconstruct, "compute_device", lambda: "cpu")
    folder = tmp_path / "reconstructions/capture"
    folder.mkdir(parents=True)
    old = {"model": "facebook/VGGT-1B", "points": "old-artifact"}
    (folder / "cloud.json").write_text(json.dumps(old))
    def decoded(*args, **kwargs):
        for i in range(2):
            (folder / "vggt_omega/frames" / f"{i}.jpg").touch()
        return type("Result", (), {"returncode": 0})()
    monkeypatch.setattr(reconstruct.subprocess, "run", decoded)
    def fail(*args):
        raise RuntimeError("Omega inference reached")
    monkeypatch.setattr(reconstruct, "infer_images", fail)
    with pytest.raises(RuntimeError, match="Omega inference reached"):
        reconstruct.reconstruct({"id": "capture", "path": "video"}, lambda stage: None)
    assert json.loads((folder / "cloud.json").read_text()) == old
