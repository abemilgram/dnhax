import json
import pytest
from backend import models, reconstruct, store


def test_model_selection_and_missing_explicit_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ROOT", tmp_path)
    monkeypatch.setattr(models.Path, "home", lambda: tmp_path)
    monkeypatch.delenv("VGGT_OMEGA_CHECKPOINT", raising=False)
    monkeypatch.setenv("SIMV1_MODEL", "auto")
    assert models.model_config()["key"] == "vggt"
    omega = tmp_path / "models/vggt-omega/vggt_omega_1b_512.pt"
    omega.parent.mkdir(parents=True)
    omega.touch()
    assert models.model_config()["key"] == "vggt_omega"
    monkeypatch.setenv("SIMV1_MODEL", "vggt")
    assert models.model_config()["key"] == "vggt"
    monkeypatch.setenv("SIMV1_MODEL", "amb3r")
    amb3r = models.model_config()
    assert amb3r["key"] == "amb3r"
    assert amb3r["checkpoint"] == tmp_path / "models/amb3r/amb3r.pt"
    amb3r_weights = tmp_path / "models/amb3r/amb3r.pt"
    amb3r_weights.parent.mkdir(parents=True, exist_ok=True)
    amb3r_weights.touch()
    monkeypatch.setenv("SIMV1_MODEL", "auto")
    assert models.model_config()["key"] == "vggt_omega"
    omega.unlink()
    assert models.model_config()["key"] == "vggt"
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


def test_reconstruct_vflips_and_keeps_every_frame(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    monkeypatch.delenv("SIMV1_MAX_FRAMES", raising=False)
    monkeypatch.setattr(reconstruct, "compute_device", lambda: "cpu")
    monkeypatch.setattr(reconstruct.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    seen = {}

    def decoded(args, **kwargs):
        seen["args"] = list(args)
        out = tmp_path / "reconstructions/capture" / models.model_config()["key"] / "frames"
        out.mkdir(parents=True, exist_ok=True)
        for index in range(3):
            (out / f"{index:04d}.jpg").touch()
        return type("Result", (), {"returncode": 0, "stderr": ""})()

    monkeypatch.setattr(reconstruct.subprocess, "run", decoded)

    def fail(*args):
        raise RuntimeError("kept every extracted frame")

    monkeypatch.setattr(reconstruct, "infer_images", fail)
    with pytest.raises(RuntimeError, match="kept every extracted frame"):
        reconstruct.reconstruct({"id": "capture", "path": "video"}, lambda stage: None)
    assert "vflip" in seen["args"][seen["args"].index("-vf") + 1]
    assert "-frames:v" not in seen["args"]
