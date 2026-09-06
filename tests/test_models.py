import pytest
from backend import models, reconstruct, store


def test_unset_model_selects_amb3r(tmp_path, monkeypatch):
    monkeypatch.setattr(models, "ROOT", tmp_path)
    monkeypatch.delenv("SIMV1_MODEL", raising=False)
    config = models.model_config()
    assert config["key"] == "amb3r"
    assert config["checkpoint"] == tmp_path / "models/amb3r/amb3r.pt"


def test_explicit_amb3r_is_accepted(monkeypatch):
    monkeypatch.setenv("SIMV1_MODEL", "amb3r")
    assert models.model_config()["key"] == "amb3r"


@pytest.mark.parametrize("selected", ["auto", "vggt", "vggt_omega"])
def test_removed_model_selectors_are_rejected(monkeypatch, selected):
    monkeypatch.setenv("SIMV1_MODEL", selected)
    with pytest.raises(ValueError, match="Only SIMV1_MODEL=amb3r"):
        models.model_config()


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
