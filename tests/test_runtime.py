from types import SimpleNamespace
import pytest
from backend.runtime import select_device, frame_limit, release_memory


def fake_torch(cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
    )


def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch):
    monkeypatch.delenv("SIMV1_DEVICE", raising=False)
    assert select_device(fake_torch(True, True)) == "cuda"
    assert select_device(fake_torch(False, True)) == "mps"
    assert select_device(fake_torch()) == "cpu"


def test_explicit_device_never_silently_falls_back():
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        select_device(fake_torch(), "mps")
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        select_device(fake_torch(), "cuda")
    with pytest.raises(ValueError):
        select_device(fake_torch(), "metal")


def test_frame_limit_is_unlimited_by_default(monkeypatch):
    monkeypatch.delenv("SIMV1_MAX_FRAMES", raising=False)
    assert [frame_limit(d) for d in ["cuda", "mps", "cpu"]] == [None, None, None]
    monkeypatch.setenv("SIMV1_MAX_FRAMES", "4")
    assert frame_limit("mps") == 4
    monkeypatch.setenv("SIMV1_MAX_FRAMES", "0")
    assert frame_limit("mps") is None
    monkeypatch.setenv("SIMV1_MAX_FRAMES", "unlimited")
    assert frame_limit("cuda") is None
    monkeypatch.setenv("SIMV1_MAX_FRAMES", "1")
    with pytest.raises(ValueError, match="unlimited"):
        frame_limit("mps")


def test_cleanup_targets_only_selected_backend():
    calls = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(empty_cache=lambda: calls.append("cuda")),
        mps=SimpleNamespace(empty_cache=lambda: calls.append("mps")),
    )
    release_memory(torch, "mps")
    assert calls == ["mps"]
    release_memory(torch, "cpu")
    assert calls == ["mps"]
