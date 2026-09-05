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


def test_bounded_frame_defaults_and_override(monkeypatch):
    monkeypatch.delenv("SIMV1_MAX_FRAMES", raising=False)
    assert [frame_limit(d) for d in ["cuda", "mps", "cpu"]] == [24, 8, 2]
    monkeypatch.setenv("SIMV1_MAX_FRAMES", "4")
    assert frame_limit("mps") == 4
    monkeypatch.setenv("SIMV1_MAX_FRAMES", "0")
    with pytest.raises(ValueError):
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


def test_portable_forward_uses_real_torch_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    from backend.runtime import predict_geometry

    def forbidden(*args, **kwargs):
        raise AssertionError("Non-CUDA forward touched CUDA")

    monkeypatch.setattr(torch.cuda, "is_bf16_supported", forbidden)

    class Model:
        training = False

        def aggregator(self, images):
            assert images.shape == (1, 2, 3, 8, 8)
            return [images.mean(dim=(-2, -1))], 1

        def camera_head(self, features, patch_token_start):
            return features[-1]

        def dense_head(self, features, images, patch_token_start):
            return images.mean(2)[..., None], torch.ones((1, 2, 8, 8))

        def __call__(self, images):
            raise AssertionError("Called hardcoded upstream forward")

    images = torch.rand(2, 3, 8, 8)
    with torch.inference_mode():
        result = predict_geometry(Model(), images, "cpu")
    assert result["depth"].shape == (1, 2, 8, 8, 1)
    assert torch.isfinite(result["depth"]).all()
    assert result["pose_enc"].dtype == torch.float32


def test_cuda_forward_is_preserved():
    from backend.runtime import predict_geometry

    sentinel = object()
    assert predict_geometry(lambda images: sentinel, None, "cuda") is sentinel
