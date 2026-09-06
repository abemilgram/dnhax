import numpy as np
import pytest
from fastapi.testclient import TestClient
from backend import store, joint
from backend.api import app
from backend.worker import process


@pytest.fixture
def captures(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "ROOT", tmp_path)
    client = TestClient(app)
    ids = [client.post('/api/captures', data={'source': source},
                       files={'file': ('video.mp4', b'test')}).json()['id'] for source in ('A', 'B')]
    return client, {'capture_a': ids[0], 'capture_b': ids[1]}


def test_joint_accepts_unreconstructed_captures_and_deduplicates(captures):
    client, payload = captures
    first = client.post('/api/joint', json=payload)
    assert first.status_code == 202
    assert client.post('/api/joint', json=payload).json() == first.json()
    assert store.claim()['kind'] == 'joint'
    assert store.claim() is None
    assert client.get('/api/state').json()['scenes'] == []


def test_joint_rejects_missing_repeated_and_swapped_sources(captures):
    client, payload = captures
    assert client.post('/api/joint', json={**payload, 'capture_a': 'missing'}).status_code == 404
    assert client.post('/api/joint', json={**payload, 'capture_a': payload['capture_b']}).status_code == 422
    assert client.post('/api/joint', json={'capture_a': payload['capture_b'], 'capture_b': payload['capture_a']}).status_code == 422
    assert store.claim() is None


def test_joint_uses_one_sequence_and_preserves_shared_camera_coordinates(captures, monkeypatch):
    client, payload = captures
    monkeypatch.setenv('SIMV1_JOINT_FRAMES_PER_SOURCE', '2')
    monkeypatch.setattr(joint, 'compute_device', lambda: 'cpu')
    def candidates(capture, folder, count):
        folder.mkdir(parents=True)
        return [{'path': folder / f'{i}.jpg', 't': float(i * 5),
                 'sharpness': 1, 'descriptors': None} for i in range(2)]
    monkeypatch.setattr(joint, 'sample_candidates', candidates)
    calls = []
    def predict(images, device, progress):
        calls.append(images)
        n = len(images)
        ex = np.tile(np.eye(4)[:3], (n, 1, 1))
        ex[:, 0, 3] = np.arange(n) * 10
        return {'model': 'facebook/VGGT-Omega', 'model_variant': '1B-512',
                'depth': np.ones((n, 8, 8)), 'confidence': np.ones((n, 8, 8)),
                'extrinsics': ex, 'intrinsics': np.tile(np.eye(3), (n, 1, 1)),
                'rgb': np.tile(np.arange(n)[:, None, None, None] / 4, (1, 3, 8, 8))}
    monkeypatch.setattr(joint, 'infer_images', predict)
    client.post('/api/joint', json=payload)
    process(store.claim())
    state = client.get('/api/state').json()
    scene = state['scenes'][0]
    assert len(calls) == 1 and len(calls[0]) == 4
    assert [p.parts[-3] for p in calls[0]] == ['A', 'A', 'B', 'B']
    assert scene['reconstruction']['method'] == 'joint_vggt'
    assert scene['reconstruction']['model'] == 'facebook/VGGT-Omega'
    assert scene['diagnostics'] is None
    assert scene['reconstruction']['overlap_verified'] is False
    assert state['jobs'][0]['status'] == 'completed'
    a, b = scene['clouds']
    assert a['count'] == b['count'] == 128
    assert a['capture_id'] == payload['capture_a'] and b['capture_id'] == payload['capture_b']
    assert b['cameras'][0]['world_to_camera'][0][3] == 20
    for c in (a, b):
        assert c['model'] == 'facebook/VGGT-Omega'
        np.testing.assert_array_equal(c['transform'], np.eye(4))
        assert len(client.get(c['colors']).content) == c['count'] * 3
    xyz_a = np.frombuffer(client.get(a['points']).content, dtype='<f4').reshape(-1, 3)
    xyz_b = np.frombuffer(client.get(b['points']).content, dtype='<f4').reshape(-1, 3)
    np.testing.assert_allclose(xyz_b - xyz_a, np.tile([-20, 0, 0], (128, 1)))
    # Retry creates another scene, preserving previous artifacts and provenance.
    client.post('/api/joint', json=payload)
    process(store.claim())
    assert client.get('/api/state').json()['scenes'][1] == scene


def test_failed_joint_never_publishes_a_scene(captures, monkeypatch):
    client, payload = captures
    def fail(*args):
        raise RuntimeError('Insufficient usable video')
    monkeypatch.setattr(joint, 'reconstruct_joint', fail)
    client.post('/api/joint', json=payload)
    with pytest.raises(RuntimeError, match='Insufficient'):
        process(store.claim())
    assert client.get('/api/state').json()['scenes'] == []


def test_keyframes_include_anchor_and_span_clip():
    frames = [{'t': float(i), 'sharpness': 100, 'path': str(i)} for i in range(12)]
    selected = joint.select_frames(frames, 4, 3)
    assert selected[0]['t'] == 3
    assert len({f['t'] for f in selected}) == 4
    assert max(f['t'] for f in selected) >= 10
    assert min(f['t'] for f in selected) <= 1


def test_joint_frame_budget_is_separate_and_bounded(monkeypatch):
    monkeypatch.setenv('SIMV1_MAX_FRAMES', '2')
    monkeypatch.delenv('SIMV1_JOINT_FRAMES_PER_SOURCE', raising=False)
    assert joint.frames_per_source() == 4
    for value in ('0', '1', '9', 'garbage'):
        monkeypatch.setenv('SIMV1_JOINT_FRAMES_PER_SOURCE', value)
        with pytest.raises(ValueError, match='2 to 8'):
            joint.frames_per_source()
