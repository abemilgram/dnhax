import numpy as np
import pytest

from backend.live_geometry import camera_locations, apply_similarity
from backend import live, store
from backend.worker import process
from test_live import rig, seed, state, upload


def test_camera_center_and_alignment():
    rotation = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 1]])
    center = np.array([2., 3, 4])
    prediction = {"extrinsics": np.array([np.column_stack((rotation, -rotation @ center))]),
                  "depth": np.ones((1, 2, 2))}
    frames = [{"id": 1, "source": "A", "epoch": "one", "seq": 2, "captured": 3., "received": 4.}]
    sample = camera_locations(prediction, frames)["samples"][0]
    np.testing.assert_allclose(sample["position"], center)
    np.testing.assert_allclose(np.array(sample["camera_to_world"])[:3, :3], rotation.T)
    matrix = np.eye(4)
    matrix[:3, :3] = 2 * rotation
    matrix[:3, 3] = [10, 20, 30]
    apply_similarity(prediction, matrix)
    aligned = camera_locations(prediction, frames)["samples"][0]
    np.testing.assert_allclose(aligned["position"], matrix[:3, :3] @ center + matrix[:3, 3])
    np.testing.assert_allclose(np.array(aligned["camera_to_world"])[:3, :3], rotation @ rotation.T)
    prediction["extrinsics"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        camera_locations(prediction, frames)


def test_location_stream_publishes_with_geometry(rig, monkeypatch):
    client, session, _ = rig
    url = f"/api/live/sessions/{session['id']}/camera-locations"
    assert client.get(url).json() == {"updates": [], "cursor": 0}
    assert client.get(url + "?after_batch=-1").status_code == 422
    assert client.get('/api/live/sessions/missing/camera-locations').status_code == 404
    seed(rig)
    live.schedule()
    assert client.get(url).json()["updates"] == []
    process(store.claim())
    scene = state(rig)["scene"]
    result = client.get(url).json()
    update = result["updates"][0]
    assert result["cursor"] == 1
    assert update["scene_id"] == scene["id"]
    assert update["segment"] == scene["live"]["segment"]
    assert update["samples"] == scene["live"]["camera_locations"]["samples"]
    for cloud in scene["clouds"]:
        for camera in cloud["cameras"]:
            frame_id = int(camera["frame"].rsplit('/', 1)[-1].split('.')[0])
            sample = next(s for s in update["samples"] if s["frame_id"] == frame_id)
            ex = np.array(camera["world_to_camera"])
            np.testing.assert_allclose(ex[:, :3] @ sample["position"] + ex[:, 3], 0, atol=1e-6)
    assert client.get(url + '?after_batch=1').json() == {"updates": [], "cursor": 1}
    # Rejected continuity must not mix coordinate segments.
    monkeypatch.setattr('backend.live_geometry.continuity', lambda *args: (np.eye(4), {"status": "new_segment", "reason": "test"}))
    upload(rig, 'A', 2)
    live.schedule()
    process(store.claim())
    second = client.get(url + '?after_batch=1').json()["updates"][0]
    assert second["batch"] == 2
    assert second["segment"] != update["segment"]
    shift = np.eye(4)
    shift[:3, 3] = [10, 20, 30]
    monkeypatch.setattr('backend.live_geometry.continuity', lambda *args: (shift, {"status": "accepted", "reason": "test"}))
    upload(rig, 'A', 3)
    live.schedule()
    process(store.claim())
    third = client.get(url + '?after_batch=2').json()["updates"][0]
    assert third["segment"] == second["segment"]
    assert all(s["position"][1:] == [20., 30.] for s in third["samples"])
    client.post(f"/api/live/sessions/{session['id']}/cancel", headers={"X-Live-Token": session["token"]})
    assert client.get(url + '?after_batch=3').json()["updates"] == []
