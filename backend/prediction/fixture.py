"""Synthetic pixel fixture, deliberately separate from the engine and real evidence.

This is an invented two-exit room in meters, not a surveyed Dust II model.
Known positions below generate test pixels only. The engine receives boxes.
"""

import numpy as np
from .schema import Camera, Detection, EngineConfig, FramePacket, ScenePrior


def scene() -> ScenePrior:
    return ScenePrior.model_validate({
        "id": "synthetic-two-exit-room-v1", "coordinate_frame": "synthetic-room-meters",
        "provenance": "Hand-authored synthetic unit-test geometry; not measured Dust II geometry.",
        "scale_source": "Synthetic meter coordinates defined by the test fixture author.", "synthetic": True,
        "nodes": [{"id": "entry", "xyz": [0, 0, 0]}, {"id": "junction", "xyz": [0, 2, 0]},
                  {"id": "exit_a", "xyz": [4, 2, 0]}, {"id": "exit_b", "xyz": [0, 10, 0]}],
        "edges": [{"a": "entry", "b": "junction"}, {"a": "junction", "b": "exit_a"},
                  {"a": "junction", "b": "exit_b"}],
        "surfaces": [{"id": "floor", "z": 0, "polygon": [[-3, -4], [8, -4], [8, 12], [-3, 12]]}],
        "walls": [{"id": "east-lower", "minimum": [2, -.1, 0], "maximum": [2.2, 1.2, 3]},
                  {"id": "east-upper", "minimum": [2, 2.8, 0], "maximum": [2.2, 8.5, 3]},
                  {"id": "west", "minimum": [-2.2, -.1, 0], "maximum": [-2, 8.5, 3]}],
        "intents": [{"id": "exit_a", "label": "exit A", "kind": "route", "goal_node": "exit_a", "prior": .45},
                    {"id": "exit_b", "label": "exit B", "kind": "route", "goal_node": "exit_b", "prior": .45},
                    {"id": "hold", "label": "hold or wait", "kind": "hold", "prior": .1}]
    })


def camera(sensor: str, position=None, target=None) -> Camera:
    if position is None:
        position = [0, -6, 6] if sensor == "drone" else [6, -2, 1.6]
    if target is None:
        target = [0, 1, 0] if sensor == "drone" else [4, 2, 1]
    position, target = np.array(position, dtype=float), np.array(target, dtype=float)
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, forward])
    pose[:3, 3] = position
    return Camera(sensor_id=sensor, coordinate_frame=scene().coordinate_frame,
                  calibration_id="synthetic-pinhole-v1", calibration_provenance="Exact synthetic camera for unit tests only.",
                  width=1280, height=720, intrinsics=[[650, 0, 640], [0, 650, 360], [0, 0, 1]],
                  camera_to_world=pose.tolist(), position_std_m=.06, pixel_std=1, available_t=0)


def packet(t: float, sensor: str, xyz=None, *, complete=False, camera_override=None) -> FramePacket:
    from .geometry import project_world
    cam = camera_override or camera(sensor)
    detections = []
    if xyz is not None:
        foot = project_world(cam, np.array(xyz, dtype=float))
        head = project_world(cam, np.array(xyz, dtype=float) + [0, 0, 1.7])
        if foot is None or head is None:
            raise ValueError("Fixture target behind camera")
        detections = [Detection(box=(float(foot[0] - 12), float(min(head[1], foot[1] - 2)),
                                     float(foot[0] + 12), float(foot[1])), confidence=.9, ground_contact_visible=True)]
    return FramePacket(frame_id=f"{sensor}-{t:g}", clock_id="fixture-clock", t=t, available_t=t,
                       camera=cam, evidence_source="synthetic_fixture", detector="synthetic-projected-box-v1",
                       detections=detections, coverage_complete=complete,
                       detection_probability=.85 if complete else None,
                       coverage_assumption="Synthetic fixture assumes 0.85 person detection probability within unobstructed view." if complete else None)


def config() -> EngineConfig:
    return EngineConfig(particles=32, speed_mps=2, speed_std_mps=.2,
                        hysteresis_margin=.02, hysteresis_dwell_s=.3)


def replay_packets() -> list[FramePacket]:
    # No future fixture route is passed to the engine: just these causal pixel packets.
    return [packet(0, "drone", [0, 0, 0]), packet(.1, "player"),
            packet(1, "player", complete=False), packet(2, "player", complete=False),
            packet(3.5, "player", complete=True), packet(4, "player", complete=True),
            packet(4.6, "player", complete=True),
            packet(5, "drone", [0, 10, 0], camera_override=camera("drone", [0, 7, 6], [0, 10, 0]))]
