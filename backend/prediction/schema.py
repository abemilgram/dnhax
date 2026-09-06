"""Explicit metric geometry, source-clock and pixel-evidence contracts.

World axes are X/Y on the ground plane and Z up. Cameras use OpenCV axes.
No enemy entity identifier or server-transform ingestion exists in this contract.
"""

from typing import Annotated, Literal
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

Number = Annotated[float, Field(allow_inf_nan=False)]
Vec2 = tuple[Number, Number]
Vec3 = tuple[Number, Number, Number]
Matrix = list[list[Number]]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NavNode(Contract):
    id: str = Field(min_length=1, max_length=80)
    xyz: Vec3


class NavEdge(Contract):
    a: str
    b: str


class Surface(Contract):
    id: str
    polygon: list[Vec2] = Field(min_length=3, max_length=64)
    z: Number


class Wall(Contract):
    id: str
    minimum: Vec3
    maximum: Vec3

    @model_validator(mode="after")
    def extent(self):
        if any(a >= b for a, b in zip(self.minimum, self.maximum)):
            raise ValueError("Wall bounds must have positive extent on all axes")
        return self


class Intent(Contract):
    id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=100)
    kind: Literal["hold", "route"]
    goal_node: str | None = None
    prior: float = Field(default=1, gt=0, le=100, allow_inf_nan=False)

    @model_validator(mode="after")
    def goal(self):
        if (self.kind == "route") != (self.goal_node is not None):
            raise ValueError("Route intents require a goal; hold intents have no goal")
        return self


class ScenePrior(Contract):
    id: str = Field(min_length=1, max_length=80)
    coordinate_frame: str = Field(min_length=1, max_length=100)
    units: Literal["meters"] = "meters"
    provenance: str = Field(min_length=8, max_length=2000)
    scale_source: str = Field(min_length=8, max_length=500)
    synthetic: bool = False
    nodes: list[NavNode] = Field(min_length=2, max_length=128)
    edges: list[NavEdge] = Field(min_length=1, max_length=512)
    surfaces: list[Surface] = Field(min_length=1, max_length=64)
    walls: list[Wall] = Field(default_factory=list, max_length=128)
    intents: list[Intent] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def references(self):
        coordinates = [v for n in self.nodes for v in n.xyz]
        coordinates += [v for s in self.surfaces for p in s.polygon for v in p]
        coordinates += [s.z for s in self.surfaces]
        coordinates += [v for w in self.walls for p in (w.minimum, w.maximum) for v in p]
        if any(abs(v) > 10_000 for v in coordinates):
            raise ValueError("Scene coordinates must be within 10km of the shared origin")
        ids = [n.id for n in self.nodes]
        if len(ids) != len(set(ids)):
            raise ValueError("Navigation node IDs must be unique")
        if len({i.id for i in self.intents}) != len(self.intents):
            raise ValueError("Intent IDs must be unique")
        if not any(i.kind == "hold" for i in self.intents):
            raise ValueError("Retain a hold hypothesis")
        lookup = {n.id: np.asarray(n.xyz) for n in self.nodes}
        for e in self.edges:
            if e.a not in ids or e.b not in ids or e.a == e.b:
                raise ValueError("Edges must connect two distinct existing nodes")
            if np.linalg.norm(lookup[e.a] - lookup[e.b]) > 200:
                raise ValueError("Split navigation edges longer than 200 meters into shorter segments")
        if any(i.goal_node not in ids for i in self.intents if i.kind == "route"):
            raise ValueError("Unknown intent goal node")
        return self


class Camera(Contract):
    sensor_id: str = Field(min_length=1, max_length=80)
    coordinate_frame: str
    calibration_id: str = Field(min_length=1, max_length=100)
    calibration_provenance: str = Field(min_length=8, max_length=1000)
    width: int = Field(ge=16, le=8192)
    height: int = Field(ge=16, le=8192)
    intrinsics: Matrix
    camera_to_world: Matrix
    position_std_m: float = Field(default=0.25, ge=0.01, le=20, allow_inf_nan=False)
    pixel_std: float = Field(default=3, ge=0.5, le=100, allow_inf_nan=False)
    available_t: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def matrices(self):
        k, p = np.asarray(self.intrinsics), np.asarray(self.camera_to_world)
        if k.shape != (3, 3) or p.shape != (4, 4):
            raise ValueError("Camera needs 3x3 intrinsics and 4x4 pose")
        if not np.allclose(k[2], [0, 0, 1]) or k[0, 0] <= 0 or k[1, 1] <= 0 or abs(np.linalg.det(k)) < 1e-8:
            raise ValueError("Invalid pinhole intrinsics")
        r = p[:3, :3]
        if not np.allclose(p[3], [0, 0, 0, 1]) or not np.allclose(r.T @ r, np.eye(3), atol=1e-5) or not np.isclose(np.linalg.det(r), 1, atol=1e-5):
            raise ValueError("Camera pose must be rigid; apply metric scale to position, not rotation")
        return self


class Detection(Contract):
    box: tuple[Number, Number, Number, Number]
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    label: Literal["person"] = "person"
    ground_contact_visible: bool = False

    @model_validator(mode="after")
    def extent(self):
        if self.box[0] >= self.box[2] or self.box[1] >= self.box[3]:
            raise ValueError("Box must be [left, top, right, bottom] with positive area")
        return self


class FramePacket(Contract):
    frame_id: str = Field(min_length=1, max_length=160)
    clock_id: str = Field(min_length=1, max_length=100)
    t: float = Field(ge=0, allow_inf_nan=False)
    available_t: float = Field(ge=0, allow_inf_nan=False)
    clock_uncertainty_s: float = Field(default=0, ge=0, le=5, allow_inf_nan=False)
    camera: Camera
    evidence_source: Literal["video_detector", "synthetic_fixture"]
    detector: str = Field(min_length=1, max_length=200)
    detections: list[Detection] = Field(default_factory=list, max_length=64)
    coverage_complete: bool = False
    detection_probability: float | None = Field(default=None, gt=0, lt=1, allow_inf_nan=False)
    coverage_assumption: str | None = Field(default=None, min_length=8, max_length=500)

    @model_validator(mode="after")
    def validity(self):
        if self.available_t < self.t or self.available_t < self.camera.available_t:
            raise ValueError("Evidence cannot be available before capture or calibration")
        if self.coverage_complete and (self.detection_probability is None or self.coverage_assumption is None):
            raise ValueError("Negative evidence requires an explicit detection/coverage assumption")
        for d in self.detections:
            x1, y1, x2, y2 = d.box
            if x1 < 0 or y1 < 0 or x2 > self.camera.width or y2 > self.camera.height:
                raise ValueError("Boxes must use the calibrated full-frame pixel coordinates")
        return self


class Observation(Contract):
    """Internal result of pixel projection; never an HTTP ingestion type."""
    sensor_id: str
    frame_id: str
    detection_index: int | None = Field(default=None, ge=0, le=63)
    t: Number
    xyz: Vec3
    covariance_xy: Matrix
    confidence: Number
    surface_id: str


class EngineConfig(Contract):
    prediction_model: Literal["route_bank", "branching"] = "route_bank"
    horizon_s: float = Field(default=10, ge=2, le=15, allow_inf_nan=False)
    step_s: float = Field(default=0.2, ge=0.1, le=0.5, allow_inf_nan=False)
    particles: int = Field(default=64, ge=8, le=512)
    speed_mps: float = Field(default=2.5, gt=0.1, le=8, allow_inf_nan=False)
    speed_std_mps: float = Field(default=0.6, ge=0, le=3, allow_inf_nan=False)
    stale_after_s: float = Field(default=0.5, gt=0, le=5, allow_inf_nan=False)
    expire_after_s: float = Field(default=20, gt=1, le=120, allow_inf_nan=False)
    max_frame_lag_s: float = Field(default=1, ge=0, le=5, allow_inf_nan=False)
    max_clock_uncertainty_s: float = Field(default=0.1, ge=0, le=1, allow_inf_nan=False)
    acceleration_std_mps2: float = Field(default=1, gt=0, le=10, allow_inf_nan=False)
    association_gate: float = Field(default=9.21, gt=0, le=30, allow_inf_nan=False)
    min_detection_confidence: float = Field(default=0.3, ge=0, le=1, allow_inf_nan=False)
    max_tracks: int = Field(default=16, ge=1, le=64)
    hysteresis_margin: float = Field(default=0.05, ge=0, le=1, allow_inf_nan=False)
    hysteresis_dwell_s: float = Field(default=0.4, ge=0, le=5, allow_inf_nan=False)
    player_sensor_id: str = "player"
    random_seed: int = Field(default=41, ge=0, le=2**31-1)
    branch_exploration_probability: float = Field(default=.2, ge=0, le=1, allow_inf_nan=False)
    branch_stop_rate_per_s: float = Field(default=.12, ge=0, le=2, allow_inf_nan=False)
    branch_reverse_rate_per_s: float = Field(default=.04, ge=0, le=2, allow_inf_nan=False)

    @model_validator(mode="after")
    def limits(self):
        if self.expire_after_s <= self.stale_after_s:
            raise ValueError("Expiry must be later than stale threshold")
        if self.particles * len(np.arange(0, self.horizon_s + self.step_s, self.step_s)) * self.max_tracks > 400_000:
            raise ValueError("Simulation budget too large; reduce particles, horizon or tracks")
        return self
