"""Recorded image inputs and explicit readiness gates for video-driven inference."""

from pathlib import Path
from typing import Literal
from pydantic import Field, model_validator
from .schema import Camera, Contract, EngineConfig, ScenePrior


class ImageFrame(Contract):
    frame_id: str = Field(min_length=1, max_length=160)
    sensor_id: str = Field(min_length=1, max_length=80)
    image: str = Field(min_length=1, max_length=500)
    t: float = Field(ge=0, allow_inf_nan=False)
    original_frame_index: int | None = Field(default=None, ge=0)
    observation_usable: bool = True
    unusable_reason: str | None = Field(default=None, min_length=8, max_length=500)
    camera: Camera | None = None

    @model_validator(mode="after")
    def identity(self):
        if not self.observation_usable and not self.unusable_reason:
            raise ValueError("Excluded observation frames need a reason")
        if self.camera and self.camera.sensor_id != self.sensor_id:
            raise ValueError("Camera sensor does not match the image source")
        path = Path(self.image)
        if path.is_absolute() or ".." in path.parts or ":" in self.image:
            raise ValueError("Images must be relative files inside the take directory")
        return self


class RecordedTake(Contract):
    id: str = Field(min_length=1, max_length=100)
    clock_id: str = Field(min_length=1, max_length=100)
    provenance: str = Field(min_length=12, max_length=2000)
    timing_provenance: str = Field(min_length=12, max_length=2000)
    synchronization_verified: bool = False
    geometry_verified: bool = False
    geometry_validation_provenance: str | None = Field(default=None, min_length=20, max_length=2000)
    clock_uncertainty_s: float = Field(ge=0, le=5, allow_inf_nan=False)
    frames: list[ImageFrame] = Field(min_length=1, max_length=20_000)
    prior: ScenePrior | None = None
    config: EngineConfig = Field(default_factory=EngineConfig)
    # Model confidence is not ground-contact truth. This opt-in rule is always
    # reported as provisional until compared with independent pixel labels.
    ground_contact_policy: Literal["reject_unknown", "pose_ankles_provisional_v1"] = "reject_unknown"

    @model_validator(mode="after")
    def frames_and_prior(self):
        if self.geometry_verified and (self.prior is None or not self.geometry_validation_provenance):
            raise ValueError("Verified geometry requires a map and independent scale/landmark validation provenance")
        identities = [(f.sensor_id, f.frame_id) for f in self.frames]
        if len(identities) != len(set(identities)):
            raise ValueError("Take contains duplicate source/frame IDs")
        if len({f.sensor_id for f in self.frames}) > 8:
            raise ValueError("At most eight sensors per take")
        if self.prior and self.prior.synthetic:
            raise ValueError("Real-image ingestion cannot use a synthetic map prior")
        if self.prior and any(f.camera and f.camera.coordinate_frame != self.prior.coordinate_frame for f in self.frames):
            raise ValueError("Camera and map coordinate frames differ")
        return self
