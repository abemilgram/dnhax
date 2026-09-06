"""Deterministic geometric belief tracking for fictional tactical scenes.

This package estimates motion state only.  It intentionally exposes no aim,
fire, target-selection, or control outputs.
"""

from .cues import CueConfig, CueReducer
from .detector import Detector, PixelDetection, TimestampedFrame
from .map import TacticalMap, load_map
from .navgraph import IntentDefinition, NavGraph, NavNode
from .planner import PlannerConfig, TacticalPlanner, plan_trajectories
from .projection import (
    CameraIntrinsics,
    CameraPose,
    FixedAerialProjector,
    FixedPoseProvider,
    HomographyCalibration,
    PoseProvider,
    RayGroundCalibration,
    RayGroundProjector,
)
from .scoring import (
    evaluate_route,
    has_line_of_sight,
    interpolate_polyline,
    polyline_length,
    turn_cost,
)
from .schema import (
    ActorState,
    Cue,
    EvidenceState,
    IntentKind,
    LifecycleState,
    Observation,
    PlanRanking,
    RiskBand,
    ScoreComponents,
    SensorSpec,
    TimedPoint,
    TrackSnapshot,
    TrajectoryCandidate,
)
from .tracker import TacticalTracker, TrackerConfig
from .tape import FeedState, ReplayIndex, TacticalTape, load_tape

__all__ = [
    "ActorState",
    "CameraIntrinsics",
    "CameraPose",
    "Cue",
    "CueConfig",
    "CueReducer",
    "Detector",
    "EvidenceState",
    "FeedState",
    "FixedAerialProjector",
    "FixedPoseProvider",
    "HomographyCalibration",
    "IntentDefinition",
    "IntentKind",
    "LifecycleState",
    "NavGraph",
    "NavNode",
    "Observation",
    "PlanRanking",
    "PlannerConfig",
    "PixelDetection",
    "PoseProvider",
    "RayGroundCalibration",
    "RayGroundProjector",
    "ReplayIndex",
    "RiskBand",
    "ScoreComponents",
    "SensorSpec",
    "TacticalMap",
    "TacticalTape",
    "TacticalPlanner",
    "TacticalTracker",
    "TimedPoint",
    "TimestampedFrame",
    "TrackerConfig",
    "TrackSnapshot",
    "TrajectoryCandidate",
    "evaluate_route",
    "has_line_of_sight",
    "interpolate_polyline",
    "load_map",
    "load_tape",
    "plan_trajectories",
    "polyline_length",
    "turn_cost",
]
