"""Deterministic geometric belief tracking for fictional tactical scenes.

This package estimates motion state only.  It intentionally exposes no aim,
fire, target-selection, or control outputs.
"""

from .cues import CueConfig, CueReducer
from .map import TacticalMap, load_map
from .navgraph import IntentDefinition, NavGraph, NavNode
from .planner import PlannerConfig, TacticalPlanner, plan_trajectories
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

__all__ = [
    "ActorState",
    "Cue",
    "CueConfig",
    "CueReducer",
    "EvidenceState",
    "IntentDefinition",
    "IntentKind",
    "LifecycleState",
    "NavGraph",
    "NavNode",
    "Observation",
    "PlanRanking",
    "PlannerConfig",
    "RiskBand",
    "ScoreComponents",
    "SensorSpec",
    "TacticalMap",
    "TacticalPlanner",
    "TacticalTracker",
    "TimedPoint",
    "TrackerConfig",
    "TrackSnapshot",
    "TrajectoryCandidate",
    "evaluate_route",
    "has_line_of_sight",
    "interpolate_polyline",
    "load_map",
    "plan_trajectories",
    "polyline_length",
    "turn_cost",
]
