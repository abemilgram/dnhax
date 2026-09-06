"""Deterministic geometric belief tracking for fictional tactical scenes.

This package estimates motion state only.  It intentionally exposes no aim,
fire, target-selection, or control outputs.
"""

from .map import TacticalMap, load_map
from .schema import (
    EvidenceState,
    LifecycleState,
    Observation,
    SensorSpec,
    TrackSnapshot,
)
from .tracker import TacticalTracker, TrackerConfig

__all__ = [
    "EvidenceState",
    "LifecycleState",
    "Observation",
    "SensorSpec",
    "TacticalMap",
    "TacticalTracker",
    "TrackerConfig",
    "TrackSnapshot",
    "load_map",
]
