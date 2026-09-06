export type Vec2 = [number, number];
export type Vec3 = [number, number, number];
export type Matrix4 = [
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
  [number, number, number, number],
];

export type EvidenceState = 'observed' | 'stale' | 'conflicting';
export type LifecycleState = 'tentative' | 'confirmed';
export type FeedState = 'online' | 'occluded' | 'offline';
export type IntentKind = 'HOLD' | 'CROSS' | 'FLANK';
export type RiskBand = 'low' | 'medium' | 'high';

export interface ReplayState {
  id: string;
  duration: number;
  position: number;
  playing: boolean;
}

export interface TacticalObstacle {
  id: string;
  min: Vec3;
  max: Vec3;
}

export interface TacticalNode {
  id: string;
  xyz: Vec3;
}

export interface TacticalEdge {
  id: string;
  from: string;
  to: string;
  cost: number;
}

export interface TacticalZone {
  id: string;
  min: Vec3;
  max: Vec3;
}

export interface TacticalMap {
  name: string;
  coordinate_system: 'right-handed:x-east,y-up,z-north';
  obstacles: TacticalObstacle[];
  nodes: TacticalNode[];
  edges: TacticalEdge[];
  zones: TacticalZone[];
}

export interface ActorState {
  t: number;
  xyz: Vec3;
  velocity_xz: Vec2;
}

export interface FeedSnapshot {
  sensor_id: string;
  state: FeedState;
}

export interface TrackSnapshot {
  track_id: number;
  t: number;
  xyz: Vec3;
  velocity_xz: Vec2;
  covariance: Matrix4;
  evidence: EvidenceState;
  lifecycle: LifecycleState;
  hits: number;
  misses: number;
  expected_visible_misses: number;
}

export interface TimedPoint {
  t: number;
  xyz: Vec3;
}

export interface ScoreComponents {
  route_length: number;
  turn_cost: number;
  los_fraction: number;
  exposure_fraction: number;
  time_in_open: number;
  open_fraction: number;
  uncertainty_risk: number;
  risk_std: number;
  exposure_cvar90: number;
  reach_probability: number;
  progress: number;
  invalid_fraction: number;
  risk_score: number;
}

export interface TrajectoryCandidate {
  intent: IntentKind;
  valid: boolean;
  utility: number;
  score: ScoreComponents;
  route: TimedPoint[];
  reasons: string[];
}

export interface PlanRanking {
  t: number;
  cycle_index: number;
  candidates: TrajectoryCandidate[];
}

export interface TacticalCue {
  revision: number;
  sequence: number;
  t: number;
  intent: IntentKind;
  risk_band: RiskBand;
  risk: number;
  route_valid: boolean;
  tracking_degraded: boolean;
  reasons: string[];
  route: TimedPoint[];
}

export interface TacticalSnapshot {
  schema_version: 1;
  revision: number;
  replay: ReplayState;
  map: TacticalMap;
  actor: ActorState;
  feeds: FeedSnapshot[];
  tracks: TrackSnapshot[];
  ghosts: TrackSnapshot[];
  ranking: PlanRanking | null;
  cue_history: TacticalCue[];
}
