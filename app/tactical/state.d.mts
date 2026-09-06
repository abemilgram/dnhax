import type {
  Matrix4,
  RiskBand,
  TacticalCue,
  TacticalSnapshot,
} from './types';

export function acceptSnapshot(
  current: TacticalSnapshot | null,
  incoming: TacticalSnapshot,
): TacticalSnapshot;

export function mergeCueHistory(
  current: TacticalCue[],
  incoming: TacticalCue[],
): TacticalCue[];

export function applyCue(
  current: TacticalSnapshot | null,
  cue: TacticalCue,
): TacticalSnapshot | null;

export function timelineAdditions(
  current: TacticalCue[],
  snapshot: TacticalSnapshot,
): TacticalCue[];

export function riskBandLabel(band: RiskBand): string;
export function rankedIntentLabel(intent: string, rank: number): string;

export function covarianceEllipse95(covariance: Matrix4): {
  major: number;
  minor: number;
  angle: number;
};
