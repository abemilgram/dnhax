import assert from 'node:assert/strict';
import test from 'node:test';
import {
  acceptSnapshot,
  applyCue,
  covarianceEllipse95,
  mergeCueHistory,
  rankedIntentLabel,
  riskBandForScore,
  riskBandLabel,
  timelineAdditions,
} from '../app/tactical/state.mjs';

const cue = (sequence, revision = sequence) => ({
  sequence,
  revision,
  t: sequence / 10,
  intent: 'HOLD',
  risk_band: 'low',
  risk: 0.1,
  route_valid: true,
  tracking_degraded: false,
  reasons: [],
  route: [],
});

const snapshot = (revision, cueHistory = []) => ({
  schema_version: 1,
  revision,
  cue_history: cueHistory,
});

test('rejects stale and duplicate snapshot revisions', () => {
  const current = snapshot(8);
  assert.equal(acceptSnapshot(current, snapshot(7)), current);
  assert.equal(acceptSnapshot(current, snapshot(8)), current);
  assert.equal(acceptSnapshot(current, snapshot(9)).revision, 9);
});

test('deduplicates cues by stable sequence', () => {
  const first = cue(1);
  const merged = mergeCueHistory([first], [cue(1, 99), cue(2)]);
  assert.deepEqual(merged.map((item) => item.sequence), [1, 2]);
  assert.equal(merged[0], first);
});

test('stale cue events cannot repopulate a post-seek snapshot', () => {
  const afterSeek = snapshot(20, [cue(1)]);
  assert.equal(applyCue(afterSeek, cue(3, 19)), afterSeek);
  assert.deepEqual(
    applyCue(afterSeek, cue(2, 21)).cue_history.map((item) => item.sequence),
    [1, 2],
  );
});

test('newer snapshots authoritatively truncate timeline after rewind', () => {
  const beforeSeek = snapshot(20, [cue(1), cue(2), cue(3)]);
  const afterSeek = snapshot(21, [cue(1)]);
  assert.equal(acceptSnapshot(beforeSeek, afterSeek), afterSeek);
  assert.deepEqual(
    acceptSnapshot(beforeSeek, afterSeek).cue_history.map(
      (item) => item.sequence,
    ),
    [1],
  );
});

test('provides explicit risk and ranked intent labels', () => {
  assert.equal(riskBandLabel('low'), 'Low geometric risk');
  assert.equal(riskBandLabel('medium'), 'Moderate geometric risk');
  assert.equal(riskBandLabel('high'), 'High geometric risk');
  assert.equal(rankedIntentLabel('FLANK', 2), '2. FLANK');
  assert.equal(riskBandForScore(0.249), 'low');
  assert.equal(riskBandForScore(0.25), 'medium');
  assert.equal(riskBandForScore(0.549), 'medium');
  assert.equal(riskBandForScore(0.55), 'high');
});

test('computes a rotated 95 percent covariance ellipse', () => {
  const covariance = [
    [4, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
  const ellipse = covarianceEllipse95(covariance);
  assert.ok(ellipse.major > ellipse.minor);
  assert.ok(ellipse.major > 4.8);
  assert.ok(ellipse.angle > 0);
});

test('unchanged snapshots add no timeline entries', () => {
  const existing = [cue(1), cue(2)];
  assert.deepEqual(timelineAdditions(existing, snapshot(10, existing)), []);
  assert.deepEqual(
    timelineAdditions(existing, snapshot(11, [...existing, cue(3)])).map(
      (item) => item.sequence,
    ),
    [3],
  );
});
