import assert from 'node:assert/strict';
import test from 'node:test';
import { nextSceneId, sceneLabel } from '../app/scene-selection.mjs';

const a = { id: 'a', title: 'Capture A', clouds: [{ source: 'A' }] };
const b = { id: 'b', title: 'Capture B', clouds: [{ source: 'B' }] };
const combined = {
  id: 'combined',
  title: 'Room / independent captures',
  clouds: [...a.clouds, ...b.clouds],
};

test('opening the app prefers real A+B even when a single capture is newer', () => {
  assert.equal(nextSceneId([b, a, combined], '', ''), combined.id);
});
test('a newly finished single capture preserves the combined view', () => {
  assert.equal(
    nextSceneId([b, combined, a], combined.id, combined.id),
    combined.id,
  );
});
test('selecting a single capture survives polling unchanged scene history', () => {
  assert.equal(nextSceneId([combined, b, a], a.id, combined.id), a.id);
});
test('a newly combined or registered result opens automatically', () => {
  assert.equal(nextSceneId([combined, b, a], b.id, b.id), combined.id);
  const registered = { ...combined, id: 'registered' };
  assert.equal(
    nextSceneId([registered, combined, b], combined.id, b.id),
    registered.id,
  );
});
test('single capture, removed selection, and empty workspace fallbacks', () => {
  assert.equal(nextSceneId([b, a], '', ''), b.id);
  assert.equal(nextSceneId([combined, b], 'removed', b.id), combined.id);
  assert.equal(nextSceneId([], a.id, a.id), '');
});
test('scene labels distinguish real pairs, single captures, and sample pairs', () => {
  assert.equal(sceneLabel(combined), 'Combined A + B');
  assert.equal(sceneLabel(a), 'Capture A');
  assert.equal(sceneLabel(b), 'Capture B');
  assert.equal(sceneLabel({ ...combined, sample: true }), 'Sample A + B');
});
