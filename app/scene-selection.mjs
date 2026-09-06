/** @param {import('./types').Scene | undefined} scene */
export function isCombinedScene(scene) {
  return (
    !!scene &&
    ['A', 'B'].every((source) =>
      scene.clouds.some((cloud) => cloud.source === source),
    )
  );
}

/** @param {import('./types').Scene} scene */
export function sceneLabel(scene) {
  return scene.sample
    ? 'Sample A + B'
    : scene.reconstruction?.method === 'joint_vggt'
      ? 'Joint A + B'
      : isCombinedScene(scene)
        ? 'Combined A + B'
        : scene.title;
}

/**
 * Prefer a real combined scene on entry; later single-capture results must not
 * replace a combined view. A newly combined/registered scene still opens.
 * @param {import('./types').Scene[]} scenes Newest first.
 * @param {string} selectedId
 * @param {string} previousLatestId
 */
export function nextSceneId(scenes, selectedId, previousLatestId) {
  const newest = scenes[0];
  if (!newest) return '';
  const selected = scenes.find((scene) => scene.id === selectedId);
  if (!selected) {
    return (
      scenes.find((scene) => !scene.sample && isCombinedScene(scene)) || newest
    ).id;
  }
  if (newest.id === previousLatestId) return selected.id;
  if (isCombinedScene(selected) && !newest.sample && !isCombinedScene(newest)) {
    return selected.id;
  }
  return newest.id;
}
