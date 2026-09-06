const ELLIPSE_95_SCALE = Math.sqrt(5.991464547107979);

export function acceptSnapshot(current, incoming) {
  if (!incoming || typeof incoming.revision !== 'number') return current;
  if (current && incoming.revision <= current.revision) return current;
  return incoming;
}

export function mergeCueHistory(current, incoming) {
  const bySequence = new Map(current.map((cue) => [cue.sequence, cue]));
  for (const cue of incoming) {
    if (!bySequence.has(cue.sequence)) bySequence.set(cue.sequence, cue);
  }
  return [...bySequence.values()].sort((a, b) => a.sequence - b.sequence);
}

export function applyCue(current, cue) {
  if (!current || cue.revision <= current.revision) return current;
  const cueHistory = mergeCueHistory(current.cue_history, [cue]);
  return cueHistory.length === current.cue_history.length
    ? current
    : { ...current, cue_history: cueHistory };
}

export function timelineAdditions(current, snapshot) {
  const known = new Set(current.map((cue) => cue.sequence));
  return snapshot.cue_history.filter((cue) => !known.has(cue.sequence));
}

export function riskBandLabel(band) {
  if (band === 'low') return 'Low geometric risk';
  if (band === 'medium') return 'Moderate geometric risk';
  if (band === 'high') return 'High geometric risk';
  return 'Unknown geometric risk';
}

export function rankedIntentLabel(intent, rank) {
  return `${rank}. ${intent}`;
}

export function covarianceEllipse95(covariance) {
  const xx = Math.max(0, Number(covariance?.[0]?.[0]) || 0);
  const xz = Number(covariance?.[0]?.[1]) || 0;
  const zz = Math.max(0, Number(covariance?.[1]?.[1]) || 0);
  const trace = xx + zz;
  const difference = xx - zz;
  const root = Math.sqrt(Math.max(0, difference * difference + 4 * xz * xz));
  const majorVariance = Math.max(0, (trace + root) / 2);
  const minorVariance = Math.max(0, (trace - root) / 2);
  return {
    major: ELLIPSE_95_SCALE * Math.sqrt(majorVariance),
    minor: ELLIPSE_95_SCALE * Math.sqrt(minorVariance),
    angle: 0.5 * Math.atan2(2 * xz, difference),
  };
}
