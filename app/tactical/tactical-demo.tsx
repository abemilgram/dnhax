'use client';

import { useRef, useState } from 'react';
import { Pause, Play, RotateCcw } from 'lucide-react';
import TacticalViewer from './tactical-viewer';
import { useTacticalStream } from './use-tactical-stream';
import {
  covarianceEllipse95,
  rankedIntentLabel,
  riskBandForScore,
  riskBandLabel,
} from './state.mjs';
import type {
  TacticalCue,
  TacticalSnapshot,
  TrajectoryCandidate,
} from './types';

function formatClock(value: number) {
  return `${value.toFixed(1).padStart(4, '0')}s`;
}

function FreshnessPanel({ snapshot }: { snapshot: TacticalSnapshot }) {
  return (
    <section className="tactical-panel tactical-feeds" aria-labelledby="feed-title">
      <div className="tactical-panel-heading">
        <div>
          <p className="tactical-kicker">INPUT FRESHNESS</p>
          <h2 id="feed-title">Feed state</h2>
        </div>
        <span>{snapshot.feeds.filter((feed) => feed.state === 'online').length}/{snapshot.feeds.length} fresh</span>
      </div>
      {snapshot.feeds.map((feed) => (
        <div className="tactical-feed-row" key={feed.sensor_id}>
          <span className={`freshness-mark ${feed.state}`} aria-hidden="true" />
          <strong>{feed.sensor_id}</strong>
          <span>{feed.state}</span>
        </div>
      ))}
    </section>
  );
}

function BeliefPanel({ snapshot }: { snapshot: TacticalSnapshot }) {
  const beliefs = [...snapshot.tracks, ...snapshot.ghosts];
  return (
    <section className="tactical-panel tactical-beliefs" aria-labelledby="belief-title">
      <div className="tactical-panel-heading">
        <div>
          <p className="tactical-kicker">BELIEF STATE</p>
          <h2 id="belief-title">Tracked hypotheses</h2>
        </div>
        <span>{beliefs.length} active</span>
      </div>
      {beliefs.length ? beliefs.map((belief) => {
        const ellipse = covarianceEllipse95(belief.covariance);
        return (
          <article className={`belief-row ${belief.evidence}`} key={belief.track_id}>
            <div>
              <strong>Belief {belief.track_id}</strong>
              <span>{belief.lifecycle}</span>
            </div>
            <div className="belief-evidence">
              <b>{belief.evidence}</b>
              <span>
                95% uncertainty {ellipse.major.toFixed(2)} × {ellipse.minor.toFixed(2)} m
              </span>
            </div>
          </article>
        );
      }) : <p className="tactical-empty">No tracked hypotheses at this replay beat.</p>}
    </section>
  );
}

function RiskBar({ value, label }: { value: number; label: string }) {
  const bounded = Math.max(0, Math.min(1, value));
  return (
    <div className="risk-component">
      <span>{label}</span>
      <span className="risk-track" aria-hidden="true">
        <i style={{ width: `${bounded * 100}%` }} />
      </span>
      <b>{Math.round(bounded * 100)}%</b>
    </div>
  );
}

function FutureCard({
  candidate,
  rank,
}: {
  candidate: TrajectoryCandidate;
  rank: number;
}) {
  const band = riskBandForScore(candidate.score.risk_score);
  return (
    <article className={`future-card rank-${rank}`}>
      <div className="future-card-heading">
        <div>
          <span className="future-rank">{rank === 1 ? 'SOLID' : rank === 2 ? 'DASHED' : 'DOTTED'}</span>
          <h3>{rankedIntentLabel(candidate.intent, rank)}</h3>
        </div>
        <span className={`risk-pill ${band}`}>{riskBandLabel(band)}</span>
      </div>
      <RiskBar value={candidate.score.exposure_fraction} label="Exposure" />
      <RiskBar value={candidate.score.open_fraction} label="Open space" />
      <RiskBar value={candidate.score.uncertainty_risk} label="Uncertainty" />
      <RiskBar value={1 - candidate.score.reach_probability} label="Non-reach" />
      <footer>
        <span>{candidate.score.route_length.toFixed(1)} m route</span>
        <span>{candidate.valid ? 'geometrically valid' : 'invalid geometry'}</span>
      </footer>
    </article>
  );
}

function FuturesPanel({ snapshot }: { snapshot: TacticalSnapshot }) {
  return (
    <section className="tactical-futures" aria-labelledby="futures-title">
      <div className="tactical-section-heading">
        <div>
          <p className="tactical-kicker">RANKED FUTURES</p>
          <h2 id="futures-title">HOLD / CROSS / FLANK</h2>
        </div>
        <span>geometric scoring · no automatic control</span>
      </div>
      <div className="future-grid">
        {snapshot.ranking?.candidates.slice(0, 3).map((candidate, index) => (
          <FutureCard
            candidate={candidate}
            rank={index + 1}
            key={candidate.intent}
          />
        ))}
      </div>
    </section>
  );
}

function CueTimeline({
  snapshot,
  cues,
}: {
  snapshot: TacticalSnapshot;
  cues: TacticalCue[];
}) {
  return (
    <section className="tactical-panel tactical-timeline" aria-labelledby="timeline-title">
      <div className="tactical-panel-heading">
        <div>
          <p className="tactical-kicker">CHANGE-ONLY CUES</p>
          <h2 id="timeline-title">Decision timeline</h2>
        </div>
        <span>{cues.length} changes</span>
      </div>
      <div role="log" aria-live="polite" aria-relevant="additions">
        {cues.length ? cues.slice().reverse().map((cue) => (
          <article className="cue-entry" key={cue.sequence}>
            <time>{formatClock(cue.t)}</time>
            <div>
              <strong>{cue.intent}</strong>
              <span>{riskBandLabel(cue.risk_band)}</span>
              {cue.tracking_degraded && <em>degraded evidence</em>}
            </div>
          </article>
        )) : (
          <p className="tactical-empty">
            Awaiting the first change at {formatClock(snapshot.replay.position)}.
          </p>
        )}
      </div>
    </section>
  );
}

export default function TacticalDemo() {
  const {
    snapshot,
    connection,
    error,
    controlPending,
    start,
    pause,
    restart,
    seek,
  } = useTacticalStream();
  const [sliderValue, setSliderValue] = useState<number | null>(null);
  const seekDraft = useRef<number | null>(null);

  if (!snapshot) {
    return (
      <main className="tactical-shell">
        <div className="tactical-loading" role={error ? 'alert' : 'status'}>
          <span className="tactical-pulse" />
          <h1>Tactical Brain</h1>
          <p>{error || 'Connecting to deterministic replay…'}</p>
        </div>
      </main>
    );
  }

  const displayedPosition = sliderValue ?? snapshot.replay.position;
  const commitSeek = () => {
    if (seekDraft.current === null) return;
    const position = Math.round(seekDraft.current * 10) / 10;
    seekDraft.current = null;
    setSliderValue(null);
    void seek(position);
  };

  return (
    <main className="tactical-shell">
      <header className="tactical-header">
        <div>
          <p className="tactical-kicker">DNHACKS DEFENSE / FICTIONAL METRIC TWIN</p>
          <h1>Tactical Brain</h1>
        </div>
        <div className="replay-status">
          <span className={`stream-state ${connection}`}>{connection}</span>
          <strong>{formatClock(snapshot.replay.position)}</strong>
          <span>of {formatClock(snapshot.replay.duration)}</span>
          <b>DETERMINISTIC GOLDEN-TAPE</b>
        </div>
      </header>

      {error && <div className="tactical-error" role="alert">{error}</div>}

      <section className="tactical-stage">
        <TacticalViewer snapshot={snapshot} />
        <div className="tactical-side-stack">
          <FreshnessPanel snapshot={snapshot} />
          <BeliefPanel snapshot={snapshot} />
          <CueTimeline snapshot={snapshot} cues={snapshot.cue_history} />
        </div>
      </section>

      <section className="playback-panel" aria-label="Replay controls">
        <div className="playback-buttons">
          <button type="button" disabled={controlPending || snapshot.replay.playing} onClick={() => void start()}>
            <Play size={15} /> Start
          </button>
          <button type="button" disabled={controlPending || !snapshot.replay.playing} onClick={() => void pause()}>
            <Pause size={15} /> Pause
          </button>
          <button type="button" disabled={controlPending} onClick={() => void restart()}>
            <RotateCcw size={15} /> Restart
          </button>
        </div>
        <label>
          <span>Replay position</span>
          <input
            type="range"
            min={0}
            max={snapshot.replay.duration}
            step={0.1}
            value={displayedPosition}
            onChange={(event) => {
              const value = Number(event.target.value);
              seekDraft.current = value;
              setSliderValue(value);
            }}
            onPointerUp={commitSeek}
            onKeyUp={commitSeek}
            onBlur={commitSeek}
            aria-valuetext={`${formatClock(displayedPosition)} of ${formatClock(snapshot.replay.duration)}`}
          />
          <output>{formatClock(displayedPosition)}</output>
        </label>
      </section>

      <FuturesPanel snapshot={snapshot} />
      <p className="tactical-disclaimer">
        Deterministic fictional replay for geometric decision support. Drag to orbit; scroll to zoom.
      </p>
    </main>
  );
}
