'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from '@/components/ui/select';
import {
  Box,
  Layers3,
  Upload,
  ArrowUpRight,
  FlaskConical,
  CheckCircle2,
  Link2,
} from 'lucide-react';
import Viewer from './viewer';
import Capture from './capture';
import FrameInspector from './frame-inspector';
import type { State } from './types';
const empty: State = {
  captures: [],
  jobs: [],
  scenes: [],
  worker: { online: false, device: 'Connecting…' },
  mode: 'submitted captures',
};
async function request(path: string, body?: unknown) {
  const response = await fetch(
    '/api/' + path,
    body === undefined
      ? undefined
      : {
          method: 'POST',
          headers:
            body instanceof FormData
              ? {}
              : { 'Content-Type': 'application/json' },
          body: body instanceof FormData ? body : JSON.stringify(body),
        },
  );
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      detail =
        typeof payload.detail === 'string'
          ? payload.detail
          : JSON.stringify(payload.detail);
    } catch {}
    throw Error(detail);
  }
  return response.json();
}
export default function Home() {
  const [state, setState] = useState<State>(empty);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState('alignment');
  const [sceneId, setSceneId] = useState('');
  const [aligned, setAligned] = useState(true);
  const [visible, setVisible] = useState<Record<string, boolean>>({
    A: true,
    B: true,
  });
  const [pointSize, setPointSize] = useState(1.5);
  const [colorBySource, setColorBySource] = useState(false);
  const [landmarks, setLandmarks] = useState('');
  const [pickSource, setPickSource] = useState<string>();
  const pendingTarget = useRef<number[] | null>(null);
  const [pairs, setPairs] = useState<{
    source_points: number[][];
    target_points: number[][];
  }>({ source_points: [], target_points: [] });
  const [threshold, setThreshold] = useState('0.08');
  const latest = useRef('');
  const [connected, setConnected] = useState(false);
  const refresh = useCallback(async () => {
    const data = (await request('state')) as State;
    setState(data);
    setConnected(true);
    if (data.scenes[0] && latest.current !== data.scenes[0].id) {
      latest.current = data.scenes[0].id;
      setSceneId(data.scenes[0].id);
    }
    return data;
  }, []);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        await refresh();
      } catch {
        if (!cancelled) setConnected(false);
      }
      if (!cancelled) timer = setTimeout(poll, 2500);
    }
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [refresh]);
  const scene = state.scenes.find((s) => s.id === sceneId) || state.scenes[0];
  const diagnostics = scene?.diagnostics;
  const active = state.jobs.some((j) =>
    ['queued', 'running'].includes(j.status),
  );
  useEffect(() => {
    setPickSource(undefined);
    pendingTarget.current = null;
    setPairs({ source_points: [], target_points: [] });
    setLandmarks('');
  }, [scene?.id]);
  async function act(path: string, body: unknown = {}) {
    setBusy(true);
    setError('');
    try {
      const result = await request(path, body);
      await refresh();
      return result;
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed.');
      return null;
    } finally {
      setBusy(false);
    }
  }
  const a = state.captures.find((c) => c.source === 'A');
  const b = state.captures.find((c) => c.source === 'B');
  function onPick(point: number[]) {
    if (pickSource === 'A') {
      pendingTarget.current = point;
      setPickSource('B');
    } else if (pickSource === 'B' && pendingTarget.current) {
      const next = {
        source_points: [...pairs.source_points, point],
        target_points: [...pairs.target_points, pendingTarget.current],
      };
      setPairs(next);
      setLandmarks(JSON.stringify(next, null, 2));
      pendingTarget.current = null;
      setPickSource('A');
    }
  }
  async function align() {
    if (!scene) return;
    try {
      const data = JSON.parse(landmarks);
      const value = Number(threshold);
      if (!Number.isFinite(value) || value <= 0)
        throw Error('Enter a positive threshold in reconstruction units.');
      await act('register', { scene_id: scene.id, ...data, threshold: value });
      setPickSource(undefined);
      setAligned(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Invalid landmark JSON.');
    }
  }
  // Optional browser agent surface shares the read path used by the interface.
  useEffect(() => {
    const context = (
      document as unknown as {
        modelContext?: {
          registerTool: (
            tool: unknown,
            options: unknown,
          ) => Promise<void> | void;
        };
      }
    ).modelContext;
    if (!context) return;
    const lifecycle = new AbortController();
    try {
      void Promise.resolve(
        context.registerTool(
          {
            name: 'read_room_workspace',
            description:
              'Read current room capture, job, and scene metadata and refresh the visible workspace.',
            inputSchema: {
              type: 'object',
              properties: {},
              additionalProperties: false,
            },
            annotations: { readOnlyHint: true },
            execute: async (input: unknown) => {
              if (
                !input ||
                typeof input !== 'object' ||
                Array.isArray(input) ||
                Object.keys(input).length
              )
                throw Error('Expected an empty object.');
              return refresh();
            },
          },
          { signal: lifecycle.signal },
        ),
      ).catch(() => {});
    } catch {}
    return () => lifecycle.abort();
  }, [refresh]);
  return (
    <main className="shell">
      <header>
        <div className="brand">
          <Box /> simv1 <span>ROOM RECONSTRUCTION</span>
        </div>
        <span className="connection">
          {connected
            ? state.worker.online
              ? 'Worker connected'
              : 'API online · worker offline'
            : 'Connecting to local server'}
        </span>
      </header>
      <div className="titlebar">
        <div>
          <p className="eyebrow">WORKSPACE / ROOM 001</p>
          <h1>One room. Two perspectives.</h1>
        </div>
        <Button size="lg" onClick={() => setView('sources')}>
          <Upload /> Add capture
        </Button>
      </div>
      {error && (
        <div className="error" role="alert">
          {error}
          <button
            style={{ float: 'right' }}
            onClick={() => setError('')}
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}
      {!connected && (
        <output className="note">
          Waiting for the backend. Start the API on the compute laptop to upload
          captures and process scenes.
        </output>
      )}
      <Tabs value={view} onValueChange={(value) => setView(String(value))}>
        <TabsList className="navigation">
          <TabsTrigger value="sources">01 / Sources</TabsTrigger>
          <TabsTrigger value="alignment">02 / Alignment</TabsTrigger>
          <TabsTrigger value="explore">03 / Explore</TabsTrigger>
        </TabsList>
        <TabsContent value="sources">
          <div className="source-grid">
            {['A', 'B'].map((source) => (
              <Capture
                key={source}
                source={source}
                capture={source === 'A' ? a : b}
                busy={busy}
                onUpload={async (source, file) => {
                  const form = new FormData();
                  form.append('source', source);
                  form.append('file', file);
                  await act('captures', form);
                }}
                onReconstruct={(capture_id) =>
                  void act('reconstruct', { capture_id })
                }
              />
            ))}
          </div>
          <div className="sample-bar">
            <div>
              <strong>Ready to compare both reconstructions?</strong>
              <p>
                Create a scene from the latest completed A and B captures, then
                supply matching landmarks.
              </p>
            </div>
            <Button
              variant="outline"
              disabled={!a || !b || busy || active}
              onClick={() => {
                void act('pair', { capture_a: a?.id, capture_b: b?.id });
                setView('alignment');
              }}
            >
              <Link2 /> Combine A + B
            </Button>
          </div>
        </TabsContent>
        {['alignment', 'explore'].map((tab) => (
          <TabsContent key={tab} value={tab}>
            <div className="workbench">
              <section className="viewport">
                {scene ? (
                  <Viewer
                    scene={scene}
                    aligned={aligned && !!diagnostics}
                    visible={visible}
                    pointSize={pointSize}
                    colorBySource={colorBySource}
                    onPick={pickSource ? onPick : undefined}
                    pickSource={pickSource}
                  />
                ) : (
                  <div className="empty-scene">
                    <Layers3 size={40} />
                    <h2>Bring the room into view.</h2>
                    <p>
                      Add two captures, or load a labeled sample to try the
                      viewer and alignment diagnostics.
                    </p>
                    <Button
                      disabled={busy || !connected}
                      onClick={() => void act('sample')}
                    >
                      <FlaskConical /> Load sample scene
                    </Button>
                  </div>
                )}
              </section>
              <aside className="inspector">
                <p className="eyebrow">
                  {tab === 'alignment'
                    ? 'ALIGNMENT INSPECTOR'
                    : 'SCENE EXPLORER'}
                </p>
                <div className="project-info">
                  <h2>{scene ? scene.title : 'No scene yet'}</h2>
                </div>
                {scene?.sample && (
                  <span className="badge sample">
                    <FlaskConical size={12} /> Synthetic sample
                  </span>
                )}
                {scene && <p className="small">{scene.provenance}</p>}
                {state.scenes.length > 0 && (
                  <Select
                    value={scene?.id}
                    onValueChange={(v) => {
                      if (v) setSceneId(v);
                    }}
                  >
                    <SelectTrigger style={{ width: '100%', marginTop: 18 }}>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {state.scenes.map((s) => (
                        <SelectItem key={s.id} value={s.id}>
                          {s.sample ? 'Sample' : 'Capture'} ·{' '}
                          {new Date(s.created * 1000).toLocaleTimeString()}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
                <label className="toggle-row" htmlFor={`alignment-${tab}`}>
                  Apply alignment
                  <Switch
                    id={`alignment-${tab}`}
                    checked={aligned}
                    onCheckedChange={setAligned}
                    disabled={!diagnostics}
                  />
                </label>
                {['A', 'B'].map((source) => (
                  <label className="toggle-row" key={source}>
                    <span>
                      <i className={`dot ${source === 'B' ? 'b' : ''}`} />
                      Source {source}
                    </span>
                    <Switch
                      checked={visible[source]}
                      onCheckedChange={(v) =>
                        setVisible({ ...visible, [source]: v })
                      }
                    />
                  </label>
                ))}
                {tab === 'alignment' ? (
                  <>
                    <div className="metric">
                      <span>Registration</span>
                      <strong
                        className={
                          diagnostics?.status === 'accepted' ? 'ready' : ''
                        }
                      >
                        {diagnostics?.status === 'accepted' ? (
                          <>
                            <CheckCircle2
                              size={16}
                              style={{ display: 'inline', marginRight: 6 }}
                            />
                            Accepted
                          </>
                        ) : diagnostics ? (
                          'Provisional'
                        ) : (
                          'Awaiting landmarks'
                        )}
                      </strong>
                    </div>
                    <div className="metric">
                      <span>Median alignment error</span>
                      <strong>
                        {diagnostics
                          ? diagnostics.median_error.toFixed(4)
                          : '—'}{' '}
                        <small className="small">
                          {diagnostics ? 'units' : ''}
                        </small>
                      </strong>
                    </div>
                    <div className="metric">
                      <span>Held-out error / inlier ratio</span>
                      <strong>
                        {diagnostics
                          ? `${diagnostics.heldout_error.toFixed(4)} / ${(diagnostics.inlier_ratio * 100).toFixed(0)}%`
                          : '—'}
                      </strong>
                    </div>
                    {diagnostics && (
                      <p className="small">
                        {diagnostics.inliers}/{diagnostics.fit_count} fitting
                        pairs · {diagnostics.heldout_count} held out
                        <br />
                        P90: {diagnostics.p90_error.toFixed(4)} units
                        <br />
                        Relative scale B → A:{' '}
                        {diagnostics.relative_scale.toFixed(4)}
                      </p>
                    )}
                    {scene && scene.clouds.length === 2 && (
                      <details className="landmarks">
                        <summary>Landmark alignment</summary>
                        <p className="small">
                          Pick the same physical points in A then B. Use at
                          least 8 pairs across the scene. Coordinates always
                          refer to the original clouds.
                        </p>
                        <div className="toolbar">
                          <Button
                            variant="outline"
                            onClick={() => {
                              setAligned(false);
                              setPickSource('A');
                            }}
                          >
                            Pick pairs
                          </Button>
                          {pickSource && (
                            <Button
                              variant="outline"
                              onClick={() => setPickSource(undefined)}
                            >
                              Stop picking
                            </Button>
                          )}
                        </div>
                        <p className="small">
                          {pairs.source_points.length} pairs selected{' '}
                          {pickSource ? `· next: source ${pickSource}` : ''}
                        </p>
                        {scene.landmarks && (
                          <Button
                            variant="outline"
                            onClick={() =>
                              setLandmarks(
                                JSON.stringify(scene.landmarks, null, 2),
                              )
                            }
                          >
                            Load existing pairs
                          </Button>
                        )}
                        <label className="small" htmlFor={`landmarks-${tab}`}>
                          Landmark pairs (B source → A target)
                        </label>
                        <textarea
                          id={`landmarks-${tab}`}
                          value={landmarks}
                          onChange={(e) => setLandmarks(e.target.value)}
                          placeholder={
                            '{"source_points": [[x,y,z], …], "target_points": [[x,y,z], …]}'
                          }
                        />
                        <label className="small">
                          Inlier threshold (reconstruction units)
                          <input
                            type="number"
                            min="0.000001"
                            step="0.01"
                            value={threshold}
                            onChange={(e) => setThreshold(e.target.value)}
                            style={{
                              width: '100%',
                              border: '1px solid #bbc9cc',
                              padding: 8,
                              margin: '8px 0',
                            }}
                          />
                        </label>
                        <Button
                          disabled={busy || active || !landmarks}
                          onClick={() => void align()}
                        >
                          Fit and validate
                        </Button>
                      </details>
                    )}
                  </>
                ) : (
                  <>
                    <label className="toggle-row" htmlFor="color-by-source">
                      Color by source
                      <Switch
                        id="color-by-source"
                        checked={colorBySource}
                        onCheckedChange={setColorBySource}
                      />
                    </label>
                    <div className="slider-label">
                      <span>Surface coverage</span>
                      <span>{pointSize}×</span>
                    </div>
                    <Slider
                      aria-label="Surface coverage"
                      min={0.5}
                      max={3}
                      step={0.25}
                      value={[pointSize]}
                      onValueChange={(v) =>
                        setPointSize(Array.isArray(v) ? v[0] : v)
                      }
                    />
                    <div className="metric">
                      <span>Displayed points</span>
                      <strong>
                        {scene?.clouds
                          .reduce((total, c) => total + c.count, 0)
                          .toLocaleString() || '—'}
                      </strong>
                    </div>
                    {scene?.clouds.map((c) => (
                      <p key={c.source}>
                        <a className="download" href={c.ply} download>
                          Download source {c.source} PLY ↗
                        </a>
                      </p>
                    ))}
                    {scene?.clouds.map((c) => (
                      <FrameInspector key={scene.id + c.source} cloud={c} />
                    ))}
                    {scene && (
                      <a
                        className="download"
                        href={`/api/artifacts/scenes/${scene.id}/manifest.json`}
                        download
                      >
                        Download scene manifest ↗
                      </a>
                    )}
                  </>
                )}
                <div className="note">
                  {scene?.scale_source || 'Scale is uncalibrated.'}
                  <br />
                  Distances are not meters.
                </div>
              </aside>
            </div>
          </TabsContent>
        ))}
      </Tabs>
      <div className="sample-bar">
        <div>
          <strong>Test the workflow without a GPU.</strong>
          <p>
            The sample uses synthetic room surfaces and computed alignment
            diagnostics. It is not a video reconstruction.
          </p>
        </div>
        <Button
          variant="outline"
          disabled={busy || active || !connected}
          onClick={() => {
            void act('sample');
            setView('alignment');
          }}
        >
          <FlaskConical /> Load sample <ArrowUpRight size={15} />
        </Button>
      </div>
      <section className="job-list" aria-live="polite">
        <p className="eyebrow">PROCESSING ACTIVITY</p>
        {state.jobs.length ? (
          state.jobs.slice(0, 5).map((job) => (
            <div className="job" key={job.id}>
              <span>
                {job.kind} · {job.stage}
                {job.error && (
                  <>
                    <br />
                    <em>{job.error}</em>
                  </>
                )}
              </span>
              <span
                className={
                  job.status === 'failed'
                    ? 'failed'
                    : job.status === 'completed'
                      ? 'ready'
                      : 'muted'
                }
              >
                {job.status}
              </span>
            </div>
          ))
        ) : (
          <p className="small">
            No jobs yet. Submit a capture or load the sample.
          </p>
        )}
      </section>
      <footer>
        Submitted captures → independent reconstruction → validated alignment
        <span>{state.worker.device}</span>
      </footer>
    </main>
  );
}
