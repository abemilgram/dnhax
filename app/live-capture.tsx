'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import type { Scene } from './types';

type LiveState = {
  id: string;
  status: string;
  processing: string;
  error?: string;
  latest_batch: number;
  config: { frame_budget: number; model: string };
  scene?: Scene;
  sources: {
    source: string;
    status: string;
    stale: boolean;
    last_received?: number;
    accepted: number[];
    skipped: number;
  }[];
  batches: { number: number; status: string; error?: string }[];
};
type Frame = { seq: number; blob: Blob; time: number };
type Producer = {
  token: string;
  epoch: string;
  media: MediaStream;
  queue: Frame[];
  seq: number;
  sent: number;
  skipped: number;
  pumping: boolean;
  stopping: boolean;
  timer?: ReturnType<typeof setTimeout>;
  beat?: ReturnType<typeof setInterval>;
  stopped?: Promise<void>;
  deadline?: number;
  abort?: AbortController;
};
class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}
async function api<T = Record<string, unknown>>(
  path: string,
  body?: unknown,
  token?: string,
): Promise<T> {
  const response = await fetch('/api/live/' + path, {
    method: body === undefined ? 'GET' : 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { 'X-Live-Token': token } : {}),
    },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(10000),
  });
  const data = (await response.json()) as { detail?: string };
  if (!response.ok)
    throw new ApiError(
      typeof data.detail === 'string'
        ? data.detail
        : `Request failed (${response.status})`,
      response.status,
    );
  return data as T;
}

function LiveSource({
  session,
  source,
}: {
  session: LiveState;
  source: string;
}) {
  const video = useRef<HTMLVideoElement>(null);
  const run = useRef<Producer | null>(null);
  const alive = useRef(true);
  const [active, setActive] = useState(false);
  const [starting, setStarting] = useState(false);
  const [takeover, setTakeover] = useState(false);
  const [error, setError] = useState('');
  const [counts, setCounts] = useState({ sent: 0, skipped: 0, queued: 0 });
  const base = `sessions/${session.id}/sources/${source}`;
  const report = (p: Producer) => {
    if (alive.current)
      setCounts({ sent: p.sent, skipped: p.skipped, queued: p.queue.length });
  };
  async function pump(p: Producer) {
    if (p.pumping) return;
    p.pumping = true;
    try {
      while (p.queue.length && alive.current) {
        if (p.deadline && Date.now() > p.deadline) break;
        const frame = p.queue[0];
        try {
          const abort = new AbortController();
          p.abort = abort;
          const timeout = setTimeout(() => abort.abort(), 5000);
          let response: Response;
          try {
            response = await fetch(
              `/api/live/${base}/frames/${p.epoch}/${frame.seq}`,
              {
                method: 'PUT',
                headers: {
                  'Content-Type': 'image/jpeg',
                  'X-Live-Token': p.token,
                  'X-Capture-Time': String(frame.time),
                },
                body: frame.blob,
                signal: abort.signal,
              },
            );
          } finally {
            clearTimeout(timeout);
          }
          if (response.ok) {
            p.queue.shift();
            p.sent++;
            if (alive.current) setError('');
          } else if (response.status === 410) {
            p.queue.shift();
            p.skipped++;
          } else if (response.status === 409) {
            if (alive.current)
              setError('Source ownership changed or session stopped.');
            void stop();
            break;
          } else {
            const data = (await response.json()) as { detail?: string };
            if (response.status !== 429 && response.status < 500) {
              p.queue.shift();
              p.skipped++;
              if (alive.current) setError(data.detail || 'Frame rejected.');
            } else throw Error(data.detail || 'Upload delayed; retrying.');
          }
        } catch (e) {
          if (alive.current)
            setError(
              e instanceof Error ? e.message : 'Upload delayed; retrying.',
            );
          await new Promise((resolve) => setTimeout(resolve, 1000));
        }
        report(p);
      }
    } finally {
      p.pumping = false;
    }
  }
  function stop(): Promise<void> {
    const p = run.current;
    if (!p) return Promise.resolve();
    if (p.stopped) return p.stopped;
    p.stopping = true;
    p.deadline = Date.now() + 4500;
    clearTimeout(p.timer);
    clearInterval(p.beat);
    p.media.getTracks().forEach((track) => track.stop());
    p.stopped = (async () => {
      void pump(p);
      while (p.pumping && Date.now() < p.deadline!)
        await new Promise((resolve) => setTimeout(resolve, 100));
      p.abort?.abort();
      p.skipped += p.queue.length;
      p.queue = [];
      try {
        await api(
          base + '/stop',
          { final_seq: p.seq - 1, skipped: p.skipped },
          p.token,
        );
      } catch (e) {
        if (alive.current)
          setError(
            e instanceof Error ? e.message : 'Could not acknowledge stop.',
          );
      }
      if (run.current === p) run.current = null;
      if (alive.current) {
        setActive(false);
        report(p);
      }
    })();
    return p.stopped;
  }
  const stopRef = useRef(stop);
  stopRef.current = stop;
  useEffect(() => {
    alive.current = true;
    return () => {
      alive.current = false;
      void stopRef.current();
    };
  }, []);
  useEffect(() => {
    if (session.status !== 'open') void stopRef.current();
  }, [session.status]);
  async function start(mode: 'camera' | 'screen') {
    setStarting(true);
    setError('');
    let media: MediaStream | undefined;
    try {
      if (!window.isSecureContext || !navigator.mediaDevices)
        throw Error('Live capture requires localhost or trusted HTTPS.');
      media =
        mode === 'camera'
          ? await navigator.mediaDevices.getUserMedia({
              video: true,
              audio: false,
            })
          : await navigator.mediaDevices.getDisplayMedia({
              video: true,
              audio: false,
            });
      if (!alive.current) {
        media.getTracks().forEach((t) => t.stop());
        return;
      }
      const claim = await api<{ token: string; epoch: string }>(
        base + '/claim',
        { takeover },
      );
      const p: Producer = {
        ...claim,
        media,
        queue: [],
        seq: 0,
        sent: 0,
        skipped: 0,
        pumping: false,
        stopping: false,
      };
      run.current = p;
      if (!video.current) throw Error('Preview is unavailable.');
      video.current.srcObject = media;
      await video.current.play();
      setActive(true);
      report(p);
      media.getTracks().forEach((t) =>
        t.addEventListener(
          'ended',
          () => {
            void stopRef.current();
          },
          { once: true },
        ),
      );
      const canvas = document.createElement('canvas');
      async function sample() {
        if (p.stopping || !alive.current) return;
        const preview = video.current;
        if (preview && preview.readyState >= 2 && preview.videoWidth > 0) {
          const scale = Math.min(
            1,
            960 / Math.max(preview.videoWidth, preview.videoHeight),
          );
          canvas.width = Math.round(preview.videoWidth * scale);
          canvas.height = Math.round(preview.videoHeight * scale);
          canvas
            .getContext('2d')!
            .drawImage(preview, 0, 0, canvas.width, canvas.height);
          const time = preview.currentTime;
          const blob = await new Promise<Blob | null>((resolve) =>
            canvas.toBlob(resolve, 'image/jpeg', 0.8),
          );
          if (blob && !p.stopping) {
            // Index zero belongs to an in-flight request and is never replaced.
            if (p.queue.length >= 5) {
              p.queue.splice(1, 1);
              p.skipped++;
            }
            p.queue.push({ seq: p.seq++, blob, time });
            report(p);
            void pump(p);
          }
        }
        if (!p.stopping)
          p.timer = setTimeout(() => {
            void sample();
          }, 1000);
      }
      p.beat = setInterval(() => {
        void api(base + '/heartbeat', {}, p.token)
          .then((data) => {
            if (data.status !== 'open') void stopRef.current();
          })
          .catch((e) => {
            if (alive.current) setError(e.message);
            if (e instanceof ApiError && e.status === 409)
              void stopRef.current();
          });
      }, 5000);
      void sample();
    } catch (e) {
      media?.getTracks().forEach((t) => t.stop());
      if (run.current) void stopRef.current();
      if (alive.current)
        setError(e instanceof Error ? e.message : 'Could not start capture.');
    } finally {
      if (alive.current) setStarting(false);
    }
  }
  const remote = session.sources.find((s) => s.source === source);
  return (
    <section className="source-card">
      <header>
        <h2>
          <i className={`dot ${source === 'B' ? 'b' : ''}`} /> Source {source}
        </h2>
        <span className="badge">
          {active
            ? 'Sending live images'
            : remote?.stale
              ? 'Waiting / stale'
              : remote?.status || 'Ready'}
        </span>
      </header>
      <video
        ref={video}
        muted
        playsInline
        aria-label={`Live source ${source} preview`}
        style={{ display: active ? 'block' : 'none' }}
      />
      <div className="toolbar">
        <Button
          disabled={active || starting || session.status !== 'open'}
          onClick={() => void start('camera')}
        >
          Start camera
        </Button>
        <Button
          variant="outline"
          disabled={active || starting || session.status !== 'open'}
          onClick={() => void start('screen')}
        >
          Start screen
        </Button>
        {active && (
          <Button variant="destructive" onClick={() => void stop()}>
            Stop source
          </Button>
        )}
      </div>
      {!active && (
        <label className="toggle-row">
          Take over this source{' '}
          <Switch
            checked={takeover}
            onCheckedChange={setTakeover}
            aria-label={`Take over source ${source}`}
          />
        </label>
      )}
      <p className="small">
        {counts.sent} sent · {counts.queued} pending · {counts.skipped} skipped
      </p>
      {remote?.last_received && (
        <p className="small">
          Last server receipt:{' '}
          {new Date(remote.last_received * 1000).toLocaleTimeString()}
          {remote.stale ? ' · stale' : ''}
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  );
}

export default function LiveCapture({
  follow,
  onFollow,
  onScene,
}: {
  follow: boolean;
  onFollow: (value: boolean) => void;
  onScene: (scene: Scene) => void;
}) {
  const [session, setSession] = useState<LiveState | null>(null);
  const [identity, setIdentity] = useState('');
  const [token, setToken] = useState('');
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [budget, setBudget] = useState(4);
  const [history, setHistory] = useState<Scene[]>([]);
  const callbacks = useRef({ follow, onScene });
  callbacks.current = { follow, onScene };
  const delivered = useRef('');
  const refresh = useCallback(async () => {
    let id = identity;
    if (!id) {
      const sessions = await api<LiveState[]>('sessions');
      id =
        sessions.find((s: LiveState) => ['open', 'stopping'].includes(s.status))
          ?.id || sessions[0]?.id;
      if (!id) return;
      setIdentity(id);
      setToken(localStorage.getItem('simv1-live-owner-' + id) || '');
    }
    const data: LiveState = await api<LiveState>(`sessions/${id}`);
    setSession(data);
    if (
      data.scene &&
      callbacks.current.follow &&
      delivered.current !== data.scene.id
    ) {
      delivered.current = data.scene.id;
      callbacks.current.onScene(data.scene);
    }
  }, [identity]);
  useEffect(() => {
    let ended = false;
    let timer: ReturnType<typeof setTimeout>;
    async function poll() {
      try {
        await refresh();
      } catch (e) {
        if (!ended)
          setError(
            e instanceof Error ? e.message : 'Live connection unavailable.',
          );
      }
      if (!ended)
        timer = setTimeout(() => {
          void poll();
        }, 1000);
    }
    void poll();
    return () => {
      ended = true;
      clearTimeout(timer);
    };
  }, [refresh]);
  useEffect(() => {
    if (follow && session?.scene) {
      delivered.current = session.scene.id;
      onScene(session.scene);
    }
  }, [follow, onScene, session?.scene?.id]);
  async function action(name: string) {
    setBusy(true);
    setError('');
    try {
      if (name === 'create') {
        const result = await api<{ id: string; token: string }>('sessions', {
          request_key: crypto.randomUUID(),
          frame_budget: budget,
        });
        localStorage.setItem('simv1-live-owner-' + result.id, result.token);
        setToken(result.token);
        setIdentity(result.id);
        setHistory([]);
        onFollow(true);
      } else {
        await api(`sessions/${identity}/${name}`, {}, token);
        await refresh();
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Live operation failed.');
    } finally {
      setBusy(false);
    }
  }
  return (
    <section className="live-panel">
      <div className="sample-bar">
        <div>
          <strong>Live batch reconstruction</strong>
          <p>
            Images upload while capture runs. Each completed batch replaces the
            preview; intermediate frames may be skipped.
          </p>
        </div>
        <label className="toggle-row">
          Follow live{' '}
          <Switch
            checked={follow}
            onCheckedChange={onFollow}
            aria-label="Follow live scene updates"
          />
        </label>
      </div>
      <div className="toolbar">
        {(!session || ['completed', 'cancelled'].includes(session.status)) && (
          <>
            <label className="small">
              Frames per batch{' '}
              <select
                value={budget}
                onChange={(e) => setBudget(Number(e.target.value))}
              >
                <option value={4}>4 · conservative</option>
                <option value={8}>8 · more memory</option>
              </select>
            </label>
            <Button disabled={busy} onClick={() => void action('create')}>
              Create live session
            </Button>
          </>
        )}
        {session && (
          <span className="badge">
            {session.status} · {session.processing} · batch{' '}
            {session.latest_batch} · {session.config.model}
          </span>
        )}
        {session && token && ['open', 'stopping'].includes(session.status) && (
          <>
            <Button
              variant="outline"
              disabled={busy || session.status === 'stopping'}
              onClick={() => void action('stop')}
            >
              Stop and finish batch
            </Button>
            <Button
              variant="outline"
              disabled={busy}
              onClick={() => void action('cancel')}
            >
              Cancel processing
            </Button>
            {session.processing === 'paused_error' && (
              <Button disabled={busy} onClick={() => void action('resume')}>
                Retry processing
              </Button>
            )}
          </>
        )}
        {session?.scene && token && (
          <Button
            variant="outline"
            disabled={busy}
            onClick={() => void action(`scenes/${session.scene!.id}/pin`)}
          >
            Save latest snapshot
          </Button>
        )}
        {session && (
          <Button
            variant="outline"
            onClick={async () => {
              try {
                const data = await api<{ scenes: Scene[] }>(
                  `sessions/${identity}/scenes`,
                );
                setHistory(data.scenes);
              } catch (e) {
                setError(String(e));
              }
            }}
          >
            Scene history
          </Button>
        )}
      </div>
      {history.length > 0 && (
        <div className="toolbar">
          {history.map((scene) => (
            <Button
              key={scene.id}
              variant="outline"
              onClick={() => {
                onFollow(false);
                onScene(scene);
              }}
            >
              {scene.title}
            </Button>
          ))}
        </div>
      )}
      {session && (
        <div className="source-grid">
          {['A', 'B'].map((source) => (
            <LiveSource
              key={session.id + source}
              session={session}
              source={source}
            />
          ))}
        </div>
      )}
      {session?.scene?.live && (
        <p className="small">
          Last preview:{' '}
          {new Date(session.scene.created * 1000).toLocaleTimeString()} ·{' '}
          {session.scene.live.elapsed_seconds.toFixed(1)}s processing ·{' '}
          {session.scene.live.continuity.reason}
        </p>
      )}
      {session?.error && (
        <p className="error" role="alert">
          {session.error}
        </p>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      <p className="small">
        Use the same page on your second device and start the other source.
        Trusted HTTPS is required on LAN devices. Sessions stop after 10
        minutes. Keep capture tabs active.
      </p>
    </section>
  );
}
