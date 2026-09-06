'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import { applyCue } from './state.mjs';
import type { TacticalCue, TacticalSnapshot } from './types';

type ConnectionState = 'connecting' | 'live' | 'reconnecting' | 'offline';
type TacticalControl = 'start' | 'pause' | 'restart' | 'seek';

async function readSnapshot(
  path: string,
  init?: RequestInit,
): Promise<TacticalSnapshot> {
  const headers = new Headers(init?.headers);
  if (init?.body) headers.set('Content-Type', 'application/json');
  const response = await fetch(path, {
    ...init,
    headers,
  });
  if (!response.ok) {
    let detail = `Tactical API request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      if (typeof payload.detail === 'string') detail = payload.detail;
    } catch {}
    throw new Error(detail);
  }
  return (await response.json()) as TacticalSnapshot;
}

export function useTacticalStream() {
  const [snapshot, setSnapshot] = useState<TacticalSnapshot | null>(null);
  const [connection, setConnection] =
    useState<ConnectionState>('connecting');
  const [error, setError] = useState('');
  const [controlPending, setControlPending] = useState(false);
  const revision = useRef(0);
  const mounted = useRef(true);

  const publish = useCallback((incoming: TacticalSnapshot) => {
    if (!mounted.current) return;
    setSnapshot((current) => {
      if (incoming.revision <= revision.current) return current;
      revision.current = incoming.revision;
      return incoming;
    });
  }, []);

  const publishCue = useCallback((cue: TacticalCue) => {
    if (!mounted.current) return;
    setSnapshot((current) => {
      if (cue.revision <= revision.current) return current;
      const next = applyCue(current, cue);
      revision.current = cue.revision;
      return next;
    });
  }, []);

  useEffect(() => {
    mounted.current = true;
    const abort = new AbortController();
    let source: EventSource | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | undefined;
    let delay = 500;
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      setConnection(revision.current ? 'reconnecting' : 'connecting');
      source = new EventSource(
        `/api/tactical/events?since=${encodeURIComponent(revision.current)}`,
      );
      source.onopen = () => {
        delay = 500;
        setConnection('live');
        setError('');
      };
      source.addEventListener('snapshot', (event) => {
        try {
          publish(JSON.parse((event as MessageEvent<string>).data));
        } catch {
          setError('Received an invalid tactical snapshot.');
        }
      });
      source.addEventListener('cue', (event) => {
        try {
          publishCue(JSON.parse((event as MessageEvent<string>).data));
        } catch {
          setError('Received an invalid tactical cue.');
        }
      });
      source.onerror = () => {
        source?.close();
        source = null;
        if (stopped) return;
        setConnection('reconnecting');
        reconnectTimer = setTimeout(connect, delay);
        delay = Math.min(delay * 2, 8000);
      };
    };

    void readSnapshot('/api/tactical/state', { signal: abort.signal })
      .then((initial) => {
        publish(initial);
        setError('');
      })
      .catch((reason: unknown) => {
        if (!abort.signal.aborted) {
          setConnection('offline');
          setError(
            reason instanceof Error
              ? reason.message
              : 'Could not reach the tactical API.',
          );
        }
      })
      .finally(connect);

    return () => {
      stopped = true;
      mounted.current = false;
      abort.abort();
      source?.close();
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, [publish, publishCue]);

  const control = useCallback(
    async (operation: TacticalControl, position?: number) => {
      setControlPending(true);
      setError('');
      try {
        const incoming = await readSnapshot(`/api/tactical/${operation}`, {
          method: 'POST',
          body:
            operation === 'seek'
              ? JSON.stringify({ position })
              : JSON.stringify({}),
        });
        publish(incoming);
        return incoming;
      } catch (reason) {
        setError(
          reason instanceof Error ? reason.message : 'Playback control failed.',
        );
        return null;
      } finally {
        if (mounted.current) setControlPending(false);
      }
    },
    [publish],
  );

  return {
    snapshot,
    connection,
    error,
    controlPending,
    start: () => control('start'),
    pause: () => control('pause'),
    restart: () => control('restart'),
    seek: (position: number) => control('seek', position),
  };
}
