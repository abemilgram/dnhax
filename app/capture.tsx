'use client';
/* eslint-disable jsx-a11y/media-has-caption -- Room geometry previews are silent; audio is not used. */
import { useEffect, useRef, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Upload, Monitor, Camera, Square, Trash2 } from 'lucide-react';
import type { State } from './types';
export default function Capture({
  source,
  capture,
  busy,
  onUpload,
  onReconstruct,
  onDelete,
}: {
  source: string;
  capture?: State['captures'][number];
  busy: boolean;
  onUpload: (source: string, file: File) => Promise<void>;
  onReconstruct: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [recording, setRecording] = useState(false);
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState('');
  const recorder = useRef<MediaRecorder | null>(null);
  const stream = useRef<MediaStream | null>(null);
  const preview = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    if (!file) {
      setUrl('');
      return;
    }
    const value = URL.createObjectURL(file);
    setUrl(value);
    return () => URL.revokeObjectURL(value);
  }, [file]);
  useEffect(
    () => () => {
      if (recorder.current) {
        recorder.current.onstop = null;
        if (recorder.current.state !== 'inactive') recorder.current.stop();
      }
      stream.current?.getTracks().forEach((t) => t.stop());
    },
    [],
  );
  async function record(mode: 'screen' | 'camera') {
    setError('');
    setStarting(true);
    try {
      if (!window.isSecureContext)
        throw Error(
          'Recording requires localhost or trusted HTTPS. You can upload an existing video over LAN HTTP.',
        );
      if (mode === 'screen' && !navigator.mediaDevices?.getDisplayMedia)
        throw Error(
          'Screen recording is unavailable in this browser. Open this app in Chrome or Edge, or upload an existing recording.',
        );
      if (mode === 'camera' && !navigator.mediaDevices?.getUserMedia)
        throw Error('Camera recording is unavailable in this browser.');
      if (typeof MediaRecorder === 'undefined')
        throw Error(
          'This browser cannot record clips. Upload an existing recording instead.',
        );
      const media =
        mode === 'screen'
          ? await navigator.mediaDevices.getDisplayMedia({
              video: true,
              audio: false,
            })
          : await navigator.mediaDevices.getUserMedia({
              video: true,
              audio: false,
            });
      stream.current = media;
      const chunks: BlobPart[] = [];
      const mime = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find(
        (t) => MediaRecorder.isTypeSupported(t),
      );
      const instance = new MediaRecorder(media, mime ? { mimeType: mime } : {});
      recorder.current = instance;
      // The browser's Stop sharing control must finalize the clip too.
      media.getVideoTracks().forEach((track) => {
        track.addEventListener(
          'ended',
          () => {
            if (instance.state !== 'inactive') instance.stop();
          },
          { once: true },
        );
      });
      instance.ondataavailable = (event) => {
        if (event.data.size) chunks.push(event.data);
      };
      instance.onstop = () => {
        media.getTracks().forEach((t) => t.stop());
        setRecording(false);
        if (preview.current) preview.current.srcObject = null;
        const type = instance.mimeType;
        setFile(
          new File(
            chunks,
            `room-${source}-${mode}-${Date.now()}.${type.includes('mp4') ? 'mp4' : 'webm'}`,
            { type },
          ),
        );
      };
      setRecording(true);
      instance.start(1000);
      if (preview.current) {
        preview.current.srcObject = media;
        void preview.current.play();
      }
    } catch (e) {
      stream.current?.getTracks().forEach((t) => t.stop());
      setRecording(false);
      setError(e instanceof Error ? e.message : 'Recording could not start.');
    } finally {
      setStarting(false);
    }
  }
  return (
    <section className="source-card">
      <header>
        <h2>
          <i className={`dot ${source === 'B' ? 'b' : ''}`} />
          Source {source}
        </h2>
        <span className="badge">
          {capture ? 'Capture stored' : 'No capture'}
        </span>
      </header>
      <p>A short, steady room walkthrough with overlap between sources.</p>
      <video
        ref={preview}
        src={recording ? undefined : url || capture?.url}
        controls={!recording}
        muted
        playsInline
        style={{ display: recording || url || capture ? 'block' : 'none' }}
      />
      <div className="record-controls">
        <Button
          variant="outline"
          disabled={busy || recording || starting}
          onClick={() => void record('screen')}
        >
          <Monitor /> Record screen
        </Button>
        <Button
          variant="outline"
          disabled={busy || recording || starting}
          onClick={() => void record('camera')}
        >
          <Camera /> Record camera
        </Button>
        {recording && (
          <Button
            variant="destructive"
            onClick={() => recorder.current?.stop()}
          >
            <Square /> Stop recording
          </Button>
        )}
      </div>
      <label className="upload">
        <Upload size={22} />
        <span>Choose a video file</span>
        <input
          aria-label={`Video for source ${source}`}
          type="file"
          accept="video/mp4,video/quicktime,video/webm,video/x-matroska"
          disabled={busy || recording}
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
      </label>
      {file && (
        <p>
          {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
        </p>
      )}
      <div className="toolbar">
        <Button
          disabled={!file || busy || recording}
          onClick={async () => {
            if (file) {
              await onUpload(source, file);
            }
          }}
        >
          <Upload /> Submit capture
        </Button>
        <Button
          variant="outline"
          disabled={!capture || busy}
          onClick={() => capture && onReconstruct(capture.id)}
        >
          Reconstruct capture
        </Button>
        {capture && (
          <Button
            variant="destructive"
            disabled={busy}
            onClick={() => onDelete(capture.id)}
          >
            <Trash2 /> Delete stored capture
          </Button>
        )}
      </div>
      <p className="video-note">
        Record your camera, or choose a screen, window, or tab. Recording
        continues until you stop it or stop sharing. Nothing is uploaded until
        you submit. Maximum upload: 512 MB.
      </p>
      {capture && <p className="small">Stored: {capture.name}</p>}
      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}
    </section>
  );
}
