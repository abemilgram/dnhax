'use client';
import { useState } from 'react';
import Image from 'next/image';
import { Slider } from '@/components/ui/slider';
import type { Cloud } from './types';
export default function FrameInspector({ cloud }: { cloud: Cloud }) {
  const [index, setIndex] = useState(0);
  const frames = cloud.cameras || [];
  const frame = frames[Math.min(index, frames.length - 1)];
  if (!frame) return null;
  return (
    <details style={{ marginTop: 22 }}>
      <summary style={{ fontSize: 14, cursor: 'pointer' }}>
        Source {cloud.source} · supporting frames
      </summary>
      <Image
        unoptimized
        width={512}
        height={342}
        src={frame.frame}
        alt={`Source ${cloud.source} captured frame at approximately ${frame.t} seconds`}
        style={{ width: '100%', borderRadius: 6, marginTop: 12 }}
      />
      <div className="slider-label">
        <span>
          Frame {index + 1} / {frames.length}
        </span>
        <span>~{frame.t}s</span>
      </div>
      <Slider
        aria-label={`Supporting frame for source ${cloud.source}`}
        min={0}
        max={Math.max(1, frames.length - 1)}
        step={1}
        value={[index]}
        onValueChange={(v) => setIndex(Array.isArray(v) ? v[0] : v)}
      />
      <p className="small">Camera parameters are preserved in the manifest.</p>
    </details>
  );
}
