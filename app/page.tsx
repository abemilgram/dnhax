'use client';

import { useState } from 'react';
import ReconstructionWorkspace from './reconstruction-workspace';
import TacticalDemo from './tactical/tactical-demo';

type WorkspaceMode = 'tactical' | 'reconstruction';

export default function Home() {
  const [mode, setMode] = useState<WorkspaceMode>('tactical');
  return (
    <div className={`mode-root mode-${mode}`}>
      <nav className="mode-switch" aria-label="Workspace mode">
        <strong>DNHACKS DEFENSE</strong>
        <fieldset aria-label="Choose workspace">
          <button
            type="button"
            aria-pressed={mode === 'tactical'}
            onClick={() => setMode('tactical')}
          >
            Tactical Brain
          </button>
          <button
            type="button"
            aria-pressed={mode === 'reconstruction'}
            onClick={() => setMode('reconstruction')}
          >
            Reconstruction Lab
          </button>
        </fieldset>
      </nav>
      {mode === 'tactical' ? <TacticalDemo /> : <ReconstructionWorkspace />}
    </div>
  );
}
