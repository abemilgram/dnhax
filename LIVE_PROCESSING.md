# Live batch processing

The live mode captures independent JPEG images while a single resident model processes small, overlapping windows. The viewer replaces its preview after each successful batch. It does not accumulate a permanent whole-room map or guarantee video-rate updates.

## Start

Use the existing virtual environment and launcher. The model selector now also discovers `~/Downloads/vggt_omega_1b_512.pt` when no project-local Omega checkpoint exists. An explicit `VGGT_OMEGA_CHECKPOINT` still takes priority.

```sh
cd ~/Downloads/simv1
.venv/bin/python -m pip install -r requirements.txt
npm run build
SIMV1_MODEL=vggt_omega SIMV1_DEVICE=mps .venv/bin/python scripts/serve.py
```

Use your existing trusted HTTPS launcher/certificate for capture from other devices; camera and screen access require HTTPS or localhost. Both browsers open the same server URL.

1. Click **Create live session**. Start with four frames per batch.
2. Start source A's camera or screen on one device, then source B on the other.
3. Keep the tabs active and move slowly through overlapping views. Images upload during capture. No video is recorded or archived by live mode.
4. The Alignment/Explore viewer follows completed previews. Turn off **Follow live** to hold the current version, or open **Scene history** to inspect another version.
5. **Stop source** releases that device's camera and flushes a bounded queue. The browser that created the session can **Stop and finish batch**, **Cancel processing**, **Retry processing** after an error, or **Save latest snapshot**.

The owner token is kept only in that browser's local storage. Source takeover must be explicit; a new owner fences the previous device. This remains a trusted-LAN application, not an authenticated public service.

## Limits and behavior

- One live session and one GPU worker at a time; an OS lock prevents duplicate workers.
- One sampled JPEG/second/source, maximum 960px long edge, four pending plus one in-flight upload. Intermediate unsent images may be skipped.
- Four frames/batch by default; eight is an explicit higher-memory choice. No silent CPU fallback.
- The scheduler snapshots current candidates when dispatching work. It alternates live and submitted jobs if both need the GPU.
- Reuses weights between model-compatible batches. CUDA/MPS dispatch and pinned model implementations are retained.
- Repeated image geometry provides a heuristic similarity fit between windows. Failed continuity opens a labeled new segment and resets the view. This does not certify A/B alignment, absolute scale or physical accuracy.
- At most 200,000 preview points/source. Rich geometry and camera artifacts retain provenance; PLY downloads contain the preview density shown.
- A source becomes stale after 10 seconds without receipt. Stale sources are omitted from fresh open-session batches; final draining may include the last stopped inputs.
- Sessions stop after 10 minutes. Retains 60 recent input candidates/source plus active overlap frames, five latest previews and pinned scenes. Submitted captures/scenes are excluded from live cleanup. Manual corrections referencing live geometry keep those buffers alive.
- Ingestion returns explicit backpressure below 5 GiB free disk or above 2 GiB session storage. Free storage or stop the session; the app does not delete unrelated files.
- A 900-second live job watchdog terminates a stuck worker. Override with `SIMV1_LIVE_JOB_TIMEOUT` only after profiling. Restart the launcher and resume after a watchdog error.
- Interrupted work is recorded as failed and replaced with fresh eligible inputs after restart. Cancelled in-flight work cannot publish a new latest scene. Missing weights or inference errors preserve the last good preview.

## Profiling

Run against two submitted captures. Output is isolated from workspace scenes. The first run is cold; the second reuses the model. Start with four frames; request `--budgets 4 8` only when memory allows it.

```sh
SIMV1_MODEL=vggt_omega SIMV1_DEVICE=mps .venv/bin/python scripts/profile_live.py \
  --capture-a /absolute/path/to/a.mov --capture-b /absolute/path/to/b.mov \
  --output /absolute/path/to/profile-results --budgets 4
```

Timing includes model load separately from preprocessing/inference/CPU copies. It excludes live upload, selection and viewer rendering. CPU copies synchronize inference timing. A successful hardware run validates execution, not geometric accuracy.

## Validation

```sh
.venv/bin/python -m pytest -q
node --test tests/scene-selection.test.mjs
npm run typecheck
npm run build
```

Tests exercise duplicate/conflicting uploads, takeover, bounded queue scheduling, newest-frame selection, stop/drain, cancellation while inference is executing, stale sources, restart recovery, pinned retention, low-disk backpressure and camera/geometry continuity. Browser permission flows, cross-device behavior and a sustained camera session still require the intended browsers and devices.

## Measured Omega run on this Mac

Four-frame Omega / MPS benchmark on two existing room captures: cold call 379.01 seconds (267.15 seconds loading and 101.57 seconds preprocessing/inference/CPU copies), then a warm call of 109.98 seconds with zero model-load time. Both calls used the same resident model; allocated model memory after each call was 4.58 GB and driver memory was 8.88 GB. Checkpoint SHA-256 matched `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.

These measurements establish that resident Omega executes here. Under those conditions, expect approximately two-minute reconstruction updates after warm-up, plus upload/export/viewer time. Eight-frame performance and a sustained two-device browser session have not been validated. Storage/file hydration slowed the cold run substantially; no faster cadence is promised.
