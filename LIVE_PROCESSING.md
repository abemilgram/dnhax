# Live batch processing

Live mode captures independent JPEG images while one resident AMB3R runtime
processes small, overlapping windows. The viewer replaces its preview after
each successful batch. It does not accumulate a permanent whole-room map or
guarantee video-rate updates.

Live reconstruction is AMB3R-only and CUDA-only. Missing CUDA, AMB3R source,
dependencies, or checkpoint and all load/inference failures are explicit.
There is no VGGT, VGGT-Omega, MPS, CPU, or sample fallback.

## Start

Use `Dockerfile.amb3r` on an NVIDIA CUDA host and the trusted HTTPS
launcher/certificate for capture from other devices:

```sh
SIMV1_MODEL=amb3r SIMV1_DEVICE=cuda python scripts/serve.py
```

1. Click **Create live session**. Start with four frames per batch.
2. Start source A's camera or screen on one device, then source B on the other.
3. Keep the tabs active and move slowly through overlapping views. Images
   upload during capture; live mode does not archive video.
4. The viewer follows completed previews. Turn off **Follow live** to hold the
   current version, or open **Scene history** to inspect another version.
5. **Stop source** releases that device's media and flushes a bounded queue.
   The session owner can finish, cancel, retry after an error, or save the
   latest snapshot.

The owner token is kept only in the creating browser's local storage. Source
takeover must be explicit. This remains a trusted-LAN application, not an
authenticated public service.

## Limits and behavior

- One live session and one CUDA worker at a time; an OS lock prevents duplicate workers.
- One sampled JPEG/second/source, maximum 960px long edge, four pending plus one in-flight upload.
- Four frames/batch by default; eight is an explicit higher-memory choice.
- The scheduler snapshots candidates at dispatch and alternates live and submitted work.
- AMB3R weights remain resident across compatible batches.
- An out-of-memory batch may retry once with fewer frames on the same AMB3R/CUDA runtime. It never changes model or device.
- Failed continuity opens a labeled new segment instead of claiming fusion.
- At most 200,000 preview points/source.
- A source becomes stale after 10 seconds without receipt and is omitted from fresh open-session batches.
- Sessions stop after 10 minutes and retain bounded inputs and previews.
- Ingestion reports explicit disk backpressure; it does not delete unrelated files.
- A 900-second watchdog terminates stuck live work. Override `SIMV1_LIVE_JOB_TIMEOUT` only after profiling.
- Missing weights or inference errors preserve the last good preview and record the failed attempt.

## Profiling

Run against two submitted captures on the AMB3R CUDA host. Output is isolated
from workspace scenes. The first run is cold; the second reuses AMB3R.

```sh
SIMV1_MODEL=amb3r SIMV1_DEVICE=cuda python scripts/profile_live.py \
  --capture-a /absolute/path/to/a.mov \
  --capture-b /absolute/path/to/b.mov \
  --output /absolute/path/to/profile-results \
  --budgets 4
```

Timing separates model load from inference. A successful hardware run validates
execution, not geometric accuracy or a specific update cadence.

## Validation

```sh
python -m pytest -q
npm run test:frontend
npm run typecheck
npm run lint
npm run build
```

Tests exercise ingestion, ownership, bounded scheduling, selection, draining,
cancellation, stale sources, recovery, retention, disk backpressure, and
camera/geometry continuity. Browser permissions, cross-device behavior, real
CUDA inference, and sustained camera sessions still require target hardware.
