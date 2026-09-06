# Live batch processing build spec

Status: historical implementation specification. The shipped runtime is now
AMB3R-only and CUDA-only; model/device alternatives described by older
validation records are not selectable.
Target: `/Users/jeffdai/Downloads/simv1`.

## Outcome and scope

Two capture browsers continuously send sampled room images to the existing compute laptop. A single warm GPU worker reconstructs bounded, overlapping image batches. The viewer automatically displays completed scene versions while capture continues. Stop ends capture and processes one final eligible batch.

“Live” means continuous ingestion with periodic reconstruction updates. It does not establish a video-rate reconstruction guarantee. The first release shows the current reconstruction window; persistent whole-room accumulation is a separate milestone because it needs stronger drift correction and fusion.

Keep the existing React/Three.js frontend, FastAPI service, SQLite metadata, local artifact storage, and AMB3R CUDA adapter. Preserve submitted-video processing, sample scenes, exports, and manual landmark correction. One active live session, A/B sources, one GPU worker, and a mostly static room are the initial operating envelope. The API and worker remain on the compute host, using the existing trusted LAN HTTPS deployment.

## What the repository already provides

| Existing component | Current behavior | Required change |
| --- | --- | --- |
| `app/capture.tsx` | Camera/screen recording accumulates chunks until stop, uploads only on Submit | Separate live mode that samples images and uploads during capture |
| `backend/api.py` | Complete video upload; explicit reconstruct/joint requests | Session, source ownership, frame ingest, live state and stop APIs |
| `backend/store.py` | Durable FIFO queue, atomic claiming, immutable scene records | Versioned migrations, bounded scheduling, ownership leases, durable batch/publication links |
| `backend/reconstruct.py` | `infer_images()` loads and releases a model on every call | Worker-owned model runtime reused across jobs |
| `backend/joint.py` | Selects full-clip frames, runs A/B in one sequence | Reusable image-based selection and joint inference without FFmpeg |
| `backend/worker.py` | One worker; abandoned running jobs become failed | Live scheduling, controlled recovery, stop/drain and publication fencing |
| `app/page.tsx` | Polls full state every 2.5 seconds | Compact live state plus explicit Follow live / inspect history behavior |
| `app/viewer.tsx` | Recreates renderer and resets camera on `scene.id` change | Persistent renderer with atomic geometry replacement |

`VALIDATION.md` contains pre-cutover measurements from removed runtime backends.
They are historical evidence only and do not establish current AMB3R
performance. This spec therefore begins with profiling and treats timing
settings below as provisional.

## User flow

1. The compute browser creates a live session and shows A/B connection status, pinned AMB3R/CUDA configuration and worker readiness.
2. Each capture browser joins A or B and clicks Start camera or Start screen. This click obtains browser media permission; joining a session never starts recording automatically.
3. Each source shows its local preview, sampled/sent/skipped counts, connection state and last acknowledged frame. Live mode clearly says that images upload while capture is active.
4. Once enough varied frames arrive, the worker publishes a first preview. Additional frames continue arriving during inference.
5. The viewer shows batch number, source ages, processing stage, last completed update and geometry continuity status. Users can orbit, hide sources, change point size or pause Follow live.
6. Stop source releases that device's media tracks and flushes its bounded upload queue. Stop session asks all connected producers to stop, closes ingestion after a short drain window, then processes one final eligible snapshot. The final state records missing producers, frame gaps and failed/ineligible tails.
7. Save snapshot pins an immutable scene and its required artifacts. Existing PLY and manifest downloads remain available.

A source can start alone. A second source joins when it has enough useful frames and visible overlap. A disconnected source is marked stale; its old images must not make the scene appear freshly captured by both devices.

## Data flow

```text
A / B media streams
  -> sampled JPEG images + source sequence numbers
  -> bounded upload queues and idempotent HTTP ingestion
  -> durable per-source frame rings
  -> worker snapshots a bounded overlapping image sequence
  -> resident AMB3R inference on CUDA
  -> continuity check + bounded preview export
  -> immutable scene publication + atomic latest-scene pointer
  -> viewer preloads buffers and swaps the completed scene
```

Use HTTP multipart uploads and short polling for the first release. They fit the existing deployment and allow explicit retry/backpressure. WebSockets, WebRTC transport and server-sent events are unnecessary for the initial cadence. Add SSE later only if polling measurements justify it.

## Capture and ingest contract

Sample the granted media stream through a video element and canvas into independent JPEGs. Start at one frame/second/source, JPEG quality 0.8, maximum long edge 960 pixels; the model adapter retains responsibility for its required preprocessing. These are configurable starting values, not accuracy claims.

Do not submit each existing `MediaRecorder.start(1000)` blob as a separate video. The recording standard says individual timeslice blobs need not be playable; only the combined completed recording must be playable. Independent JPEGs remove that container dependency. See [W3C MediaStream Recording](https://www.w3.org/TR/mediastream-recording/).

Each frame carries session ID, source ID, producer epoch, increasing sequence number, media-relative capture time, dimensions and content digest. The server adds receipt time. Ordering uses epoch/sequence, not cross-device wall clocks. Reconnects that preserve a media stream keep the epoch; a new stream receives a new epoch. Store capture and receipt timestamps separately. Show server receipt age as such; do not label it true capture latency without clock-offset estimation.

Use one in-flight upload and at most four pending frames per producer. While offline or throttled, discard the oldest unsent non-anchor candidates and retain recent images. This is intentionally sampled preview processing, not archival recording. Expose skipped counts. Retry a dispatched frame with the identical identity and bytes until acknowledged, expired by the session policy, or explicitly abandoned.

Initial server limits: 2 MiB encoded/frame, 2 megapixels decoded/frame, two producers, 60 candidate frames/source, and a 10-minute session duration. The server validates content type, actual decode, dimensions and digest; enforce body limits while reading, before buffering an oversized request. Acknowledgment means the file is durable and its DB record committed. Write a temporary file and rename it before metadata insertion; clean unreferenced files after crashes.

Same identity and digest returns the original acknowledgment; conflicting content returns 409. Out-of-order frames may be accepted, but cannot move the current input watermark backward. Maintain accepted ranges/gaps rather than treating the highest sequence as proof that all lower sequences arrived. Return a suggested sample interval and `Retry-After` on 429. Never hold a SQLite transaction during upload or image decoding.

One producer owns each source through a renewable lease and opaque token. Use 5-second heartbeats and a 20-second expiry initially. Explicit takeover rotates the token/epoch and fences the previous producer. Tokens protect accidental producer collisions; they are not a replacement for authentication if public hosting is introduced.

## Batch selection and scheduling

Only one GPU job runs globally. Maintain at most one running live batch and one pending refresh marker for the active session. New images update the candidate ring; they do not each enqueue a GPU job. On claim, resolve the newest eligible candidates and persist an immutable batch input list. Never modify the inputs of a running batch.

Start with a benchmarked budget of four total frames on constrained devices and eight when measured memory and latency permit. For two-source eight-frame windows, select four frames/source: up to two retained overlap frames plus two fresh candidates. For four-frame windows, use one retained plus one fresh frame/source. Bootstrap uses fresh frames with temporal diversity. At least two distinct usable images are required for any inference; two-source mode requires at least two/source. These settings are separate from existing single-clip and joint-clip limits.

Reuse sharpness, temporal spread and reciprocal SIFT appearance cues from `joint.py`; compute descriptors once per candidate. Reject duplicate/near-static and unusably blurred frames using thresholds calibrated in the profiling milestone. If no usable new content exists, show Waiting for useful frames rather than repeatedly reconstructing the same input. Selection, input order and reason codes are persisted.

Retain a small overlap set from the most recent accepted batch; keeping the first image identical helps but does not guarantee identical scale or geometry across inference calls. For A/B, choose shared-looking views when available. Appearance matches alone do not certify alignment. Insufficient cross-source overlap produces an explicit warning and separate-source preview option, never a claimed validated joint map.

Trigger when the worker is free, useful new frames exist and the minimum dispatch interval has elapsed. A timeout may dispatch a smaller valid batch; it must not duplicate frames to meet a budget. After completion, snapshot the newest candidates and supersede intermediate work. A stalled B source never blocks A indefinitely: default stale threshold is 10 seconds since receipt, show B's age, and omit stale B from fresh geometry unless explicitly shown as historical.

Live work and existing submitted jobs share the worker. A running job is not preempted. Alternate eligible live and submitted jobs when both queues are nonempty so neither starves. Report the extra delay from submitted work. Require an OS-level worker lock as well as a heartbeat; the existing heartbeat check alone is not an atomic singleton guarantee.

## Resident model runtime and failure behavior

Introduce a worker-owned `ModelRuntime` with load, infer and unload methods. Cache exactly one AMB3R model keyed by model variant, checkpoint digest, CUDA device and precision. Release per-batch inputs/predictions after export; retain weights across batches. Persist the actual preprocessing policy and precision in every batch manifest. Configuration, dependency, checkpoint, load, and inference errors fail explicitly without another model or device.

Pin model/checkpoint settings for a live session. Configuration changes start a new session. Existing submitted jobs may cause a cache change at job boundaries, with the load delay reported. All inference stays inside the worker; API handlers never load the model.

On OOM: release batch tensors, keep the last good preview, mark the attempt failed, unload/reload if necessary, and retry at most once at the next lower valid frame budget. Record both attempts and the reduced inputs. If the minimum budget also fails, pause processing and show the failure. Never silently switch to CPU or fabricate sample geometry. Configure a measured per-job watchdog so a hung GPU call cannot leave the session “running” indefinitely; process termination is required for a native call that cannot be cooperatively cancelled.

Worker restarts leave the prior scene visible. Recover interrupted batches as retryable only when inputs still exist, retry limits allow it and the session is eligible; otherwise record failure and move to fresh work. Stop/cancel checks happen before inference and publication. In-flight inference may finish, but a cancelled session's result cannot advance its latest-scene pointer.

## Geometry across batches

All A/B frames in one window continue to enter one sequence, as the current joint adapter does. Never concatenate independently predicted windows with identity transforms and call that a stable map. The current AMB3R adapter offers per-call joint predictions, not persistent mapping state.

For continuity, use identical retained frame IDs and matching preprocessed pixel coordinates to obtain 3D correspondences between successive predictions. Fit a positive-scale similarity transform from the new window into the previous accepted scene frame. Use confidence/depth-edge filtering, spatially distributed sampling, robust fitting and a held-out subset. Evaluate each shared frame, spatial support, scale change, inlier ratio and normalized held-out residual. Keep continuity diagnostics distinct from the existing manual A/B registration diagnostics.

Provisional gates for calibration: at least two shared frames, 100 valid spatially distributed correspondences, at least 60% fitting inliers, median held-out residual below 2% of robust scene extent and scale ratio between 0.8 and 1.25. Passing these is an internal continuity heuristic, not measured physical accuracy. A single-source minimum two-frame window may lack enough overlap to pass; it must still work as a clearly labeled new preview segment. Tune gates against representative static-room sequences before enabling automatic continuity by default.

On acceptance, apply one transform to the whole new window and its cameras. On failure, preserve the last accepted view, mark continuity lost, and offer the fresh window as a separate segment. Follow live can enter that segment with an explicit camera reset and warning. Do not silently append it to historical geometry. Scene scale remains arbitrary.

V1 replaces the visible window rather than appending unlimited points. Whole-room accumulation later requires keyframe/submap retention, deduplication, drift/loop correction, moving-object rejection and bounded fusion. Merely retaining every PLY is not that feature.

## Persistence, publication and retention

Add a versioned migration runner; do not rely solely on extending `CREATE TABLE IF NOT EXISTS` in each connection. Enable foreign-key enforcement on every connection. Add:

| Entity | Essential fields and constraints |
| --- | --- |
| `live_sessions` | ID, status, pinned config, creation/end time, latest scene/batch, generation fence, stop deadline |
| `live_sources` | Session/source unique key, producer epoch, token hash, lease expiry, status, accepted ranges, last receipt |
| `live_frames` | ID, session/source/epoch/sequence unique key, digest/path, timestamps, dimensions, quality metadata |
| `live_batches` | Session/monotonic batch number unique key, status, input digest, model/config digest, attempts, timestamps, output scene, error |
| `live_batch_frames` | Batch/frame/order, source, selection role; immutable after claim |
| Job additions | Session/batch/attempt reference, retry count, cancellation state, output scene ID |
| Scene additions | Session/batch/segment, input provenance, continuity transform/diagnostics, preview/export policy |

Session lifecycle: `open -> stopping -> completed`; processing can be `warming`, `waiting`, `running`, `paused_error` or `draining` within that lifecycle. Cancellation is terminal. A completed session may have a failed final batch, recorded explicitly rather than hidden by “completed.” A disconnected producer is not the same as a failed worker.

Write all scene artifacts under an attempt-specific staging location and finalize them before publication. In one SQLite transaction insert the immutable scene, link its successful batch/job, and conditionally advance the session pointer only if its generation is current and batch number is newer. A unique successful output per batch makes crash retries idempotent. Consumers only see published artifact records. This closes the existing gap between scene publication and job completion.

Keep each source's newest 60 candidates plus explicitly referenced anchor/batch/scene frames. Keep the newest five unpinned live previews and the final scene by default; pinning retains a scene and all referenced assets. Initial per-session storage quota: 2 GiB, including intermediates, with a server-wide minimum free-space reserve of 5 GiB. At limits, garbage-collect unreferenced data, then pause ingestion visibly if still full. Stop at 10 minutes by default rather than growing storage indefinitely. Never garbage-collect existing submitted captures/scenes as part of live retention. Artifact serving must enforce publication/allowed roots and traversal protection for new paths.

## API contract

| Endpoint | Behavior |
| --- | --- |
| `POST /api/live/sessions` | Create one active session; return ID, pinned config, initial limits and control token; idempotency key required |
| `POST /api/live/sessions/{id}/sources/{A\|B}/claim` | Claim or explicitly take over producer ownership; return epoch and token |
| `POST /api/live/sessions/{id}/sources/{source}/heartbeat` | Renew lease; return source/session state and server time |
| `PUT /api/live/sessions/{id}/sources/{source}/frames/{epoch}/{seq}` | Idempotently ingest JPEG plus metadata; return accepted identity and pacing hints |
| `GET /api/live/sessions/{id}` | Compact status, source ages/gaps, current job, latest scene reference, revision and stop state; support ETag/304 |
| `POST /api/live/sessions/{id}/sources/{source}/stop` | Idempotently close producer with final sequence and known skipped ranges |
| `POST /api/live/sessions/{id}/stop` | Idempotently enter stopping; default 5-second ingress drain deadline; coalesce tail into at most one final eligible batch |
| `POST /api/live/sessions/{id}/cancel` | End ingestion and cancel pending work; fence publication from in-flight work |
| `GET /api/live/sessions/{id}/scenes?cursor=...` | Paginated history so inspection is not limited to `/api/state`'s ten newest scenes |
| `POST /api/live/sessions/{id}/scenes/{scene_id}/pin` | Idempotently retain a published scene and its referenced artifacts |

Return 404 for unknown resources, 409 for ownership/state/content conflicts, 413 for limits, 415/422 for bad image/metadata and 429 for temporary backpressure. Mutations require the appropriate source or session control token. Keep existing APIs compatible; add only a lightweight live-session summary to `/api/state`.

## Viewer and performance contract

Mount renderer, orbit controls and interaction handlers once. Load a new scene into a separate bounded buffer set, validate counts/finite geometry, then swap on an animation frame. Keep the previous scene visible during transfer and on failure. Abort obsolete downloads and fence callbacks by requested scene ID. Dispose replaced GPU buffers after a successful swap. Persist orbit, source visibility, RGB/source colors and point-size controls across updates within a stable segment; reset only on first load, explicit Fit view or a segment change.

Start with 200,000 preview points/source and deterministic confidence-aware/spatial sampling. At six float/byte components, positions plus RGB cost about 3 MB/source (12 position bytes + 3 color bytes per point), or about 6 MB for two sources before overhead. Budget for both current and incoming geometry plus any float color conversion. Keep richer geometry artifacts only within the retention quota; mark preview PLY exports with their actual density. Existing one-million-point source exports remain available for submitted jobs.

Poll compact live status every second while following, with backoff when disconnected. Polling history does not switch the user's selected scene. Stop existing scene-change effects from resetting A/B toggles on every live version. Manual landmark picking pins/pauses the inspected version so points cannot change mid-selection.

Measure separately: cold load, preprocessing, selection, queue wait, inference, continuity, export, publication, download and buffer swap. Record per-source newest-input age, batch frame budget, preview bytes, dropped frames, process memory and device memory where supported. Synchronize GPU timing when benchmarking. Do not infer latency from frame count or advertise a five-second cadence from the historical 55-second run.

Provisional release targets: healthy-LAN ingest acknowledgment p95 under 1 second at configured limits; publication-to-display p95 under 2 seconds for the bounded preview on the test LAN; no more than one pending refresh; no unbounded age/backlog or memory growth in a 10-minute run. Reconstruction cadence is a measured hardware/model profile. Compute scene freshness from the selected frames and queue/inference/export/display timings; report stale content explicitly. With a slow device, skip intermediate candidate windows instead of promising to process every sampled frame.

## Build sequence and acceptance gates

| Milestone | Deliverable | Required evidence |
| --- | --- | --- |
| 1. Profile and resident runtime | `backend/model_runtime.py`, adapter refactor, repeatable profiling script | Cold versus warm runs at 4/8 total frames on the actual target model/device; repeated inference keeps correct provenance and releases per-batch memory; select default budget from results |
| 2. Durable live capture | `app/live-capture.tsx`, `backend/live_api.py`, migrations, frame storage | Two producers upload independently; duplicates, conflicting retries, out-of-order frames, takeover, disconnects and quotas behave as specified |
| 3. Bounded scheduler | `backend/live.py`, worker/store changes | Artificially slow inference with fast producers stays at one running batch and one marker; selects newest eligible inputs; submitted jobs get turns; stop produces at most one final batch |
| 4. Joint previews and continuity | Reuse `joint.py`, add `backend/live_geometry.py` | Known synthetic similarity recovery plus static-room replay; disjoint/degenerate/poor-overlap windows do not silently fuse; OOM retry is bounded |
| 5. Continuous viewer | Refactor `viewer.tsx`; live state/types/history | Full buffers swap atomically, camera stays stable in a segment, history remains selected, A/B controls persist, stale downloads cannot overwrite a newer view |
| 6. Recovery and hardware validation | Cleanup/recovery, documentation and operational limits | Crash injection around claim/artifact write/DB publication; restart without duplicate scenes; stop/cancel races; 10-minute A/B session with a disconnect and device stop |

Use fake inference for deterministic queue/failure tests and labeled synthetic geometry for transform tests. Real target-hardware runs are required for latency, memory, model compatibility and geometry quality; mocks cannot establish those. Validate browser camera and screen permission/end-sharing behavior on the intended browsers, including mobile camera if used. Observe background-tab throttling and expose capture pauses rather than claiming uninterrupted browser capture.

Regression checks during implementation: backend pytest suite, Node scene-selection tests, TypeScript check and production build. Extend tests for live behavior; retain all submitted-clip compatibility checks. No runtime tests are claimed by this specification-only change.

The first shippable version completes all six milestones for bounded live previews. A seamless, continually growing room map is a subsequent build with its own geometry and accuracy acceptance criteria.
