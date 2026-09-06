# Prediction engine

This branch adds a stateful prediction service alongside the existing `macoslive`
reconstruction service. The recorded-pixel detector pipeline now runs in an
isolated environment on the existing cloud GPU. See [PERCEPTION.md](PERCEPTION.md)
for measured detection results, commands, and the remaining geometry gates.

The product contract is: video detections → geometric observations → uncertain
tracks → bounded route hypotheses → change-only cues. The intended host is a
controlled CS2 scene. Enemy game transforms, the scripted bot route, future video
frames, and demo entity data are not accepted as inference inputs.

## Existing project context

- Original implementation baseline: `macoslive`, commit
  `4a6fa51d8f30bd2c779dfc2213560098429c76e3`. Before publication this branch was
  rebased onto `e4a4625`, retaining the newer AMB3R reconstruction changes and
  tactical interface.
- The existing tactical interface uses its separate `/api/tactical` service.
  This package exposes `/api/prediction` and the recorded-image runner; both
  routers are registered. The tactical UI has not been connected to this
  package's branching forecasts. API coexistence is not an end-to-end UI handoff.
- Existing live reconstruction exports point clouds, camera intrinsics and poses,
  capture timestamps, and a continuity segment. Its scene units are arbitrary.
  A point cloud is not automatically a walkable navigation mesh.
- The current scenario plan is **Dust II B tunnels**. Exit A is the Upper Tunnels
  mouth into B; exit B is the Lower Tunnels route toward Mid. The player begins on
  B and later rotates through B Doors / CT Mid. Those are scenario intentions,
  not measurements or model observations.
- The annotated radar is an approximate planning drawing in radar pixels. It
  provides no validated metric scale, traversable graph, floor heights, or camera
  calibration. Upper and Lower Tunnels must not collapse onto one flat surface.
- Earlier Mirage recordings are technical capture tests. They are not the
  Dust II golden take and do not validate this predictor.
- The sibling bot task has since reported five verified physical Dust II bot
  traversals and a 21.77-second 720p30 bot-POV control-proof clip. It reports that
  static navigation geometry is available on Windows, but no metric alignment,
  calibrated aerial visibility or synchronized paired footage has been validated.
- The existing Runpod report confirms an RTX PRO 6000 Blackwell running VGGT-1B:
  warm two-frame outer calls measured 210ms median / 217ms p95 (10 repeats,
  294x518 model tensors). A separate recorded-feed transport test decoded and
  acknowledged 100/100 frames per feed at 10fps, with approximately 52–55ms median
  replay-release-to-ACK latency. These are component benchmarks, not a connected
  detector/belief/cue pipeline or live camera-to-cue measurements. Evidence:
  `/Users/abe/Documents/New project/runpod-realtime/results/REPORT.md`.
  A separate pixel detector workspace now runs on that GPU; the existing transport
  endpoint still returns ACKs, not integrated inference responses.

## What runs

`backend/prediction/` contains strict contracts, pixel-to-floor projection,
anonymous Kalman tracking, navigation-graph rollouts, cue hysteresis, a live-scene
camera adapter, a synthetic replay and a FastAPI router.

Projection requires visible ground contact, undistorted calibrated pinhole pixels and an explicitly metric
scene with XY on the ground and Z up. Bbox bottom-centre rays intersect candidate
walkable surfaces. Invalid or ambiguous geometry is rejected rather than silently
converting image coordinates into meters. The pose convention follows the
[OpenCV pinhole model](https://docs.opencv.org/5.0/main_modules/calib.html).

The tracker maintains position, velocity, uncertainty and observation age. It
coasts during occlusion. The Gaussian estimate itself is not a navigation-constrained
distribution; the response includes the last observed point for a ghost marker
and separate geometry-constrained future paths. Do not draw the Gaussian mean as
a known enemy position. Same-time multi-camera fusion is conservative about shared
projection errors; see the discussion of unknown correlation in
[this distributed filtering paper](https://arxiv.org/abs/2105.15061).
Cross-floor reacquisition requires a traversable graph connection and enough
elapsed travel time under the configured speed assumptions. Simultaneous
detections on different floors remain separate hypotheses. Identity across a
long occlusion is still uncertain; appearance-based re-identification is absent.

The default `route_bank` model samples a starting position and speed, follows a shortest
path on the supplied graph, and reports conditional arrival-time quantiles. HOLD
remains available. This is a coarse graph prototype, not a learned world dynamics
model or a full triangle-navmesh planner. Slower movement is sampled, but explicit
return/flank alternatives must be authored as additional route intents. It does
not predict a moving player's future actions.

Hypotheses remain anchored to the last observation and expire at their configured
horizon (at most 15 seconds). Ticks update the display without inventing fresh
evidence. A new matched observation refreshes the horizon. Travel intervals can
be negative once a route would already have arrived; do not clamp these into a
new countdown.

## Branching model

The optional `branching` model adds a finite, weighted ensemble of graph
trajectories. Each proposal samples position uncertainty and speed, chooses
branches, and can slow, pause, or reverse. Segments follow the validated graph
after a checked connector from the observed point; exact turns are retained so
interpolation cannot cut a corner. This is an authored stochastic movement model,
not a trained dynamics network. Its behavior parameters are assumptions to test
against a separate take.

During occlusion, multiple positions survive in `tracks[].map_belief`. Spatial
bins retain representative actual particle positions, with omitted and unsupported
mass exposed. Future snapshots at +3, +6, and +10 seconds are supplied only when
they fit within the remaining forecast horizon. Arrival summaries distinguish
already arrived, expected later within the horizon, and no arrival within that
horizon. Quantiles are conditional on arrivals; missing arrivals never become a
zero-second countdown. Model mass is not an empirically calibrated probability.

A new projected observation scores previous futures and draws a new ensemble
around that measurement, preserving relative intent support. This is measurement
reanchoring, not persistent full-state particle filtering or learned intent
recognition. Same-time views do not count as independent motion evidence. An
observation after forecast expiry resets behavior support to its disclosed priors.
Only qualified visible misses reweight individual trajectories during occlusion.
The record includes evidence that last changed the weights. Ticks never resample
or add evidence.

Same-time position proposals use the tracker's measurement-level covariance
intersection, not its unconstrained coasting mean. The first processed view at a
new capture timestamp supplies the motion-support update; subsequent same-time
views refine the position proposals. Merge equal-time frames in deterministic
`sensor_id`, then `frame_id` order (the recorded runner does this). The motion
posterior is not claimed to be invariant to arbitrary same-time arrival order.
`map_belief.exit_forecasts` aggregates actual simulated exit visits across the
behavior mixture; individual intent support is not an exit-arrival probability.

For this model, legacy `xyz` is a **last-observed ghost marker**; spatial belief
is in `map_belief`. The Gaussian used for anonymous association is exposed
separately and may cross walls. Do not render it as a known location. Cross-view
identity, occlusion duration, detector reliability and metric alignment still
need real-data validation. Counterfactual camera planning is not implemented.

Prediction and geometry run in Python/NumPy on CPU. The cloud GPU runs the
detector (and the separate reconstruction worker). These graph simulations do
not launch game instances. CPU and GPU stages can reside on the same cloud host;
having them installed together does not establish connected live video-to-cue
latency.

## Integrity rules

1. `relative_support` is **not a calibrated exit probability**. `rank_score` is a
   heuristic that combines support and geometric exposure to a fresh player pose.
   Equal exit priors mean equal assumed support, not an empirically established
   50% chance. Display conditional time ranges, not a falsely precise countdown.
2. Empty frames do not automatically prove absence. Negative evidence requires
   complete detector coverage, a disclosed assumed detection probability and
   visibility of the hypothesized location. Updates are limited to once per
   second per source to reduce repeated-frame overconfidence. This cooldown is a
   heuristic; detector errors still require empirical calibration.
3. Only empty frames currently contribute absence evidence. Frames containing
   people may have ambiguous associations, so this version does not infer other
   missing people from them. Person detections have no automatic enemy/team label.
4. Each session fixes its evidence mode, clock and coordinate frame. Synthetic
   fixtures cannot mix with video evidence. Strict HTTP schemas reject entity
   positions and undeclared fields. The server cannot independently prove that a
   caller actually used a video detector: retain original frames and detector logs
   for audit.
5. Camera calibration must be available by the time its evidence is processed.
   A reconstructed batch cannot become an earlier camera pose merely because
   some images in that batch were captured earlier. Rejected continuity requires
   a new metric alignment and a new predictor session.
6. A geometry model needs walls and ceiling/floor occluders as well as walkable
   surfaces. Verify visibility around stairs and tunnel roofs before presenting
   real-game results.
7. Scripted capture, synthetic replay and live inference are different evaluation
   modes. Label them. A staged player action does not prove a person reacted to
   a live model cue.

## Run locally

Use the project's Python environment with its existing requirements:

```sh
python -m backend.prediction.replay --output work/prediction-replay.json
python -m backend.prediction.replay --prediction-model branching --output work/prediction-branching.json
python -m pytest tests/test_prediction*.py -q
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000 --workers 1
```

The replay fabricates pixel boxes in an invented two-exit room. Its report labels
that fact and measures only the local engine call. It measures no cloud, detector,
video decoder or network performance. Keep the report as a regression fixture,
not as a product demonstration.

To select the same model for actual recorded images, set
`"config": {"prediction_model": "branching", "particles": 32}` in the take manifest, retaining
any other existing configuration fields, or use the command-line override:

```sh
python -m backend.prediction.perception --take path/to/take.json --weights path/to/yolo11m-pose.pt --device 0 --prediction-model branching --output work/take-branching
```

The API accepts `config.prediction_model` when creating a session. Omission keeps
the existing `route_bank` behavior. Both entry points preserve calibration,
source-clock, synchronization and ground-contact checks. Selecting a richer
model cannot make an uncalibrated take ready. The Windows capture script needs
no changes for this selection. Compare models using identical input pixels,
geometry and parameters, and retain separate result directories.

### Cloud verification

The isolated release is `/workspace/cs2-perception/service-branching-v1` on the
existing cloud host, using `/workspace/cs2-perception/.venv/bin/python`. The
separate `/workspace/cs2-perception/service` directory remains available for the
coordinator's footage evaluation. Run the commands above from the release
directory. These are installed code paths, not a claim that live streams are
connected to the prediction API.

A synthetic benchmark on the host's AMD EPYC 9355 CPU ran five repeats of the
eight-frame test replay after a warmup. At 32 particles per intent, three intents,
one tracked person and a ten-second horizon (96 proposed trajectories), ingests
with new projected observations measured **65.88 ms median / 80.28 ms p95**
(10 samples). Empty-frame ingests measured **8.45 / 10.15 ms** (30 samples).
This measures projection, belief, simulation and response preparation; it excludes
video detection, network transport and JSON transmission. It is not a real CS2
forecast accuracy result or a many-track throughput guarantee. The global
configuration default remains 64 particles; the timing above uses the explicit
32-particle preset.

Evidence: `work/branching/cloud-benchmark.json`, `source-manifest.json`, and
`synthetic-replay.json`. The GPU detector remains the existing YOLO11 worker;
this release adds no trained dynamics model or GPU simulation dependency.

Validation after rebasing onto the current project: **226 backend tests and
13 frontend state tests pass** in the project environment.
New checks cover graph branches and reversals, obstacle-safe event segments,
censored arrivals, conserved occupancy mass, forecast expiry, simultaneous-view
position uncertainty, duplicate/tick invariance, API selection and the recorded
runner's calibration gates. Two existing dependency deprecation warnings remain.

The API is documented at `/docs` when the backend runs. A single persistent worker
owns in-memory sessions. Restarts lose sessions; multiple independent workers
cannot share a session. Do not send these requests to the existing Runpod job
handler: it handles reconstruction jobs, not this stateful loop.

1. `POST /api/prediction/sessions` with `prior`, `clock_id`, `evidence_source` and
   optional `config`. Save the returned `id` and `token`.
2. `POST /api/prediction/sessions/{id}/frames` with `{ "packet": ..., "now": ... }`.
   Include `X-Prediction-Token` on all session requests.
3. Poll `GET /api/prediction/sessions/{id}` or submit
   `POST /api/prediction/sessions/{id}/tick` with `{ "now": ... }` to advance the
   source-clock display between frames. Deduplicate cues using `event_cursor` and
   each event's `id`. The most recent 256 events are retained.
4. `DELETE /api/prediction/sessions/{id}` ends a session. Unused sessions expire
   after 30 minutes; at most eight sessions and eight sensors per session exist.
   Frame IDs remain protected against reuse for the session lifetime. A session
   accepts at most 20,000 distinct frames; start a new session for another take.

Bind locally during development. For cluster operation, put normal deployment
authentication and TLS in front of the service; the returned session token guards
an individual session and is not cluster-wide access control.

## Detector handoff

Each frame packet contains:

```json
{
  "frame_id": "drone-000041",
  "clock_id": "capture-take-01",
  "t": 4.1,
  "available_t": 4.18,
  "clock_uncertainty_s": 0.01,
  "evidence_source": "video_detector",
  "detector": "actual-model-name-and-weights-digest",
  "detections": [{"box": [100, 80, 150, 220], "confidence": 0.86, "label": "person", "ground_contact_visible": true}],
  "coverage_complete": false,
  "camera": "replace with a validated Camera object; see /docs"
}
```

The example documents fields; the string in `camera` deliberately is not a valid
calibration. Camera objects require full K, camera-to-world matrix, matching image
dimensions, shared frame ID, calibration provenance, uncertainty and availability.
Convert crop/letterbox detector boxes back to the calibrated image coordinates.
The ground-contact flag defaults to false: an ordinary person box alone does not
establish that its bottom edge is the person's feet. Establish visible feet using
keypoints/segmentation or reviewed annotations before setting this flag. Boxes
with unknown ground contact are rejected, including partial bodies behind cover.
Game camera/player pose may be a disclosed input; enemy pose may not.

Use one agreed time origin. `t` is capture time, `available_t` is when detector
and calibration outputs were ready, and request `now` is processing time in that
same clock. `now` must be monotonic; default allowed capture lag is one second.
Merge both sensors by capture time before ingestion. Late packets behind the
capture watermark are rejected in this version; there is no silent smoothing
with future evidence or implicit clock conversion. Presentation ticks do not
prevent ordinary delayed frames from being accepted.

## Next real-data gate

1. Record the authored Dust II take with both views and a visible synchronization
   event. Keep bot scripts and entity telemetry in an evaluation-only directory.
2. Define the coarse metric site prior: floor polygons, floor heights, walls,
   slabs, graph connections, alternative goals and disclosed priors. Measure
   scale/alignment using held-out landmarks. Do not populate it from radar pixels
   and claim surveyed distances.
3. Calibrate both image streams to that shared frame, or align accepted
   `macoslive` geometry segments using the explicit adapter. Preserve calibration
   availability and preprocessing dimensions.
4. Run the actual detector over the recording, save its boxes and source timing,
   and feed only those packets to the engine. Overlay projected observations on
   the video and map to inspect misses and alignment errors.
5. Measure observed → stale → route reranking → reacquisition, no repeated cue
   for a repeated frame, conditional arrival error, and false cues on held-out
   takes. Tune on one take and report separately on another.
6. Run the same service and detector beside the cloud GPU, then measure real
   end-to-end p50/p95 latency. GPU detector/reconstruction capacity and the small
   geometry simulation can scale independently.
