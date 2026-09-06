# Recorded pixels to prediction: accuracy work

The recorded-image path is implemented: decoded RGB pixels → resident person
detector → explicit ground-contact policy → calibrated floor projection →
Kalman belief → route hypotheses/cues. A mock calibrated image exercises the
whole software path. Actual Dust II pixels have reached the GPU detector.
Actual Dust II metric tracking and predictions remain gated by unvalidated
geometry. Passing software tests is not an end-to-end accuracy result.

The same runner accepts `--prediction-model branching` (or the take's
`config.prediction_model`) to select stochastic graph trajectories, multimodal
occupancy and censored arrival summaries. The default remains `route_bank`.
This selection does not bypass the real-footage geometry and synchronization
gates; existing uncalibrated recordings still produce no metric predictions.
See [PREDICTION.md](PREDICTION.md) for the model contract and CPU/GPU split.

## What is measured

### Current input: the newer world-facing drone render

The current technical input is `drone_world_facing.mp4`: 1920x1080, 30 fps,
840 frames, 28 seconds, SHA-256
`3d0f37fa5535a20ef881e1035503d2ddcb24c5051e661603e1de117f0bc4222d`.
It is a newer render of the same source demo, not a new human encounter.
All 840 frames reached the cloud detector. The previous sky-away exclusion
was not reused because this camera path keeps world landmarks visible.

With the same YOLO11m-pose model at input size 1280 and confidence 0.25,
warm detector processing measured **7.59 ms median / 9.78 ms p95**. The offline
frame-read/detector/diagnostic loop measured **25.34 / 37.35 ms**. These exclude
transport and prediction; the first frame included 1.95 seconds of cold setup.

A blind 19-frame development review matched five of six clear-person boxes
at IoU >=0.5, left one partial figure unscored, and found three false boxes in
twelve empty frames. The last actual-person detection was frame 108 (3.6 s),
while the person was still visibly present at frame 120 (4.0 s). A separate
inspection of all 46 transit candidate boxes across 45 frames identified them
as static scenery or vegetation false alarms. They remain in the diagnostic
overlay. Equal model settings do not isolate the cause of the improvement:
the two renders differ in resolution, path and rendering details.

No complete watched-exit coverage was certified. Negative evidence stays
disabled, ground-contact policy stays `reject_unknown`, and all frames produced
zero metric projections and predictor updates pending camera/map validation.
The detailed local report is
`../outputs/cloud-drone-newest-20260906/REPORT.md`; its source, frame outputs,
labels and overlays are local evaluation artifacts and are not in this Git
commit. The earlier measurements below are historical development comparisons.

### Full-frame-rate audit of the corrected technical take

The coordinator subsequently tested all 840 frames of the same corrected
28-second clip, using a snapshot of the preserved cloud detector service.
Warm detector processing measured 5.30 ms median / 5.92 ms p95; the offline
JPEG/detector/diagnostic loop measured 15.28 / 51.09 ms. These exclude transport
and prediction. The first frame included 1.46 seconds of cold setup.

The last actual-person detection occurred at frame 90 (3.0 s), although the
person remained clearly visible at frames 120 and 128 (4.0 and 4.27 s).
Treat this as a detector miss, not verified entry into cover. A blind 12-frame
review found six clear-person instances, four empty frames and two ambiguous
frames excluded from scoring: four of six clear-person frames had boxes, three
met IoU >=0.5, one localized the real person at IoU 0.426, and two were complete
misses. The earlier 16-frame development comparison again found six of nine
matches and no false positives. These are development-take results.

Three building-detail false alarms were inside the 330-frame excluded transit
interval. No frame produced a metric projection or prediction: the shared map
and camera calibration remain unvalidated. Negative-evidence updates remain
disabled in the recorded runner. Preserve that guard while improving doorway
detection; an empty detector output must not trigger an exit inference.

Report and artifacts: `../outputs/cloud-drone-fullrate-20260906/REPORT.md`.
The source is the existing corrected technical take, not a new human run.

### Earlier sampled-frame comparison

The preferred corrected clip was tested on 112 sampled frames. Sixteen frames
were separately annotated, with 9 visible person instances and 7 empty frames.
A second independent enlarged-image review tightened the small-person boxes
and clarified that the foot label is the midpoint between boot contact points.
The original labels and initial results remain preserved; use the `*-qa.json`
artifacts for the reviewed comparison. Both label sets remain approximate.

| Model/run | Hits | Misses | False positives | Processing median / p95 |
|---|---:|---:|---:|---:|
| YOLO11m full frame | 6 | 3 | 0 | 14.90 / 22.30 ms |
| YOLO11m with fixed crops | 7 | 2 | 13 | 37.90 / 43.59 ms |
| YOLO11x full frame | 6 | 3 | 0 | 19.41 / 22.12 ms |

The crop experiment adds architectural false alarms and remains disabled by
default. The larger model does not improve recall on these labels. Keep the
medium full-frame baseline for the next paired human take; no detector is yet
validated for the final demo. These same-scenario frames are development data
after model selection. Counts above cover 16 labeled frames; timings cover all
112 processed frames and exclude transport/prediction.

Reviewed evidence: `work/perception/corrected-drone/comparison-qa.json` and each
run's `evaluation-qa.json`. Medium-model bbox contact-proxy error is 3.45 px
median across 6 matched visible-contact labels. World-coordinate error remains
unmeasured. The [official YOLO11 model documentation](https://docs.ultralytics.com/models/yolo11/)
lists the two pretrained pose variants used here; no general-dataset benchmark
is substituted for our CS2 measurements.

### Earlier development sample

The original provisional drone video was sampled at 2 fps: 56 frames, with the
camera-away/artifact interval 6–17 seconds excluded from observation evidence.
Sixteen original frames were independently reviewed before inspecting detector
output, including entry/doorway contact, two empty frames, and the later hover.
Image hashes, source IDs, and timestamps were checked before evaluation.

With YOLO11m-pose at 1280 pixels and confidence 0.25, this development sample
contains 14 visible person instances: **8 hits, 6 misses, 0 false positives**
at IoU ≥ 0.5 (57.1% recall). This does not establish general precision or recall.
The matched visible-foot subset has 7 labels, with 2.58 px median / 5.81 px p90
error for the bbox-bottom-center proxy. That proxy is not a metric position.

The opt-in pose-foot policy also produced one false visible-contact candidate
among 8 matched people (5 true positives, 2 false negatives, 1 false positive).
High ankle confidence therefore does not establish unobscured ground contact.
The default policy rejects unknown contact; the opt-in policy stays provisional.

GPU decode/detection/diagnostic-write wall time for that offline run was
15.94 ms median / 18.67 ms p95 across 56 frames. The first call includes cold
setup. These figures exclude video transport and do not measure live cue latency.

Evidence under `work/perception/` (ignored generated artifacts):

- `entry-preview/`: original image, independent single-image label, GPU result.
- `drone-proof/gpu-result/`: all 56 detector rows and diagnostic overlays.
- `drone-proof/evaluation.json`: independent 16-frame detection comparison.
- `drone-proof/ground-policy-evaluation.json`: provisional contact-policy check.
- `corrected-drone/`: preferred corrected footage sampling and separate evaluation.

The first 16 frames are development data after inspecting these failures. Further
frames from the corrected render are within the same recorded scenario, not
an independent-take generalization benchmark. Capture a fresh take for that claim.

## Reproduce inference

The cloud workspace is `/workspace/cs2-perception`; its separate virtual
environment reuses the installed GPU Torch runtime without replacing the existing
reconstruction or transport service. The model remains resident through a take.

Installed model: official `yolo11m-pose.pt`, Ultralytics 8.4.142, Torch 2.8.0+cu128.
Weights SHA-256:
`29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2`.
The adapter requires an existing local checkpoint and records its fingerprint.
The optional larger trial uses `yolo11x-pose.pt`, SHA-256
`013c43543b0751b8918486ba96e01ee44a59040683a07f96cc22bcc2cb7785f8`.

```sh
python -m backend.prediction.media \
  --video drone_corrected.mp4 --output take --sensor drone \
  --fps 5 --time-offset 0

python -m backend.prediction.perception \
  --take take/take.json --weights /path/to/yolo11m-pose.pt \
  --device 0 --output results/baseline

python -m backend.prediction.evaluation \
  results/baseline/frames.jsonl evaluation-only-annotations.json \
  --output results/baseline/evaluation.json
```

`media` writes a sampled-frame manifest. Wrap its `frames` in a `RecordedTake`
with an explicit clock, provenance, uncertainty, and exclusion intervals. The
existing `work/perception/corrected-drone/take.json` is a concrete detection-only
example. Preserve the original source video, exact frame indices, PTS and hashes.
An output directory should belong to one run; do not mix model variants.

`take-native-clock-candidate.json` joins all 112 sampled corrected images to
the corresponding rows of the 840-row native camera file. The companion
`camera-association-audit.json` binds source hashes, original indices and video
PTS to native camera timestamps. This uses a disclosed nominal scale only and
leaves `geometry_verified=false`; it cannot produce accepted metric tracks.

`run_take` is a **capture-time offline replay**. It reorders the recorded manifest
by source time and evaluates belief at those times. Detector processing delay is
measured separately, not simulated in belief timestamps. It is not a live service
or proof of deadline performance. The existing live transport still needs an
inference worker that records actual capture, arrival and result times.

## Geometry and camera gates

`hlae_camera.py` imports AdvancedFX v2.191.1 `.cam` v2 rows. It preserves native
game time, subtracts an explicit shared origin, converts Source roll/pitch/yaw to
OpenCV axes, and treats the exported FOV as the aspect-scaled horizontal image
FOV. A 1920×1080→1280×720 scale-only derivative uses intrinsics for 1280×720;
the FOV must not be aspect-scaled a second time.

Importer source-format verification is not a calibration certificate. Its unit
scale is caller supplied. `calibration.py` evaluates a fixed pose against separate
static landmark observations, with fit and holdout reporting; it does not tune
the pose to those holdouts. Pixel agreement alone cannot establish metric scale.

Current preferred capture bundle:
`/Users/abe/Documents/New project/outputs/dust2-drone/corrected-proof/`.
Use `drone_corrected_native.cam` with `drone_corrected.mp4`, and `ct_native.cam`
with `ct_proof.mp4`. Each clip and sidecar has 840 frames/rows. The derivative is
scale-only, with no crop/trim/drop/retime. Native game-time increments vary.
Recorder provenance supports row-order association; there are no embedded frame
IDs and no shared visible jump anchor. Cross-camera visual sync is **unverified**.

The static nav dump has 2242 area centers/bounds. It lacks polygons, adjacency,
collision surfaces, floor/roof occluders, independently surveyed landmarks and
metric scale. Do not infer traversability from overlapping bounds or invent
walls from a radar image. Upper and Lower Tunnels need separate floor geometry.

Actual-image prediction requires an explicit `geometry_verified` assertion and
`geometry_validation_provenance`, alongside the map/camera/clock checks. This is
an operator assertion backed by retained evidence, not cryptographic proof.
It is false in all current real takes. Evaluation labels stay outside inference.
Camera-away/excluded frames never enter the predictor. Complete visibility and
detector miss assumptions have not been calibrated, so negative evidence is off.

## Order of remaining accuracy checks

1. Record the actual human course run with both player and drone views. The
   current technical pair has a mostly stationary player; it cannot validate
   the intended player rotation and alternate-exit reveal. Prepare now and
   record when the human is ready. Use the same source demo for both rendered
   views, with camera sidecars and source time; clearly label this as replay.
   Measure person/foot detection on that fresh paired take.
2. Obtain static floors/walls/roofs and connectivity; validate camera projections
   using independent static landmark pixels and a documented common scale.
3. Join actual frame cameras and source clocks; measure pixel→world error and
   cross-view track consistency on the recorded pair.
4. Evaluate stale/occluded tracking, hold/return alternatives, conditional arrival
   time error, and missed-exit updates using validated visibility assumptions.
5. Measure actual cloud camera-to-cue latency and complete the acted two-exit take.

Relative route support is still not a calibrated probability. Scripted route and
enemy server transforms remain outside inference. The fixed-CT technical pair
does not yet contain the human rotation, alternate-exit reveal or final encounter.
