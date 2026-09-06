# dnhax — simv1 civilian room reconstruction

**macos branch:** Apple Silicon setup and validation limits are in [MACOS.md](MACOS.md). CUDA remains supported.

A local three-device demo for submitted room captures. Two browsers upload independent walkthroughs to a CUDA or Apple Silicon processing computer. A GPU worker reconstructs each capture; the viewer supports manual landmark registration, validation, and inspection of the combined point clouds.

For a containerized CUDA deployment on RunPod, see [RUNPOD.md](RUNPOD.md).

**Live batch capture is now available alongside submitted clips. See [LIVE_PROCESSING.md](LIVE_PROCESSING.md) for setup and limits. Live AR overlays, automatic MASt3R matching, VLM analysis, metric calibration and persistent whole-room fusion are not implemented.**

## Tactical Brain golden-tape demo

The default interface is a deterministic six-second replay over an original
fictional metric twin. It contains no CS2 entity transforms or copied map
geometry, and it provides no targeting, auto-aim, or automatic movement
output. A live detector and calibration adapter are future work; the current
demo reads only the checked-in golden tape.

Run the API and frontend as described under Development, then use **Start**,
**Pause**, **Restart**, or the 0.1-second seek slider. The frontend reads
`GET /api/tactical/state`, resumes updates from
`GET /api/tactical/events?since=<revision>`, and posts playback changes to
`/api/tactical/start`, `/pause`, `/restart`, and `/seek`. Use the top mode
switch to open the existing **Reconstruction Lab** without running its polling
or capture components in Tactical Brain mode.

## What works without CUDA

- Capture upload (MP4/MOV/WebM/MKV, up to 512 MB).
- Optional browser screen/window/tab recording; nothing uploads before Submit.
- SQLite-backed processing queue, worker heartbeat, explicit job errors.
- Labeled synthetic room fixture with real computed registration diagnostics.
- Orbit/zoom viewer; source isolation, before/after transform, point size, source/RGB color.
- Point picking for paired landmarks, or JSON landmark import.
- Robust positive-scale similarity registration and held-out validation.
- Immutable scene versions, binary PLY export, transform/provenance manifests.

## Deployment on the compute laptop

Use Python 3.11 or 3.12 for the CUDA environment and Node.js >=22.13. Install FFmpeg on PATH for real video processing.

Windows PowerShell:

```powershell
cd C:\path\to\simv1
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
npm ci
npm run build
.\.venv\Scripts\python.exe scripts\serve.py
```

macOS/Linux (sample and registration also run on CPU; see MACOS.md for Apple GPU inference):

```sh
cd /path/to/simv1
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
npm ci
npm run build
.venv/bin/python scripts/serve.py
```

The launcher starts the API/static frontend on port **8000** and a separate worker. Ctrl+C stops both. Only one worker is supported. Metadata and artifacts persist under `data/`; do not commit or share that directory unintentionally. Set `SIMV1_DATA` before launch to choose another storage directory. The example environment file is documentation; it is not automatically loaded.

Connect all three laptops to the same trusted LAN. Open `http://<NVIDIA-laptop-IP>:8000` in each browser. Find the server's address with `ipconfig` on Windows. Allow the app's TCP port through the server firewall on the private network. The network must permit client-to-client traffic. No account login is implemented, so everyone who can reach this endpoint can read and submit project data. Do not expose it to the public internet.

The two capture laptops need only browsers. All three browsers share one room workspace and its A/B sources.

### Screen recording and HTTPS

File uploads work on LAN HTTP. Browser screen recording generally requires localhost or HTTPS trusted by the client. For in-page screen recording on the other laptops, provision a certificate for the server's actual LAN hostname/IP and trust its issuing CA on each client, then run:

```sh
python scripts/serve.py --cert /path/to/cert.pem --key /path/to/key.pem
```

Open the corresponding `https://` address. Do not copy a CA private key to the clients. Click **Record screen** and select a screen, window, or tab in the browser picker. Stopping sharing finalizes the clip, just like the app’s Stop recording button. Screen capture availability depends on the browser; if the embedded browser does not support it, open the URL in Chrome or Edge. Recording is explicitly user-initiated and is never a background continuous upload.

## AMB3R reconstruction on RunPod

The `AMB3R` branch uses the full AMB3R model and its unordered SfM pipeline on
CUDA. Use the pinned [RunPod image setup](RUNPOD.md); this is not a macOS
backend. Frames from A and B enter one image pool, while simv1 retains the
input-index manifest needed to split the resulting shared world points back
into their sources. AMB3R-SfM requires a connected visual-overlap graph and
does not natively model camera IDs, synchronization, dynamic objects, or
separate disconnected maps.

The released geometry remains in reconstruction units. AMB3R's metric-depth
head does not metric-scale the SfM world points or camera translations. The
upstream source and checkpoint also lack a stated license, so this integration
is an evaluation build until those rights are clarified.

## Optional VGGT reconstruction

Approved **VGGT-Omega 1B-512** is also supported for single and joint A+B reconstruction on MPS/CUDA. See [the Omega setup instructions](MACOS.md#approved-vggt-omega-weights). Install `requirements-vggt-omega.txt` and the approved checkpoint. Auto model selection prefers installed Omega weights; set `SIMV1_MODEL=vggt` to keep public VGGT. Results record the actual model and preserve older scenes.

The sample path is independent of the model. No model weights are bundled or downloaded automatically.

1. For NVIDIA CUDA, install the matching PyTorch build using the [official PyTorch installer](https://pytorch.org/get-started/locally/) in the worker's Python environment. For Apple MPS, follow [MACOS.md](MACOS.md).
2. Install the pinned public [VGGT implementation](https://github.com/facebookresearch/vggt): `python -m pip install -r requirements-vggt.txt`.
3. Download the public `facebook/VGGT-1B` weights: `python scripts/download_vggt.py`. The checkpoint uses the CC-BY-NC-4.0 license.
4. Optionally set `VGGT_CHECKPOINT` to another compatible `.safetensors` or `.pt` checkpoint path. Otherwise the downloader's `models/vggt-1b/model.safetensors` path is used.
5. Verify CUDA:

```sh
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

PowerShell launch example:

```powershell
.\.venv\Scripts\python.exe scripts\serve.py
```

Single-capture reconstruction extracts a bounded number of frames at 1 fps (defaults: CUDA 24, MPS 8, CPU 2) from the beginning of the submitted clip. Joint A+B reconstruction selects frames across both full clips as described below. Both paths confidence-filter geometry and export at most 1,000,000 displayed points per source. Depth, confidence, and predicted camera matrices are preserved alongside the clouds. Video timestamps are approximate frame-sampling timestamps.

The adapter follows the official model API. Memory use, geometry quality, and latency depend on the selected device and clip. A missing model, missing checkpoint, explicitly requested unavailable device, or failed decode produces an explicit failed job and never a sample reconstruction. The app does not promise live processing performance.

## Demo walkthrough

1. Open the app and click **Load sample**. Wait for the worker; a synthetic room appears.
2. In **Alignment**, toggle Apply alignment to inspect independent frames versus registered geometry. Diagnostics are computed from the fixture's known correspondence pairs, not measured video performance.
3. In **Explore**, isolate each source, change point size, and download original PLYs and the scene manifest.
4. For real captures, submit one room clip to A and one to B. Both clips need recognizable shared content.
5. Click **Reconstruct A + B together**. The selected model processes keyframes from both videos together and exports a **Joint A + B** scene with separate source labels in shared coordinates. You do not need to reconstruct the captures separately first.
6. In **Explore**, toggle A/B or enable source colors to inspect overlap. Joint predictions are not independently validated alignment measurements. If shared features are scarce, the scene displays a warning.
7. For an optional manual correction, open **Optional landmark correction**, click **Pick pairs**, and select the same landmark in A then B for at least eight distributed pairs. Click **Fit and validate** to publish a new version. The manual correction remains a separate RANSAC similarity fit; the joint pipeline does not use RANSAC.

To compare the previous independent pipeline, use **Reconstruct capture** on each video, then **Compare existing independent reconstructions → Combine existing A + B**. That path only places the clouds together until you supply landmark pairs.

Registration uses a deterministic train/held-out split. At least two pairs are held out. The heuristic accepted status requires >=60% training inliers and median held-out residual below the chosen threshold. Inspect the geometry: a low error on a small shared patch is not proof of global accuracy. A global similarity transform cannot remove internal reconstruction distortion. No absolute meters are claimed.

The viewer opens the latest real two-source scene when one is available. Later single-capture results do not replace a selected combined scene. The scene menu distinguishes **Joint A + B**, **Combined A + B**, and single captures. Each source switch shows its point count; an absent source is labeled **Not in this scene** and cannot be toggled. Scene changes show all included sources, and **Displayed points** counts only sources currently enabled.

### Joint keyframe selection and limits

`SIMV1_JOINT_FRAMES_PER_SOURCE` defaults to **unlimited** (every 1 fps frame). Set an integer of at least 2 to cap each source. This is independent of `SIMV1_MAX_FRAMES`, which likewise defaults to unlimited for single-capture jobs. When a cap is set, the selector samples at least twelve candidates across each full clip, chooses an A/B anchor pair using reciprocal SIFT descriptor matches, then selects additional frames using temporal spread and sharpness. The match count is an appearance cue, not a geometric overlap certificate. No extra model weights are needed.

All selected A/B images enter one model invocation. Splitting the output by source preserves the predicted shared coordinates without recentering or fitting a transform. New jobs write new scene artifacts; existing independent reconstructions remain unchanged. Joint manifests include selected camera frames/timestamps, source IDs, device, elapsed time, and an explicit unverified-quality note. Disjoint views, repeated textures, blur, or inconsistent depth can still produce poor geometry; the app does not claim an accepted registration merely because inference completes.

## Layout and containers

```text
app/                  React frontend, Three.js viewer, capture controls
components/ui/        Starter UI primitives
backend/api.py        Uploads, jobs, state, artifact serving
backend/worker.py     Single worker, sample generation, job execution
backend/reconstruct.py Shared inference/export boundary
backend/amb3r_runtime.py AMB3R and AMB3R-SfM CUDA adapter
backend/joint.py      Joint A+B keyframe selection and reconstruction
backend/geometry.py   Similarity fitting, RANSAC, PLY output
backend/store.py      SQLite metadata and immutable publication
scripts/serve.py      Local LAN launcher and process cleanup
scripts/download_vggt.py Public pinned checkpoint downloader
scripts/download_amb3r.py Official AMB3R checkpoint downloader
dist/client/          Built static frontend (generated)
data/                 Local captures and artifacts (ignored)
tests/                Backend integration and geometry checks
```

The API serves static `dist/client/` and `/api/*` on the same origin; there is no separate frontend server in the built deployment. The worker accesses the local SQLite queue and filesystem. It is never directly exposed over the LAN.

### Development

Run these in separate terminals:

```sh
python -m uvicorn backend.api:app --host 127.0.0.1 --port 8000
python -m backend.worker
npm run dev
```

Vite runs at port 5173 and proxies `/api` to 8000. Use the built launcher for the shared LAN demo.

### Verification

```sh
python -m pytest -q
node --test tests/scene-selection.test.mjs
npm run typecheck
npm run build
```

Tests cover transform recovery, outliers, degenerate landmarks, exclusive job claiming, sample publication, binary artifacts, upload validation, traversal protection, version preservation, and no fake reconstruction on failure.

Browser interaction/screen-capture testing and real CUDA inference remain hardware checks. An optional read-only WebMCP workspace tool is feature-detected; it is not required for the app and has not been validated in a WebMCP-capable browser.

## Appearance and density

Real reconstructions default to their source image RGB colors. Color by source remains available for alignment inspection. The viewer converts stored sRGB bytes to linear vertex colors and uses round, depth-tested point footprints with perspective sizing. Surface coverage controls their size relative to estimated cloud spacing; this is a rendering heuristic, not a reconstructed mesh or trained Gaussian splat.

New CUDA reconstructions retain all eligible pixels before the 1-million-point-per-source export limit, reject depth discontinuities above a 3% local relative jump, and remove the bottom 20% of remaining confidence values. Existing cached reconstructions keep their original density; submit the clip again to create a fresh capture if needed. Sample geometry remains synthetic and does not acquire photographic colors.

### Live camera location stream

Each new live batch includes `live.camera_locations` in its scene manifest. The viewer's **Cameras on/off** control overlays positions, schematic viewing directions, and within-batch paths on the point cloud. Paths are separated by source and capture epoch. Coordinate readouts show the latest sample per source in the displayed batch.

Poll `GET /api/live/sessions/{session_id}/camera-locations?after_batch=0` for a separate location stream. The response contains `updates` in ascending batch order (at most 20) and a `cursor`; pass that cursor as `after_batch` on the next poll. Each update includes `scene_id`, `batch`, `segment`, `coordinate_system`, `units`, and `samples`. Join `scene_id` to geometry from the existing scene history endpoint or session state. Only completed, published scenes are returned. The stream follows scene retention (normally five recent batches plus pinned snapshots); it is not a durable full-session trajectory, and older pre-feature scenes have no pose samples.

Each sample includes `frame_id`, `source`, `epoch`, `seq`, capture media time `captured`, server receipt time `received`, `position` (x/y/z), and a row-major 4×4 `camera_to_world` matrix. Camera axes are right/down/forward (OpenCV). For world-to-camera extrinsics `[R|t]`, position is `-Rᵀt`. The same continuity similarity is applied to the geometry and camera poses before publication, so they share segment coordinates. A new `segment` means coordinates must not be joined to a previous segment without another alignment. Repeated frames can receive revised estimates in later batches.

These are geometry-derived positions in arbitrary reconstruction units, not GPS latitude/longitude or measured meters. Absolute geolocation requires an external geographic reference and scale/orientation calibration. Screen capture estimates describe the imaged scene's camera, not the physical laptop's location. Updates arrive at reconstruction batch cadence, not at camera frame rate.
