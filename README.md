# dnhax — simv1 civilian room reconstruction

**macos branch:** Apple Silicon setup and validation limits are in [MACOS.md](MACOS.md). CUDA remains supported.

A local three-laptop demo for submitted room captures. Two browsers upload independent walkthroughs to an NVIDIA processing laptop. A CUDA worker reconstructs each capture; the viewer supports manual landmark registration, validation, and inspection of the combined point clouds.

**This release processes submitted clips. It does not implement continuous keyframe streaming, live AR overlays, automatic MASt3R matching, VLM analysis, or metric calibration.**

## What works without CUDA

- Capture upload (MP4/MOV/WebM/MKV, up to 512 MB).
- Optional browser screen/window/tab recording of a clip up to 60 seconds; nothing uploads before Submit.
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

## Optional real CUDA reconstruction

The sample path is independent of the model. No model weights are bundled or downloaded automatically.

1. Install the CUDA-enabled PyTorch build matching the NVIDIA laptop using the [official PyTorch installer](https://pytorch.org/get-started/locally/), into the same Python environment used by the worker.
2. Install the [official VGGT-Ω repository](https://github.com/facebookresearch/vggt-omega) and its dependencies into that environment, following its README. Request access to the **VGGT-Omega-1B-512** checkpoint and comply with its license. Record the installed repository revision for reproducibility.
3. Set `VGGT_OMEGA_CHECKPOINT` to the actual `.pt` checkpoint path before starting the launcher.
4. Verify CUDA:

```sh
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

PowerShell example:

```powershell
$env:VGGT_OMEGA_CHECKPOINT = 'C:\models\vggt-omega\model.pt'
.\.venv\Scripts\python.exe scripts\serve.py
```

The adapter extracts a bounded number of frames at 1 fps (defaults: CUDA 24, MPS 8, CPU 2) (the beginning of the submitted clip), independently processes each capture, confidence-filters its geometry, and exports at most 1,000,000 displayed points per source. Depth, confidence, and predicted camera matrices are preserved under the reconstruction directory. This is a fixed sampling baseline, not intelligent keyframe selection. Video timestamps are approximate frame-sampling timestamps.

**CUDA inference is not validated on this Mac.** The adapter follows the official model API, but installation, checkpoint format, memory use, geometry quality, and latency must be verified on the actual NVIDIA machine. A missing model, missing checkpoint, explicitly requested unavailable device, or failed decode produces an explicit failed job and never a sample reconstruction. The app does not promise live processing performance.

## Demo walkthrough

1. Open the app and click **Load sample**. Wait for the worker; a synthetic room appears.
2. In **Alignment**, toggle Apply alignment to inspect independent frames versus registered geometry. Diagnostics are computed from the fixture's known correspondence pairs, not measured video performance.
3. In **Explore**, isolate each source, change point size, and download original PLYs and the scene manifest.
4. For real captures, submit one room clip to A and one to B. Click **Reconstruct capture** for each and wait for both jobs to complete.
5. Click **Combine A + B**. This publishes the two independent maps without claiming alignment.
6. Under **Landmark alignment**, click **Pick pairs**. Select the same physical landmark first in A, then B; repeat for at least 8 well-distributed pairs. Toggle source visibility if useful. Clicking stores original cloud coordinates, even if previously transformed.
7. Adjust the inlier threshold in A's reconstruction units and click **Fit and validate**. A new immutable scene version is published. Alternatively paste JSON with `source_points` (B) and `target_points` (A), arrays of matching `[x,y,z]` coordinates.

Registration uses a deterministic train/held-out split. At least two pairs are held out. The heuristic accepted status requires >=60% training inliers and median held-out residual below the chosen threshold. Inspect the geometry: a low error on a small shared patch is not proof of global accuracy. A global similarity transform cannot remove internal reconstruction distortion. No absolute meters are claimed.

## Layout and containers

```text
app/                  React frontend, Three.js viewer, capture controls
components/ui/        Starter UI primitives
backend/api.py        Uploads, jobs, state, artifact serving
backend/worker.py     Single worker, sample generation, job execution
backend/reconstruct.py Optional batch VGGT-Ω CUDA adapter
backend/geometry.py   Similarity fitting, RANSAC, PLY output
backend/store.py      SQLite metadata and immutable publication
scripts/serve.py      Local LAN launcher and process cleanup
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
npm run typecheck
npm run build
```

Tests cover transform recovery, outliers, degenerate landmarks, exclusive job claiming, sample publication, binary artifacts, upload validation, traversal protection, version preservation, and no fake reconstruction on failure.

Browser interaction/screen-capture testing and real CUDA inference remain hardware checks. An optional read-only WebMCP workspace tool is feature-detected; it is not required for the app and has not been validated in a WebMCP-capable browser.

## Appearance and density

Real reconstructions default to their source image RGB colors. Color by source remains available for alignment inspection. The viewer converts stored sRGB bytes to linear vertex colors and uses round, depth-tested point footprints with perspective sizing. Surface coverage controls their size relative to estimated cloud spacing; this is a rendering heuristic, not a reconstructed mesh or trained Gaussian splat.

New CUDA reconstructions retain all eligible pixels before the 1-million-point-per-source export limit, reject depth discontinuities above a 3% local relative jump, and remove the bottom 20% of remaining confidence values. Existing cached reconstructions keep their original density; submit the clip again to create a fresh capture if needed. Sample geometry remains synthetic and does not acquire photographic colors.
