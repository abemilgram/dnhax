# Run the macos branch on Apple Silicon

This branch adds an **experimental MPS inference path** for submitted room captures and retains CUDA. It is not a validated real-time pipeline. The interface, upload server, sample scenes, and landmark registration also run without a GPU.

## Install

Use native **arm64 Python 3.11 or 3.12**, Git, Node.js >=22.13, and FFmpeg. With Homebrew already installed:

```sh
brew install python@3.11 ffmpeg node
```

```sh
git clone --branch macos https://github.com/abemilgram/dnhax.git
cd dnhax
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-macos.txt
npm ci
npm run build
```

The macOS requirements pin the public upstream VGGT revision and NumPy 1.26.4. No NVIDIA CUDA toolkit or compiled CUDA extension is needed by this path. Download the public **facebook/VGGT-1B** checkpoint (CC-BY-NC-4.0) into the ignored local model directory:

```sh
.venv/bin/python scripts/download_vggt.py
```

## Check and launch

Run in a normal macOS terminal:

```sh
export SIMV1_DEVICE=mps
export SIMV1_MAX_FRAMES=4
.venv/bin/python scripts/check_compute.py
.venv/bin/python scripts/serve.py --port 48371
```

The check performs a small matrix operation on the selected device and checks that the package and default checkpoint file exist. It does **not** load or validate the checkpoint. Exit 2 means the device works but the package or checkpoint is missing; exit 1 means the device check failed. Set `VGGT_CHECKPOINT` only to use another compatible `.safetensors` or `.pt` checkpoint.

Open `http://localhost:48371`, submit a short room clip, and select **Reconstruct capture**. Start with four frames; inspect geometry and memory before increasing `SIMV1_MAX_FRAMES`. No specific RAM size or speed is guaranteed. MPS currently uses float32 for compatibility, so it may consume more memory than CUDA mixed precision.

`SIMV1_DEVICE=auto` chooses CUDA, MPS, then CPU. Explicit `mps` fails clearly if unavailable. `SIMV1_DEVICE=cpu` permits a slow compatibility run (two frames by default). Sample scenes and registration still work if no model is installed. OOM errors do not silently retry on CPU. Don't disable PyTorch memory safeguards to force a run.

To allow PyTorch's optional fallback for certain unsupported MPS operations, set `PYTORCH_ENABLE_MPS_FALLBACK=1` **before starting Python**. It is not enabled automatically and does not guarantee every unsupported operation can run. It can be much slower.

For another laptop, use the server's LAN address. Screen recording there still requires trusted HTTPS or a localhost SSH tunnel, as described in README.md.

### Same-Wi-Fi phone capture over HTTPS

Install `mkcert`, then use the macOS launcher to create a LAN certificate, expose only its public CA certificate on port 8001, and run the app over HTTPS on port 8000:

```sh
brew install mkcert
.venv/bin/python scripts/download_vggt.py
.venv/bin/python scripts/serve_macos_https.py
```

The launcher prints three exact URLs: the public CA certificate, the Mac app, and the phone app. On an iPhone or iPad, open the CA URL in Safari, install the downloaded profile under **Settings → General → VPN & Device Management**, then enable it under **Settings → General → About → Certificate Trust Settings**. Open the printed phone app URL after trust is enabled. Android setting names vary; install the downloaded file as a CA certificate before opening the HTTPS URL.

The generated certificate, private key, and public phone certificate stay under ignored `work/` directories. The temporary certificate server exposes only the public CA file and stops with the app. Never copy or serve the private key. Mobile browsers may support camera recording without supporting screen recording.

## Implementation and validation limits

`backend/runtime.py` selects the device, releases device-specific caches, and dispatches inference. CUDA uses upstream `forward` unchanged. MPS/CPU directly call the same aggregator and camera/depth heads without upstream's hard-coded CUDA autocast contexts. The model and inputs stay float32; no upstream files or global Torch functions are modified. The optional point and tracking heads are disabled.

The MPS device check and real checkpoint inference must pass on the target Mac before treating the setup as ready. Reconstruction accuracy, memory consumption, and latency still depend on the capture and selected frame count. `VALIDATION.md` records the checks performed for this branch.

Existing cached reconstructions are reused; submit a new capture when comparing devices or frame settings. `compute_device`, `frame_count`, and precision are stored on new reconstruction records.

### Joint A+B on the Apple GPU

Upload both videos, then click **Reconstruct A + B together** under Sources. Each new joint job samples candidates across both full clips, chooses shared-looking anchor frames and temporally distributed sharp frames, and runs one VGGT sequence. Default: four frames per source, eight total. The two source clouds retain the resulting shared coordinate system; no RANSAC transform is applied by this path.

Set `SIMV1_JOINT_FRAMES_PER_SOURCE=2` before restarting the worker for a smaller four-frame-total run. The allowed range is 2–8 per source; the single-capture `SIMV1_MAX_FRAMES` limit does not affect joint jobs. Joint jobs always create a new scene, so retries with changed settings do not reuse old single-capture geometry. Results remain labeled as unverified predictions: recognizable overlap is required for a reliable reconstruction.
