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

The macOS requirements pin the audited upstream VGGT-Ω revision and NumPy 1.26.4 to satisfy its `numpy<2` requirement. No NVIDIA CUDA toolkit or compiled CUDA extension is needed by this path. Model weights are separate: request the official **VGGT-Omega-1B-512** checkpoint under its license and download it yourself.

## Check and launch

Run in a normal macOS terminal:

```sh
export SIMV1_DEVICE=mps
export SIMV1_MAX_FRAMES=4
export VGGT_OMEGA_CHECKPOINT='/absolute/path/to/model.pt'
.venv/bin/python scripts/check_compute.py
.venv/bin/python scripts/serve.py --port 48371
```

The check performs a small matrix operation on the selected device and checks that the package and checkpoint file exist. It does **not** load or validate the checkpoint. Exit 2 means the device works but the package or checkpoint is missing; exit 1 means the device check failed.

Open `http://localhost:48371`, submit a short room clip, and select **Reconstruct capture**. Start with four frames; inspect geometry and memory before increasing `SIMV1_MAX_FRAMES`. No specific RAM size or speed is guaranteed. MPS currently uses float32 for compatibility, so it may consume more memory than CUDA mixed precision.

`SIMV1_DEVICE=auto` chooses CUDA, MPS, then CPU. Explicit `mps` fails clearly if unavailable. `SIMV1_DEVICE=cpu` permits a slow compatibility run (two frames by default). Sample scenes and registration still work if no model is installed. OOM errors do not silently retry on CPU. Don't disable PyTorch memory safeguards to force a run.

To allow PyTorch's optional fallback for certain unsupported MPS operations, set `PYTORCH_ENABLE_MPS_FALLBACK=1` **before starting Python**. It is not enabled automatically and does not guarantee every unsupported operation can run. It can be much slower.

For another laptop, use the server's LAN address. Screen recording there still requires trusted HTTPS or a localhost SSH tunnel, as described in README.md.

### Same-Wi-Fi phone capture over HTTPS

Install `mkcert`, then use the macOS launcher to create a LAN certificate, expose only its public CA certificate on port 8001, and run the app over HTTPS on port 8000:

```sh
brew install mkcert
export VGGT_OMEGA_CHECKPOINT='/absolute/path/to/vggt_omega_1b_512.pt'
.venv/bin/python scripts/serve_macos_https.py
```

The launcher prints three exact URLs: the public CA certificate, the Mac app, and the phone app. On an iPhone or iPad, open the CA URL in Safari, install the downloaded profile under **Settings → General → VPN & Device Management**, then enable it under **Settings → General → About → Certificate Trust Settings**. Open the printed phone app URL after trust is enabled. Android setting names vary; install the downloaded file as a CA certificate before opening the HTTPS URL.

The generated certificate, private key, and public phone certificate stay under ignored `work/` directories. The temporary certificate server exposes only the public CA file and stops with the app. Never copy or serve the private key. Mobile browsers may support camera recording without supporting screen recording.

## Implementation and validation limits

`backend/runtime.py` selects the device, releases device-specific caches, and dispatches inference. CUDA uses upstream `forward` unchanged. MPS/CPU directly call the same aggregator and camera/depth heads without upstream's hard-coded CUDA autocast contexts. The model and inputs stay float32; no upstream files or global Torch functions are modified. The optional text-alignment head is not used by this application.

This development session reports MPS as unavailable. Device selection and non-CUDA dispatch are tested, but **full checkpoint inference on Apple GPU, reconstruction accuracy, memory consumption, and latency remain unverified**. `VALIDATION.md` records local checks. If the check fails in this environment, try a native terminal on the target Mac before assuming the hardware is unsupported.

Existing cached reconstructions are reused; submit a new capture when comparing devices or frame settings. `compute_device`, `frame_count`, and precision are stored on new reconstruction records.
