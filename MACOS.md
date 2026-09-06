# Run the UI and tactical replay on macOS

Video-to-3D reconstruction is not supported on macOS. The only selectable
reconstruction backend is AMB3R, and AMB3R requires NVIDIA CUDA. CPU and Apple
MPS requests fail explicitly; the runtime does not substitute VGGT,
VGGT-Omega, another device, or sample geometry.

macOS can run:

- the Tactical Brain golden-tape UI and API;
- capture upload and queue management;
- sample geometry, scene viewing, exports, and landmark registration;
- frontend and backend development tests.

## Install

Use native arm64 Python 3.11 or 3.12, Git, Node.js >=22.13, and FFmpeg:

```sh
brew install python@3.11 ffmpeg node
git clone https://github.com/abemilgram/dnhax.git
cd dnhax
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
npm ci
npm run build
```

Historical macOS/VGGT requirement and download files remain in the repository
only for artifact compatibility and prior validation records. They are not a
selectable reconstruction path.

## Launch

```sh
.venv/bin/python scripts/serve.py --port 48371
```

Open `http://localhost:48371`. The default Tactical Brain replay, sample jobs,
and registration work without AMB3R. Submitting a single or joint
reconstruction job on this Mac records an explicit CUDA/AMB3R failure.

For another laptop, use the server's LAN address. Screen recording there still
requires trusted HTTPS or a localhost SSH tunnel, as described in README.md.

## Same-Wi-Fi phone capture over HTTPS

Install `mkcert`, then use the macOS launcher to create a LAN certificate,
expose only its public CA certificate on port 8001, and run the app over HTTPS
on port 8000:

```sh
brew install mkcert
.venv/bin/python scripts/serve_macos_https.py
```

The launcher prints the public CA certificate URL and the Mac and phone app
URLs. On iPhone or iPad, install the downloaded profile under **Settings →
General → VPN & Device Management**, enable it under **Settings → General →
About → Certificate Trust Settings**, then open the phone app URL. Android
setting names vary.

Generated certificates stay under ignored `work/` directories. The temporary
certificate server exposes only the public CA file and stops with the app.
Never copy or serve the private key. Mobile browsers may support camera
recording without supporting screen recording.

The HTTPS launcher validates the AMB3R-only model selector during startup, but
that does not make reconstruction available on macOS. Use
`Dockerfile.amb3r` on an NVIDIA CUDA host for video-to-3D.
