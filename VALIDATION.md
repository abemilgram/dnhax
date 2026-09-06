# Validation — 2026-09-05

- Backend tests: 11 passed (including dense export color correspondence and depth-edge filtering).
- Python undefined/unused-name checks: passed.
- TypeScript type check: passed.
- Application lint: passed; generated UI primitives are excluded. React Compiler checks are disabled because the project does not enable React Compiler.
- Production static export: passed. Three.js produces a bundle-size advisory; this is not a build failure.
- Dependency audit after updates: 0 reported vulnerabilities.
- Built same-origin frontend/API: HTTP 200; all seven directly referenced JS/CSS assets served successfully.
- LAN address responded on the host. Access from the other physical laptops still needs verification.
- Launcher started API and worker; a sample job completed and binary point counts matched the manifest.
- Sample registration: synthetic geometry, not evidence of real capture accuracy.

Not verified here: NVIDIA CUDA inference, model checkpoint loading, real-video reconstruction quality, browser interactions/screen-sharing permission flows, cross-laptop connectivity, or the optional WebMCP integration. These need the target hardware and browser. No claim of live reconstruction or automatic cross-capture matching is made.

## macos branch

- 17 backend tests pass, including device selection, strict device errors, frame limits, targeted cache cleanup, non-CUDA head dispatch using real CPU tensors, and unchanged CUDA forward dispatch.
- The pinned public VGGT source installed successfully on macOS arm64 / Python 3.11 with PyTorch 2.11.0, and `pip check` reports no broken requirements.
- The public `facebook/VGGT-1B` safetensors checkpoint downloaded at the pinned Hugging Face revision. Its size is 5,026,367,224 bytes and its SHA-256 matches the repository metadata: `f164acf60724910d8fe1578bb499d800850c7bb0948db7555c413f9fbe60467e`.
- The compute checker passed a matrix operation on Apple MPS and found both the installed package and checkpoint.
- Full checkpoint inference completed on an Apple M4 Pro GPU for two independent submitted clips at two 518px frames each. The exports contain 415,881 and 401,209 filtered colored points, camera predictions, depth, confidence, and `compute_device: mps` provenance.
- These checks establish that the integration executes on this Mac. They do not establish reconstruction accuracy or performance across other captures, frame counts, or Apple Silicon models.

## Combined-scene visibility

- Both saved clouds were fetched through the running trusted HTTPS API and checked: A contains 415,881 finite, non-zero points; B contains 401,209. Color buffer lengths match both manifests.
- Six Node regression tests pass for default combined-scene selection, preservation after a later single reconstruction, explicit single-scene selection, newly combined/registered results, missing selections, and distinct labels.
- TypeScript, lint, and production build pass. The running HTTPS app serves the rebuilt viewer and all seven directly referenced JS/CSS assets. The default-scene helper selects both real clouds using the current API state.
