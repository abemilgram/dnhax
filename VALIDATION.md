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
- macOS arm64 / Python 3.11 dependency resolution succeeded for requirements-macos.txt. This is dependency resolution, not a full installation test.
- The compute checker passed its CPU matrix operation. It correctly reported missing model installation/checkpoint.
- This session is arm64 with PyTorch 2.11.0; MPS is built but unavailable. Full model-weight inference on MPS remains unverified.
- Actual upstream model components at reduced width ran on CPU with synthetic initialized weights and finite camera/depth/confidence outputs. This checks dispatch compatibility only, not checkpoint correctness or reconstruction quality.
