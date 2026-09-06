# RunPod deployment

This image runs the static frontend, FastAPI API, and the single supported
GPU worker in one pod. Captures are uploaded to the pod, AMB3R creates the
point clouds there, and browser clients fetch the resulting artifacts. This
branch runs the full AMB3R model through its unordered AMB3R-SfM pipeline.

Do not point the existing macoslive VGGT RunPod endpoint at this branch.
Build a separate image and pod when you want to evaluate AMB3R.

## Build and publish

Choose a registry path and publish the image from the repository root:

```sh
docker build --platform linux/amd64 -t ghcr.io/OWNER/simv1:latest .
docker push ghcr.io/OWNER/simv1:latest
```

The runtime is based on PyTorch 2.7.1 and CUDA 12.8. It compiles PyTorch3D for
SM 12.0, installs matching torch-scatter and spconv builds, pins AMB3R source
to commit `92c4081`, and disables FlashAttention because AMB3R's published
FlashAttention version has no Blackwell kernel. Model weights are not embedded
in the image. The first image build is consequently long.

## RunPod template

Configure a GPU Pod, not a Serverless endpoint:

- **Container image:** the image published above
- **HTTP port:** `8000`
- **Network volume mount:** `/workspace`
- **Container disk:** at least 60 GB
- **Volume size:** enough for model weights, uploaded captures, and scenes
- **Command:** leave empty to use the image default

The image defaults are:

```env
SIMV1_DEVICE=cuda
SIMV1_MODEL=amb3r
AMB3R_CHECKPOINT=/workspace/models/amb3r/amb3r.pt
AMB3R_ROOT=/opt/amb3r
SIMV1_DATA=/workspace/simv1-data
SIMV1_RETENTION_HOURS=24
PORT=8000
```

On first start, the roughly 4.1 GB official `amb3r.pt` checkpoint is downloaded
from the authors' Google Drive release into the network volume. Later pods
reuse it. Set `SIMV1_DOWNLOAD_MODEL=0` to require a pre-provisioned checkpoint.
Google Drive is not a production-grade model registry; mirror the checkpoint
and pin its SHA-256 before relying on unattended production starts.

For approved VGGT-Omega weights, configure:

```env
SIMV1_MODEL=vggt_omega
VGGT_OMEGA_CHECKPOINT=/workspace/models/vggt-omega/vggt_omega_1b_512.pt
HF_TOKEN=your-read-token
```

Omega access must already be approved on Hugging Face. Store `HF_TOKEN` as a
RunPod secret rather than in the image or repository.

The worker removes completed submitted-capture data older than
`SIMV1_RETENTION_HOURS`. Set it to `0` to disable scheduled cleanup. The
interface also provides direct controls to delete one stored capture, delete
ordinary data older than 24 hours, or reset the complete workspace. Resetting
workspace data does not remove model weights from `/workspace/models`.

## Verify the GPU

The startup log prints the GPU name, compute capability, PyTorch version, and
CUDA version before downloading weights. An RTX PRO 6000 Blackwell should
report compute capability `(12, 0)`.

Set RunPod's health-check path to `/ping`. Model loading happens in the worker
on the first reconstruction rather than in the HTTP health check.

RunPod's HTTP proxy forwards its HTTPS URL to port 8000. The application has no
account authentication, so do not distribute that URL outside the intended
group. Keep one pod attached to a given `SIMV1_DATA` directory because the API
and worker share a SQLite database and filesystem lock.

AMB3R's repository and released checkpoint do not currently state a license.
Treat this image as an evaluation build until the authors grant the rights
needed for the intended deployment.
