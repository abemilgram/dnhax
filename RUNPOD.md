# RunPod deployment

## Queue worker

Build `Dockerfile` and deploy it as endpoint type Queue.
The container starts `handler.py`, which registers the Runpod handler.
Smoke-test input: `{"input":{"kind":"sample"}}`.

The handler runs the existing processor synchronously and returns the local job
ID and newly published scene manifests. Supported kinds are `sample`,
`reconstruct`, `joint`, `pair`, and `register`; `payload` uses the same fields as
the local processor. Invalid input or processing failures fail the Runpod job.

This lightweight image supports sample, pair, and registration work; Tactical
Brain's separate golden replay also needs no reconstruction model. The queue
image does not contain AMB3R. A `reconstruct` or `joint` request therefore
fails explicitly unless CUDA and the complete AMB3R source, dependencies, and
checkpoint were separately provisioned. Use `Dockerfile.amb3r` for video-to-3D;
no other model is selected as a fallback.

The existing application is a local SQLite/filesystem workspace. Capture and
scene IDs must already exist in the worker's SIMV1_DATA directory; submitting an
ID from a laptop does not upload its video. Set SIMV1_DATA to a mounted persistent
workspace when retaining data and use a single worker. Do not run the local
worker concurrently against that workspace. This adapter does not run the live
batch scheduler or migrate the frontend to Runpod's queue API.

Artifact paths in scene manifests are relative to the existing application's
`/api/artifacts/` route; the queue container does not host that HTTP route.
Retrieve artifacts from the workspace or serve them through the existing API.

## AMB3R GPU Pod

`Dockerfile.amb3r` runs the static frontend, FastAPI API, and the single supported
GPU worker in one pod. Captures are uploaded to the pod, AMB3R creates the
point clouds there, and browser clients fetch the resulting artifacts. This
branch runs the full AMB3R model through its unordered AMB3R-SfM pipeline.

Do not use the lightweight queue image for reconstruction. Build a separate
AMB3R image and GPU pod.

## Build and publish

Choose a registry path and publish the image from the repository root:

```sh
docker build --platform linux/amd64 -f Dockerfile.amb3r -t ghcr.io/OWNER/simv1:amb3r .
docker push ghcr.io/OWNER/simv1:amb3r
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

Any `SIMV1_MODEL` value other than `amb3r` is rejected. Missing source,
dependencies, checkpoint, CUDA, or inference failures terminate the
reconstruction job explicitly.

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
