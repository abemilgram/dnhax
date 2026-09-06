# RunPod deployment

This image runs the static frontend, FastAPI API, and the single supported
GPU worker in one pod. Captures are uploaded to the pod, VGGT creates the
point clouds there, and browser clients fetch the resulting artifacts.

## Build and publish

Choose a registry path and publish the image from the repository root:

```sh
docker build --platform linux/amd64 -t ghcr.io/OWNER/simv1:latest .
docker push ghcr.io/OWNER/simv1:latest
```

The runtime is based on the PyTorch 2.7.1 CUDA 12.8 development image, which
includes Blackwell support and the compiler toolchain needed for experimental
AMB3R dependencies. Model weights are deliberately not embedded in the image.

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
SIMV1_MODEL=vggt
VGGT_CHECKPOINT=/workspace/models/vggt-1b/model.safetensors
SIMV1_DATA=/workspace/simv1-data
SIMV1_RETENTION_HOURS=24
PORT=8000
```

On the first start, the public VGGT-1B checkpoint is downloaded to the network
volume. Later pods reuse it. Set `SIMV1_DOWNLOAD_MODEL=0` to require a
pre-provisioned checkpoint instead.

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

RunPod's HTTP proxy forwards its HTTPS URL to port 8000. The application has no
account authentication, so do not distribute that URL outside the intended
group. Keep one pod attached to a given `SIMV1_DATA` directory because the API
and worker share a SQLite database and filesystem lock.
