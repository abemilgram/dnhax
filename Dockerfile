# syntax=docker/dockerfile:1.7

FROM node:22-bookworm-slim AS frontend

WORKDIR /src
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST=12.0 \
    HF_HOME=/workspace/.cache/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-vggt.txt requirements-vggt-omega.txt ./
RUN python -m pip install \
        -r requirements.txt \
        -r requirements-vggt.txt \
        -r requirements-vggt-omega.txt

COPY . .
COPY --from=frontend /src/dist/client /app/dist/client

ENV SIMV1_DEVICE=cuda \
    SIMV1_MODEL=vggt \
    SIMV1_DATA=/workspace/simv1-data \
    VGGT_CHECKPOINT=/workspace/models/vggt-1b/model.safetensors \
    SIMV1_RETENTION_HOURS=24 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15m --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/state', timeout=3)"

CMD ["python", "-m", "scripts.runpod_start"]
