# Base image with PyTorch 2.4.1 + CUDA 12.1 + cuDNN 9 runtime
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Build tools are not part of final image
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Split into two layers: general deps (change less often) vs. the
# torch-scatter/torch-geometric pair, which MUST be installed in this
# order and pinned to the exact torch+CUDA build.
# Cache mount speeds up rebuilds without bloating this (discarded) stage.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install e3nn biopython tqdm scipy scikit-learn h5py py3Dmol hdbscan

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch-scatter \
        -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch-geometric \
        -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

## Runtime image
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Pull in only the installed packages, not the build toolchain
COPY --from=builder /opt/conda /opt/conda

RUN useradd -ms /bin/bash appuser

COPY --chown=appuser:appuser Model/ /home/appuser/Model/

USER appuser
WORKDIR /home/appuser/Model