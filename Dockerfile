# syntax=docker/dockerfile:1.7

# ============================================================
# Builder
# ============================================================
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Pinned to upstream main HEADs; override with --build-arg if needed
ARG GAFL_COMMIT=018247ade22812407b6dfb9093944f442a606aa0
ARG BBFLOW_COMMIT=f8628c950bbcd577bf294338c9ef330ed4cb9bbd

# Build-time tools only
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Enforce numpy<2 across every subsequent pip install: GATr's (patched)
# metadata requires it, but unpinned transitive deps would otherwise
# pull in numpy 2.x and break `pip check`.
RUN echo "numpy<2" > /tmp/constraints.txt
ENV PIP_CONSTRAINT=/tmp/constraints.txt

# ------------------------------------------------------------
# General dependencies
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        numpy==1.26.4 \
        e3nn \
        biopython \
        tqdm \
        scipy \
        scikit-learn \
        h5py \
        py3Dmol \
        hdbscan

# ------------------------------------------------------------
# PyG packages matching the exact PyTorch/CUDA build
# ------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
        torch-scatter \
        -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch-geometric \
        -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# ------------------------------------------------------------
# BBFlow dependencies
# ------------------------------------------------------------
# Cloned into /opt (not /tmp): bbflow/gafl's setup.py rely on implicit
# namespace packages, which only resolve correctly with an editable
# install (`pip install -e .`), so the source tree must persist at runtime.
RUN git clone \
        https://github.com/graeter-group/bbflow.git \
        /opt/bbflow

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /opt/bbflow/install_utils/requirements.txt

# ------------------------------------------------------------
# GAFL
# ------------------------------------------------------------
RUN git clone \
        https://github.com/hits-mli/gafl.git \
        /opt/gafl && \
    cd /opt/gafl && \
    git checkout "${GAFL_COMMIT}" && \
    bash install_gatr.sh && \
    SITE_PACKAGES=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") && \
    METADATA=$(ls "${SITE_PACKAGES}"/gatr-*.dist-info/METADATA) && \
    sed -i \
        -e '/^Requires-Dist: xformers/d' \
        -e 's/^Requires-Dist: numpy.*/Requires-Dist: numpy<2/' \
        "$METADATA" && \
    pip install --no-deps -e .

# ------------------------------------------------------------
# BBFlow
# ------------------------------------------------------------
RUN cd /opt/bbflow && \
    git checkout "${BBFLOW_COMMIT}" && \
    pip install --no-deps -e .

# ------------------------------------------------------------
# Verify dependency consistency and critical versions
# ------------------------------------------------------------
RUN pip install --force-reinstall --no-cache-dir ninja && \
    pip check

RUN python - <<'PY'
import torch
import torch_scatter
import torch_geometric
import gafl
import bbflow

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("torch-scatter:", torch_scatter.__version__)
print("torch-geometric:", torch_geometric.__version__)
print("GAFL:", gafl.__file__)
print("BBFlow:", bbflow.__file__)
PY


# ============================================================
# Runtime
# ============================================================
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /opt/conda /opt/conda
# gafl/bbflow are installed editable, so their source trees must ship too
COPY --from=builder /opt/gafl /opt/gafl
COPY --from=builder /opt/bbflow /opt/bbflow

RUN useradd -ms /bin/bash appuser

COPY --chown=appuser:appuser Model/ /home/appuser/Model/

USER appuser
WORKDIR /home/appuser/Model