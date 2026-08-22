FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /opt

# ------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Keep the PyTorch 2.4.1 / CUDA 12.1 stack from Simorgh
# ------------------------------------------------------------
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        torch==2.4.1 \
        torchvision==0.19.1 \
        torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# ------------------------------------------------------------
# Simorgh dependencies
# ------------------------------------------------------------
RUN python -m pip install \
    e3nn \
    glob2 \
    tqdm \
    h5py \
    numpy==2.2.6 \
    scipy==1.15.3 \
    wandb \
    scikit-learn==1.7.2 \
    biopython

# ------------------------------------------------------------
# PyTorch Geometric / torch-scatter
# Must match torch 2.4.1 + CUDA 12.1
# ------------------------------------------------------------
RUN python -m pip install \
    torch-scatter \
    -f https://data.pyg.org/whl/torch-2.4.1+cu121.html && \
    python -m pip install \
    torch-geometric \
    -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# ------------------------------------------------------------
# BBFlow
# ------------------------------------------------------------
RUN git clone --depth 1 \
    https://github.com/graeter-group/bbflow.git \
    /opt/bbflow

# Install BBFlow's non-PyTorch requirements.
# Do this BEFORE reinstalling/pinning the PyTorch stack.
RUN python -m pip install \
    -r /opt/bbflow/install_utils/requirements.txt

# ------------------------------------------------------------
# GAFL
# GATr must be patched before installing GAFL.
# ------------------------------------------------------------
RUN git clone --depth 1 \
    https://github.com/hits-mli/gafl.git \
    /opt/gafl && \
    cd /opt/gafl && \
    bash install_gatr.sh && \
    python -m pip install -e .

# ------------------------------------------------------------
# Install BBFlow
# ------------------------------------------------------------
RUN cd /opt/bbflow && \
    python -m pip install -e .

# ------------------------------------------------------------
# Re-pin the critical numerical/PyTorch stack.
# Prevent BBFlow/GAFL requirements from silently changing it.
# ------------------------------------------------------------
RUN python -m pip install --force-reinstall \
    numpy==2.2.6 \
    scipy==1.15.3 \
    scikit-learn==1.7.2 && \
    python -m pip install --force-reinstall \
    torch==2.4.1 \
    torchvision==0.19.1 \
    torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu121 && \
    python -m pip install --force-reinstall \
    torch-scatter \
    -f https://data.pyg.org/whl/torch-2.4.1+cu121.html

# ------------------------------------------------------------
# Download BBFlow multimer-0.2 model without loading it
# ------------------------------------------------------------
RUN python - <<'PY'
from pathlib import Path
import requests

root = Path("/opt/bbflow")
model_dir = root / "models" / "bbflow-multimer-0.2"
model_dir.mkdir(parents=True, exist_ok=True)

ckpt_url = "https://keeper.mpdl.mpg.de/f/6a7b1fedd8ac4dda8e22/?dl=1"
config_url = "https://keeper.mpdl.mpg.de/f/346a25bca784417a8e3d/?dl=1"

ckpt = model_dir / "bbflow-multimer-0.2.ckpt"
config = model_dir / "config.yaml"

def download(url, path):
    print(f"Downloading {url}")
    r = requests.get(
        url,
        stream=True,
        headers={"User-Agent": "Wget/1.21"},
    )
    r.raise_for_status()

    with open(path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

    print(f"Downloaded {path}: {path.stat().st_size / 1024**2:.1f} MB")

download(config_url, config)
download(ckpt_url, ckpt)
PY

RUN python - <<'PY'
from pathlib import Path

p = Path("/opt/bbflow/models/bbflow-multimer-0.2/bbflow-multimer-0.2.ckpt")

assert p.exists(), "Checkpoint does not exist"
assert p.stat().st_size > 10_000_000, f"Checkpoint suspiciously small: {p.stat().st_size} bytes"

with open(p, "rb") as f:
    magic = f.read(4)

assert magic == b"PK\x03\x04", f"Unexpected checkpoint header: {magic!r}"

print(f"BBFlow checkpoint OK: {p.stat().st_size / 1024**2:.1f} MB")
PY
# ------------------------------------------------------------
# Verification
# ------------------------------------------------------------
RUN python - <<'PY'
import torch
import numpy
import scipy
import sklearn
import e3nn
import torch_geometric
import torch_scatter
import gatr
import gafl
import bbflow

print("Python:", __import__("sys").version)
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("NumPy:", numpy.__version__)
print("SciPy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("torch-geometric:", torch_geometric.__version__)
print("torch-scatter:", torch_scatter.__version__)
print("e3nn:", e3nn.__version__)
print("GATr:", getattr(gatr, "__version__", "installed"))
print("GAFL: installed")
print("BBFlow: installed")
PY

CMD ["python"]