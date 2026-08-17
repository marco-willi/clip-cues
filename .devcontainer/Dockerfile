FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# System dependencies. CUDA is deliberately NOT installed here: the cu128 torch wheels that
# `uv sync --group gpu` installs into .venv bundle their own CUDA runtime and cuDNN, and the
# GPU driver is injected at runtime by the NVIDIA Container Toolkit (see .devcontainer/gpu/).
# What remains are the non-CUDA runtime deps: image codecs for PIL/OpenCV, build tools, make.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget ca-certificates git build-essential pkg-config \
    ffmpeg libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 \
    make graphviz \
    openssh-client rsync \
    bash bash-completion \
    && rm -rf /var/lib/apt/lists/*

# uv drives dependency management (pyproject.toml + uv.lock). Pre-installed so post-create
# can sync immediately.
RUN pip install --upgrade pip uv

WORKDIR /workspace

# JupyterLab, TensorBoard
EXPOSE 8888 6006

CMD ["/bin/bash"]
