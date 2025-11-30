# Dockerfile - reproducible environment for QLoRA / LoRA training
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-lc"]

# Minimal system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl wget ca-certificates python3 python3-dev python3-pip python3-venv \
    libssl-dev libcurl4-openssl-dev libxml2-dev pkg-config unzip zip \
    && rm -rf /var/lib/apt/lists/*

# Use python3 as default python
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Upgrade pip, install wheel
RUN python -m pip install --upgrade pip wheel setuptools

# Pin protobuf first to avoid GetPrototype errors
RUN python -m pip install --no-cache-dir "protobuf==3.20.3"

# Compatible HF stack that works together
RUN python -m pip install --no-cache-dir \
    "huggingface_hub==0.19.4" \
    "transformers==4.38.2" \
    "accelerate==0.20.3" \
    "peft==0.5.0" \
    "sentence-transformers==2.2.2" \
    "datasets" \
    "safetensors==0.3.0" \
    "tokenizers" \
    "tqdm"

# bitsandbytes wheel (attempt install)
# bitsandbytes often needs CUDA11.8-compatible wheel; pip will try to pull appropriate wheel.
RUN python -m pip install --no-cache-dir bitsandbytes || true

# Additional niceties
RUN python -m pip install --no-cache-dir ftfy regex

# create workdir and copy repo (user will mount instead if desired)
WORKDIR /workspace
COPY . /workspace

# set PATH
ENV PYTHONPATH=/workspace:$PYTHONPATH

# default command
CMD ["bash"]
