# ALFWorld EnvTuning Experiments
# Base: CUDA 12.4 + Ubuntu 22.04 (matches torch 2.6.0+cu124)
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git wget curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# Create venv
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip

# Install Python deps (frozen from working environment)
COPY verl-agent/requirements_frozen.txt /tmp/requirements_frozen.txt
RUN pip install --no-cache-dir -r /tmp/requirements_frozen.txt || true

# Install flash-attn prebuilt wheel
RUN pip install --no-cache-dir https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

# Copy code
COPY verl-agent /workspace/verl-agent
COPY alfworld_augment /workspace/alfworld_augment

# Install verl-agent
WORKDIR /workspace/verl-agent
RUN pip install -e .

# Download ALFWorld game data
RUN python -c "import subprocess; subprocess.run(['alfworld-download'], check=True)"

# Environment variables
ENV CUDA_HOME="/usr/local/cuda"
ENV VLLM_ATTENTION_BACKEND=XFORMERS
ENV WANDB_API_KEY="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"

# Model is mounted at runtime, not baked into image (too large)
# docker run --gpus all -v /path/to/Qwen3-8B:/models/Qwen3-8B ...

WORKDIR /workspace/verl-agent
CMD ["bash"]
