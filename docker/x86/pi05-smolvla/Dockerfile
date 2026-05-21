# ATC Benchmark Image: pi0.5 + SmolVLA
# Models: pi0.5 (pi05.py, pi05_resource.py), SmolVLA (smolvla.py, smolvla_resource.py)
# GPU: RTX 4090, RTX 5090 (sm_89, sm_120)
# Requires: nvidia-container-toolkit on host

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    CUDA_HOME=/usr/local/cuda

# System dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git wget curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

# Install Nsight Systems CLI
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    nsight-systems-2025.3.2 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install common dependencies
RUN pip install --no-cache-dir \
    tyro numpy Pillow psutil huggingface_hub \
    "transformers>=4.52.0,<5" tokenizers \
    accelerate safetensors

# Install pi0.5 dependencies (openpi)
RUN pip install --no-cache-dir \
    git+https://github.com/physical-intelligence/openpi.git || \
    echo "WARNING: openpi install failed; mount repo at /repos/openpi"

# Install SmolVLA dependencies (lerobot)
RUN pip install --no-cache-dir \
    git+https://github.com/huggingface/lerobot.git || \
    echo "WARNING: lerobot install failed; mount repo at /repos/lerobot"

# Create directories
RUN mkdir -p /app /checkpoints /datasets /results /repos

# Copy ATC benchmark code (all files are in this build context)
COPY . /app/

WORKDIR /app

RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

ENTRYPOINT ["python"]
CMD ["--help"]
