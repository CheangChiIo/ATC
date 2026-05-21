# ATC Benchmark Image: OpenVLA
# Models: OpenVLA (openvla.py, openvla_resource.py)
# GPU: RTX 4090, RTX 5090 (sm_89, sm_120)
# Requires: nvidia-container-toolkit on host

FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    TZ=Asia/Shanghai \
    CUDA_HOME=/usr/local/cuda \
    PATH=/usr/local/cuda/bin:$PATH

# System dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    git wget curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && python -m pip install --upgrade pip setuptools wheel

# Install Nsight Systems CLI (for resource profiling with .nsys-rep)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    nsight-systems-2025.3.2 2>/dev/null || \
    (wget -qO- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub | \
     apt-key add - && \
     add-apt-repository -y "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
     apt-get update -qq && \
     apt-get install -y --no-install-recommends nsight-systems-2025.3.2) || true \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.8
RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install OpenVLA dependencies
RUN pip install --no-cache-dir \
    "transformers>=4.52.0,<5" \
    "accelerate>=1.0.0" \
    "peft>=0.14.0" \
    safetensors \
    timm \
    einops \
    tyro \
    numpy \
    Pillow \
    psutil \
    huggingface_hub

# Install flash-attn (pre-built wheel for CUDA 12.8 + sm_120)
RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
    pip install --no-cache-dir flash-attn || \
    echo "WARNING: flash-attn installation failed; falling back to SDPA"

# Install OpenVLA / Prismatic from source
RUN pip install --no-cache-dir \
    git+https://github.com/openvla/openvla.git || \
    echo "WARNING: openvla pip install failed; will use mounted code instead"

# Create directories
RUN mkdir -p /app /checkpoints /datasets /results

# Copy ATC benchmark code (all files are in this build context)
COPY . /app/

WORKDIR /app

# Verify imports
RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

ENTRYPOINT ["python"]
CMD ["openvla.py", "--help"]
