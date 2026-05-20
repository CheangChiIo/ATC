# ATC Benchmark Image: OpenVLA (Jetson/L4T)
# GPU: Jetson Orin (ARM64)
# Base: NVIDIA L4T PyTorch container
# Adjust the tag to match your JetPack version:
#   JetPack 6.0: l4t-pytorch:r36.3.0-pth2.1.0
#   JetPack 6.1: l4t-pytorch:r36.4.0-pth2.4.0
# Run: docker run --runtime nvidia --network host --ipc=host \
#        -v /data/checkpoints:/checkpoints -v /data/datasets:/datasets \
#        atc-openvla:jetson-l4t

ARG L4T_TAG=r36.4.0-pth2.4.0
FROM nvcr.io/nvidia/l4t-pytorch:${L4T_TAG}

ENV PYTHONUNBUFFERED=1 \
    TZ=Asia/Shanghai \
    CUDA_HOME=/usr/local/cuda

# Install system dependencies
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender-dev \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Nsight Systems CLI (Jetson version)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    nsight-systems 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

# Install OpenVLA dependencies (use pre-installed PyTorch from L4T image)
RUN pip install --no-cache-dir \
    transformers>=4.52.0 accelerate>=1.0.0 peft>=0.14.0 \
    safetensors timm einops tyro numpy Pillow psutil \
    huggingface_hub

# flash-attn for Jetson ARM64 - may need special build
# RUN pip install --no-cache-dir flash-attn --no-build-isolation || \
#     echo "WARNING: flash-attn not available for ARM64; use --attn-implementation sdpa"

# OpenVLA / Prismatic
RUN pip install --no-cache-dir \
    git+https://github.com/openvla/openvla.git || \
    echo "WARNING: openvla install failed"

RUN mkdir -p /app /checkpoints /datasets /results

COPY shared/ /app/
COPY openvla/ /app/

WORKDIR /app

RUN python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

ENTRYPOINT ["python"]
CMD ["openvla.py", "--help"]
