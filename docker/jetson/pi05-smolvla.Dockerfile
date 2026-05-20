# ATC Benchmark Image: pi0.5 + SmolVLA (Jetson/L4T)
# GPU: Jetson Orin (ARM64)
# Adjust L4T_TAG to match your JetPack version

ARG L4T_TAG=r36.4.0-pth2.4.0
FROM nvcr.io/nvidia/l4t-pytorch:${{L4T_TAG}}

ENV PYTHONUNBUFFERED=1 TZ=Asia/Shanghai

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 \
    libxrender-dev build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    nsight-systems 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (adjust per model)
RUN pip install --no-cache-dir \
    tyro numpy Pillow psutil huggingface_hub \
    transformers>=4.52.0 accelerate safetensors

RUN mkdir -p /app /checkpoints /datasets /results

COPY shared/ /app/
# COPY model-specific files here

WORKDIR /app
ENTRYPOINT ["python"]

CMD ["--help"]
