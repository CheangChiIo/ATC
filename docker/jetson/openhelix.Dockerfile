# ATC Benchmark Image: OpenHelix (Jetson/L4T)
# GPU: Jetson Orin (ARM64)
# Adjust L4T_TAG to match your JetPack version

ARG L4T_TAG=r36.4.0-pth2.4.0
FROM nvcr.io/nvidia/pytorch:24.10-py3-igpu

ENV PYTHONUNBUFFERED=1 TZ=Asia/Shanghai

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    git wget curl ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 \
    libxrender-dev build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

RUN git config --global http.version HTTP/1.1

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    nsight-systems 2>/dev/null || true \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (adjust per model)
RUN pip install --no-cache-dir \
    tyro "numpy<2" Pillow psutil huggingface_hub \
    "transformers>=4.52.0,<5" "accelerate>=1.0.0" \
    safetensors tokenizers scipy peft diffusers einops typed-argument-parser \
    absl-py blosc sentencepiece markdown2 shortuuid ftfy regex tqdm

RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

RUN pip install --no-cache-dir deepspeed || \
    echo "WARNING: deepspeed install failed; OpenHelix smoke/runtime can continue without it"

RUN mkdir -p /app /checkpoints /datasets /results /repos

COPY shared/ /app/
COPY openhelix/ /app/

WORKDIR /app
ENTRYPOINT ["python"]

CMD ["--help"]
