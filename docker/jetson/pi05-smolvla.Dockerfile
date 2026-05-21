# ATC Benchmark Image: pi0.5 + SmolVLA (Jetson/L4T)
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

# Install Python dependencies without replacing NVIDIA's preinstalled PyTorch.
RUN pip install --no-cache-dir \
    tyro "numpy<2" Pillow psutil huggingface_hub \
    "transformers>=4.52.0,<5" accelerate safetensors tokenizers

# Jetson PyTorch image is Python 3.10. Current openpi/lerobot main branches
# require newer Python, so pin the LeRobot release that still supports py310
# and install only the SmolVLA runtime dependencies needed by smolvla.py.
RUN pip install --no-cache-dir --no-deps "lerobot==0.4.4" && \
    pip install --no-cache-dir --upgrade-strategy only-if-needed \
      "datasets==4.4.1" "diffusers==0.35.2" "huggingface-hub==0.36.2" \
      "accelerate==1.13.0" "einops==0.8.2" \
      "jsonlines==4.0.0" "packaging==25.0" "draccus==0.10.0" \
      "gymnasium==1.3.0" "deepdiff==8.6.2" "imageio==2.37.3" \
      "termcolor==3.3.0" "num2words==0.5.14" "pyserial==3.5" \
      "av==15.0.0" "safetensors<1" && \
    pip install --no-cache-dir "numpy<2"

RUN mkdir -p /app /checkpoints /datasets /results /repos

COPY shared/ /app/
COPY pi05-smolvla/ /app/

WORKDIR /app
ENTRYPOINT ["python"]

CMD ["--help"]
