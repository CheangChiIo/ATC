# ATC Benchmark Image: Hume + RoboDual (Jetson/L4T)
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
    "transformers>=4.52.0,<5" accelerate safetensors tokenizers \
    timm einops beartype jaxtyping hydra-core omegaconf \
    ema_pytorch rotary-embedding-torch json_numpy sentencepiece draccus einops-exts

# Hume imports a py310-compatible LeRobot tree; install without allowing it to
# replace the NVIDIA PyTorch wheel in the Jetson base image.
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/huggingface/lerobot@768e36660d1408c71118a2760f831c037fbfa17d

# Runtime dependencies imported by LeRobot/Hume. Keep torch/torchvision/cv2
# from the NVIDIA Jetson base image; do not install packages that replace them.
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    "datasets==4.4.1" "deepdiff==8.6.2" "gymnasium==0.29.1" \
    "h5py>=3.10.0" "imageio==2.37.3" "termcolor==3.3.0" \
    "av==15.0.0" "zarr>=2.17.0,<3" "packaging==25.0" \
    "pyzmq>=26.2.1" "flask>=3.0.3" "gdown>=5.1.0"

# Install RoboDual/OpenVLA runtime dependencies without replacing PyTorch.
RUN pip install --no-cache-dir --upgrade-strategy only-if-needed \
    diffusers scipy matplotlib rich jsonlines protobuf pyyaml \
    "numpy<2"

# OpenVLA / Prismatic (RoboDual also mounts /repos/RoboDual_AGX, which contains prismatic)
RUN pip install --no-cache-dir \
    git+https://github.com/openvla/openvla.git || \
    echo "WARNING: openvla install failed; mounted RoboDual/OpenVLA code may still be used"

# RoboDual imports PEFT during inference setup; install it after the cached
# LeRobot/OpenVLA dependency layers to avoid invalidating slow GitHub pulls.
RUN pip install --no-cache-dir peft

RUN mkdir -p /app /checkpoints /datasets /results /repos

COPY shared/ /app/
COPY hume-robodual/ /app/

WORKDIR /app
ENTRYPOINT ["python"]

CMD ["--help"]
