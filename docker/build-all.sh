#!/bin/bash
# ATC Docker Image Build Script
# Must be run from the docker/ directory (where x86/ and jetson/ live).
#
# Usage:  ./build-all.sh              # build all 4 x86 images
#         ./build-all.sh --jetson     # build all 4 Jetson (ARM64) images
#         ./build-all.sh openvla      # build single model (x86)
#         ./build-all.sh --jetson openvla
#
# Requires: docker, nvidia-container-toolkit (x86)
# Run on a machine with good internet access (>= 50 Mbps recommended)

set -e

# ── guard: must be run from docker/ directory ─────────────
if [ ! -d "./x86" ] || [ ! -d "./jetson" ]; then
    echo "ERROR: build-all.sh must be run from the docker/ directory"
    echo "       where x86/ and jetson/ subdirectories exist."
    echo ""
    echo "  cd /path/to/ATC/docker"
    echo "  ./build-all.sh"
    exit 1
fi

# ── defaults ──────────────────────────────────────────────
TARGET="x86"
MODEL_FILTER=""

# ── parse args ────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --jetson) TARGET="jetson" ;;
        --help|-h)
            echo "Usage: ./build-all.sh [--jetson] [model_name]"
            echo ""
            echo "  (none)          build all 4 x86 images"
            echo "  --jetson        build all 4 Jetson (ARM64) images"
            echo ""
            echo "  model_name:  openvla | pi05-smolvla | hume-robodual | openhelix"
            echo ""
            echo "Examples:"
            echo "  ./build-all.sh                      # all x86"
            echo "  ./build-all.sh --jetson             # all Jetson"
            echo "  ./build-all.sh openvla              # x86 openvla only"
            echo "  ./build-all.sh --jetson openvla     # Jetson openvla only"
            exit 0
            ;;
        *) MODEL_FILTER="$arg" ;;
    esac
done

# ── prechecks ─────────────────────────────────────────────
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker daemon not reachable."
    echo "  sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

if [ "$TARGET" = "x86" ]; then
    if ! nvidia-ctk --version > /dev/null 2>&1; then
        echo "WARNING: nvidia-container-toolkit not found, GPU access may not work."
    fi
fi

# ── build ─────────────────────────────────────────────────
MODELS=("openvla" "pi05-smolvla" "hume-robodual" "openhelix")

echo "========================================="
echo "ATC Docker Build"
echo "Target:  $TARGET"
echo "Models:  ${MODEL_FILTER:-all}"
echo "========================================="

for model in "${MODELS[@]}"; do
    if [ -n "$MODEL_FILTER" ] && [ "$model" != "$MODEL_FILTER" ]; then
        continue
    fi

    IMAGE="atc-${model}:${TARGET}-cu128"

    if [ "$TARGET" = "x86" ]; then
        echo ""
        echo "=== Building $IMAGE ==="
        cd "x86/$model"
        docker build -t "$IMAGE" .
        cd ../..
        echo "$IMAGE: DONE"

    elif [ "$TARGET" = "jetson" ]; then
        DOCKERFILE="jetson/${model}.Dockerfile"
        if [ ! -f "$DOCKERFILE" ]; then
            echo "WARNING: $DOCKERFILE not found, skipping $IMAGE"
            continue
        fi
        L4T_TAG="${L4T_TAG:-r36.4.0-pth2.4.0}"
        echo ""
        echo "=== Building $IMAGE (L4T_TAG=$L4T_TAG) ==="
        docker build -f "$DOCKERFILE" \
            --build-arg "L4T_TAG=$L4T_TAG" \
            -t "$IMAGE" \
            shared/
        echo "$IMAGE: DONE"
    fi
done

echo ""
echo "========================================="
echo "ALL BUILDS COMPLETE"
echo "========================================="
echo ""
echo "Export for distribution:"
echo "  docker save atc-openvla:x86-cu128 | gzip > atc-openvla-x86-cu128.tar.gz"
echo ""
echo "See README.md for usage instructions."
