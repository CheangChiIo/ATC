#!/bin/bash
# ATC Docker Run Script
# Run from the docker/ directory so that default paths
# (./checkpoints ./datasets ./results) are predictable.
#
# Usage: ./run.sh <model_name> <test_type> [additional args]
#
# latency:  runs the latency benchmark, outputs JSON to /results
# resource: runs the resource benchmark with full Nsight profile,
#           outputs JSON + .nsys-rep + .sqlite to /results
#
# Examples:
#   ./run.sh openvla latency --model-id /checkpoints/openvla
#   ./run.sh openvla resource --model-id /checkpoints/openvla

# ── guard: hint if not run from docker/ ───────────────────
if [ ! -d "./x86" ] || [ ! -d "./jetson" ]; then
    echo "NOTE: run.sh is designed to run from the docker/ directory."
    echo "      Default paths (./checkpoints ./datasets ./results)"
    echo "      are relative to the current working directory."
    echo ""
fi

MODEL=${1:-openvla}
TEST_TYPE=${2:-latency}
shift 2 2>/dev/null
EXTRA_ARGS="$@"

# ── architecture auto-detect ──────────────────────────────
# Default from env, otherwise detect.  x86_64 → x86, aarch64 → jetson
if [ -z "$ATC_ARCH" ]; then
    HOST_ARCH="$(uname -m)"
    case "$HOST_ARCH" in
        x86_64|amd64)   ATC_ARCH="x86" ;;
        aarch64|arm64)  ATC_ARCH="jetson" ;;
        *) echo "WARNING: unknown arch $HOST_ARCH, defaulting to x86"; ATC_ARCH="x86" ;;
    esac
fi

CHECKPOINT_DIR=${CHECKPOINT_DIR:-./checkpoints}
DATASET_DIR=${DATASET_DIR:-./datasets}
RESULT_DIR=${RESULT_DIR:-./results}

case "$MODEL" in
    openvla)
        IMAGE="atc-openvla:${ATC_ARCH}-cu128"
        LATENCY_SCRIPT="openvla.py"
        RESOURCE_SCRIPT="openvla_resource.py"
        ;;
    pi05)
        IMAGE="atc-pi05-smolvla:${ATC_ARCH}-cu128"
        LATENCY_SCRIPT="pi05.py"
        RESOURCE_SCRIPT="pi05_resource.py"
        ;;
    smolvla)
        IMAGE="atc-pi05-smolvla:${ATC_ARCH}-cu128"
        LATENCY_SCRIPT="smolvla.py"
        RESOURCE_SCRIPT="smolvla_resource.py"
        ;;
    hume)
        IMAGE="atc-hume-robodual:${ATC_ARCH}-cu128"
        LATENCY_SCRIPT="hume.py"
        RESOURCE_SCRIPT="hume_resource.py"
        ;;
    robodual)
        IMAGE="atc-hume-robodual:${ATC_ARCH}-cu128"
        LATENCY_SCRIPT="robodual.py"
        RESOURCE_SCRIPT="robodual_resource.py"
        ;;
    openhelix)
        IMAGE="atc-openhelix:${ATC_ARCH}-cu128"
        LATENCY_SCRIPT="openhelix.py"
        RESOURCE_SCRIPT="openhelix_resource.py"
        ;;
    *)
        echo "Unknown model: $MODEL"
        echo "Available: openvla, pi05, smolvla, hume, robodual, openhelix"
        exit 1
        ;;
esac

SCRIPT="$LATENCY_SCRIPT"
RESOURCE_ARGS=""
if [ "$TEST_TYPE" = "resource" ]; then
    SCRIPT="$RESOURCE_SCRIPT"
    # Ensure Nsight reports land in the mounted /results volume so they
    # survive container exit.  --nsight-output-dir is supported by all
    # *_resource.py scripts; the wrapper also sets VLA_NSIGHT_OUTPUT_DIR
    # as a fallback for scripts that only read the env variable.
    RESOURCE_ARGS="--nsight-output-dir /results"
    export VLA_NSIGHT_OUTPUT_DIR=/results

    # Full Nsight GPU metrics require:
    #  1. Container runs as root (default in Docker)  → already satisfied
    #  2. --pid=host + --cap-add=SYS_ADMIN             → set below
    #  3. Host perf_event_paranoid <= 1               → check manually:
    #     cat /proc/sys/kernel/perf_event_paranoid
    #     If the value is >=2, run as root on the host:
    #     echo 1 > /proc/sys/kernel/perf_event_paranoid
    #  4. Some drivers set RmProfilingAdminOnly=1      → requires host root
    # Without these, Nsight may only collect CUDA timeline without
    # full SM utilization and GPU counter metrics.
fi

echo "Running: $MODEL ($TEST_TYPE)  |  arch: $ATC_ARCH ($(uname -m))"
echo "Image:  $IMAGE"
echo "Script: python $SCRIPT $RESOURCE_ARGS $EXTRA_ARGS"

# Auto-detect whether sudo is needed for docker
DOCKER_CMD="docker"
if ! docker info > /dev/null 2>&1; then
    echo "Not in docker group, trying sudo (prompts for root password)..."
    DOCKER_CMD="sudo docker"
fi

# ── platform-specific docker flags ────────────────────────
# Jetson uses --runtime nvidia + --network host, x86 uses --gpus all + --pid host
if [ "$ATC_ARCH" = "jetson" ]; then
    GPU_FLAGS="--runtime nvidia --network host"
else
    GPU_FLAGS="--gpus all --pid=host"
fi

$DOCKER_CMD run --rm $GPU_FLAGS \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --cap-add=SYS_ADMIN \
    --security-opt seccomp=unconfined \
    -e VLA_NSIGHT_OUTPUT_DIR=/results \
    -v "$CHECKPOINT_DIR:/checkpoints" \
    -v "$DATASET_DIR:/datasets" \
    -v "$RESULT_DIR:/results" \
    "$IMAGE" \
    python "$SCRIPT" $RESOURCE_ARGS $EXTRA_ARGS
