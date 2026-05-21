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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

_pick_default_input_dir() {
    local docker_dir="$1"
    local repo_dir="$2"
    if [ -d "$docker_dir" ] && [ -n "$(find "$docker_dir" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "$docker_dir"
    elif [ -d "$repo_dir" ]; then
        echo "$repo_dir"
    else
        echo "$docker_dir"
    fi
}

DEFAULT_CHECKPOINT_DIR="$(_pick_default_input_dir "$SCRIPT_DIR/checkpoints" "$PROJECT_ROOT/checkpoints")"
DEFAULT_DATASET_DIR="$(_pick_default_input_dir "$SCRIPT_DIR/datasets" "$PROJECT_ROOT/datasets")"
DEFAULT_RESULT_DIR="$PROJECT_ROOT/results"
if [ ! -d "$DEFAULT_RESULT_DIR" ]; then
    DEFAULT_RESULT_DIR="$SCRIPT_DIR/results"
fi
DEFAULT_REPO_DIR="$PROJECT_ROOT/external_repos"
if [ ! -d "$DEFAULT_REPO_DIR" ]; then
    DEFAULT_REPO_DIR="$SCRIPT_DIR/repos"
fi
DEFAULT_HF_CACHE_DIR="$SCRIPT_DIR/hf_cache"
if [ ! -d "$DEFAULT_HF_CACHE_DIR" ]; then
    DEFAULT_HF_CACHE_DIR="$PROJECT_ROOT/hf_cache"
fi
if [ ! -d "$DEFAULT_HF_CACHE_DIR" ]; then
    DEFAULT_HF_CACHE_DIR="$PROJECT_ROOT/code/.cache/huggingface"
fi

CHECKPOINT_DIR=${CHECKPOINT_DIR:-$DEFAULT_CHECKPOINT_DIR}
DATASET_DIR=${DATASET_DIR:-$DEFAULT_DATASET_DIR}
RESULT_DIR=${RESULT_DIR:-$DEFAULT_RESULT_DIR}
REPO_DIR=${REPO_DIR:-$DEFAULT_REPO_DIR}
HF_CACHE_DIR=${HF_CACHE_DIR:-$DEFAULT_HF_CACHE_DIR}
DEFAULT_ATC_TMP_DIR="$PROJECT_ROOT/.tmp"
ATC_TMP_DIR=${ATC_TMP_DIR:-$DEFAULT_ATC_TMP_DIR}
ATC_HF_OFFLINE=${ATC_HF_OFFLINE:-1}

mkdir -p "$RESULT_DIR" "$ATC_TMP_DIR"
export TMPDIR="$ATC_TMP_DIR"
export TMP="$ATC_TMP_DIR"
export TEMP="$ATC_TMP_DIR"

_warn_if_missing_dir() {
    local label="$1"
    local dir="$2"
    if [ ! -d "$dir" ]; then
        echo "WARNING: $label directory does not exist on host: $dir"
    fi
}

_run_host_root_cmd() {
    if [ "$(id -u)" -eq 0 ]; then
        sh -c "$1"
    else
        echo "sudo required: $2"
        sudo sh -c "$1"
    fi
}

_prepare_nsight_host_permissions() {
    if [ "$TEST_TYPE" != "resource" ]; then
        return 0
    fi

    echo "Nsight resource mode: checking host profiling permissions..."
    if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
        local perf_value
        perf_value="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
        echo "Host perf_event_paranoid=$perf_value"
        case "$perf_value" in
            ''|*[!0-9-]*) ;;
            *)
                if [ "$perf_value" -gt 1 ]; then
                    _run_host_root_cmd \
                        "echo 1 > /proc/sys/kernel/perf_event_paranoid" \
                        "set /proc/sys/kernel/perf_event_paranoid to 1 for Nsight profiling"
                fi
                ;;
        esac
    else
        echo "WARNING: cannot read /proc/sys/kernel/perf_event_paranoid; Nsight counters may be limited."
    fi

    if [ -d /proc/driver/nvidia/params ] && grep -R "RmProfilingAdminOnly: 1" /proc/driver/nvidia/params >/dev/null 2>&1; then
        echo "WARNING: NVIDIA driver reports RmProfilingAdminOnly=1."
        echo "         On Jetson this may require rebooting with a driver/module setting changed; continuing with current permissions."
    fi
}

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

_warn_if_missing_dir "checkpoint" "$CHECKPOINT_DIR"
_warn_if_missing_dir "dataset" "$DATASET_DIR"
_warn_if_missing_dir "result" "$RESULT_DIR"
_warn_if_missing_dir "external repo" "$REPO_DIR"
_warn_if_missing_dir "Hugging Face cache" "$HF_CACHE_DIR"
_warn_if_missing_dir "ATC tmp" "$ATC_TMP_DIR"
_prepare_nsight_host_permissions

NSIGHT_DOCKER_FLAGS=""
CONTAINER_NSYS_PATH=""
if [ "$TEST_TYPE" = "resource" ]; then
    if command -v nsys >/dev/null 2>&1; then
        HOST_NSYS_REAL="$(readlink -f "$(command -v nsys)")"
        case "$HOST_NSYS_REAL" in
            /opt/nvidia/nsight-systems/*)
                NSIGHT_DOCKER_FLAGS="$NSIGHT_DOCKER_FLAGS -v /opt/nvidia/nsight-systems:/opt/nvidia/nsight-systems:ro"
                CONTAINER_NSYS_PATH="$HOST_NSYS_REAL"
                ;;
            /usr/local/cuda/*)
                NSIGHT_DOCKER_FLAGS="$NSIGHT_DOCKER_FLAGS -v /usr/local/cuda:/usr/local/cuda:ro"
                CONTAINER_NSYS_PATH="$HOST_NSYS_REAL"
                ;;
        esac
    fi
    if [ -z "$CONTAINER_NSYS_PATH" ]; then
        if [ "$ATC_ARCH" = "jetson" ]; then
            CONTAINER_NSYS_PATH="/usr/lib/aarch64-linux-gnu/nsight-systems/target-linux-armv8/nsys"
        else
            CONTAINER_NSYS_PATH="/usr/local/cuda/bin/nsys"
        fi
    fi
    echo "Nsight executable inside container: $CONTAINER_NSYS_PATH"
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
# Nsight resource profiling needs host PID visibility. Keep --pid=host for
# latency too so a single image/run path behaves the same on Jetson and x86.
if [ "$ATC_ARCH" = "jetson" ]; then
    GPU_FLAGS="--runtime nvidia --network host --pid=host"
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
    -e VLA_NSIGHT_NSYS_PATH="$CONTAINER_NSYS_PATH" \
    -e TMPDIR=/atc_tmp \
    -e TMP=/atc_tmp \
    -e TEMP=/atc_tmp \
    -e XDG_CACHE_HOME=/hf_cache/.cache \
    -e HF_HOME=/hf_cache \
    -e HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    -e TRANSFORMERS_CACHE=/hf_cache/hub \
    -e HF_HUB_OFFLINE="$ATC_HF_OFFLINE" \
    -e TRANSFORMERS_OFFLINE="$ATC_HF_OFFLINE" \
    -e HUME_REPO_SRC=/repos/hume_repo_check/src \
    -e ROBODUAL_REPO=/repos/RoboDual_AGX \
    -e OPENHELIX_REPO=/repos/openhelix_repo_check \
    -v "$CHECKPOINT_DIR:/checkpoints" \
    -v "$DATASET_DIR:/datasets" \
    -v "$RESULT_DIR:/results" \
    -v "$REPO_DIR:/repos" \
    -v "$HF_CACHE_DIR:/hf_cache" \
    -v "$ATC_TMP_DIR:/atc_tmp" \
    $NSIGHT_DOCKER_FLAGS \
    "$IMAGE" \
    "$SCRIPT" $RESOURCE_ARGS $EXTRA_ARGS
