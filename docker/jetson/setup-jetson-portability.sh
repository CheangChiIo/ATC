#!/usr/bin/env bash
# Prepare a Jetson host for the ATC Docker benchmark package.
# Run from the docker/ directory after copying the ATC package to the target Jetson.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_MOUNT="$(df -P "$PROJECT_ROOT" | awk 'NR==2 {print $6}')"

if [ "${PROJECT_MOUNT:-/}" = "/" ]; then
    DEFAULT_DOCKER_DATA_ROOT="$PROJECT_ROOT/.docker-data"
else
    DEFAULT_DOCKER_DATA_ROOT="$PROJECT_MOUNT/docker"
fi

ATC_DOCKER_DATA_ROOT="${ATC_DOCKER_DATA_ROOT:-$DEFAULT_DOCKER_DATA_ROOT}"
ATC_TMP_DIR="${ATC_TMP_DIR:-$PROJECT_ROOT/.tmp}"
ATC_HOST_CACHE_DIR="${ATC_HOST_CACHE_DIR:-$PROJECT_ROOT/.host-cache}"
ATC_HF_CACHE_DIR="${ATC_HF_CACHE_DIR:-$PROJECT_ROOT/hf_cache}"
ASSUME_YES=0
MOVE_HOST_CACHES=0
PERSIST_NSIGHT=0

usage() {
    cat <<USAGE
Usage: bash jetson/setup-jetson-portability.sh [options]

Options:
  --yes                 Do not ask before restarting Docker.
  --move-host-caches    Move /tmp/nvidia, root pip cache, and user CLIP cache to the ATC disk.
  --persist-nsight      Persist kernel.perf_event_paranoid=1 via /etc/sysctl.d/99-atc-nsight.conf.
  -h, --help            Show this help.

Environment overrides:
  ATC_DOCKER_DATA_ROOT  Docker daemon data-root. Default: same filesystem as ATC project.
  ATC_TMP_DIR           Host tmp directory mounted into containers. Default: PROJECT_ROOT/.tmp
  ATC_HOST_CACHE_DIR    Host cache migration target. Default: PROJECT_ROOT/.host-cache
  ATC_HF_CACHE_DIR      Hugging Face cache target. Default: PROJECT_ROOT/hf_cache
USAGE
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --yes) ASSUME_YES=1 ;;
        --move-host-caches) MOVE_HOST_CACHES=1 ;;
        --persist-nsight) PERSIST_NSIGHT=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

need_sudo() {
    if [ "$(id -u)" -eq 0 ]; then
        "$@"
    else
        sudo "$@"
    fi
}

confirm() {
    if [ "$ASSUME_YES" -eq 1 ]; then
        return 0
    fi
    printf '%s [y/N] ' "$1"
    read -r answer
    case "$answer" in
        y|Y|yes|YES) return 0 ;;
        *) return 1 ;;
    esac
}

move_or_link_dir() {
    local src="$1"
    local dst="$2"
    local owner="${3:-}"
    if [ -L "$src" ]; then
        echo "Already linked: $src -> $(readlink "$src")"
        return 0
    fi
    need_sudo mkdir -p "$(dirname "$dst")"
    if [ -d "$src" ]; then
        echo "Moving $src -> $dst"
        need_sudo mkdir -p "$dst"
        need_sudo rsync -a "$src"/ "$dst"/
        need_sudo rm -rf "$src"
    elif [ -e "$src" ]; then
        echo "Skipping non-directory path: $src"
        return 0
    else
        echo "Creating cache target for $src -> $dst"
        need_sudo mkdir -p "$dst"
    fi
    need_sudo ln -s "$dst" "$src"
    if [ -n "$owner" ]; then
        need_sudo chown -h "$owner" "$src" || true
        need_sudo chown -R "$owner" "$dst" || true
    fi
}

configure_docker_data_root() {
    echo "Configuring Docker data-root: $ATC_DOCKER_DATA_ROOT"
    need_sudo mkdir -p "$ATC_DOCKER_DATA_ROOT"
    need_sudo python3 - "$ATC_DOCKER_DATA_ROOT" <<'PY'
import json
import sys
from pathlib import Path

data_root = sys.argv[1]
path = Path('/etc/docker/daemon.json')
if path.exists():
    try:
        cfg = json.loads(path.read_text() or '{}')
    except json.JSONDecodeError as exc:
        raise SystemExit(f'Cannot parse {path}: {exc}')
else:
    cfg = {}
cfg['data-root'] = data_root
if Path('/usr/bin/nvidia-container-runtime').exists():
    runtimes = cfg.setdefault('runtimes', {})
    runtimes.setdefault('nvidia', {'path': '/usr/bin/nvidia-container-runtime', 'runtimeArgs': []})
    cfg.setdefault('default-runtime', 'nvidia')
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + '\n')
PY
    if command -v systemctl >/dev/null 2>&1; then
        if confirm "Restart Docker now? This stops running containers."; then
            need_sudo systemctl restart docker
        else
            echo "Docker was not restarted. Restart it before building/loading images."
        fi
    else
        echo "systemctl not found; restart Docker manually before building/loading images."
    fi
}

prepare_project_dirs() {
    mkdir -p "$ATC_TMP_DIR" "$ATC_HF_CACHE_DIR" "$PROJECT_ROOT/results" "$PROJECT_ROOT/checkpoints" "$PROJECT_ROOT/datasets" "$PROJECT_ROOT/external_repos"
    echo "Project root:      $PROJECT_ROOT"
    echo "Project mount:     $PROJECT_MOUNT"
    echo "ATC tmp:           $ATC_TMP_DIR"
    echo "HF cache:          $ATC_HF_CACHE_DIR"
    echo "Host cache root:   $ATC_HOST_CACHE_DIR"
}

prepare_nsight_permissions() {
    if [ -r /proc/sys/kernel/perf_event_paranoid ]; then
        value="$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo unknown)"
        echo "perf_event_paranoid=$value"
        case "$value" in
            ''|*[!0-9-]*) ;;
            *)
                if [ "$value" -gt 1 ]; then
                    echo "sudo required: set perf_event_paranoid=1 for Nsight profiling"
                    need_sudo sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid'
                fi
                ;;
        esac
    fi
    if [ "$PERSIST_NSIGHT" -eq 1 ]; then
        echo "Persisting perf_event_paranoid=1"
        printf 'kernel.perf_event_paranoid=1\n' | need_sudo tee /etc/sysctl.d/99-atc-nsight.conf >/dev/null
        need_sudo sysctl --system >/dev/null || true
    fi
    if command -v nsys >/dev/null 2>&1; then
        echo "nsys: $(readlink -f "$(command -v nsys)")"
    else
        echo "WARNING: host nsys not found. run.sh will fall back to container nsys if available."
    fi
}

move_host_caches_if_requested() {
    if [ "$MOVE_HOST_CACHES" -ne 1 ]; then
        echo "Host cache migration skipped. Use --move-host-caches to move /tmp/nvidia and pip/CLIP caches."
        return 0
    fi
    need_sudo mkdir -p "$ATC_HOST_CACHE_DIR/tmp" "$ATC_HOST_CACHE_DIR/root-cache" "$ATC_HOST_CACHE_DIR/user-cache"
    move_or_link_dir /tmp/nvidia "$ATC_HOST_CACHE_DIR/tmp/nvidia" root:root
    move_or_link_dir /root/.cache/pip "$ATC_HOST_CACHE_DIR/root-cache/pip" root:root
    move_or_link_dir "$HOME/.cache/clip" "$ATC_HOST_CACHE_DIR/user-cache/clip" "$(id -un):$(id -gn)"
}

prepare_project_dirs
configure_docker_data_root
prepare_nsight_permissions
move_host_caches_if_requested

echo
echo "Done. Current Docker root:"
docker info --format '  {{.DockerRootDir}}' 2>/dev/null || echo '  Docker not reachable yet.'
echo "Run resource tests with ./run.sh <model> resource ...; sudo prompts are expected when host profiling permissions need changes."
