from __future__ import annotations

import bisect
from contextlib import contextmanager
import json
import math
import os
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch


LEGACY_COMPONENT_STAGE_KEYS = (
    "vision_encoder",
    "llm_backbone",
    "action_expert",
    "dit",
)
DUAL_SYSTEM_COMPONENT_STAGE_KEYS = (
    "system2_vision_encoder",
    "system2_inference",
    "system_bridge",
    "system1_vision_encoder",
    "system1_action_expert",
)
LEGACY_RESOURCE_STAGE_KEYS = (
    "data_processing",
    *LEGACY_COMPONENT_STAGE_KEYS,
    "e2e",
)
DUAL_SYSTEM_RESOURCE_STAGE_KEYS = (
    "data_processing",
    *DUAL_SYSTEM_COMPONENT_STAGE_KEYS,
    "e2e",
)
# Backward-compatible union constants. New code should use infer_stage_keys()
# or pass stage_keys/component_stage_keys explicitly when producing JSON.
COMPONENT_STAGE_KEYS = LEGACY_COMPONENT_STAGE_KEYS + DUAL_SYSTEM_COMPONENT_STAGE_KEYS
RESOURCE_STAGE_KEYS = (
    "data_processing",
    *LEGACY_COMPONENT_STAGE_KEYS,
    *DUAL_SYSTEM_COMPONENT_STAGE_KEYS,
    "e2e",
)

# Keep the old primary metric keys that downstream plotting scripts are likely
# to read.  CPU is measured inline with psutil process.cpu_times() deltas; GPU/memory are patched in from
# Nsight.  cpu_process_percent_mean is normalized to the available logical CPU
# capacity so it stays in the familiar 0-100% range.  The raw multicore
# "core-equivalent" value is still emitted as cpu_process_core_percent_mean.
RESOURCE_SERIES_KEYS = (
    # Duration is kept only as the measurement-window denominator for resource
    # metrics.  Official latency should come from the separate latency scripts.
    "duration_ms",
    "gpu_profile_duration_ms",
    "cpu_pytorch_duration_ms",
    "cpu_process_percent_mean",
    "cpu_process_core_percent_mean",
    # Nsight GPU profile metrics.  gpu_util_percent_mean is kept for backward
    # compatibility; it is a GPU kernel-busy time ratio, not NVML sampling.
    "gpu_util_percent_mean",
    "gpu_busy_ratio_percent_mean",
    "sm_util_percent_mean",
    "gpu_memory_used_mb_max",
    "nsight_cuda_memory_peak_mb",
    # PyTorch allocator metrics measured in non-profiled passes.
    "torch_memory_allocated_mb_start",
    "torch_memory_allocated_mb_end",
    "torch_memory_allocated_mb_peak",
    "torch_memory_allocated_mb_increment_peak",
    "torch_memory_reserved_mb_start",
    "torch_memory_reserved_mb_end",
    "torch_memory_reserved_mb_peak",
    "torch_memory_reserved_mb_increment_peak",
    "sample_count",
)

POSTPROCESSED_RESOURCE_KEYS = RESOURCE_SERIES_KEYS

_NSIGHT_LAUNCHED_ENV = "VLA_RESOURCE_NSIGHT_ACTIVE"
_NSIGHT_REPORT_ENV = "VLA_RESOURCE_NSIGHT_REPORT_PATH"
_NSIGHT_OUTPUT_DIR = os.environ.get("VLA_NSIGHT_OUTPUT_DIR", "nsight_reports")
_NSIGHT_GPU_METRICS_DEVICE = os.environ.get("VLA_NSIGHT_GPU_METRICS_DEVICE", "all")
_NSIGHT_TRACE = os.environ.get("VLA_NSIGHT_TRACE", "cuda,nvtx,osrt")
_NSIGHT_EXTRA_ARGS = os.environ.get("VLA_NSIGHT_EXTRA_ARGS", "")
_NSIGHT_NSYS_PATH = os.environ.get("VLA_NSIGHT_NSYS_PATH", "") or None
_NSIGHT_REPORT_PATH = os.environ.get(_NSIGHT_REPORT_ENV)
_RESOURCE_BACKEND = os.environ.get("VLA_RESOURCE_BACKEND", "full")

_STAGE_NVTX_TO_KEY = {
    "data_processing": "data_processing",
    "e2e": "e2e",
    "component/vision_encoder": "vision_encoder",
    "component/llm_backbone": "llm_backbone",
    "component/action_expert": "action_expert",
    "component/dit": "dit",
    "component/system2_vision_encoder": "system2_vision_encoder",
    "component/system2_inference": "system2_inference",
    "component/system_bridge": "system_bridge",
    "component/system1_vision_encoder": "system1_vision_encoder",
    "component/system1_action_expert": "system1_action_expert",
}


def infer_stage_keys(
    components: dict[str, Any] | None = None,
    model_info: dict[str, Any] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Return (resource_stage_keys, component_stage_keys, stage_schema).

    Legacy scripts keep the old resource JSON clean by default. Dual-system
    scripts are selected when model_info['schema_family'] == 'dual_system' or
    when dual-system component keys are present in the measured components.
    """
    model_info = model_info or {}
    components = components or {}
    schema_family = str(model_info.get("schema_family") or "").lower()
    has_dual = schema_family in {"dual_system", "dual-system", "dual"}
    if not has_dual:
        has_dual = any(
            key in components or f"{key}_calls" in components
            for key in DUAL_SYSTEM_COMPONENT_STAGE_KEYS
        )
    if has_dual:
        return DUAL_SYSTEM_RESOURCE_STAGE_KEYS, DUAL_SYSTEM_COMPONENT_STAGE_KEYS, "dual_system"
    return LEGACY_RESOURCE_STAGE_KEYS, LEGACY_COMPONENT_STAGE_KEYS, "legacy"


def _available_logical_cpu_count() -> int:
    try:
        if hasattr(os, "sched_getaffinity"):
            return max(1, len(os.sched_getaffinity(0)))
    except Exception:
        pass
    try:
        import psutil

        return max(1, int(psutil.cpu_count(logical=True) or 1))
    except Exception:
        return max(1, int(os.cpu_count() or 1))


def _normalize_backend(backend: str | None) -> str:
    """Normalize resource orchestration backend.

    full (default): run the complete three-run resource protocol:
      1) Nsight GPU profile for GPU busy / SM / CUDA memory;
      2) non-profiled E2E CPU + PyTorch allocator memory;
      3) non-profiled component CPU + PyTorch allocator memory.

    nsight: run only the Nsight GPU profile path.
    cpu_pytorch: run only the non-profiled CPU + PyTorch allocator path.
    none: keep only boundary duration/NVTX markers for debugging.
    """
    value = (backend or "full").strip().lower()
    aliases = {
        "default": "full",
        "all": "full",
        "hybrid": "full",
        "final": "full",
        "nsys": "nsight",
        "pytorch": "cpu_pytorch",
        "cpu": "cpu_pytorch",
        "cpu-pytorch": "cpu_pytorch",
        "disable": "none",
        "disabled": "none",
        "off": "none",
        "false": "none",
        "0": "none",
    }
    value = aliases.get(value, value)
    if value not in {"full", "nsight", "cpu_pytorch", "none"}:
        raise ValueError(
            f"Unsupported resource_sampler_backend={backend!r}. Use 'full', 'nsight', 'cpu_pytorch', or 'none'."
        )
    return value


def configure_resource_sampler(
    backend: str = "full",
    *,
    nsight_output_dir: str | None = None,
    nsight_gpu_metrics_device: str = "all",
    nsight_trace: str = "cuda,nvtx,osrt",
    nsight_extra_args: str = "",
    nsight_nsys_path: str | None = None,
    **_: Any,
):
    """Configure resource measurement.

    The default `full` backend produces a merged JSON from three independent
    runs so intrusive Nsight profiling does not contaminate CPU/PyTorch memory
    statistics.  Nsight-derived metrics are used only for GPU timeline/hardware
    counters and CUDA memory residency.  CPU and PyTorch allocated/reserved
    memory are measured in regular non-profiled runs.
    """
    global _RESOURCE_BACKEND
    global _NSIGHT_OUTPUT_DIR
    global _NSIGHT_GPU_METRICS_DEVICE
    global _NSIGHT_TRACE
    global _NSIGHT_EXTRA_ARGS
    global _NSIGHT_NSYS_PATH

    _RESOURCE_BACKEND = _normalize_backend(backend)
    if nsight_output_dir:
        _NSIGHT_OUTPUT_DIR = nsight_output_dir
    _NSIGHT_GPU_METRICS_DEVICE = str(nsight_gpu_metrics_device or "").strip()
    _NSIGHT_TRACE = str(nsight_trace or "cuda,nvtx,osrt").strip()
    _NSIGHT_EXTRA_ARGS = str(nsight_extra_args or "").strip()
    _NSIGHT_NSYS_PATH = nsight_nsys_path or _NSIGHT_NSYS_PATH


def requested_resource_backend() -> str:
    return _normalize_backend(_RESOURCE_BACKEND)


def resource_run_kind() -> str:
    """Return the current child/orchestrator run kind."""
    return os.environ.get("VLA_RESOURCE_RUN_KIND", "direct").strip().lower() or "direct"


def inline_resource_backend() -> str:
    return "psutil-cpu-times-delta+pytorch-allocator-peaks"


def nsight_is_active() -> bool:
    return os.environ.get(_NSIGHT_LAUNCHED_ENV) == "1"


def should_measure_data_processing(args: Any) -> bool:
    kind = resource_run_kind()
    if kind == "cpu_pytorch_e2e":
        return False
    if kind == "cpu_pytorch_components":
        return bool(getattr(args, "measure_data_processing", False))
    return bool(getattr(args, "measure_data_processing", False))


def should_measure_e2e(args: Any) -> bool:
    kind = resource_run_kind()
    if kind == "cpu_pytorch_components":
        return False
    if kind == "cpu_pytorch_e2e":
        return bool(getattr(args, "measure_e2e", False))
    return bool(getattr(args, "measure_e2e", False))


def should_measure_components(args: Any) -> bool:
    kind = resource_run_kind()
    if kind == "cpu_pytorch_e2e":
        return False
    if kind == "cpu_pytorch_components":
        return bool(getattr(args, "measure_components", False))
    return bool(getattr(args, "measure_components", False))


def _make_nsight_report_base(report_prefix: str, output_dir: str | None = None) -> str:
    out_dir = Path(output_dir or _NSIGHT_OUTPUT_DIR).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_prefix = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in report_prefix)
    return str(out_dir / f"{safe_prefix}_{timestamp}_{os.getpid()}")


def nsight_report_path() -> str | None:
    return _NSIGHT_REPORT_PATH or os.environ.get(_NSIGHT_REPORT_ENV)


def nsight_metadata() -> dict[str, Any]:
    report = nsight_report_path()
    return {
        "requested": requested_resource_backend() in {"full", "nsight"},
        "active": nsight_is_active(),
        "run_kind": resource_run_kind(),
        "nsys_path": _NSIGHT_NSYS_PATH or shutil.which("nsys"),
        "report_path": report,
        "expected_report_file": f"{report}.nsys-rep" if report else None,
        "gpu_metrics_device": _NSIGHT_GPU_METRICS_DEVICE or None,
        "trace": _NSIGHT_TRACE,
        "extra_args": _NSIGHT_EXTRA_ARGS,
        "inline_fallback_backend": inline_resource_backend(),
        "resource_metrics_source": "three_run_full_resource_protocol" if requested_resource_backend() == "full" else "single_run_resource_protocol",
    }


def _append_output_json_arg(argv: list[str], output_json: str) -> list[str]:
    """Return argv with a single output-json value, compatible with argparse/tyro."""
    result: list[str] = []
    skip_next = False
    replaced = False
    for idx, item in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if item in {"--output-json", "--output_json"}:
            result.extend([item, output_json])
            skip_next = True
            replaced = True
            continue
        if item.startswith("--output-json="):
            result.append(f"--output-json={output_json}")
            replaced = True
            continue
        if item.startswith("--output_json="):
            result.append(f"--output_json={output_json}")
            replaced = True
            continue
        result.append(item)
    if not replaced:
        result.extend(["--output-json", output_json])
    return result


def _child_env(kind: str, backend: str, *, report_base: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["VLA_RESOURCE_RUN_KIND"] = kind
    env["VLA_RESOURCE_BACKEND"] = backend
    env["VLA_NSIGHT_OUTPUT_DIR"] = _NSIGHT_OUTPUT_DIR
    env["VLA_NSIGHT_GPU_METRICS_DEVICE"] = _NSIGHT_GPU_METRICS_DEVICE
    env["VLA_NSIGHT_TRACE"] = _NSIGHT_TRACE
    env["VLA_NSIGHT_EXTRA_ARGS"] = _NSIGHT_EXTRA_ARGS
    if _NSIGHT_NSYS_PATH:
        env["VLA_NSIGHT_NSYS_PATH"] = _NSIGHT_NSYS_PATH
    if report_base is not None:
        env[_NSIGHT_REPORT_ENV] = report_base
    return env


def _run_plain_child(kind: str, output_json: str) -> None:
    cmd = [sys.executable, *_append_output_json_arg(sys.argv, output_json)]
    env = _child_env(kind, "cpu_pytorch")
    print("\n" + "=" * 100)
    print(f"Launching non-profiled {kind} resource pass.")
    print("Command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    print("=" * 100 + "\n")
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _run_nsight_child(nsys: str, report_base: str, output_json: str) -> None:
    cmd = [
        nsys,
        "profile",
        "--force-overwrite=true",
        f"--trace={_NSIGHT_TRACE}",
        "--sample=process-tree",
        "--cpuctxsw=process-tree",
        "--cuda-memory-usage=true",
    ]
    if _NSIGHT_GPU_METRICS_DEVICE and _NSIGHT_GPU_METRICS_DEVICE.lower() not in {"none", "false", "off", "0"}:
        cmd.append(f"--gpu-metrics-devices={_NSIGHT_GPU_METRICS_DEVICE}")
    if _NSIGHT_EXTRA_ARGS:
        cmd.extend(shlex.split(_NSIGHT_EXTRA_ARGS))
    cmd.extend(["--output", report_base, sys.executable, *_append_output_json_arg(sys.argv, output_json)])

    env = _child_env("nsight_gpu", "nsight", report_base=report_base)
    env[_NSIGHT_LAUNCHED_ENV] = "1"

    print("\n" + "=" * 100)
    print("Launching Nsight GPU resource pass. CPU/PyTorch allocator values from this pass are not used as main metrics.")
    print(f"Nsight report base: {report_base}")
    print("Command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    print("=" * 100 + "\n")
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def maybe_reexec_under_nsight(args: Any, report_prefix: str) -> list[str]:
    """Launch the requested resource protocol before the model is loaded.

    In default `full` mode this parent process runs three independent passes and
    then writes one merged JSON:
      - Nsight GPU pass: GPU busy ratio, SM utilization, CUDA memory peak.
      - regular E2E pass: CPU utilization + PyTorch allocated/reserved peak.
      - regular component pass: per-stage CPU utilization + PyTorch allocator peaks.

    Child passes return immediately from this function and execute the unchanged
    model/stage code below.
    """
    warnings: list[str] = []
    kind = resource_run_kind()
    if kind in {"nsight_gpu", "cpu_pytorch_e2e", "cpu_pytorch_components", "cpu_pytorch_all"}:
        configure_resource_sampler(
            backend=os.environ.get("VLA_RESOURCE_BACKEND", _RESOURCE_BACKEND),
            nsight_output_dir=os.environ.get("VLA_NSIGHT_OUTPUT_DIR", _NSIGHT_OUTPUT_DIR),
            nsight_gpu_metrics_device=os.environ.get("VLA_NSIGHT_GPU_METRICS_DEVICE", _NSIGHT_GPU_METRICS_DEVICE),
            nsight_trace=os.environ.get("VLA_NSIGHT_TRACE", _NSIGHT_TRACE),
            nsight_extra_args=os.environ.get("VLA_NSIGHT_EXTRA_ARGS", _NSIGHT_EXTRA_ARGS),
            nsight_nsys_path=os.environ.get("VLA_NSIGHT_NSYS_PATH", "") or _NSIGHT_NSYS_PATH,
        )
        return warnings

    configure_resource_sampler(
        backend=getattr(args, "resource_sampler_backend", _RESOURCE_BACKEND),
        nsight_output_dir=getattr(args, "nsight_output_dir", _NSIGHT_OUTPUT_DIR),
        nsight_gpu_metrics_device=getattr(args, "nsight_gpu_metrics_device", _NSIGHT_GPU_METRICS_DEVICE),
        nsight_trace=getattr(args, "nsight_trace", _NSIGHT_TRACE),
        nsight_extra_args=getattr(args, "nsight_extra_args", _NSIGHT_EXTRA_ARGS),
        nsight_nsys_path=getattr(args, "nsight_nsys_path", _NSIGHT_NSYS_PATH),
    )
    if nsight_is_active():
        return warnings

    backend = requested_resource_backend()
    if backend == "none":
        warnings.append("resource_sampler_backend='none': only boundary duration markers are recorded.")
        return warnings
    if backend == "cpu_pytorch":
        warnings.append("resource_sampler_backend='cpu_pytorch': running a single non-profiled CPU/PyTorch allocator pass.")
        return warnings

    if bool(getattr(args, "nsight_disable_auto_launch", False)):
        warnings.append(
            "Nsight auto-launch is disabled. The script will run inline and will not produce Nsight GPU metrics."
        )
        return warnings

    nsys = _NSIGHT_NSYS_PATH or shutil.which("nsys")
    if not nsys:
        raise SystemExit(
            "ERROR: 'nsys' was not found. Install Nsight Systems, set --nsight-nsys-path, "
            "or run with --resource-sampler-backend cpu_pytorch for CPU/PyTorch-only metrics."
        )

    output_json = getattr(args, "output_json", None)
    if not output_json:
        output_json = f"{report_prefix}_resource.json"
        print(f"No --output-json was provided; using {output_json!r} for the merged resource JSON.")

    output_path = Path(os.path.abspath(os.path.expanduser(output_json)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stem = output_path.with_suffix("")
    gpu_json = f"{stem}.gpu_profile.tmp.json"
    e2e_json = f"{stem}.cpu_pytorch_e2e.tmp.json"
    comp_json = f"{stem}.cpu_pytorch_components.tmp.json"

    report_base = _make_nsight_report_base(report_prefix, _NSIGHT_OUTPUT_DIR)
    _run_nsight_child(nsys, report_base, gpu_json)
    try:
        finalize_json_with_nsight(
            gpu_json,
            report_base=report_base,
            nsys_path=nsys,
            include_samples=bool(getattr(args, "include_samples_in_json", False)),
        )
    except Exception as exc:
        print(f"WARNING: Nsight post-processing failed: {exc}")

    if backend == "full":
        _run_plain_child("cpu_pytorch_e2e", e2e_json)
        _run_plain_child("cpu_pytorch_components", comp_json)
        merge_full_resource_json(
            final_path=str(output_path),
            gpu_profile_json=gpu_json,
            cpu_e2e_json=e2e_json,
            cpu_components_json=comp_json,
            include_samples=bool(getattr(args, "include_samples_in_json", False)),
        )
    else:
        # Backward-compatible single Nsight output.
        Path(gpu_json).replace(output_path)
        print(f"\nWrote Nsight-only resource JSON to: {output_path}")

    raise SystemExit(0)


@contextmanager
def nvtx_range(name: str):
    pushed = False
    try:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(str(name))
            pushed = True
    except Exception:
        pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except Exception:
                pass


def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_count(value: int | None, default: int) -> int:
    return int(default if value is None else value)


class ResourceMonitor:
    """Non-profiled CPU + PyTorch allocator interval meter.

    CPU utilization is computed from psutil process.cpu_times() deltas.
    PyTorch memory records start/end/peak allocated/reserved memory over the
    same measurement window.  GPU busy/SM/CUDA-memory metrics are filled later
    from Nsight in the GPU profiling pass.
    """

    def __init__(self, sample_interval_ms: float = 0.0, gpu_index: int = 0):
        self.sample_interval_s = max(float(sample_interval_ms), 0.0) / 1000.0
        self.gpu_index = int(gpu_index)
        self.samples: list[dict[str, float]] = []
        self.warnings: list[str] = []
        self._psutil_warned = False
        self._cpu_capacity = _available_logical_cpu_count()
        self._cpu_start_wall: float | None = None
        self._cpu_start_time: float | None = None
        self._cpu_end_wall: float | None = None
        self._cpu_end_time: float | None = None
        self._cpu_core_percent_mean = math.nan
        self._cpu_process_percent_mean = math.nan
        self._torch_start = _torch_memory_stats()
        self._torch_end = _torch_memory_stats()
        try:
            import psutil

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
        except Exception as exc:
            self._psutil = None
            self._process = None
            self._warn_once(f"psutil CPU-time sampling unavailable: {exc}")

    def _warn_once(self, message: str):
        if not self._psutil_warned:
            self.warnings.append(message)
            self._psutil_warned = True

    @classmethod
    def sampler_backend(cls) -> str:
        backend = requested_resource_backend()
        if backend == "full":
            return "full:nsight-gpu+nonprofiled-cpu-pytorch"
        if backend == "nsight":
            return "nsight-systems-gpu-profile"
        if backend == "cpu_pytorch":
            return "nonprofiled-psutil-cpu-times+pytorch-allocator"
        return "none"

    @classmethod
    def inline_sampler_backend(cls) -> str:
        return "psutil-cpu-times-delta+pytorch-allocator-peaks"

    @staticmethod
    def _cpu_time_seconds(cpu_times: Any) -> float:
        return float(getattr(cpu_times, "user", 0.0)) + float(getattr(cpu_times, "system", 0.0))

    def _capture_cpu_start(self):
        self.samples = []
        self._cpu_start_wall = time.perf_counter()
        self._cpu_start_time = None
        self._cpu_end_wall = None
        self._cpu_end_time = None
        self._cpu_core_percent_mean = math.nan
        self._cpu_process_percent_mean = math.nan
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception as exc:
                self.warnings.append(f"torch.cuda.reset_peak_memory_stats failed: {exc}")
        self._torch_start = _torch_memory_stats()
        self._torch_end = dict(self._torch_start)
        if self._psutil is None or self._process is None:
            return
        try:
            self._cpu_start_time = self._cpu_time_seconds(self._process.cpu_times())
        except Exception as exc:
            self._warn_once(f"psutil CPU-time sampling failed at stage start: {exc}")

    def _capture_cpu_stop(self):
        if self._psutil is not None and self._process is not None:
            self._cpu_end_wall = time.perf_counter()
            try:
                self._cpu_end_time = self._cpu_time_seconds(self._process.cpu_times())
            except Exception as exc:
                self._warn_once(f"psutil CPU-time sampling failed at stage stop: {exc}")
            if self._cpu_start_wall is not None and self._cpu_start_time is not None and self._cpu_end_time is not None:
                elapsed = max(self._cpu_end_wall - self._cpu_start_wall, 1e-9)
                cpu_delta = max(0.0, self._cpu_end_time - self._cpu_start_time)
                core_percent = cpu_delta / elapsed * 100.0
                total_capacity_percent = core_percent / max(1, self._cpu_capacity)
                self._cpu_core_percent_mean = float(core_percent)
                self._cpu_process_percent_mean = float(min(max(total_capacity_percent, 0.0), 100.0))
                self.samples = [
                    {
                        "timestamp_start": float(self._cpu_start_wall),
                        "timestamp_end": float(self._cpu_end_wall),
                        "wall_time_s": float(elapsed),
                        "cpu_time_delta_s": float(cpu_delta),
                        "cpu_process_percent": self._cpu_process_percent_mean,
                        "cpu_process_core_percent": self._cpu_core_percent_mean,
                    }
                ]
        self._torch_end = _torch_memory_stats()

    def cpu_metrics(self) -> tuple[float, float]:
        """Return (total-capacity %, raw core-equivalent %)."""
        return self._cpu_process_percent_mean, self._cpu_core_percent_mean

    def torch_metrics(self) -> dict[str, float]:
        start_alloc = self._torch_start.get("torch_memory_allocated_mb", math.nan)
        end_alloc = self._torch_end.get("torch_memory_allocated_mb", math.nan)
        peak_alloc = self._torch_end.get("torch_memory_peak_allocated_mb", math.nan)
        start_res = self._torch_start.get("torch_memory_reserved_mb", math.nan)
        end_res = self._torch_end.get("torch_memory_reserved_mb", math.nan)
        peak_res = self._torch_end.get("torch_memory_peak_reserved_mb", math.nan)
        return {
            "torch_memory_allocated_mb_start": float(start_alloc),
            "torch_memory_allocated_mb_end": float(end_alloc),
            "torch_memory_allocated_mb_peak": float(peak_alloc),
            "torch_memory_allocated_mb_increment_peak": (
                float(max(peak_alloc - start_alloc, 0.0)) if math.isfinite(peak_alloc) and math.isfinite(start_alloc) else math.nan
            ),
            "torch_memory_reserved_mb_start": float(start_res),
            "torch_memory_reserved_mb_end": float(end_res),
            "torch_memory_reserved_mb_peak": float(peak_res),
            "torch_memory_reserved_mb_increment_peak": (
                float(max(peak_res - start_res, 0.0)) if math.isfinite(peak_res) and math.isfinite(start_res) else math.nan
            ),
        }

    def start(self):
        self._capture_cpu_start()

    def stop(self):
        self._capture_cpu_stop()


def _torch_memory_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "torch_memory_allocated_mb": math.nan,
            "torch_memory_reserved_mb": math.nan,
            "torch_memory_peak_allocated_mb": math.nan,
            "torch_memory_peak_reserved_mb": math.nan,
        }
    return {
        "torch_memory_allocated_mb": float(torch.cuda.memory_allocated() / 1024**2),
        "torch_memory_reserved_mb": float(torch.cuda.memory_reserved() / 1024**2),
        "torch_memory_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / 1024**2),
        "torch_memory_peak_reserved_mb": float(torch.cuda.max_memory_reserved() / 1024**2),
    }


def _nan_stat(values: list[float], fn: Callable[[np.ndarray], float]) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(fn(arr))


def _empty_sample(
    duration_ms: float = math.nan,
    *,
    cpu_process_percent_mean: float = math.nan,
    cpu_process_core_percent_mean: float = math.nan,
    sample_count: float = 0.0,
    torch_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    metrics = {
        "duration_ms": float(duration_ms),
        "gpu_profile_duration_ms": math.nan,
        "cpu_pytorch_duration_ms": float(duration_ms) if resource_run_kind().startswith("cpu_pytorch") or requested_resource_backend() == "cpu_pytorch" else math.nan,
        "cpu_process_percent_mean": float(cpu_process_percent_mean),
        "cpu_process_core_percent_mean": float(cpu_process_core_percent_mean),
        "gpu_util_percent_mean": math.nan,
        "gpu_busy_ratio_percent_mean": math.nan,
        "sm_util_percent_mean": math.nan,
        "gpu_memory_used_mb_max": math.nan,
        "nsight_cuda_memory_peak_mb": math.nan,
        "torch_memory_allocated_mb_start": math.nan,
        "torch_memory_allocated_mb_end": math.nan,
        "torch_memory_allocated_mb_peak": math.nan,
        "torch_memory_allocated_mb_increment_peak": math.nan,
        "torch_memory_reserved_mb_start": math.nan,
        "torch_memory_reserved_mb_end": math.nan,
        "torch_memory_reserved_mb_peak": math.nan,
        "torch_memory_reserved_mb_increment_peak": math.nan,
        "sample_count": float(sample_count),
    }
    if torch_metrics:
        for key, value in torch_metrics.items():
            metrics[key] = float(value) if value is not None else math.nan
    return metrics


def measure_callable(
    monitor: ResourceMonitor,
    fn: Callable[[], Any],
    *,
    sync_cuda_before: bool = False,
    sync_cuda_after: bool = False,
    stage_name: str | None = None,
) -> tuple[Any, dict[str, float]]:
    if sync_cuda_before:
        cuda_synchronize()
    monitor.start()
    start = time.perf_counter()
    with nvtx_range(stage_name or "nsight_stage"):
        result = fn()
        if sync_cuda_after:
            cuda_synchronize()
    duration_ms = (time.perf_counter() - start) * 1000.0
    monitor.stop()
    cpu_process_percent, cpu_core_percent = monitor.cpu_metrics()
    sample = _empty_sample(
        duration_ms,
        cpu_process_percent_mean=cpu_process_percent,
        cpu_process_core_percent_mean=cpu_core_percent,
        sample_count=float(len(monitor.samples)),
        torch_metrics=monitor.torch_metrics(),
    )
    return result, sample


def stage_samples_to_metrics(samples: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray([sample.get(key, math.nan) for sample in samples], dtype=float)
        for key in RESOURCE_SERIES_KEYS
    }


class StageResourceHooks:
    def __init__(self, sample_interval_ms: float = 0.0, stage_keys: Iterable[str] | None = None):
        self.sample_interval_ms = float(sample_interval_ms)
        self.component_stage_keys = tuple(stage_keys or LEGACY_COMPONENT_STAGE_KEYS)
        self.active = False
        self.records: dict[str, list[dict[str, float]]] = {}
        self.calls: dict[str, int] = {}
        self.warnings: list[str] = []

    def start_iteration(self):
        self.active = True
        self.records = {stage: [] for stage in self.component_stage_keys}
        self.calls = {stage: 0 for stage in self.component_stage_keys}

    def finish_iteration(self) -> dict[str, Any]:
        self.active = False
        return {"records": {stage: list(v) for stage, v in self.records.items()}, "calls": dict(self.calls)}

    def start(self) -> dict[str, Any]:
        monitor = ResourceMonitor(sample_interval_ms=self.sample_interval_ms)
        monitor.start()
        return {
            "monitor": monitor,
            "start_time": time.perf_counter(),
        }

    def stop(self, stage: str, handle: dict[str, Any], duration_ms: float):
        self.calls[stage] = self.calls.get(stage, 0) + 1
        monitor = handle["monitor"]
        monitor.stop()
        self.warnings.extend(monitor.warnings)
        cpu_process_percent, cpu_core_percent = monitor.cpu_metrics()
        self.records.setdefault(stage, []).append(
            _empty_sample(
                duration_ms,
                cpu_process_percent_mean=cpu_process_percent,
                cpu_process_core_percent_mean=cpu_core_percent,
                sample_count=float(len(monitor.samples)),
                torch_metrics=monitor.torch_metrics(),
            )
        )

    @contextmanager
    def measure(self, stage: str):
        handle = self.start()
        try:
            with nvtx_range(f"component/{stage}"):
                try:
                    yield
                finally:
                    # Keep the old boundary: component timing includes the CUDA sync
                    # that proves kernels launched in the stage have completed.
                    cuda_synchronize()
        finally:
            duration_ms = (time.perf_counter() - handle["start_time"]) * 1000.0
            self.stop(stage, handle, duration_ms)


def _aggregate_stage_samples(samples: list[dict[str, float]]) -> dict[str, float]:
    if not samples:
        return _empty_sample()
    durations = np.asarray([sample.get("duration_ms", math.nan) for sample in samples], dtype=float)
    finite_duration = durations[np.isfinite(durations)]
    weights = np.where(np.isfinite(durations) & (durations > 0), durations, 0.0)
    cpu_values = np.asarray([sample.get("cpu_process_percent_mean", math.nan) for sample in samples], dtype=float)
    cpu_core_values = np.asarray([sample.get("cpu_process_core_percent_mean", math.nan) for sample in samples], dtype=float)
    sample_counts = np.asarray([sample.get("sample_count", 0.0) for sample in samples], dtype=float)
    finite_counts = sample_counts[np.isfinite(sample_counts)]

    def weighted_mean(values: np.ndarray) -> float:
        valid = np.isfinite(values) & (weights > 0)
        if valid.any():
            return float(np.average(values[valid], weights=weights[valid]))
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else math.nan

    torch_metrics: dict[str, float] = {}
    for key in (
        "torch_memory_allocated_mb_start",
        "torch_memory_allocated_mb_end",
        "torch_memory_reserved_mb_start",
        "torch_memory_reserved_mb_end",
    ):
        values = np.asarray([sample.get(key, math.nan) for sample in samples], dtype=float)
        finite = values[np.isfinite(values)]
        torch_metrics[key] = float(np.mean(finite)) if finite.size else math.nan
    for key in (
        "torch_memory_allocated_mb_peak",
        "torch_memory_allocated_mb_increment_peak",
        "torch_memory_reserved_mb_peak",
        "torch_memory_reserved_mb_increment_peak",
    ):
        values = np.asarray([sample.get(key, math.nan) for sample in samples], dtype=float)
        finite = values[np.isfinite(values)]
        torch_metrics[key] = float(np.max(finite)) if finite.size else math.nan

    return _empty_sample(
        float(np.sum(finite_duration)) if finite_duration.size else math.nan,
        cpu_process_percent_mean=weighted_mean(cpu_values),
        cpu_process_core_percent_mean=weighted_mean(cpu_core_values),
        sample_count=float(np.sum(finite_counts)) if finite_counts.size else 0.0,
        torch_metrics=torch_metrics,
    )


def records_to_component_metrics(
    records: list[dict[str, Any]],
    stage_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    components: dict[str, Any] = {}
    component_stage_keys = tuple(stage_keys or LEGACY_COMPONENT_STAGE_KEYS)
    for stage in component_stage_keys:
        per_iteration_samples = [
            _aggregate_stage_samples(record.get("records", {}).get(stage, []))
            for record in records
        ]
        components[stage] = stage_samples_to_metrics(per_iteration_samples)
        components[f"{stage}_calls"] = np.asarray(
            [record.get("calls", {}).get(stage, 0) for record in records],
            dtype=float,
        )
    return components


def measure_idle_resource_baseline(*, duration_ms: float, warmup_ms: float, sample_interval_ms: float):
    # Baseline subtraction is not meaningful here because final metrics are
    # derived from per-stage NVTX ranges. Keep a no-op for CLI compatibility.
    if warmup_ms > 0:
        time.sleep(float(warmup_ms) / 1000.0)
    return _empty_sample(float(duration_ms)), [
        "Baseline resource sampling is disabled in Nsight+psutil-CPU-time mode; no NVML/nvidia-smi baseline is taken."
    ]


def subtract_baseline_from_metrics(metrics: dict[str, Any] | None, baseline: dict[str, float] | None):
    return metrics


def subtract_baseline_from_components(components: dict[str, Any] | None, baseline: dict[str, float] | None):
    return components


def build_components(data_processing_metrics, component_metrics, e2e_metrics) -> dict[str, Any]:
    components: dict[str, Any] = {}
    if data_processing_metrics is not None:
        components["data_processing"] = data_processing_metrics
    if component_metrics is not None:
        for key, value in component_metrics.items():
            components[key] = value
    if e2e_metrics is not None:
        components["e2e"] = e2e_metrics
    return components


def _finite_series(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def _series_stats(values: Any) -> dict[str, float | int | list[float] | None]:
    arr = _finite_series(values)
    if arr.size == 0:
        return {"mean": None, "p5": None, "p95": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }


def _format_series(values: Any, suffix: str, include_p95: bool) -> str:
    stats = _series_stats(values)
    if stats["mean"] is None:
        return "n/a"
    if include_p95:
        return f"{stats['p5']:.1f} / {stats['mean']:.1f} / {stats['p95']:.1f}{suffix}"
    return f"{stats['mean']:.1f}{suffix}"


def print_resource_summary(label: str, metrics: dict[str, Any], include_p95: bool = True):
    print(f"  {label} (resource window stats; official latency is measured separately):")
    if "duration_ms" in metrics:
        print(f"    Window duration: {_format_series(metrics.get('duration_ms', []), ' ms', include_p95)}")
    if "gpu_busy_ratio_percent_mean" in metrics:
        print(f"    GPU busy:       {_format_series(metrics['gpu_busy_ratio_percent_mean'], '%', include_p95)}")
    elif "gpu_util_percent_mean" in metrics:
        print(f"    GPU busy:       {_format_series(metrics['gpu_util_percent_mean'], '%', include_p95)}")
    if "sm_util_percent_mean" in metrics:
        print(f"    SM util:        {_format_series(metrics['sm_util_percent_mean'], '%', include_p95)}")
    if "nsight_cuda_memory_peak_mb" in metrics:
        print(f"    Nsight CUDA mem:{_format_series(metrics['nsight_cuda_memory_peak_mb'], ' MB peak', include_p95)}")
    elif "gpu_memory_used_mb_max" in metrics:
        print(f"    GPU memory peak:{_format_series(metrics['gpu_memory_used_mb_max'], ' MB', include_p95)}")
    if "torch_memory_allocated_mb_peak" in metrics:
        print(f"    Torch alloc pk: {_format_series(metrics['torch_memory_allocated_mb_peak'], ' MB', include_p95)}")
    if "torch_memory_reserved_mb_peak" in metrics:
        print(f"    Torch reserv pk:{_format_series(metrics['torch_memory_reserved_mb_peak'], ' MB', include_p95)}")
    if "cpu_process_percent_mean" in metrics:
        print(f"    CPU total:      {_format_series(metrics['cpu_process_percent_mean'], '%', include_p95)}")


def print_baseline_summary(baseline: dict[str, float], label: str = "Idle baseline"):
    print(f"  {label}: skipped; Nsight+psutil-CPU-time mode has no baseline sampler.")


def print_markdown_table(components: dict[str, Any], device_name: str, title: str, include_p95: bool = True):
    print("\n" + title)
    print("Inline table contains duration and psutil CPU-time before Nsight post-processing. Final GPU/memory p5/mean/p95 metrics are written to JSON after .nsys-rep export.")


def _summarize_metric_dict(metrics: dict[str, Any], include_samples: bool) -> dict[str, Any]:
    summary = {}
    for key in RESOURCE_SERIES_KEYS:
        if key not in metrics:
            continue
        stats = _series_stats(metrics[key])
        if include_samples:
            arr = np.asarray(metrics[key], dtype=float)
            stats["samples"] = [float(x) if math.isfinite(float(x)) else None for x in arr.tolist()]
        summary[key] = stats
    return summary


def _stage_metrics_for_json(
    components: dict[str, Any],
    *,
    include_samples: bool,
    raw: bool,
    stage_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    payload = {}
    resource_stage_keys = tuple(stage_keys or LEGACY_RESOURCE_STAGE_KEYS)
    for stage in resource_stage_keys:
        if stage not in components:
            continue
        metrics = components[stage]
        payload[stage] = _summarize_metric_dict(metrics, include_samples)
    return payload


def _calls_for_json(
    components: dict[str, Any],
    include_samples: bool,
    component_stage_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    calls = {}
    for stage in tuple(component_stage_keys or LEGACY_COMPONENT_STAGE_KEYS):
        key = f"{stage}_calls"
        if key not in components:
            continue
        stats = _series_stats(components[key])
        if include_samples:
            arr = np.asarray(components[key], dtype=float)
            stats["samples"] = [float(x) for x in arr.tolist()]
        calls[key] = stats
    return calls


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


PLOT_STAGE_LABELS = {
    "data_processing": "Data Processing",
    "vision_encoder": "Vision Encoder",
    "llm_backbone": "LLM Backbone",
    "action_expert": "Action Expert",
    "dit": "DiT/Decode-only",
    "system2_vision_encoder": "System2 Vision Encoder",
    "system2_inference": "System2 Inference",
    "system_bridge": "System Bridge / Projector",
    "system1_vision_encoder": "System1 Vision Encoder",
    "system1_action_expert": "System1 Action Expert",
    "e2e": "E2E",
}

RESOURCE_METRIC_UNITS = {
    "duration_ms": "ms",
    "gpu_profile_duration_ms": "ms",
    "cpu_pytorch_duration_ms": "ms",
    "cpu_process_percent_mean": "%",
    "cpu_process_core_percent_mean": "%",
    "gpu_util_percent_mean": "%",
    "gpu_busy_ratio_percent_mean": "%",
    "sm_util_percent_mean": "%",
    "gpu_memory_used_mb_max": "MB",
    "nsight_cuda_memory_peak_mb": "MB",
    "torch_memory_allocated_mb_start": "MB",
    "torch_memory_allocated_mb_end": "MB",
    "torch_memory_allocated_mb_peak": "MB",
    "torch_memory_allocated_mb_increment_peak": "MB",
    "torch_memory_reserved_mb_start": "MB",
    "torch_memory_reserved_mb_end": "MB",
    "torch_memory_reserved_mb_peak": "MB",
    "torch_memory_reserved_mb_increment_peak": "MB",
    "sample_count": "count",
}


def _copy_summary_fields_for_plot(stats: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}mean": _json_safe_value(stats.get("mean")),
        f"{prefix}p5": _json_safe_value(stats.get("p5")),
        f"{prefix}p95": _json_safe_value(stats.get("p95")),
        f"{prefix}std": _json_safe_value(stats.get("std")),
        f"{prefix}min": _json_safe_value(stats.get("min")),
        f"{prefix}max": _json_safe_value(stats.get("max")),
        f"{prefix}n": _json_safe_value(stats.get("n")),
    }


def _make_plot_base_record(*, model_name, device_name, hardware_name, model_info, benchmark) -> dict[str, Any]:
    return {
        "schema_version": "vla_plot_v1",
        "model": model_name,
        "device_name": device_name,
        "hardware_name": hardware_name,
        "model_path": model_info.get("model_path"),
        "checkpoint_dir": model_info.get("checkpoint_dir"),
        "train_config": model_info.get("train_config"),
        "dataset_path": model_info.get("dataset_path"),
        "dataset_root": model_info.get("dataset_root"),
        "embodiment_tag": model_info.get("embodiment_tag"),
        "example_source": model_info.get("example_source"),
        "action_horizon": model_info.get("action_horizon"),
        "denoising_steps": model_info.get("denoising_steps"),
        "unnorm_key": model_info.get("unnorm_key"),
        "e2e_inference_path": benchmark.get("e2e_inference_path"),
        "num_steps": benchmark.get("num_steps"),
        "resource_sampler_backend": benchmark.get("resource_sampler_backend"),
        "metrics_source": benchmark.get("resource_metrics_source"),
    }


def _make_resource_plot_records(
    *,
    model_name,
    device_name,
    hardware_name,
    model_info,
    benchmark,
    resource_metrics,
    calls,
    metrics_source: str,
    stage_keys: Iterable[str] | None = None,
):
    base = _make_plot_base_record(
        model_name=model_name,
        device_name=device_name,
        hardware_name=hardware_name,
        model_info=model_info,
        benchmark=benchmark,
    )
    records: list[dict[str, Any]] = []
    for stage in tuple(stage_keys or LEGACY_RESOURCE_STAGE_KEYS):
        if stage not in resource_metrics:
            continue
        stage_metrics = resource_metrics[stage]
        for metric_name, stats in stage_metrics.items():
            if not isinstance(stats, dict):
                continue
            record = dict(base)
            record.update(
                {
                    "metric_family": "resource",
                    "metric_name": metric_name,
                    "metrics_source": metrics_source,
                    "unit": RESOURCE_METRIC_UNITS.get(metric_name),
                    "stage": stage,
                    "stage_label": PLOT_STAGE_LABELS.get(stage, stage),
                }
            )
            record.update(_copy_summary_fields_for_plot(stats, prefix="value_"))
            if "samples" in stats:
                record["samples"] = _json_safe_value(stats["samples"])
            records.append(record)
    for call_name, stats in calls.items():
        stage = call_name.removesuffix("_calls")
        record = dict(base)
        record.update(
            {
                "metric_family": "calls",
                "metric_name": call_name,
                "metrics_source": metrics_source,
                "unit": "count",
                "stage": stage,
                "stage_label": PLOT_STAGE_LABELS.get(stage, stage),
            }
        )
        record.update(_copy_summary_fields_for_plot(stats, prefix="value_"))
        if "samples" in stats:
            record["samples"] = _json_safe_value(stats["samples"])
        records.append(record)
    return records


def build_memory_summary(components: dict[str, Any], include_samples: bool = False) -> dict[str, Any]:
    e2e = components.get("e2e", {})
    mem = e2e.get("gpu_memory_used_mb_max") if isinstance(e2e, dict) else None
    result = {
        "definitions": {
            "e2e_nsight_gpu_memory_peak_mb": (
                "Peak CUDA memory usage derived from Nsight CUDA memory records inside E2E NVTX ranges. "
                "If null, the local Nsight export did not expose parseable CUDA memory usage records."
            ),
        },
        "e2e_nsight_gpu_memory_peak_mb": _series_stats(mem) if mem is not None else _series_stats([]),
    }
    if include_samples and mem is not None:
        result["e2e_nsight_gpu_memory_peak_mb"]["samples"] = _json_safe_value(mem)
    return _json_safe_value(result)


def write_resource_results_json(
    path: str,
    *,
    model_name: str,
    device_name: str,
    hardware_name: str,
    model_info: dict[str, Any],
    args: Any,
    components: dict[str, Any],
    sample_interval_ms: float,
):
    include_samples = bool(getattr(args, "include_samples_in_json", False))
    benchmark = {
        "default_measured_iterations": int(getattr(args, "num_iterations")),
        "default_warmup": int(getattr(args, "warmup")),
        "include_p95": bool(getattr(args, "include_p95", True)),
        "resource_sample_interval_ms": float(sample_interval_ms),
        "cpu_capacity_logical_cores": _available_logical_cpu_count(),
        "resource_subtract_baseline": False,
        "resource_baseline_duration_ms": 0.0,
        "resource_baseline_warmup_ms": 0.0,
        "measure_data_processing": bool(getattr(args, "measure_data_processing", False)),
        "measure_e2e": bool(getattr(args, "measure_e2e", False)),
        "measure_components": bool(getattr(args, "measure_components", False)),
        "e2e_inference_path": getattr(args, "e2e_inference_path", None),
        "num_steps": model_info.get("denoising_steps"),
        "gpu_sampler_backend": "nsight-systems" if requested_resource_backend() == "nsight" else None,
        "inline_cpu_sampler_backend": ResourceMonitor.inline_sampler_backend(),
        "inline_gpu_sampler_backend": None,
        "resource_sampler_backend": requested_resource_backend(),
        "resource_run_kind": resource_run_kind(),
        "resource_sampler_fallback_backend": ResourceMonitor.inline_sampler_backend(),
        "resource_metrics_source": (
            "nsight_child_duration_skeleton" if resource_run_kind() == "nsight_gpu"
            else "nonprofiled_cpu_times_pytorch_allocator"
        ),
        "nsight": nsight_metadata(),
        "resource_sample_interval_mode": "psutil_cpu_times_delta_pytorch_allocator",
    }
    resource_stage_keys, component_stage_keys, stage_schema = infer_stage_keys(components, model_info)
    benchmark["stage_schema"] = stage_schema
    resource_metrics = _stage_metrics_for_json(
        components,
        include_samples=include_samples,
        raw=False,
        stage_keys=resource_stage_keys,
    )
    calls = _calls_for_json(components, include_samples, component_stage_keys)
    memory_summary = build_memory_summary(components, include_samples=include_samples)

    payload = {
        "schema_version": "vla_resource_v2_child_pass",
        "model": model_name,
        "hardware": {"device_name": device_name, "torch_cuda_device_name": hardware_name},
        "model_info": model_info,
        "benchmark": benchmark,
        "resource_metrics": resource_metrics,
        "resource_metrics_raw": resource_metrics,
        "environment_baseline": None,
        "model_loaded_baseline": None,
        "resource_baseline": None,
        "memory_summary": memory_summary,
        "calls": calls,
        "stage_hook_modules": components.get("_stage_hook_modules", {}),
        "warnings": components.get("_resource_warnings", []) + components.get("_stage_hook_warnings", []),
        "notes": {
            "resource_metrics": (
                "Child pass writes resource-window statistics. In full mode, final paper metrics are merged from separate Nsight GPU and non-profiled CPU/PyTorch passes."
            ),
            "cpu_process_percent_mean": (
                "Process CPU utilization normalized by available logical CPU capacity, so values are in the 0-100% "
                "whole-machine scale. The raw multi-core equivalent is stored in cpu_process_core_percent_mean."
            ),
            "cpu_process_core_percent_mean": (
                "Raw process CPU core-equivalent utilization. A value of 2400% means roughly 24 logical cores saturated."
            ),
            "gpu_util_percent_mean": (
                "Nsight-derived GPU kernel-busy ratio within the NVTX range: union of profiled CUDA kernel time "
                "divided by NVTX wall time. It is not NVML/nvidia-smi utilization."
            ),
            "gpu_memory_used_mb_max": (
                "Nsight-derived peak CUDA memory usage inside the NVTX range when CUDA memory records are parseable."
            ),
            "measurement_protocol": "Default full mode separates intrusive Nsight GPU profiling from non-profiled CPU/PyTorch allocator measurement.",
        },
    }
    payload["plot_schema_version"] = "vla_plot_v1"
    payload["plot_stage_order"] = list(resource_stage_keys)
    payload["plot_stage_labels"] = PLOT_STAGE_LABELS
    payload["plot_records"] = _make_resource_plot_records(
        model_name=model_name,
        device_name=device_name,
        hardware_name=hardware_name,
        model_info=model_info,
        benchmark=benchmark,
        resource_metrics=resource_metrics,
        calls=calls,
        metrics_source="duration_only",
        stage_keys=resource_stage_keys,
    )
    payload["plot_records_raw"] = list(payload["plot_records"])
    payload["plot_records_memory_summary"] = []

    payload = _json_safe_value(payload)
    output_path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote duration skeleton JSON to: {output_path}")


# ------------------------- Nsight SQLite post-processing -------------------------

def _run_export_commands(nsys: str, report_file: str, sqlite_path: str) -> tuple[str | None, list[str]]:
    warnings: list[str] = []
    if os.path.exists(sqlite_path) and os.path.getsize(sqlite_path) > 0:
        return sqlite_path, warnings
    candidates = [
        [nsys, "export", "--type", "sqlite", "--force-overwrite=true", "--output", sqlite_path, report_file],
        [nsys, "export", "-t", "sqlite", "--force-overwrite=true", "-o", sqlite_path, report_file],
        [nsys, "export", "-t", "sqlite", "-o", sqlite_path, report_file],
    ]
    for cmd in candidates:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            possible = [sqlite_path, f"{sqlite_path}.sqlite"]
            for path in possible:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    return path, warnings
            warnings.append(f"Nsight export succeeded but no SQLite file was found for command: {' '.join(cmd)}")
        else:
            warnings.append(
                "Nsight export command failed: "
                + " ".join(shlex.quote(p) for p in cmd)
                + f"\nstdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
            )
    return None, warnings


def _tables(conn: sqlite3.Connection) -> list[str]:
    return [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')]


def _first_col(cols: Iterable[str], candidates: Iterable[str]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for candidate in candidates:
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def _string_map(conn: sqlite3.Connection) -> dict[int, str]:
    result: dict[int, str] = {}
    for table in _tables(conn):
        if table.lower() not in {"stringids", "string_ids", "strings"}:
            continue
        cols = _columns(conn, table)
        id_col = _first_col(cols, ["id", "stringId", "key"])
        val_col = _first_col(cols, ["value", "string", "name", "text"])
        if not id_col or not val_col:
            continue
        try:
            for sid, value in conn.execute(f'SELECT "{id_col}", "{val_col}" FROM "{table}"'):
                if sid is not None and value is not None:
                    result[int(sid)] = str(value)
        except Exception:
            pass
    return result


def _enum_map(conn: sqlite3.Connection, table: str) -> dict[int, str]:
    result: dict[int, str] = {}
    if table not in _tables(conn):
        return result
    cols = _columns(conn, table)
    id_col = _first_col(cols, ["id", "value"])
    name_col = _first_col(cols, ["name", "label"])
    if not id_col or not name_col:
        return result
    try:
        for enum_id, name in conn.execute(f'SELECT "{id_col}", "{name_col}" FROM "{table}"'):
            if enum_id is not None and name is not None:
                result[int(enum_id)] = str(name)
    except Exception:
        pass
    return result


def _resolve_text(value: Any, strings: dict[int, str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    if isinstance(value, str):
        return value
    try:
        return strings.get(int(value), str(value))
    except Exception:
        return str(value)


def _extract_nvtx_ranges(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    resource_stage_keys: Iterable[str] | None = None,
) -> tuple[dict[str, list[tuple[int, int]]], list[str]]:
    warnings: list[str] = []
    selected_stage_keys = tuple(resource_stage_keys or RESOURCE_STAGE_KEYS)
    ranges: dict[str, list[tuple[int, int]]] = {stage: [] for stage in selected_stage_keys}
    for table in _tables(conn):
        if "nvtx" not in table.lower():
            continue
        cols = _columns(conn, table)
        start_col = _first_col(cols, ["start", "startTime", "timestamp"])
        end_col = _first_col(cols, ["end", "endTime"])
        if not start_col or not end_col:
            continue
        text_col = _first_col(cols, ["text", "name", "message", "domain", "range"])
        text_id_col = _first_col(cols, ["textId", "nameId", "messageId", "stringId"])
        if not text_col and not text_id_col:
            continue
        select_cols = [start_col, end_col]
        if text_col:
            select_cols.append(text_col)
        elif text_id_col:
            select_cols.append(text_id_col)
        query = "SELECT " + ", ".join(f'"{c}"' for c in select_cols) + f' FROM "{table}"'
        try:
            for row in conn.execute(query):
                start, end, text_value = row[0], row[1], row[2]
                if start is None or end is None or int(end) <= int(start):
                    continue
                name = _resolve_text(text_value, strings)
                if name in _STAGE_NVTX_TO_KEY:
                    stage_key = _STAGE_NVTX_TO_KEY[name]
                    if stage_key in ranges:
                        ranges[stage_key].append((int(start), int(end)))
        except Exception as exc:
            warnings.append(f"Failed reading NVTX table {table}: {exc}")
    found = sum(len(v) for v in ranges.values())
    if found == 0:
        warnings.append("No parseable NVTX stage ranges were found; check that torch.cuda.nvtx markers were captured.")
    for stage in ranges:
        ranges[stage].sort(key=lambda x: x[0])
    return ranges, warnings


def _extract_kernel_intervals(
    conn: sqlite3.Connection,
    start_bound: int | None = None,
    end_bound: int | None = None,
) -> tuple[list[tuple[int, int]], list[str]]:
    warnings: list[str] = []
    intervals: list[tuple[int, int]] = []
    for table in _tables(conn):
        low = table.lower()
        if "kernel" not in low or "summary" in low or "stats" in low:
            continue
        cols = _columns(conn, table)
        start_col = _first_col(cols, ["start", "startTime", "timestamp"])
        end_col = _first_col(cols, ["end", "endTime"])
        if not start_col or not end_col:
            continue
        try:
            query = f'SELECT "{start_col}", "{end_col}" FROM "{table}"'
            where: list[str] = []
            params: list[int] = []
            if start_bound is not None:
                where.append(f'"{end_col}" > ?')
                params.append(int(start_bound))
            if end_bound is not None:
                where.append(f'"{start_col}" < ?')
                params.append(int(end_bound))
            if where:
                query += " WHERE " + " AND ".join(where)
            for start, end in conn.execute(query, params):
                if start is not None and end is not None and int(end) > int(start):
                    intervals.append((int(start), int(end)))
        except Exception as exc:
            warnings.append(f"Failed reading CUDA kernel table {table}: {exc}")
    intervals.sort(key=lambda x: x[0])
    if not intervals:
        warnings.append("No CUDA kernel intervals were found; GPU busy percent will be null.")
    return intervals, warnings


_INTERVAL_INDEX_CACHE: dict[int, tuple[int, list[int], list[int]]] = {}


def _interval_index(intervals: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    cache_key = id(intervals)
    cached = _INTERVAL_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] == len(intervals):
        return cached[1], cached[2]
    starts = [s for s, _ in intervals]
    prefix_max_ends: list[int] = []
    current = 0
    for _, e in intervals:
        current = max(current, e)
        prefix_max_ends.append(current)
    _INTERVAL_INDEX_CACHE[cache_key] = (len(intervals), starts, prefix_max_ends)
    return starts, prefix_max_ends


def _union_overlap_duration(intervals: list[tuple[int, int]], start: int, end: int) -> int:
    _, prefix_max_ends = _interval_index(intervals)
    first_possible = bisect.bisect_right(prefix_max_ends, start)
    clipped: list[tuple[int, int]] = []
    for s, e in intervals[first_possible:]:
        if s >= end:
            break
        cs, ce = max(s, start), min(e, end)
        if ce > cs:
            clipped.append((cs, ce))
    if not clipped:
        return 0
    total = 0
    cur_s, cur_e = clipped[0]
    for s, e in clipped[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total


def _extract_memory_events(conn: sqlite3.Connection, strings: dict[int, str]) -> tuple[list[tuple[int, int]], list[str]]:
    """Return CUDA device-memory deltas as (timestamp_ns, delta_bytes).

    Nsight exports memcpy/memset activity tables with `bytes` columns too.  Those
    are traffic volumes, not allocations, so they must not be folded into memory
    residency.  Restrict this parser to CUDA_GPU_MEMORY_USAGE_EVENTS (or a table
    with the same explicit memoryOperationType column) and decode the operation
    through ENUM_CUDA_DEV_MEM_EVENT_OPER.
    """
    warnings: list[str] = []
    events: list[tuple[int, int]] = []
    operation_names = _enum_map(conn, "ENUM_CUDA_DEV_MEM_EVENT_OPER")
    for table in _tables(conn):
        low = table.lower()
        if low != "cuda_gpu_memory_usage_events":
            continue
        cols = _columns(conn, table)
        time_col = _first_col(cols, ["start", "timestamp", "time", "startTime"])
        size_col = _first_col(cols, ["bytes", "size", "memorySize", "memSize"])
        op_col = _first_col(cols, ["memoryOperationType", "memoryOperationTypeId"])
        if not time_col or not size_col or not op_col:
            warnings.append(f"Skipping CUDA memory table {table}: missing timestamp/bytes/operation columns.")
            continue
        select_cols = [time_col, size_col, op_col]
        try:
            query = "SELECT " + ", ".join(f'"{c}"' for c in select_cols) + f' FROM "{table}"'
            for row in conn.execute(query):
                ts, size, operation = row[0], row[1], row[2]
                if ts is None or size is None:
                    continue
                delta = int(size)
                op_text = operation_names.get(int(operation), _resolve_text(operation, strings) or "").lower()
                if any(token in op_text for token in ("deallocation", "free", "dealloc", "release")):
                    delta = -abs(delta)
                elif any(token in op_text for token in ("allocation", "alloc", "malloc", "reserve", "create")):
                    delta = abs(delta)
                else:
                    warnings.append(
                        f"Skipping CUDA memory event with unrecognized operation {operation!r} ({op_text!r})."
                    )
                    continue
                events.append((int(ts), delta))
        except Exception as exc:
            warnings.append(f"Failed reading CUDA memory table {table}: {exc}")
    events.sort(key=lambda x: x[0])
    if not events:
        warnings.append("No parseable CUDA memory allocation/free events were found; GPU memory peak will be null.")
    return events, warnings


def _looks_like_sm_metric(name: str) -> bool:
    n = name.lower().replace("-", " ").replace("_", " ")
    if "sm" not in n and "streaming multiprocessor" not in n:
        return False
    negative = ("tensor", "dram", "l2", "pcie", "copy", "mem", "memory", "warp occupancy")
    if any(token in n for token in negative):
        return False
    positive = (
        "sm active",
        "sms active",
        "sm activity",
        "sm issue",
        "sm utilization",
        "sm util",
        "sm throughput",
        "active sm",
    )
    return any(token in n for token in positive) or "sm__throughput" in name.lower()


def _looks_like_primary_sm_metric(name: str) -> bool:
    n = name.lower().replace("-", " ").replace("_", " ")
    negative = ("tensor", "dram", "l2", "pcie", "copy", "mem", "memory", "warp occupancy")
    if any(token in n for token in negative):
        return False
    positive = ("sms active", "sm active", "sm activity", "sm utilization", "sm util", "sm throughput")
    return any(token in n for token in positive) or "sm__throughput" in name.lower()


def _extract_sm_metric_samples(
    conn: sqlite3.Connection,
    strings: dict[int, str],
    start_bound: int | None = None,
    end_bound: int | None = None,
) -> tuple[list[tuple[int, int | None, float]], list[str]]:
    """Best-effort extraction of Nsight Systems GPU SM utilization samples.

    Nsight Systems SQLite schemas differ across releases. This parser searches
    GPU metric tables dynamically and accepts common name/value/timestamp column
    variants. Values are normalized to percent if they look like 0-1 fractions.
    """
    warnings: list[str] = []
    samples: list[tuple[int, int | None, float]] = []
    metric_name_maps: dict[str, dict[int, str]] = {}
    tables = _tables(conn)

    for table in tables:
        low = table.lower()
        if "metric" not in low or not any(token in low for token in ("gpu", "dcgm", "counter")):
            continue
        cols = _columns(conn, table)
        id_col = _first_col(cols, ["id", "metricId", "metric", "value"])
        name_col = _first_col(cols, ["name", "metricName", "label", "description", "text"])
        if id_col and name_col:
            try:
                metric_name_maps[table] = {
                    int(mid): str(name)
                    for mid, name in conn.execute(f'SELECT "{id_col}", "{name_col}" FROM "{table}"')
                    if mid is not None and name is not None
                }
            except Exception:
                pass

    combined_metric_names: dict[int, str] = {}
    for mapping in metric_name_maps.values():
        combined_metric_names.update(mapping)

    for table in tables:
        low = table.lower()
        if "metric" not in low and "counter" not in low:
            continue
        cols = _columns(conn, table)
        value_col = _first_col(cols, ["value", "metricValue", "rawValue", "avg", "average", "sampleValue"])
        time_col = _first_col(cols, ["timestamp", "time", "start", "startTime", "rawTimestamp"])
        end_col = _first_col(cols, ["end", "endTime", "stop", "stopTime"])
        if not value_col or not time_col:
            continue
        text_col = _first_col(cols, ["name", "metricName", "metric", "description", "label", "text"])
        id_col = _first_col(cols, ["metricId", "metric_id", "metricNameId", "nameId", "type", "typeId"])
        table_is_sm = _looks_like_sm_metric(table)
        select_cols = [time_col]
        if end_col:
            select_cols.append(end_col)
        select_cols.append(value_col)
        if text_col:
            select_cols.append(text_col)
        elif id_col:
            select_cols.append(id_col)
        try:
            query = "SELECT " + ", ".join(f'"{c}"' for c in select_cols) + f' FROM "{table}"'
            params: list[Any] = []
            where: list[str] = []
            if id_col and not table_is_sm and combined_metric_names:
                primary_metric_ids = [
                    metric_id
                    for metric_id, metric_name in combined_metric_names.items()
                    if _looks_like_primary_sm_metric(metric_name)
                ]
                matching_metric_ids = primary_metric_ids or [
                    metric_id
                    for metric_id, metric_name in combined_metric_names.items()
                    if _looks_like_sm_metric(metric_name)
                ]
                if not matching_metric_ids:
                    continue
                placeholders = ",".join("?" for _ in matching_metric_ids)
                where.append(f'"{id_col}" IN ({placeholders})')
                params.extend(matching_metric_ids)
            if start_bound is not None:
                where.append(f'"{time_col}" >= ?')
                params.append(int(start_bound))
            if end_bound is not None:
                where.append(f'"{time_col}" <= ?')
                params.append(int(end_bound))
            if where:
                query += " WHERE " + " AND ".join(where)
            for row in conn.execute(query, params):
                idx = 0
                start = row[idx]; idx += 1
                end = None
                if end_col:
                    end = row[idx]; idx += 1
                value = row[idx]; idx += 1
                name = table
                if idx < len(row):
                    raw_name = row[idx]
                    if id_col and raw_name is not None:
                        try:
                            name = combined_metric_names.get(int(raw_name), _resolve_text(raw_name, strings) or table)
                        except Exception:
                            name = _resolve_text(raw_name, strings) or table
                    else:
                        name = _resolve_text(raw_name, strings) or table
                if not (table_is_sm or _looks_like_sm_metric(str(name))):
                    continue
                if start is None or value is None:
                    continue
                v = float(value)
                if math.isfinite(v) and 0.0 <= v <= 1.5:
                    v *= 100.0
                samples.append((int(start), int(end) if end is not None else None, float(min(max(v, 0.0), 100.0))))
        except Exception as exc:
            warnings.append(f"Failed reading GPU metric table {table}: {exc}")
    samples.sort(key=lambda x: x[0])
    if not samples:
        warnings.append("No parseable Nsight GPU SM utilization samples were found; sm_util_percent_mean will be null. Ensure GPU metrics are enabled and permitted.")
    return samples, warnings


def _sm_util_percent(samples: list[tuple[int, int | None, float]], start: int, end: int) -> float:
    if not samples:
        return math.nan
    weighted_sum = 0.0
    weight_total = 0.0
    point_values: list[float] = []
    for s, e, value in samples:
        if e is not None and e > s:
            if e <= start:
                continue
            if s >= end:
                break
            overlap = max(0, min(e, end) - max(s, start))
            if overlap > 0 and math.isfinite(value):
                weighted_sum += value * overlap
                weight_total += overlap
        else:
            if start <= s <= end and math.isfinite(value):
                point_values.append(value)
    if weight_total > 0:
        return float(weighted_sum / weight_total)
    if point_values:
        return float(np.mean(point_values))
    return math.nan


def _memory_peak_mb(events: list[tuple[int, int]], start: int, end: int) -> float:
    if not events:
        return math.nan
    current = 0
    idx = 0
    while idx < len(events) and events[idx][0] < start:
        current = max(0, current + events[idx][1])
        idx += 1
    peak = current
    while idx < len(events) and events[idx][0] <= end:
        current = max(0, current + events[idx][1])
        peak = max(peak, current)
        idx += 1
    return float(peak / 1024**2)


def _ns_to_ms(delta_ns: int | float) -> float:
    return float(delta_ns) / 1_000_000.0


def _samples_to_stats(
    samples_by_stage: dict[str, list[dict[str, float]]],
    include_samples: bool,
    resource_stage_keys: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage in tuple(resource_stage_keys or RESOURCE_STAGE_KEYS):
        stage_samples = samples_by_stage.get(stage, [])
        result[stage] = {}
        for key in RESOURCE_SERIES_KEYS:
            values = [sample.get(key, math.nan) for sample in stage_samples]
            stats = _series_stats(values)
            if include_samples:
                stats["samples"] = [float(x) if math.isfinite(float(x)) else None for x in values]
            result[stage][key] = stats
    return result


def _clean_nsight_sample(sample: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in sample.items() if not key.startswith("_")}


def _aggregate_nsight_call_group(group: list[dict[str, float]]) -> dict[str, float]:
    if not group:
        return _empty_sample()
    duration_ns = float(sum(sample.get("_duration_ns", 0.0) for sample in group))
    busy_ns = float(sum(sample.get("_gpu_busy_ns", 0.0) for sample in group))
    mem_values = [sample.get("gpu_memory_used_mb_max", math.nan) for sample in group]
    finite_mem = [value for value in mem_values if math.isfinite(value)]
    sm_values = np.asarray([sample.get("sm_util_percent_mean", math.nan) for sample in group], dtype=float)
    finite_sm = sm_values[np.isfinite(sm_values)]
    gpu_util = (
        float(min(max(busy_ns / duration_ns * 100.0, 0.0), 100.0))
        if duration_ns > 0 and busy_ns >= 0
        else math.nan
    )
    mem_peak = float(max(finite_mem)) if finite_mem else math.nan
    return {
        "duration_ms": _ns_to_ms(duration_ns),
        "gpu_profile_duration_ms": _ns_to_ms(duration_ns),
        "cpu_pytorch_duration_ms": math.nan,
        "cpu_process_percent_mean": math.nan,
        "cpu_process_core_percent_mean": math.nan,
        "gpu_util_percent_mean": gpu_util,
        "gpu_busy_ratio_percent_mean": gpu_util,
        "sm_util_percent_mean": float(np.mean(finite_sm)) if finite_sm.size else math.nan,
        "gpu_memory_used_mb_max": mem_peak,
        "nsight_cuda_memory_peak_mb": mem_peak,
        "sample_count": float(sum(sample.get("sample_count", 1.0) for sample in group)),
    }


def _aggregate_component_nsight_samples(
    samples_by_stage: dict[str, list[dict[str, float]]],
    component_call_counts: dict[str, list[int]] | None,
    warnings: list[str],
    resource_stage_keys: Iterable[str] | None = None,
    component_stage_keys: Iterable[str] | None = None,
) -> dict[str, list[dict[str, float]]]:
    selected_resource_keys = tuple(resource_stage_keys or RESOURCE_STAGE_KEYS)
    selected_component_keys = tuple(component_stage_keys or COMPONENT_STAGE_KEYS)
    aggregated: dict[str, list[dict[str, float]]] = {
        stage: [_clean_nsight_sample(sample) for sample in samples_by_stage.get(stage, [])]
        for stage in selected_resource_keys
    }
    if not component_call_counts:
        return aggregated

    for stage in selected_component_keys:
        raw = samples_by_stage.get(stage, [])
        counts = [int(count) for count in component_call_counts.get(stage, []) if int(count) >= 0]
        if not raw or not counts:
            continue
        if sum(counts) == len(raw):
            grouped = []
            index = 0
            for count in counts:
                if count == 0:
                    grouped.append(_empty_sample(sample_count=0.0))
                    continue
                grouped.append(_aggregate_nsight_call_group(raw[index : index + count]))
                index += count
            aggregated[stage] = grouped
            continue

        unique_counts = sorted(set(counts))
        if len(unique_counts) == 1 and unique_counts[0] > 0 and len(raw) % unique_counts[0] == 0:
            count = unique_counts[0]
            aggregated[stage] = [
                _aggregate_nsight_call_group(raw[index : index + count])
                for index in range(0, len(raw), count)
            ]
            warnings.append(
                f"Nsight {stage} ranges ({len(raw)}) did not exactly match child call samples "
                f"({sum(counts)}), grouped by constant call count {count}."
            )
        else:
            warnings.append(
                f"Could not aggregate Nsight {stage} ranges by iteration: "
                f"{len(raw)} ranges vs child call counts sum {sum(counts)}."
            )
    return aggregated


def parse_nsight_sqlite(
    sqlite_path: str,
    *,
    include_samples: bool = False,
    component_call_counts: dict[str, list[int]] | None = None,
    resource_stage_keys: Iterable[str] | None = None,
    component_stage_keys: Iterable[str] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    conn = sqlite3.connect(sqlite_path)
    try:
        strings = _string_map(conn)
        selected_resource_keys = tuple(resource_stage_keys or RESOURCE_STAGE_KEYS)
        selected_component_keys = tuple(component_stage_keys or COMPONENT_STAGE_KEYS)
        ranges, w = _extract_nvtx_ranges(conn, strings, selected_resource_keys)
        warnings.extend(w)
        all_ranges = [interval for stage_ranges in ranges.values() for interval in stage_ranges]
        range_start = min((start for start, _ in all_ranges), default=None)
        range_end = max((end for _, end in all_ranges), default=None)
        kernels, w = _extract_kernel_intervals(conn, range_start, range_end)
        warnings.extend(w)
        mem_events, w = _extract_memory_events(conn, strings)
        warnings.extend(w)
        sm_samples, w = _extract_sm_metric_samples(conn, strings, range_start, range_end)
        warnings.extend(w)
        # CPU process utilization is measured in non-profiled CPU/PyTorch passes.
        # Do not derive a different Nsight CPU-active metric here.
        samples_by_stage: dict[str, list[dict[str, float]]] = {stage: [] for stage in selected_resource_keys}
        for stage, stage_ranges in ranges.items():
            for start, end in stage_ranges:
                duration_ns = max(end - start, 1)
                gpu_busy_ns = _union_overlap_duration(kernels, start, end) if kernels else 0
                gpu_util = float(min(max(gpu_busy_ns / duration_ns * 100.0, 0.0), 100.0)) if kernels else math.nan
                cpu_util = math.nan
                mem_peak = _memory_peak_mb(mem_events, start, end) if mem_events else math.nan
                sm_util = _sm_util_percent(sm_samples, start, end) if sm_samples else math.nan
                samples_by_stage[stage].append(
                    {
                        "duration_ms": _ns_to_ms(duration_ns),
                        "gpu_profile_duration_ms": _ns_to_ms(duration_ns),
                        "cpu_pytorch_duration_ms": math.nan,
                        "_duration_ns": float(duration_ns),
                        "_gpu_busy_ns": float(gpu_busy_ns),
                        "cpu_process_percent_mean": cpu_util,
                        "cpu_process_core_percent_mean": math.nan,
                        "gpu_util_percent_mean": gpu_util,
                        "gpu_busy_ratio_percent_mean": gpu_util,
                        "sm_util_percent_mean": sm_util,
                        "gpu_memory_used_mb_max": mem_peak,
                        "nsight_cuda_memory_peak_mb": mem_peak,
                        "sample_count": 1.0,
                    }
                )

        samples_by_stage = _aggregate_component_nsight_samples(
            samples_by_stage,
            component_call_counts,
            warnings,
            resource_stage_keys=selected_resource_keys,
            component_stage_keys=selected_component_keys,
        )
        metrics = _samples_to_stats(samples_by_stage, include_samples, selected_resource_keys)
        return metrics, warnings
    finally:
        conn.close()


def _replace_plot_records(payload: dict[str, Any], resource_metrics: dict[str, Any]):
    benchmark = payload.get("benchmark", {})
    model_info = payload.get("model_info", {})
    hardware = payload.get("hardware", {})
    calls = payload.get("calls", {})
    resource_stage_keys, _, _ = infer_stage_keys(resource_metrics, model_info)
    payload["plot_records"] = _make_resource_plot_records(
        model_name=payload.get("model"),
        device_name=hardware.get("device_name"),
        hardware_name=hardware.get("torch_cuda_device_name"),
        model_info=model_info,
        benchmark=benchmark,
        resource_metrics=resource_metrics,
        calls=calls,
        metrics_source="nsight",
        stage_keys=resource_stage_keys,
    )
    payload["plot_records_raw"] = list(payload["plot_records"])


def _merge_psutil_cpu_stats(
    nsight_metrics: dict[str, Any],
    child_metrics: dict[str, Any],
    resource_stage_keys: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Preserve child-process psutil CPU-time statistics while using Nsight for GPU/memory."""
    merged = json.loads(json.dumps(_json_safe_value(nsight_metrics)))
    cpu_capacity = max(1, _available_logical_cpu_count())
    for stage in tuple(resource_stage_keys or RESOURCE_STAGE_KEYS):
        child_stage = child_metrics.get(stage, {}) if isinstance(child_metrics, dict) else {}
        child_cpu = child_stage.get("cpu_process_percent_mean") if isinstance(child_stage, dict) else None
        child_cpu_core = child_stage.get("cpu_process_core_percent_mean") if isinstance(child_stage, dict) else None
        if stage not in merged:
            merged[stage] = {}
        if isinstance(child_cpu, dict):
            merged[stage]["cpu_process_percent_mean"] = child_cpu
        else:
            merged[stage]["cpu_process_percent_mean"] = _series_stats([])
        if isinstance(child_cpu_core, dict):
            merged[stage]["cpu_process_core_percent_mean"] = child_cpu_core
        elif isinstance(child_cpu, dict):
            # Compatibility for old skeleton JSONs: old cpu_process_percent_mean
            # was psutil's raw core-equivalent value.
            core_stats = json.loads(json.dumps(child_cpu))
            normalized_stats = json.loads(json.dumps(child_cpu))
            for key in ("mean", "p5", "p95", "std", "min", "max"):
                value = normalized_stats.get(key)
                normalized_stats[key] = (
                    min(max(float(value) / cpu_capacity, 0.0), 100.0) if value is not None else None
                )
            if "samples" in normalized_stats:
                normalized_stats["samples"] = [
                    min(max(float(value) / cpu_capacity, 0.0), 100.0) if value is not None else None
                    for value in normalized_stats["samples"]
                ]
            merged[stage]["cpu_process_percent_mean"] = normalized_stats
            merged[stage]["cpu_process_core_percent_mean"] = core_stats
        else:
            merged[stage]["cpu_process_core_percent_mean"] = _series_stats([])
    return merged


def _component_call_counts_from_payload(
    payload: dict[str, Any],
    component_stage_keys: Iterable[str] | None = None,
) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    calls = payload.get("calls", {})
    if not isinstance(calls, dict):
        return result
    for stage in tuple(component_stage_keys or COMPONENT_STAGE_KEYS):
        stats = calls.get(f"{stage}_calls", {})
        if not isinstance(stats, dict):
            continue
        samples = stats.get("samples")
        if isinstance(samples, list) and samples:
            result[stage] = [int(round(float(value))) for value in samples if value is not None]
            continue
        mean = stats.get("mean")
        n = stats.get("n")
        if mean is not None and n:
            result[stage] = [int(round(float(mean)))] * int(n)
    return result


def _metric_stats_from(payload: dict[str, Any], stage: str, key: str) -> dict[str, Any]:
    metrics = payload.get("resource_metrics", {}) if isinstance(payload, dict) else {}
    stage_metrics = metrics.get(stage, {}) if isinstance(metrics, dict) else {}
    stats = stage_metrics.get(key) if isinstance(stage_metrics, dict) else None
    return stats if isinstance(stats, dict) else _series_stats([])


def _stage_exists(payload: dict[str, Any], stage: str) -> bool:
    metrics = payload.get("resource_metrics", {}) if isinstance(payload, dict) else {}
    return isinstance(metrics, dict) and stage in metrics


def _copy_metric_if_present(dst: dict[str, Any], payload: dict[str, Any], stage: str, src_key: str, dst_key: str | None = None):
    metrics = payload.get("resource_metrics", {}) if isinstance(payload, dict) else {}
    stage_metrics = metrics.get(stage, {}) if isinstance(metrics, dict) else {}
    if isinstance(stage_metrics, dict) and src_key in stage_metrics:
        dst[dst_key or src_key] = stage_metrics[src_key]


def _filter_resource_metrics(payload: dict[str, Any], keys: Iterable[str], stage_keys: Iterable[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for stage in tuple(stage_keys):
        if not _stage_exists(payload, stage):
            continue
        result[stage] = {key: _metric_stats_from(payload, stage, key) for key in keys}
    return result


def merge_full_resource_json(
    *,
    final_path: str,
    gpu_profile_json: str,
    cpu_e2e_json: str,
    cpu_components_json: str,
    include_samples: bool = False,
):
    """Merge the three independent resource passes into one paper-friendly JSON."""
    with open(gpu_profile_json, "r", encoding="utf-8") as f:
        gpu_payload = json.load(f)
    with open(cpu_e2e_json, "r", encoding="utf-8") as f:
        cpu_e2e_payload = json.load(f)
    with open(cpu_components_json, "r", encoding="utf-8") as f:
        cpu_comp_payload = json.load(f)

    model_info = gpu_payload.get("model_info", {})
    resource_stage_keys, component_stage_keys, stage_schema = infer_stage_keys(gpu_payload.get("resource_metrics", {}), model_info)
    stage_order = list(gpu_payload.get("plot_stage_order") or resource_stage_keys)

    gpu_keys = (
        "gpu_profile_duration_ms",
        "gpu_util_percent_mean",
        "gpu_busy_ratio_percent_mean",
        "sm_util_percent_mean",
        "gpu_memory_used_mb_max",
        "nsight_cuda_memory_peak_mb",
    )
    cpu_torch_keys = (
        "duration_ms",
        "cpu_pytorch_duration_ms",
        "cpu_process_percent_mean",
        "cpu_process_core_percent_mean",
        "torch_memory_allocated_mb_start",
        "torch_memory_allocated_mb_end",
        "torch_memory_allocated_mb_peak",
        "torch_memory_allocated_mb_increment_peak",
        "torch_memory_reserved_mb_start",
        "torch_memory_reserved_mb_end",
        "torch_memory_reserved_mb_peak",
        "torch_memory_reserved_mb_increment_peak",
        "sample_count",
    )

    unified: dict[str, Any] = {}
    for stage in stage_order:
        stage_metrics: dict[str, Any] = {}
        # GPU timeline/hardware metrics come only from Nsight.
        _copy_metric_if_present(stage_metrics, gpu_payload, stage, "duration_ms", "gpu_profile_duration_ms")
        for key in gpu_keys:
            _copy_metric_if_present(stage_metrics, gpu_payload, stage, key)
        # CPU/PyTorch allocator metrics come only from non-profiled runs.
        cpu_payload = cpu_e2e_payload if stage == "e2e" else cpu_comp_payload
        _copy_metric_if_present(stage_metrics, cpu_payload, stage, "duration_ms", "duration_ms")
        _copy_metric_if_present(stage_metrics, cpu_payload, stage, "duration_ms", "cpu_pytorch_duration_ms")
        for key in cpu_torch_keys:
            if key in {"duration_ms", "cpu_pytorch_duration_ms"}:
                continue
            _copy_metric_if_present(stage_metrics, cpu_payload, stage, key)
        if stage_metrics:
            unified[stage] = stage_metrics

    # Preserve call counts from the component CPU/PyTorch pass when available,
    # otherwise fall back to the Nsight skeleton.
    calls = cpu_comp_payload.get("calls") or gpu_payload.get("calls") or {}
    benchmark = dict(gpu_payload.get("benchmark", {}))
    benchmark.update(
        {
            "resource_metrics_source": "merged_three_run_protocol",
            "resource_sampler_backend": "full",
            "stage_schema": stage_schema,
            "latency_source": "separate_latency_scripts_not_this_resource_json",
            "resource_protocol": {
                "gpu_profile_run": "Nsight Systems NVTX run for GPU busy ratio, SM utilization, and CUDA memory peak.",
                "cpu_pytorch_e2e_run": "Non-profiled E2E run for CPU process utilization and PyTorch allocated/reserved memory peaks.",
                "cpu_pytorch_components_run": "Non-profiled component run for per-stage CPU process utilization and PyTorch allocated/reserved memory peaks.",
                "baseline_subtraction": False,
            },
            "child_json_paths": {
                "gpu_profile_json": os.path.abspath(gpu_profile_json),
                "cpu_pytorch_e2e_json": os.path.abspath(cpu_e2e_json),
                "cpu_pytorch_components_json": os.path.abspath(cpu_components_json),
            },
        }
    )

    hardware = gpu_payload.get("hardware", {})
    model_name = gpu_payload.get("model")
    plot_records = _make_resource_plot_records(
        model_name=model_name,
        device_name=hardware.get("device_name"),
        hardware_name=hardware.get("torch_cuda_device_name"),
        model_info=model_info,
        benchmark=benchmark,
        resource_metrics=unified,
        calls=calls,
        metrics_source="merged_three_run_protocol",
        stage_keys=stage_order,
    )

    warnings = []
    for payload in (gpu_payload, cpu_e2e_payload, cpu_comp_payload):
        warnings.extend(payload.get("warnings", []) if isinstance(payload.get("warnings", []), list) else [])

    gpu_profile_metrics = _filter_resource_metrics(gpu_payload, gpu_keys, stage_order)
    e2e_cpu_pytorch_metrics = _filter_resource_metrics(cpu_e2e_payload, cpu_torch_keys, ("e2e",))
    stage_cpu_pytorch_metrics = _filter_resource_metrics(cpu_comp_payload, cpu_torch_keys, stage_order)

    final_payload = {
        "schema_version": "vla_resource_v3_full_three_run",
        "model": model_name,
        "hardware": hardware,
        "model_info": model_info,
        "benchmark": benchmark,
        "resource_metrics": _json_safe_value(unified),
        "resource_metrics_raw": _json_safe_value(unified),
        "gpu_profile_metrics": _json_safe_value(gpu_profile_metrics),
        "e2e_cpu_pytorch_metrics": _json_safe_value(e2e_cpu_pytorch_metrics),
        "stage_cpu_pytorch_metrics": _json_safe_value(stage_cpu_pytorch_metrics),
        "calls": calls,
        "stage_hook_modules": cpu_comp_payload.get("stage_hook_modules", gpu_payload.get("stage_hook_modules", {})),
        "warnings": sorted(set(str(w) for w in warnings)),
        "plot_schema_version": "vla_plot_v1",
        "plot_stage_order": stage_order,
        "plot_stage_labels": PLOT_STAGE_LABELS,
        "plot_records": plot_records,
        "plot_records_raw": list(plot_records),
        "environment_baseline": None,
        "model_loaded_baseline": None,
        "resource_baseline": None,
        "memory_summary": {
            "definitions": {
                "nsight_cuda_memory_peak_mb": "Nsight CUDA allocation/free event based device-memory residency peak; not baseline-subtracted.",
                "torch_memory_allocated_mb_peak": "Non-profiled PyTorch active tensor memory peak over the measured window.",
                "torch_memory_reserved_mb_peak": "Non-profiled PyTorch caching allocator reserved-memory peak over the measured window.",
                "torch_memory_allocated_mb_increment_peak": "PyTorch allocated peak minus allocated memory at the start of the same measured window.",
                "torch_memory_reserved_mb_increment_peak": "PyTorch reserved peak minus reserved memory at the start of the same measured window.",
            },
            "e2e_nsight_cuda_memory_peak_mb": _metric_stats_from(gpu_payload, "e2e", "nsight_cuda_memory_peak_mb"),
            "e2e_torch_allocated_peak_mb": _metric_stats_from(cpu_e2e_payload, "e2e", "torch_memory_allocated_mb_peak"),
            "e2e_torch_reserved_peak_mb": _metric_stats_from(cpu_e2e_payload, "e2e", "torch_memory_reserved_mb_peak"),
            "e2e_torch_allocated_increment_peak_mb": _metric_stats_from(cpu_e2e_payload, "e2e", "torch_memory_allocated_mb_increment_peak"),
            "e2e_torch_reserved_increment_peak_mb": _metric_stats_from(cpu_e2e_payload, "e2e", "torch_memory_reserved_mb_increment_peak"),
        },
        "notes": {
            "latency": "Official latency is intentionally measured by the separate latency scripts. duration_ms in this file is only the resource-measurement window duration from the non-profiled CPU/PyTorch pass.",
            "gpu_util_percent_mean": "Backward-compatible alias for GPU kernel-busy time ratio: union of CUDA kernel execution intervals within the NVTX range divided by range wall time.",
            "gpu_busy_ratio_percent_mean": "Preferred name for GPU temporal utilization / GPU busy time ratio.",
            "sm_util_percent_mean": "Best-effort Nsight Systems GPU metric for SM utilization/SM Active over the same NVTX range. It is not baseline-subtracted.",
            "gpu_memory_used_mb_max": "Backward-compatible alias for nsight_cuda_memory_peak_mb in the merged schema.",
            "nsight_cuda_memory_peak_mb": "Nsight CUDA memory-event residency peak; not an NVML whole-device used-memory sample and not baseline-subtracted.",
            "cpu_process_percent_mean": "Non-profiled process CPU utilization from psutil process.cpu_times() delta, normalized by available logical CPU capacity to 0-100% whole-machine scale.",
            "cpu_process_core_percent_mean": "Non-profiled raw process CPU core-equivalent utilization; e.g. 2400% means roughly 24 logical cores saturated.",
            "torch_memory_allocated_reserved": "PyTorch allocator metrics come from non-profiled runs, separate from Nsight, to avoid profiler overhead contaminating these metrics.",
        },
    }

    final_payload = _json_safe_value(final_payload)
    out = os.path.abspath(os.path.expanduser(final_path))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(final_payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote merged three-run resource JSON to: {out}")


def finalize_json_with_nsight(path: str, *, report_base: str, nsys_path: str, include_samples: bool = False):
    json_path = os.path.abspath(os.path.expanduser(path))
    report_file = f"{report_base}.nsys-rep"
    sqlite_path = f"{report_base}.sqlite"
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Expected child JSON was not found: {json_path}")
    if not os.path.exists(report_file):
        raise FileNotFoundError(f"Expected Nsight report was not found: {report_file}")

    exported, warnings = _run_export_commands(nsys_path, report_file, sqlite_path)
    if exported is None:
        raise RuntimeError("Could not export Nsight report to SQLite. " + "\n".join(warnings[-3:]))

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    child_metrics = payload.get("resource_metrics", {})
    resource_stage_keys, component_stage_keys, stage_schema = infer_stage_keys(child_metrics, payload.get("model_info", {}))
    component_call_counts = _component_call_counts_from_payload(payload, component_stage_keys)
    nsight_metrics, parse_warnings = parse_nsight_sqlite(
        exported,
        include_samples=include_samples,
        component_call_counts=component_call_counts,
        resource_stage_keys=resource_stage_keys,
        component_stage_keys=component_stage_keys,
    )
    warnings.extend(parse_warnings)

    merged_metrics = _merge_psutil_cpu_stats(nsight_metrics, child_metrics, resource_stage_keys)

    payload["schema_version"] = "vla_resource_v2_nsight_cpu_times"
    benchmark = payload.setdefault("benchmark", {})
    benchmark["resource_metrics_source"] = "hybrid_cpu_times_nsight_gpu_memory"
    benchmark["stage_schema"] = stage_schema
    benchmark["resource_sampler_backend"] = "nsight"
    benchmark["resource_sampler_fallback_backend"] = "psutil-cpu-times-delta"
    benchmark["inline_cpu_sampler_backend"] = "psutil-cpu-times-delta"
    benchmark["cpu_capacity_logical_cores"] = _available_logical_cpu_count()
    benchmark["inline_gpu_sampler_backend"] = None
    benchmark["gpu_sampler_backend"] = "nsight-systems"
    benchmark["nsight"] = {
        **benchmark.get("nsight", {}),
        "requested": True,
        "active": False,
        "post_processed": True,
        "report_path": report_base,
        "expected_report_file": report_file,
        "sqlite_export_file": exported,
        "resource_metrics_source": "hybrid_cpu_times_nsight_gpu_memory",
    }

    payload["resource_metrics"] = _json_safe_value(merged_metrics)
    payload["resource_metrics_raw"] = _json_safe_value(merged_metrics)
    payload["plot_stage_order"] = list(resource_stage_keys)
    payload["memory_summary"] = {
        "definitions": {
            "e2e_nsight_gpu_memory_peak_mb": (
                "Peak CUDA device-memory residency reconstructed only from Nsight CUDA_GPU_MEMORY_USAGE_EVENTS allocation/free records inside E2E NVTX ranges."
            )
        },
        "e2e_nsight_gpu_memory_peak_mb": merged_metrics.get("e2e", {}).get("gpu_memory_used_mb_max", _series_stats([])),
    }
    payload["environment_baseline"] = None
    payload["model_loaded_baseline"] = None
    payload["resource_baseline"] = None
    payload.setdefault("notes", {})["resource_metrics"] = (
        "Hybrid metrics computed over the unchanged NVTX/stage ranges: CPU process utilization is computed inline from psutil process.cpu_times() deltas; "
        "GPU busy ratio and CUDA memory peak are patched from Nsight SQLite after the benchmark."
    )
    payload["notes"]["gpu_util_percent_mean"] = (
        "Kernel-busy ratio within the NVTX range: union of profiled CUDA kernel execution time divided by range wall time. "
        "This replaces NVML/nvidia-smi utilization and is model-process/timeline derived."
    )
    payload["notes"]["cpu_process_percent_mean"] = (
        "Process CPU utilization normalized by available logical CPU capacity, so values are in the 0-100% whole-machine scale. "
        "It is computed in the profiled child process from process.cpu_times() deltas over each original measurement window and is not derived from Nsight context-switch tables."
    )
    payload["notes"]["cpu_process_core_percent_mean"] = (
        "Raw process CPU core-equivalent utilization retained for diagnostics. For example, 2400% means roughly 24 logical cores saturated."
    )
    payload["notes"]["gpu_memory_used_mb_max"] = (
        "Peak CUDA device-memory residency reconstructed only from Nsight CUDA_GPU_MEMORY_USAGE_EVENTS allocation/free records. "
        "Memcpy/memset byte traffic is intentionally excluded."
    )
    payload["warnings"] = sorted(set(payload.get("warnings", []) + warnings))
    _replace_plot_records(payload, payload["resource_metrics"])
    payload["plot_records_memory_summary"] = []
    payload = _json_safe_value(payload)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"\nUpdated JSON with Nsight-derived metrics: {json_path}")
    print(f"Nsight report: {report_file}")
    print(f"Nsight SQLite export: {exported}")
