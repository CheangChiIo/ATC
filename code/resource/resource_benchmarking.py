from __future__ import annotations

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


COMPONENT_STAGE_KEYS = (
    "vision_encoder",
    "llm_backbone",
    "action_expert",
    "dit",
    "system2_vision_encoder",
    "system2_inference",
    "system_bridge",
    "system1_vision_encoder",
    "system1_action_expert",
)
RESOURCE_STAGE_KEYS = (
    "data_processing",
    "vision_encoder",
    "llm_backbone",
    "action_expert",
    "dit",
    "system2_vision_encoder",
    "system2_inference",
    "system_bridge",
    "system1_vision_encoder",
    "system1_action_expert",
    "e2e",
)

# Keep the old primary metric keys that downstream plotting scripts are likely
# to read.  CPU is measured inline with psutil process.cpu_times() deltas; GPU/memory are patched in from
# Nsight.  cpu_process_percent_mean is normalized to the available logical CPU
# capacity so it stays in the familiar 0-100% range.  The raw multicore
# "core-equivalent" value is still emitted as cpu_process_core_percent_mean.
RESOURCE_SERIES_KEYS = (
    "duration_ms",
    "cpu_process_percent_mean",
    "cpu_process_core_percent_mean",
    "gpu_util_percent_mean",
    "gpu_memory_used_mb_max",
    "sample_count",
)

POSTPROCESSED_RESOURCE_KEYS = (
    "duration_ms",
    "cpu_process_percent_mean",
    "cpu_process_core_percent_mean",
    "gpu_util_percent_mean",
    "gpu_memory_used_mb_max",
    "sample_count",
)

_NSIGHT_LAUNCHED_ENV = "VLA_RESOURCE_NSIGHT_ACTIVE"
_NSIGHT_REPORT_ENV = "VLA_RESOURCE_NSIGHT_REPORT_PATH"
_NSIGHT_OUTPUT_DIR = os.environ.get("VLA_NSIGHT_OUTPUT_DIR", "nsight_reports")
_NSIGHT_GPU_METRICS_DEVICE = os.environ.get("VLA_NSIGHT_GPU_METRICS_DEVICE", "all")
_NSIGHT_TRACE = os.environ.get("VLA_NSIGHT_TRACE", "cuda,nvtx,osrt")
_NSIGHT_EXTRA_ARGS = os.environ.get("VLA_NSIGHT_EXTRA_ARGS", "")
_NSIGHT_NSYS_PATH = os.environ.get("VLA_NSIGHT_NSYS_PATH", "") or None
_NSIGHT_REPORT_PATH = os.environ.get(_NSIGHT_REPORT_ENV)
_RESOURCE_BACKEND = os.environ.get("VLA_RESOURCE_BACKEND", "nsight")

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
    value = (backend or "nsight").strip().lower()
    aliases = {"disable": "none", "disabled": "none", "off": "none", "false": "none", "0": "none"}
    value = aliases.get(value, value)
    if value not in {"nsight", "none"}:
        raise ValueError(
            f"Unsupported resource_sampler_backend={backend!r}. This Nsight-only version "
            "intentionally removed NVML / nvidia-smi backends; use 'nsight' or 'none'."
        )
    return value


def configure_resource_sampler(
    backend: str = "nsight",
    *,
    nsight_output_dir: str | None = None,
    nsight_gpu_metrics_device: str = "all",
    nsight_trace: str = "cuda,nvtx,osrt",
    nsight_extra_args: str = "",
    nsight_nsys_path: str | None = None,
    **_: Any,
):
    """Configure hybrid resource measurement.

    NVML and nvidia-smi GPU samplers are intentionally not implemented in this
    file. CPU process utilization is computed inline from psutil process.cpu_times() deltas over the same
    stage boundaries; GPU busy ratio and CUDA memory peak are derived offline
    from the generated Nsight Systems report.
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


def inline_resource_backend() -> str:
    return "psutil-cpu-times-delta"


def nsight_is_active() -> bool:
    return os.environ.get(_NSIGHT_LAUNCHED_ENV) == "1"


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
        "requested": requested_resource_backend() == "nsight",
        "active": nsight_is_active(),
        "nsys_path": _NSIGHT_NSYS_PATH or shutil.which("nsys"),
        "report_path": report,
        "expected_report_file": f"{report}.nsys-rep" if report else None,
        "gpu_metrics_device": _NSIGHT_GPU_METRICS_DEVICE or None,
        "trace": _NSIGHT_TRACE,
        "extra_args": _NSIGHT_EXTRA_ARGS,
        "inline_fallback_backend": "psutil-cpu-times-delta",
        "resource_metrics_source": "hybrid_cpu_times_nsight_gpu_memory",
    }


def maybe_reexec_under_nsight(args: Any, report_prefix: str) -> list[str]:
    """Relaunch the benchmark under `nsys profile`, then post-process JSON.

    The parent process launches Nsight Systems and waits for the profiled child
    to finish. The child writes the normal JSON metadata/duration skeleton. The
    parent then exports the `.nsys-rep` to SQLite, derives per-NVTX-stage
    resource metrics, and patches the JSON in-place. This keeps all existing
    measurement boundaries inside the child process unchanged.
    """
    warnings: list[str] = []
    configure_resource_sampler(
        backend=getattr(args, "resource_sampler_backend", _RESOURCE_BACKEND),
        nsight_output_dir=getattr(args, "nsight_output_dir", _NSIGHT_OUTPUT_DIR),
        nsight_gpu_metrics_device=getattr(args, "nsight_gpu_metrics_device", _NSIGHT_GPU_METRICS_DEVICE),
        nsight_trace=getattr(args, "nsight_trace", _NSIGHT_TRACE),
        nsight_extra_args=getattr(args, "nsight_extra_args", _NSIGHT_EXTRA_ARGS),
        nsight_nsys_path=getattr(args, "nsight_nsys_path", _NSIGHT_NSYS_PATH),
    )

    if requested_resource_backend() != "nsight":
        warnings.append("resource_sampler_backend='none': only duration/NVTX markers are recorded.")
        return warnings
    if nsight_is_active():
        return warnings
    if bool(getattr(args, "nsight_disable_auto_launch", False)):
        warnings.append(
            "Nsight auto-launch is disabled. NVML/nvidia-smi fallback has been removed, "
            "so only duration/NVTX markers are available in this run."
        )
        return warnings

    nsys = _NSIGHT_NSYS_PATH or shutil.which("nsys")
    if not nsys:
        raise SystemExit(
            "ERROR: 'nsys' was not found. This Nsight-only resource benchmark has no "
            "NVML/nvidia-smi fallback. Install Nsight Systems or set --nsight-nsys-path."
        )

    report_base = _make_nsight_report_base(report_prefix, _NSIGHT_OUTPUT_DIR)
    cmd = [
        nsys,
        "profile",
        "--force-overwrite=true",
        f"--trace={_NSIGHT_TRACE}",
        "--sample=cpu",
        "--cpuctxsw=process-tree",
        "--cuda-memory-usage=true",
    ]
    if _NSIGHT_GPU_METRICS_DEVICE and _NSIGHT_GPU_METRICS_DEVICE.lower() not in {"none", "false", "off", "0"}:
        # Nsight CLI spelling changed across releases. `--gpu-metrics-device` is
        # accepted by many releases; if a local release requires a different
        # spelling, pass it through --nsight-extra-args and set this to 'none'.
        cmd.append(f"--gpu-metrics-device={_NSIGHT_GPU_METRICS_DEVICE}")
    if _NSIGHT_EXTRA_ARGS:
        cmd.extend(shlex.split(_NSIGHT_EXTRA_ARGS))
    cmd.extend(["--output", report_base, sys.executable, *sys.argv])

    env = os.environ.copy()
    env[_NSIGHT_LAUNCHED_ENV] = "1"
    env[_NSIGHT_REPORT_ENV] = report_base
    env["VLA_RESOURCE_BACKEND"] = "nsight"
    env["VLA_NSIGHT_OUTPUT_DIR"] = _NSIGHT_OUTPUT_DIR
    env["VLA_NSIGHT_GPU_METRICS_DEVICE"] = _NSIGHT_GPU_METRICS_DEVICE
    env["VLA_NSIGHT_TRACE"] = _NSIGHT_TRACE
    env["VLA_NSIGHT_EXTRA_ARGS"] = _NSIGHT_EXTRA_ARGS

    print("\n" + "=" * 100)
    print("Launching benchmark under Nsight Systems; NVML/nvidia-smi sampling is disabled; CPU uses psutil cpu_times() deltas.")
    print(f"Nsight report base: {report_base}")
    print("Command:")
    print("  " + " ".join(shlex.quote(part) for part in cmd))
    print("=" * 100 + "\n")
    completed = subprocess.run(cmd, env=env)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)

    output_json = getattr(args, "output_json", None)
    if output_json:
        try:
            finalize_json_with_nsight(
                output_json,
                report_base=report_base,
                nsys_path=nsys,
                include_samples=bool(getattr(args, "include_samples_in_json", False)),
            )
        except Exception as exc:  # Do not hide the successful benchmark run.
            print(f"WARNING: Nsight post-processing failed: {exc}")
    else:
        print(
            "No --output-json path was provided, so only the .nsys-rep report was generated. "
            "Nsight-derived p5/mean/p95 JSON metrics require --output-json."
        )
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
    """CPU-only interval meter over the original measurement windows.

    NVML and nvidia-smi GPU sampling are intentionally removed. CPU utilization
    is computed from process.cpu_times() before/after each stage, not from
    cpu_percent() sampling. GPU busy ratio and CUDA memory peak are patched into
    the JSON from Nsight after the child benchmark exits.
    """

    def __init__(self, sample_interval_ms: float = 0.0, gpu_index: int = 0):
        # Kept only for CLI compatibility. CPU is now an interval delta metric,
        # so there is no sampling interval and no background sampling thread.
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
        return "nsight-systems+psutil-cpu-times"

    @classmethod
    def inline_sampler_backend(cls) -> str:
        return "psutil-cpu-times-delta"

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
        if self._psutil is None or self._process is None:
            return
        try:
            self._cpu_start_time = self._cpu_time_seconds(self._process.cpu_times())
        except Exception as exc:
            self._warn_once(f"psutil CPU-time sampling failed at stage start: {exc}")

    def _capture_cpu_stop(self):
        if self._psutil is None or self._process is None:
            return
        self._cpu_end_wall = time.perf_counter()
        try:
            self._cpu_end_time = self._cpu_time_seconds(self._process.cpu_times())
        except Exception as exc:
            self._warn_once(f"psutil CPU-time sampling failed at stage stop: {exc}")
            return
        if self._cpu_start_wall is None or self._cpu_start_time is None:
            return
        elapsed = max(self._cpu_end_wall - self._cpu_start_wall, 1e-9)
        cpu_delta = max(0.0, self._cpu_end_time - self._cpu_start_time)
        core_percent = cpu_delta / elapsed * 100.0
        # Normalize by the logical CPU capacity so this stays in the familiar
        # 0-100% whole-machine scale. The unnormalized core-equivalent value is
        # emitted separately as cpu_process_core_percent_mean.
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

    def cpu_metrics(self) -> tuple[float, float]:
        """Return (total-capacity %, raw core-equivalent %)."""
        return self._cpu_process_percent_mean, self._cpu_core_percent_mean

    def start(self):
        self._capture_cpu_start()

    def stop(self):
        self._capture_cpu_stop()

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
) -> dict[str, float]:
    return {
        "duration_ms": float(duration_ms),
        "cpu_process_percent_mean": float(cpu_process_percent_mean),
        "cpu_process_core_percent_mean": float(cpu_process_core_percent_mean),
        "gpu_util_percent_mean": math.nan,
        "gpu_memory_used_mb_max": math.nan,
        "sample_count": float(sample_count),
    }


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
    )
    return result, sample


def stage_samples_to_metrics(samples: list[dict[str, float]]) -> dict[str, np.ndarray]:
    return {
        key: np.asarray([sample.get(key, math.nan) for sample in samples], dtype=float)
        for key in RESOURCE_SERIES_KEYS
    }


class StageResourceHooks:
    def __init__(self, sample_interval_ms: float = 0.0):
        self.sample_interval_ms = float(sample_interval_ms)
        self.active = False
        self.records: dict[str, list[dict[str, float]]] = {}
        self.calls: dict[str, int] = {}
        self.warnings: list[str] = []

    def start_iteration(self):
        self.active = True
        self.records = {stage: [] for stage in COMPONENT_STAGE_KEYS}
        self.calls = {stage: 0 for stage in COMPONENT_STAGE_KEYS}

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
    durations = np.asarray([s.get("duration_ms", math.nan) for s in samples], dtype=float)
    finite_duration = durations[np.isfinite(durations)]
    weights = np.where(np.isfinite(durations) & (durations > 0), durations, 0.0)
    cpu_values = np.asarray([s.get("cpu_process_percent_mean", math.nan) for s in samples], dtype=float)
    cpu_core_values = np.asarray([s.get("cpu_process_core_percent_mean", math.nan) for s in samples], dtype=float)
    sample_counts = np.asarray([s.get("sample_count", 0.0) for s in samples], dtype=float)
    finite_counts = sample_counts[np.isfinite(sample_counts)]

    def weighted_cpu(values: np.ndarray) -> float:
        valid = np.isfinite(values) & (weights > 0)
        if valid.any():
            return float(np.average(values[valid], weights=weights[valid]))
        finite = values[np.isfinite(values)]
        return float(np.mean(finite)) if finite.size else math.nan

    return _empty_sample(
        float(np.sum(finite_duration)) if finite_duration.size else math.nan,
        cpu_process_percent_mean=weighted_cpu(cpu_values),
        cpu_process_core_percent_mean=weighted_cpu(cpu_core_values),
        sample_count=float(np.sum(finite_counts)) if finite_counts.size else 0.0,
    )


def records_to_component_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    components: dict[str, Any] = {}
    for stage in COMPONENT_STAGE_KEYS:
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
    print(f"  {label} (Nsight-derived after post-process; child records boundary duration only):")
    print(f"    Duration:        {_format_series(metrics.get('duration_ms', []), ' ms', include_p95)}")
    if "gpu_util_percent_mean" in metrics:
        print(f"    GPU busy:        {_format_series(metrics['gpu_util_percent_mean'], '%', include_p95)}")
    if "gpu_memory_used_mb_max" in metrics:
        print(f"    GPU memory peak: {_format_series(metrics['gpu_memory_used_mb_max'], ' MB', include_p95)}")
    if "cpu_process_percent_mean" in metrics:
        print(f"    CPU total:       {_format_series(metrics['cpu_process_percent_mean'], '%', include_p95)}")


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


def _stage_metrics_for_json(components: dict[str, Any], *, include_samples: bool, raw: bool) -> dict[str, Any]:
    payload = {}
    for stage in RESOURCE_STAGE_KEYS:
        if stage not in components:
            continue
        metrics = components[stage]
        payload[stage] = _summarize_metric_dict(metrics, include_samples)
    return payload


def _calls_for_json(components: dict[str, Any], include_samples: bool) -> dict[str, Any]:
    calls = {}
    for stage in COMPONENT_STAGE_KEYS:
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
    "cpu_process_percent_mean": "%",
    "cpu_process_core_percent_mean": "%",
    "gpu_util_percent_mean": "%",
    "gpu_memory_used_mb_max": "MB",
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


def _make_resource_plot_records(*, model_name, device_name, hardware_name, model_info, benchmark, resource_metrics, calls, metrics_source: str):
    base = _make_plot_base_record(
        model_name=model_name,
        device_name=device_name,
        hardware_name=hardware_name,
        model_info=model_info,
        benchmark=benchmark,
    )
    records: list[dict[str, Any]] = []
    for stage in RESOURCE_STAGE_KEYS:
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
        "gpu_sampler_backend": "nsight-systems",
        "inline_cpu_sampler_backend": ResourceMonitor.inline_sampler_backend(),
        "inline_gpu_sampler_backend": None,
        "resource_sampler_backend": requested_resource_backend(),
        "resource_sampler_fallback_backend": "psutil-cpu-times-delta",
        "resource_metrics_source": "duration_cpu_times_before_nsight_postprocess",
        "nsight": nsight_metadata(),
        "resource_sample_interval_mode": "psutil_cpu_times_delta",
    }
    resource_metrics = _stage_metrics_for_json(components, include_samples=include_samples, raw=False)
    calls = _calls_for_json(components, include_samples)
    memory_summary = build_memory_summary(components, include_samples=include_samples)

    payload = {
        "schema_version": "vla_resource_v2_nsight_cpu_times",
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
                "Child process writes duration and psutil CPU-time statistics over the original stage windows; "
                "parent process patches GPU busy ratio and CUDA memory peak from Nsight after .nsys-rep export."
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
            "removed_backends": "NVML and nvidia-smi GPU samplers have been removed. CPU process utilization uses psutil process.cpu_times() deltas.",
        },
    }
    payload["plot_schema_version"] = "vla_plot_v1"
    payload["plot_stage_order"] = list(RESOURCE_STAGE_KEYS)
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


def _extract_nvtx_ranges(conn: sqlite3.Connection, strings: dict[int, str]) -> tuple[dict[str, list[tuple[int, int]]], list[str]]:
    warnings: list[str] = []
    ranges: dict[str, list[tuple[int, int]]] = {stage: [] for stage in RESOURCE_STAGE_KEYS}
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
                    ranges[_STAGE_NVTX_TO_KEY[name]].append((int(start), int(end)))
        except Exception as exc:
            warnings.append(f"Failed reading NVTX table {table}: {exc}")
    found = sum(len(v) for v in ranges.values())
    if found == 0:
        warnings.append("No parseable NVTX stage ranges were found; check that torch.cuda.nvtx markers were captured.")
    for stage in ranges:
        ranges[stage].sort(key=lambda x: x[0])
    return ranges, warnings


def _extract_kernel_intervals(conn: sqlite3.Connection) -> tuple[list[tuple[int, int]], list[str]]:
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
            for start, end in conn.execute(f'SELECT "{start_col}", "{end_col}" FROM "{table}"'):
                if start is not None and end is not None and int(end) > int(start):
                    intervals.append((int(start), int(end)))
        except Exception as exc:
            warnings.append(f"Failed reading CUDA kernel table {table}: {exc}")
    intervals.sort(key=lambda x: x[0])
    if not intervals:
        warnings.append("No CUDA kernel intervals were found; GPU busy percent will be null.")
    return intervals, warnings


def _union_overlap_duration(intervals: list[tuple[int, int]], start: int, end: int) -> int:
    clipped: list[tuple[int, int]] = []
    for s, e in intervals:
        if e <= start:
            continue
        if s >= end:
            break
        cs, ce = max(s, start), min(e, end)
        if ce > cs:
            clipped.append((cs, ce))
    if not clipped:
        return 0
    clipped.sort()
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


def _samples_to_stats(samples_by_stage: dict[str, list[dict[str, float]]], include_samples: bool) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for stage in RESOURCE_STAGE_KEYS:
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
    gpu_util = (
        float(min(max(busy_ns / duration_ns * 100.0, 0.0), 100.0))
        if duration_ns > 0 and busy_ns >= 0
        else math.nan
    )
    return {
        "duration_ms": _ns_to_ms(duration_ns),
        "cpu_process_percent_mean": math.nan,
        "cpu_process_core_percent_mean": math.nan,
        "gpu_util_percent_mean": gpu_util,
        "gpu_memory_used_mb_max": float(max(finite_mem)) if finite_mem else math.nan,
        "sample_count": float(sum(sample.get("sample_count", 1.0) for sample in group)),
    }


def _aggregate_component_nsight_samples(
    samples_by_stage: dict[str, list[dict[str, float]]],
    component_call_counts: dict[str, list[int]] | None,
    warnings: list[str],
) -> dict[str, list[dict[str, float]]]:
    aggregated: dict[str, list[dict[str, float]]] = {
        stage: [_clean_nsight_sample(sample) for sample in samples_by_stage.get(stage, [])]
        for stage in RESOURCE_STAGE_KEYS
    }
    if not component_call_counts:
        return aggregated

    for stage in COMPONENT_STAGE_KEYS:
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
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    conn = sqlite3.connect(sqlite_path)
    try:
        strings = _string_map(conn)
        ranges, w = _extract_nvtx_ranges(conn, strings)
        warnings.extend(w)
        kernels, w = _extract_kernel_intervals(conn)
        warnings.extend(w)
        mem_events, w = _extract_memory_events(conn, strings)
        warnings.extend(w)
        # CPU process utilization is measured inline with psutil in the child JSON.
        # Do not derive a different Nsight CPU-active metric here.
        samples_by_stage: dict[str, list[dict[str, float]]] = {stage: [] for stage in RESOURCE_STAGE_KEYS}
        for stage, stage_ranges in ranges.items():
            for start, end in stage_ranges:
                duration_ns = max(end - start, 1)
                gpu_busy_ns = _union_overlap_duration(kernels, start, end) if kernels else 0
                gpu_util = float(min(max(gpu_busy_ns / duration_ns * 100.0, 0.0), 100.0)) if kernels else math.nan
                cpu_util = math.nan
                mem_peak = _memory_peak_mb(mem_events, start, end) if mem_events else math.nan
                samples_by_stage[stage].append(
                    {
                        "duration_ms": _ns_to_ms(duration_ns),
                        "_duration_ns": float(duration_ns),
                        "_gpu_busy_ns": float(gpu_busy_ns),
                        "cpu_process_percent_mean": cpu_util,
                        "cpu_process_core_percent_mean": math.nan,
                        "gpu_util_percent_mean": gpu_util,
                        "gpu_memory_used_mb_max": mem_peak,
                        "sample_count": 1.0,
                    }
                )

        samples_by_stage = _aggregate_component_nsight_samples(
            samples_by_stage, component_call_counts, warnings
        )
        metrics = _samples_to_stats(samples_by_stage, include_samples)
        return metrics, warnings
    finally:
        conn.close()


def _replace_plot_records(payload: dict[str, Any], resource_metrics: dict[str, Any]):
    benchmark = payload.get("benchmark", {})
    model_info = payload.get("model_info", {})
    hardware = payload.get("hardware", {})
    calls = payload.get("calls", {})
    payload["plot_records"] = _make_resource_plot_records(
        model_name=payload.get("model"),
        device_name=hardware.get("device_name"),
        hardware_name=hardware.get("torch_cuda_device_name"),
        model_info=model_info,
        benchmark=benchmark,
        resource_metrics=resource_metrics,
        calls=calls,
        metrics_source="nsight",
    )
    payload["plot_records_raw"] = list(payload["plot_records"])


def _merge_psutil_cpu_stats(nsight_metrics: dict[str, Any], child_metrics: dict[str, Any]) -> dict[str, Any]:
    """Preserve child-process psutil CPU-time statistics while using Nsight for GPU/memory."""
    merged = json.loads(json.dumps(_json_safe_value(nsight_metrics)))
    cpu_capacity = max(1, _available_logical_cpu_count())
    for stage in RESOURCE_STAGE_KEYS:
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


def _component_call_counts_from_payload(payload: dict[str, Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    calls = payload.get("calls", {})
    if not isinstance(calls, dict):
        return result
    for stage in COMPONENT_STAGE_KEYS:
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

    component_call_counts = _component_call_counts_from_payload(payload)
    nsight_metrics, parse_warnings = parse_nsight_sqlite(
        exported,
        include_samples=include_samples,
        component_call_counts=component_call_counts,
    )
    warnings.extend(parse_warnings)

    child_metrics = payload.get("resource_metrics", {})
    merged_metrics = _merge_psutil_cpu_stats(nsight_metrics, child_metrics)

    payload["schema_version"] = "vla_resource_v2_nsight_cpu_times"
    benchmark = payload.setdefault("benchmark", {})
    benchmark["resource_metrics_source"] = "hybrid_cpu_times_nsight_gpu_memory"
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
