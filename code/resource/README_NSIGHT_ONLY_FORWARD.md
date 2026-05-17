# Nsight + psutil CPU-time VLA resource benchmarking patch

This patch removes the runtime NVML / nvidia-smi GPU samplers from `resource_benchmarking.py`, while using `psutil.Process.cpu_times()` before/after each stage to compute interval CPU utilization.

## What is measured

The benchmark keeps the existing data-processing, E2E, and component timing boundaries. Each measured region is emitted as an NVTX range:

- `data_processing`
- `e2e`
- `component/vision_encoder`
- `component/llm_backbone`
- `component/action_expert`
- `component/dit`

During the profiled child process, CPU process utilization is computed from `psutil.Process.cpu_times()` deltas over the same stage windows. After the child exits, the parent exports the `.nsys-rep` report to SQLite and patches `--output-json` with Nsight-derived GPU/memory statistics:

- `duration_ms`
- `gpu_util_percent_mean`: CUDA-kernel busy ratio inside the NVTX range, not NVML/nvidia-smi GPU util.
- `gpu_memory_used_mb_max`: best-effort peak CUDA memory reconstructed from Nsight CUDA memory records.
- `cpu_process_percent_mean`: process CPU utilization computed as Δ(user+system CPU time) / Δwall time, normalized by available logical CPU cores.
- `sample_count`

Each field is summarized as p5 / mean / p95 / min / max / std / n. Unsupported metrics are not faked.

## Typical command

```bash
python openvla_resource.py --output-json results/openvla_resource.json
python pi05_resource.py --output-json results/pi05_resource.json
python smolvla_resource.py --output-json results/smolvla_resource.json
```

Outputs:

- `nsight_reports/<model>_resource_<timestamp>_<pid>.nsys-rep`
- `nsight_reports/<model>_resource_<timestamp>_<pid>.sqlite`
- the requested `--output-json`, patched with hybrid psutil CPU-time + Nsight-GPU/memory p5/mean/p95 metrics

## Notes

- There is no NVML / nvidia-smi GPU fallback. If `nsys` is missing, the script exits with an error.
- Baseline subtraction is disabled by default because GPU/memory metrics are derived from NVTX ranges in the Nsight timeline and CPU is reported as process-level CPU-time-delta utilization.
- CPU percent uses psutil process.cpu_times() deltas; it is not derived from Nsight context-switch tables and not from psutil cpu_percent() sampling.
