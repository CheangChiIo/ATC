# Nsight Systems-only resource profiling patch

This version changes the resource benchmark scripts so the only generated profiling artifact is the Nsight Systems report (`.nsys-rep`).

## What changed

- Keeps the same model loading, warmup, iteration counts, and stage hook insertion points.
- Automatically launches the script under `nsys profile` by default.
- Emits NVTX ranges for:
  - `data_processing`
  - `e2e`
  - `component/vision_encoder`
  - `component/llm_backbone`
  - `component/action_expert`
  - `component/dit`
- Disables inline NVML / nvidia-smi GPU sampling; CPU is measured with psutil process.cpu_times() deltas.
- Disables resource baseline sampling.
- Disables JSON result writing, even if `--output-json` is passed.

## Typical usage

```bash
python openvla_resource.py
python pi05_resource.py
python smolvla_resource.py
```

Reports are written under:

```text
nsight_reports/<script>_<timestamp>_<pid>.nsys-rep
```

## Useful options

```bash
# Change Nsight report directory
python openvla_resource.py --nsight-output-dir /workspace/nsight_reports

# Disable GPU metrics if unsupported by your driver / permissions
python openvla_resource.py --nsight-gpu-metrics-device none

# Add Nsight arguments, for example exporting stats
python openvla_resource.py --nsight-extra-args "--stats=true"
```

## Notes

The old JSON schema is intentionally not produced in this version. Use Nsight Systems to inspect GPU utilization, CUDA kernels, memory usage, CPU sampling, and NVTX stage ranges.
