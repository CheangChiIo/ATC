#!/usr/bin/env python3
"""Shared resource benchmark runner for dual-system models.

It reuses resource_benchmarking.py's current Nsight+psutil CPU-time backend and
emits the same JSON/plot_records schema, with the dual-system stages:

  data_processing, system2_vision_encoder, system2_inference, system_bridge,
  system1_vision_encoder, system1_action_expert, e2e
"""

from __future__ import annotations

import argparse
import gc
from typing import Any

import numpy as np
import torch

import dual_system_benchmarking as ds
import resource_benchmarking as res


def _resource_model_info(spec: ds.DualSystemModelSpec, args: argparse.Namespace, adapter: ds.DualSystemAdapter) -> dict[str, Any]:
    info = ds.resolve_runtime_model_info(spec=spec, args=args, adapter=adapter)
    info["schema_family"] = "dual_system"
    return info


def benchmark_data_processing_resource(adapter, observations, *, num_iterations: int, warmup: int, sample_interval_ms: float):
    num_obs = len(observations)
    monitor = res.ResourceMonitor(sample_interval_ms=sample_interval_ms)
    gc.collect()
    for i in range(warmup):
        _ = adapter.prepare_inputs(observations[i % num_obs])
    res.cuda_synchronize()
    samples = []
    for i in range(num_iterations):
        _, sample = res.measure_callable(
            monitor,
            lambda obs=observations[i % num_obs]: adapter.prepare_inputs(obs),
            sync_cuda_before=True,
            sync_cuda_after=True,
            stage_name="data_processing",
        )
        samples.append(sample)
    return res.stage_samples_to_metrics(samples), monitor.warnings


def benchmark_e2e_resource(adapter, observations, *, num_iterations: int, warmup: int, sample_interval_ms: float):
    num_obs = len(observations)
    monitor = res.ResourceMonitor(sample_interval_ms=sample_interval_ms)
    gc.collect()
    for i in range(warmup):
        with torch.inference_mode():
            _ = ds.run_public_e2e(adapter, observations[i % num_obs])
    res.cuda_synchronize()
    samples = []
    for i in range(num_iterations):
        _, sample = res.measure_callable(
            monitor,
            lambda obs=observations[i % num_obs]: ds.run_public_e2e(adapter, obs),
            sync_cuda_before=True,
            sync_cuda_after=True,
            stage_name="e2e",
        )
        samples.append(sample)
    return res.stage_samples_to_metrics(samples), monitor.warnings


def _run_component_path_resource(adapter, prepared_inputs, recorder: res.StageResourceHooks):
    with recorder.measure("system2_vision_encoder"):
        system2_visual = adapter.run_system2_vision_encoder(prepared_inputs)
    with recorder.measure("system2_inference"):
        system2_output = adapter.run_system2_inference(prepared_inputs, system2_visual)
    with recorder.measure("system_bridge"):
        bridge_output = adapter.run_system_bridge(prepared_inputs, system2_output)
    with recorder.measure("system1_vision_encoder"):
        system1_visual = adapter.run_system1_vision_encoder(prepared_inputs)
    with recorder.measure("system1_action_expert"):
        output = adapter.run_system1_action_expert(prepared_inputs, bridge_output, system1_visual)
    return output


def benchmark_components_resource(adapter, observations, *, num_iterations: int, warmup: int, sample_interval_ms: float):
    num_obs = len(observations)
    prepared = [adapter.prepare_inputs(obs) for obs in observations]
    recorder = res.StageResourceHooks(
        sample_interval_ms=sample_interval_ms,
        stage_keys=ds.DUAL_SYSTEM_COMPONENT_KEYS,
    )
    gc.collect()
    for i in range(warmup):
        with torch.inference_mode():
            _ = ds.run_component_path(adapter, prepared[i % num_obs])
    res.cuda_synchronize()
    iter_records = []
    for i in range(num_iterations):
        recorder.start_iteration()
        with torch.inference_mode():
            _ = _run_component_path_resource(adapter, prepared[i % num_obs], recorder)
        res.cuda_synchronize()
        iter_records.append(recorder.finish_iteration())

    metrics: dict[str, Any] = {}
    for stage in ds.DUAL_SYSTEM_COMPONENT_KEYS:
        per_iter = []
        calls = []
        for item in iter_records:
            samples = item["records"].get(stage, [])
            if hasattr(res, "_aggregate_stage_samples"):
                sample = res._aggregate_stage_samples(samples)
            else:
                sample = samples[0] if samples else res._empty_sample()
            per_iter.append(sample)
            calls.append(float(item["calls"].get(stage, 0)))
        metrics[stage] = res.stage_samples_to_metrics(per_iter)
        metrics[f"{stage}_calls"] = np.asarray(calls, dtype=float)
    metrics["_stage_hook_modules"] = {stage: "explicit_adapter_boundary" for stage in ds.DUAL_SYSTEM_COMPONENT_KEYS}
    metrics["_stage_hook_warnings"] = list(getattr(recorder, "warnings", []))
    return metrics


def add_resource_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--resource-sample-interval-ms", type=float, default=5.0)
    parser.add_argument("--resource-sampler-backend", default="full", choices=["full", "nsight", "cpu_pytorch", "none"])
    parser.add_argument("--nsight-output-dir", default="nsight_reports")
    parser.add_argument("--nsight-gpu-metrics-device", default="all")
    parser.add_argument("--nsight-trace", default="cuda,nvtx,osrt")
    parser.add_argument("--nsight-extra-args", default="")
    parser.add_argument("--nsight-nsys-path", default=None)
    parser.add_argument("--nsight-disable-auto-launch", action=argparse.BooleanOptionalAction, default=False)
    # Kept because resource_benchmarking.write_resource_results_json expects it.
    parser.add_argument("--include-p95", action=argparse.BooleanOptionalAction, default=True)
    return parser


def run_resource_main(spec: ds.DualSystemModelSpec) -> None:
    parser = argparse.ArgumentParser(description=f"{spec.model_name} dual-system resource benchmark")
    ds.add_common_latency_args(parser, spec)
    parser.set_defaults(output_json=f"{spec.model_name.lower()}_resource.json")
    add_resource_args(parser)
    args = parser.parse_args()
    if args.torch_num_threads is not None:
        torch.set_num_threads(int(args.torch_num_threads))
    if args.torch_num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(args.torch_num_interop_threads))
        except RuntimeError:
            pass

    # Parent process launches Nsight and exits after post-processing. Child process continues below.
    res.maybe_reexec_under_nsight(args, report_prefix=f"{spec.model_name.lower()}_dual_system_resource")

    ds.set_seed(args.seed)
    device_name = ds.get_device_name()
    hardware_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    adapter = ds.load_adapter_from_args(args, spec)
    observations = ds.make_observations(args, spec)

    print(f"Model: {spec.model_name}")
    print(f"Device: {device_name}")
    print(f"Resource backend: {res.requested_resource_backend()} / inline={res.inline_resource_backend()}")
    print(f"Example source: {args.example_source}")

    warnings = []
    data_processing = None
    e2e = None
    comps = None
    if res.should_measure_data_processing(args):
        data_processing, w = benchmark_data_processing_resource(
            adapter,
            observations,
            num_iterations=ds.resolve_count(args.data_processing_iterations, args.num_iterations),
            warmup=ds.resolve_count(args.data_processing_warmup, args.warmup),
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        warnings.extend(w)
        res.print_resource_summary("Data Processing", data_processing, include_p95=args.include_p95)
    if res.should_measure_e2e(args):
        e2e, w = benchmark_e2e_resource(
            adapter,
            observations,
            num_iterations=ds.resolve_count(args.e2e_iterations, args.num_iterations),
            warmup=ds.resolve_count(args.e2e_warmup, args.warmup),
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        warnings.extend(w)
        res.print_resource_summary("E2E", e2e, include_p95=args.include_p95)
    if res.should_measure_components(args):
        comps = benchmark_components_resource(
            adapter,
            observations,
            num_iterations=ds.resolve_count(args.component_iterations, args.num_iterations),
            warmup=ds.resolve_count(args.component_warmup, args.warmup),
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        for stage in ds.DUAL_SYSTEM_COMPONENT_KEYS:
            res.print_resource_summary(ds.DUAL_SYSTEM_STAGE_LABELS[stage], comps[stage], include_p95=args.include_p95)

    components = res.build_components(data_processing, comps, e2e)
    components.setdefault("_resource_warnings", []).extend(warnings)
    model_info = _resource_model_info(spec, args, adapter)
    res.write_resource_results_json(
        args.output_json,
        model_name=spec.model_name,
        device_name=device_name,
        hardware_name=hardware_name,
        model_info=model_info,
        args=args,
        components=components,
        sample_interval_ms=args.resource_sample_interval_ms,
    )
