#!/usr/bin/env python3

"""
Resource benchmark script for OpenVLA inference.

This is intentionally separate from openvla.py's latency benchmark. It mirrors
the same warmup/iteration counts, E2E boundary, and phase boundaries, but records
GPU utilization, CPU utilization, and max GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import sys

import torch
import tyro

import openvla
import resource_benchmarking as res


def benchmark_data_processing_resource(
    model_id,
    processor,
    observation,
    *,
    device,
    dtype,
    center_crop=False,
    crop_scale=0.9,
    num_iterations=100,
    warmup=20,
    sample_interval_ms=5.0,
):
    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)
    monitor = res.ResourceMonitor(sample_interval_ms=sample_interval_ms)

    gc.collect()

    if warmup > 0:
        for i in range(warmup):
            obs = openvla.clone_observation(observations[i % num_obs])
            _ = openvla.prepare_model_inputs(
                model_id,
                processor,
                obs,
                device=device,
                dtype=dtype,
                center_crop=center_crop,
                crop_scale=crop_scale,
            )
        res.cuda_synchronize()
        gc.collect()

    samples = []
    for i in range(num_iterations):
        obs = openvla.clone_observation(observations[i % num_obs])
        _, sample = res.measure_callable(
            monitor,
            lambda obs=obs: openvla.prepare_model_inputs(
                model_id,
                processor,
                obs,
                device=device,
                dtype=dtype,
                center_crop=center_crop,
                crop_scale=crop_scale,
            ),
            sync_cuda_before=True,
            sync_cuda_after=True,
            stage_name="data_processing",
        )
        samples.append(sample)

    metrics = res.stage_samples_to_metrics(samples)
    return metrics, monitor.warnings


def benchmark_e2e_resource(
    model,
    model_id,
    processor,
    observation,
    *,
    device,
    dtype,
    unnorm_key,
    center_crop=False,
    crop_scale=0.9,
    do_sample=False,
    temperature=1.0,
    e2e_inference_path="manual_core",
    num_iterations=100,
    warmup=20,
    sample_interval_ms=5.0,
):
    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)
    monitor = res.ResourceMonitor(sample_interval_ms=sample_interval_ms)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if warmup > 0:
        for i in range(warmup):
            obs = openvla.clone_observation(observations[i % num_obs])
            _ = openvla.run_e2e_policy_inference(
                model,
                model_id,
                processor,
                obs,
                device=device,
                dtype=dtype,
                unnorm_key=unnorm_key,
                center_crop=center_crop,
                crop_scale=crop_scale,
                do_sample=do_sample,
                temperature=temperature,
                e2e_inference_path=e2e_inference_path,
            )
        res.cuda_synchronize()
        gc.collect()

    samples = []
    for i in range(num_iterations):
        obs = openvla.clone_observation(observations[i % num_obs])
        _, sample = res.measure_callable(
            monitor,
            lambda obs=obs: openvla.run_e2e_policy_inference(
                model,
                model_id,
                processor,
                obs,
                device=device,
                dtype=dtype,
                unnorm_key=unnorm_key,
                center_crop=center_crop,
                crop_scale=crop_scale,
                do_sample=do_sample,
                temperature=temperature,
                e2e_inference_path=e2e_inference_path,
            ),
            sync_cuda_before=True,
            sync_cuda_after=True,
            stage_name="e2e",
        )
        samples.append(sample)

    metrics = res.stage_samples_to_metrics(samples)
    return metrics, monitor.warnings


def benchmark_components_resource(
    model,
    model_id,
    processor,
    observation,
    *,
    device,
    dtype,
    unnorm_key,
    center_crop=False,
    crop_scale=0.9,
    do_sample=False,
    temperature=1.0,
    num_iterations=100,
    warmup=20,
    sample_interval_ms=5.0,
):
    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)
    recorder = res.StageResourceHooks(sample_interval_ms=sample_interval_ms)

    hook_warnings = [
        "OpenVLA has no diffusion/DiT module; the 'dit' key records cached "
        "autoregressive decoder hidden-state forwards after the first action token."
    ]
    hook_modules = {
        "vision_encoder": "vision_backbone + projector",
        "llm_backbone": "language_model decoder prefill to final hidden state, excluding lm_head",
        "action_expert": "manual action-token decode starting at first lm_head(last_hidden)",
        "dit": "cached language_model decoder forwards during action-token decode",
    }

    for i in range(warmup):
        obs = openvla.clone_observation(observations[i % num_obs])
        inputs = openvla.prepare_model_inputs(
            model_id,
            processor,
            obs,
            device=device,
            dtype=dtype,
            center_crop=center_crop,
            crop_scale=crop_scale,
        )
        with torch.inference_mode():
            _ = openvla.run_manual_action_generation(
                model,
                inputs,
                unnorm_key=unnorm_key,
                do_sample=do_sample,
                temperature=temperature,
            )
    res.cuda_synchronize()
    gc.collect()

    records = []
    for i in range(num_iterations):
        obs = openvla.clone_observation(observations[i % num_obs])
        inputs = openvla.prepare_model_inputs(
            model_id,
            processor,
            obs,
            device=device,
            dtype=dtype,
            center_crop=center_crop,
            crop_scale=crop_scale,
        )

        recorder.start_iteration()
        with torch.inference_mode():
            _ = openvla.run_manual_action_generation(
                model,
                inputs,
                unnorm_key=unnorm_key,
                timer=recorder,
                do_sample=do_sample,
                temperature=temperature,
            )
        res.cuda_synchronize()
        records.append(recorder.finish_iteration())

    components = res.records_to_component_metrics(records)
    for stage in ("vision_encoder", "llm_backbone", "action_expert"):
        calls = components[f"{stage}_calls"]
        if calls.size > 0 and max(calls) == 0:
            hook_warnings.append(
                f"Stage '{stage}' resource boundary was installed at "
                f"'{hook_modules[stage]}', but observed zero calls."
            )
    components["_stage_hook_warnings"] = hook_warnings
    components["_stage_hook_modules"] = hook_modules
    components["_resource_warnings"] = recorder.warnings
    return components


@dataclass
class ResourceBenchmarkConfig(openvla.BenchmarkConfig):
    """Configuration for OpenVLA inference resource benchmarking."""

    resource_sample_interval_ms: float = 0.0
    """Deprecated compatibility option; CPU now uses psutil process.cpu_times() before/after each stage, so no sampling interval is used."""

    resource_sampler_backend: str = "nsight"
    """Resource backend: nsight (default) or none. NVML/nvidia-smi GPU backends are removed; CPU uses psutil process.cpu_times() deltas."""

    nsight_output_dir: str = "nsight_reports"
    """Directory where Nsight Systems .nsys-rep reports are written."""

    nsight_nsys_path: str | None = None
    """Optional explicit path to the nsys executable."""

    nsight_gpu_metrics_device: str = "all"
    """Nsight Systems --gpu-metrics-device value; use 'none' to disable GPU metrics collection."""

    nsight_trace: str = "cuda,nvtx,osrt"
    """Nsight Systems trace domains."""

    nsight_extra_args: str = ""
    """Additional raw arguments appended to nsys profile, e.g. '--stats=true'."""

    nsight_disable_auto_launch: bool = False
    """Disable automatic nsys relaunch even when resource_sampler_backend='nsight'."""

    resource_subtract_baseline: bool = False
    """Deprecated compatibility option; Nsight-derived per-stage metrics are not baseline-subtracted."""

    resource_baseline_duration_ms: float = 1000.0
    """How long to sample the idle baseline before resource benchmarks."""

    resource_baseline_warmup_ms: float = 200.0
    """Idle settling time before baseline sampling starts."""


def main(args: ResourceBenchmarkConfig | None = None):
    if args is None:
        args = tyro.cli(ResourceBenchmarkConfig)

    nsight_warnings = res.maybe_reexec_under_nsight(args, "openvla_resource")

    openvla.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: No CUDA GPU detected. Benchmarking requires a GPU.")
        sys.exit(1)
    device_name = openvla.get_device_name()
    input_dtype = openvla._input_tensor_dtype(args.input_dtype)

    if args.example_source not in {"synthetic", "libero"}:
        raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))
    if args.example_source == "libero" and args.dataset_repo_id is None:
        raise ValueError("--dataset-repo-id is required when --example-source libero")
    if not (args.measure_data_processing or args.measure_e2e or args.measure_components):
        raise ValueError("At least one measurement group must be enabled.")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("--temperature must be positive when --do-sample=True.")
    if args.e2e_inference_path not in openvla.E2E_INFERENCE_PATHS:
        raise ValueError(
            f"Unsupported e2e_inference_path={args.e2e_inference_path!r}; "
            "use manual_core or public_predict_action."
        )

    data_processing_iterations = res.resolve_count(
        args.data_processing_iterations, args.num_iterations
    )
    data_processing_warmup = res.resolve_count(args.data_processing_warmup, args.warmup)
    e2e_iterations = res.resolve_count(args.e2e_iterations, args.num_iterations)
    e2e_warmup = res.resolve_count(args.e2e_warmup, args.warmup)
    component_iterations = res.resolve_count(args.component_iterations, args.num_iterations)
    component_warmup = res.resolve_count(args.component_warmup, args.warmup)

    print("=" * 100)
    print("OPENVLA INFERENCE RESOURCE BENCHMARK")
    print("=" * 100)
    print(f"Device: {device_name} ({torch.cuda.get_device_name(0)})")
    print(f"Model: {args.model_id}")
    print(f"Unnorm key: {args.unnorm_key}")
    print(f"Example source: {args.example_source}")
    print(f"Dataset repo/path: {args.dataset_repo_id}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset index: {args.dataset_index}")
    print(f"CPU method: psutil process.cpu_times() delta over each stage window")
    print(f"Resource sampler backend: {res.ResourceMonitor.sampler_backend()}")
    print("Inline GPU sampler backend: none (NVML/nvidia-smi removed)")
    nsight_meta = res.nsight_metadata()
    if nsight_meta.get("requested"):
        print(f"Nsight active: {nsight_meta.get('active')}")
        print(f"Nsight report: {nsight_meta.get('expected_report_file')}")
    print("Subtract resource baseline: False (disabled; GPU/memory are Nsight-derived, CPU is process.cpu_times() interval utilization)")
    print(f"Resource baseline duration: {args.resource_baseline_duration_ms} ms")
    print(f"E2E inference path: {args.e2e_inference_path}")
    print(f"Default measured iterations: {args.num_iterations}")
    print(f"Default warmup: {args.warmup}")
    print(f"Include p95: {args.include_p95}")
    print(
        "Measurements: "
        f"data_processing={args.measure_data_processing} "
        f"(n={data_processing_iterations}, warmup={data_processing_warmup}), "
        f"e2e={args.measure_e2e} (n={e2e_iterations}, warmup={e2e_warmup}), "
        f"components={args.measure_components} "
        f"(n={component_iterations}, warmup={component_warmup})"
    )
    print()

    resource_warnings = list(nsight_warnings)
    environment_baseline = None
    if args.resource_subtract_baseline:
        print("\n" + "-" * 50)
        print("Measuring Environment Resource Baseline (pre-model-load)...")
        print("-" * 50)
        environment_baseline, warnings = res.measure_idle_resource_baseline(
            duration_ms=args.resource_baseline_duration_ms,
            warmup_ms=args.resource_baseline_warmup_ms,
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        resource_warnings.extend(warnings)
        res.print_baseline_summary(
            environment_baseline, label="Environment baseline (pre-model-load)"
        )

    print("Loading model and processor...")
    model, processor = openvla._load_model_and_processor(args, device)
    model_device = openvla._get_model_device(model)
    action_tokens = model.get_action_dim(args.unnorm_key)
    action_horizon = 1
    print(f"Action Horizon: {action_horizon}")
    print(f"Action Tokens: {action_tokens}")
    print(f"Model device: {model_device}")

    observation = openvla.make_observation(args)

    resource_baseline = None
    if args.resource_subtract_baseline:
        print("\n" + "-" * 50)
        print("Measuring Model-Loaded Idle Resource Baseline...")
        print("-" * 50)
        resource_baseline, warnings = res.measure_idle_resource_baseline(
            duration_ms=args.resource_baseline_duration_ms,
            warmup_ms=args.resource_baseline_warmup_ms,
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        resource_warnings.extend(warnings)
        res.print_baseline_summary(
            resource_baseline, label="Model-loaded idle baseline"
        )

    data_processing_metrics = None
    if args.measure_data_processing:
        print("\n" + "-" * 50)
        print("Benchmarking Data Processing Resources...")
        print("-" * 50)
        data_processing_metrics, warnings = benchmark_data_processing_resource(
            args.model_id,
            processor,
            observation,
            device=model_device,
            dtype=input_dtype,
            center_crop=args.center_crop,
            crop_scale=args.crop_scale,
            num_iterations=data_processing_iterations,
            warmup=data_processing_warmup,
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        resource_warnings.extend(warnings)
        data_processing_metrics = res.subtract_baseline_from_metrics(
            data_processing_metrics, resource_baseline
        )
        res.print_resource_summary(
            "Data Processing", data_processing_metrics, include_p95=args.include_p95
        )

    e2e_metrics = None
    if args.measure_e2e:
        print("\n" + "-" * 50)
        print("Benchmarking E2E Inference Resources...")
        print("-" * 50)
        e2e_metrics, warnings = benchmark_e2e_resource(
            model,
            args.model_id,
            processor,
            observation,
            device=model_device,
            dtype=input_dtype,
            unnorm_key=args.unnorm_key,
            center_crop=args.center_crop,
            crop_scale=args.crop_scale,
            do_sample=args.do_sample,
            temperature=args.temperature,
            e2e_inference_path=args.e2e_inference_path,
            num_iterations=e2e_iterations,
            warmup=e2e_warmup,
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        resource_warnings.extend(warnings)
        e2e_metrics = res.subtract_baseline_from_metrics(e2e_metrics, resource_baseline)
        res.print_resource_summary("E2E", e2e_metrics, include_p95=args.include_p95)

    component_metrics = None
    if args.measure_components:
        print("\n" + "-" * 50)
        print("Benchmarking Phase Resource Breakdown...")
        print("-" * 50)
        component_metrics = benchmark_components_resource(
            model,
            args.model_id,
            processor,
            observation,
            device=model_device,
            dtype=input_dtype,
            unnorm_key=args.unnorm_key,
            center_crop=args.center_crop,
            crop_scale=args.crop_scale,
            do_sample=args.do_sample,
            temperature=args.temperature,
            num_iterations=component_iterations,
            warmup=component_warmup,
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        resource_warnings.extend(component_metrics.get("_resource_warnings", []))
        component_metrics = res.subtract_baseline_from_components(
            component_metrics, resource_baseline
        )
        for key, label in [
            ("vision_encoder", "Vision Encoder"),
            ("llm_backbone", "LLM Backbone"),
            ("action_expert", "Action Expert"),
            ("dit", "DiT/Decode-only"),
        ]:
            res.print_resource_summary(label, component_metrics[key], include_p95=args.include_p95)

    components = res.build_components(data_processing_metrics, component_metrics, e2e_metrics)
    components = res.subtract_baseline_from_components(components, resource_baseline)
    components["_environment_baseline"] = dict(environment_baseline) if environment_baseline else None
    components["_resource_warnings"] = sorted(set(resource_warnings))
    if component_metrics and component_metrics.get("_stage_hook_warnings"):
        print("  Stage hook warnings:")
        for warning in component_metrics["_stage_hook_warnings"]:
            print(f"    - {warning}")
    if components.get("_resource_warnings"):
        print("  Resource warnings:")
        for warning in components["_resource_warnings"]:
            print(f"    - {warning}")

    res.print_markdown_table(
        components,
        device_name,
        f"OpenVLA Inference Resource Usage ({action_tokens} action tokens):",
        include_p95=args.include_p95,
    )

    print("\n" + "=" * 100)
    print("DETAILED RESOURCE SUMMARY")
    print("=" * 100)
    print(f"\nHardware: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_id}")
    print(f"Unnorm key: {args.unnorm_key}")
    print(f"Action Horizon: {action_horizon}")
    print(f"Action Tokens: {action_tokens}")

    for key, label in [
        ("data_processing", "Data Processing"),
        ("e2e", "E2E"),
        ("vision_encoder", "Vision Encoder"),
        ("llm_backbone", "LLM Backbone"),
        ("action_expert", "Action Expert"),
        ("dit", "DiT/Decode-only"),
    ]:
        if key in components:
            res.print_resource_summary(label, components[key], include_p95=args.include_p95)

    if args.measure_components:
        import numpy as np

        print(
            "  Calls:           "
            f"vision={np.mean(components['vision_encoder_calls']):.1f}, "
            f"llm={np.mean(components['llm_backbone_calls']):.1f}, "
            f"action_expert={np.mean(components['action_expert_calls']):.1f}, "
            f"dit/decode={np.mean(components['dit_calls']):.1f}"
        )
    if components.get("_stage_hook_modules"):
        print("  Stage hook modules:")
        for stage, module_path in components["_stage_hook_modules"].items():
            print(f"    - {stage}: {module_path}")

    if args.output_json:
        res.write_resource_results_json(
            args.output_json,
            model_name="openvla",
            device_name=device_name,
            hardware_name=torch.cuda.get_device_name(0),
            model_info={
                "model_path": args.model_id,
                "checkpoint_dir": args.model_id,
                "train_config": None,
                "dataset_path": args.dataset_repo_id if args.example_source == "libero" else None,
                "dataset_root": args.dataset_root if args.example_source == "libero" else None,
                "embodiment_tag": None,
                "example_source": args.example_source,
                "action_horizon": int(action_horizon),
                "denoising_steps": int(action_tokens),
                "unnorm_key": args.unnorm_key,
                "e2e_inference_path": args.e2e_inference_path,
            },
            args=args,
            components=components,
            sample_interval_ms=args.resource_sample_interval_ms,
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    config = tyro.cli(ResourceBenchmarkConfig)
    main(config)
