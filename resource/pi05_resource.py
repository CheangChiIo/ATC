#!/usr/bin/env python3

"""
Resource benchmark script for OpenPI pi0.5 inference.

This stays separate from pi05.py's latency benchmark. It mirrors the same
warmup/iteration counts, E2E boundary, and component boundaries while measuring
GPU utilization, CPU utilization, and max GPU memory.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import sys
import types
from typing import Any

import torch
import tyro

import pi05
import resource_benchmarking as res


def install_vision_encoder_resource_timer(policy, recorder):
    model = getattr(policy, "_model", None)
    paligemma_with_expert = pi05._get_nested_attr(model, "paligemma_with_expert") if model else None
    if paligemma_with_expert is None or not hasattr(paligemma_with_expert, "embed_image"):
        return None, (
            "Could not find _model.paligemma_with_expert.embed_image; "
            "vision_encoder resource measurement is unavailable."
        )

    original = paligemma_with_expert.embed_image

    def measured_embed_image(*args, **kwargs):
        if not recorder.active:
            return original(*args, **kwargs)
        with recorder.measure("vision_encoder"):
            return original(*args, **kwargs)

    paligemma_with_expert.embed_image = measured_embed_image
    return original, None


def install_vlm_forward_resource_timer(policy, recorder):
    model = getattr(policy, "_model", None)
    paligemma_with_expert = pi05._get_nested_attr(model, "paligemma_with_expert") if model else None
    if paligemma_with_expert is None or not hasattr(paligemma_with_expert, "forward"):
        return None, (
            "Could not find _model.paligemma_with_expert.forward; "
            "llm_backbone/dit resource measurement is unavailable."
        )

    original = paligemma_with_expert.forward

    def measured_forward(*args, **kwargs):
        stage = pi05._classify_pi05_vlm_forward(args, kwargs) if recorder.active else None
        if stage is None:
            return original(*args, **kwargs)
        with recorder.measure(stage):
            return original(*args, **kwargs)

    paligemma_with_expert.forward = measured_forward
    return original, None


def install_action_expert_resource_timer(policy, recorder):
    model = getattr(policy, "_model", None)
    if model is None or not all(
        hasattr(model, name)
        for name in (
            "sample_noise",
            "_preprocess_observation",
            "embed_prefix",
            "denoise_step",
            "paligemma_with_expert",
            "_prepare_attention_masks_4d",
        )
    ):
        return None, None, (
            "Could not find the expected PI0Pytorch sample_actions internals; "
            "action_expert resource measurement is unavailable."
        )

    if not hasattr(model, "config") or not hasattr(model.config, "action_horizon"):
        return None, None, "PI0Pytorch config.action_horizon is unavailable."

    original_model_sample_actions = model.sample_actions
    original_policy_sample_actions = getattr(policy, "_sample_actions", None)

    @torch.no_grad()
    def measured_sample_actions(self, device, observation, noise=None, num_steps=10):
        if not recorder.active:
            return original_model_sample_actions(
                device,
                observation,
                noise=noise,
                num_steps=num_steps,
            )

        bsize = observation.state.shape[0]
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        with recorder.measure("action_expert"):
            if noise is None:
                actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
                noise = self.sample_noise(actions_shape, device)

            dt = -1.0 / num_steps
            dt = torch.tensor(dt, dtype=torch.float32, device=device)

            x_t = noise
            denoise_time = torch.tensor(1.0, dtype=torch.float32, device=device)

            while denoise_time >= -dt / 2:
                expanded_time = denoise_time.expand(bsize)
                v_t = self.denoise_step(
                    state,
                    prefix_pad_masks,
                    past_key_values,
                    x_t,
                    expanded_time,
                )
                x_t = x_t + dt * v_t
                denoise_time += dt

        return x_t

    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    bound = types.MethodType(measured_sample_actions, model)
    model.sample_actions = bound
    policy._sample_actions = bound
    return original_model_sample_actions, original_policy_sample_actions, None


def benchmark_data_processing_resource(
    policy,
    observation,
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
            obs = observations[i % num_obs]
            _ = pi05.prepare_model_inputs(policy, obs)
        res.cuda_synchronize()
        gc.collect()

    samples = []
    for i in range(num_iterations):
        obs = observations[i % num_obs]
        _, sample = res.measure_callable(
            monitor,
            lambda obs=obs: pi05.prepare_model_inputs(policy, obs),
            sync_cuda_before=True,
            sync_cuda_after=True,
            stage_name="data_processing",
        )
        samples.append(sample)

    return res.stage_samples_to_metrics(samples), monitor.warnings


def benchmark_e2e_resource(
    policy,
    observation,
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
            obs = observations[i % num_obs]
            _ = policy.infer(obs)
        res.cuda_synchronize()
        gc.collect()

    samples = []
    for i in range(num_iterations):
        obs = observations[i % num_obs]
        _, sample = res.measure_callable(
            monitor,
            lambda obs=obs: policy.infer(obs),
            sync_cuda_before=True,
            sync_cuda_after=True,
            stage_name="e2e",
        )
        samples.append(sample)

    return res.stage_samples_to_metrics(samples), monitor.warnings


def benchmark_components_resource(
    policy,
    observation,
    num_iterations=100,
    warmup=20,
    sample_interval_ms=5.0,
):
    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    recorder = res.StageResourceHooks(sample_interval_ms=sample_interval_ms)
    hook_warnings = []
    hook_modules = {}

    original_vision_encoder_method, vision_encoder_warning = install_vision_encoder_resource_timer(
        policy, recorder
    )
    if vision_encoder_warning:
        hook_warnings.append(vision_encoder_warning)
    else:
        hook_modules["vision_encoder"] = "_model.paligemma_with_expert.embed_image"

    original_vlm_forward_method, vlm_forward_warning = install_vlm_forward_resource_timer(
        policy, recorder
    )
    if vlm_forward_warning:
        hook_warnings.append(vlm_forward_warning)
    else:
        hook_modules["llm_backbone"] = "_model.paligemma_with_expert.forward:prefix_cache"
        hook_modules["dit"] = "_model.paligemma_with_expert.forward:denoise_suffix"

    (
        original_model_sample_actions,
        original_policy_sample_actions,
        action_expert_warning,
    ) = install_action_expert_resource_timer(policy, recorder)
    if action_expert_warning:
        hook_warnings.append(action_expert_warning)
    else:
        hook_modules["action_expert"] = "_model.sample_actions:denoising_loop"

    try:
        for i in range(warmup):
            obs = observations[i % num_obs]
            _, model_observation = pi05.prepare_model_inputs(policy, obs)
            with torch.inference_mode():
                _ = pi05.run_model_sample(policy, model_observation)
        res.cuda_synchronize()
        gc.collect()

        records = []
        for i in range(num_iterations):
            obs = observations[i % num_obs]
            _, model_observation = pi05.prepare_model_inputs(policy, obs)

            recorder.start_iteration()
            with torch.inference_mode():
                _ = pi05.run_model_sample(policy, model_observation)
            res.cuda_synchronize()
            records.append(recorder.finish_iteration())

        components = res.records_to_component_metrics(records)
        for stage in ("vision_encoder", "llm_backbone", "action_expert"):
            calls = components[f"{stage}_calls"]
            if stage in hook_modules and calls.size > 0 and max(calls) == 0:
                hook_warnings.append(
                    f"Stage '{stage}' resource boundary was installed at "
                    f"'{hook_modules[stage]}', but observed zero calls. "
                    "The active inference path may bypass this module."
                )

        components["_stage_hook_warnings"] = hook_warnings
        components["_stage_hook_modules"] = hook_modules
        components["_resource_warnings"] = recorder.warnings
        return components
    finally:
        pi05.restore_vision_encoder_method_timer(policy, original_vision_encoder_method)
        pi05.restore_vlm_forward_stage_timer(policy, original_vlm_forward_method)
        pi05.restore_action_expert_method_timer(
            policy, original_model_sample_actions, original_policy_sample_actions
        )


@dataclass
class ResourceBenchmarkConfig(pi05.BenchmarkConfig):
    """Configuration for OpenPI pi0.5 inference resource benchmarking."""

    resource_sample_interval_ms: float = 0.0
    """Deprecated compatibility option; CPU now uses psutil process.cpu_times() before/after each stage, so no sampling interval is used."""

    resource_sampler_backend: str = "full"
    """Resource backend: full (default), nsight, cpu_pytorch, or none. full runs Nsight GPU + non-profiled E2E/stage CPU/PyTorch passes and merges one JSON."""

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
    """Disable automatic nsys relaunch in full/nsight modes."""

    resource_subtract_baseline: bool = False
    """Deprecated compatibility option; final resource metrics are not baseline-subtracted."""

    resource_baseline_duration_ms: float = 1000.0
    """How long to sample the idle baseline before resource benchmarks."""

    resource_baseline_warmup_ms: float = 200.0
    """Idle settling time before baseline sampling starts."""


def main(args: ResourceBenchmarkConfig | None = None):
    if args is None:
        args = tyro.cli(ResourceBenchmarkConfig)

    nsight_warnings = res.maybe_reexec_under_nsight(args, "pi05_resource")

    pi05.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: No CUDA GPU detected. Benchmarking requires a GPU.")
        sys.exit(1)
    device_name = pi05.get_device_name()

    if args.example_source not in {"synthetic", "libero"}:
        raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))
    if args.example_source == "libero" and args.dataset_repo_id is None:
        raise ValueError("--dataset-repo-id is required when --example-source libero")
    if not (args.measure_data_processing or args.measure_e2e or args.measure_components):
        raise ValueError("At least one measurement group must be enabled.")

    data_processing_iterations = res.resolve_count(
        args.data_processing_iterations, args.num_iterations
    )
    data_processing_warmup = res.resolve_count(args.data_processing_warmup, args.warmup)
    e2e_iterations = res.resolve_count(args.e2e_iterations, args.num_iterations)
    e2e_warmup = res.resolve_count(args.e2e_warmup, args.warmup)
    component_iterations = res.resolve_count(args.component_iterations, args.num_iterations)
    component_warmup = res.resolve_count(args.component_warmup, args.warmup)

    print("=" * 100)
    print("PI0.5 INFERENCE RESOURCE BENCHMARK")
    print("=" * 100)
    print(f"Device: {device_name} ({torch.cuda.get_device_name(0)})")
    print(f"Train config: {args.train_config}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Example source: {args.example_source}")
    print(f"Dataset repo/path: {args.dataset_repo_id}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset index: {args.dataset_index}")
    print(f"Denoising steps: {args.num_steps}")
    print(f"PyTorch compile mode: {args.pytorch_compile_mode}")
    print(f"CPU method: psutil process.cpu_times() delta over each stage window")
    print(f"Resource sampler backend: {res.ResourceMonitor.sampler_backend()}")
    print("Inline GPU sampler backend: none (NVML/nvidia-smi removed)")
    nsight_meta = res.nsight_metadata()
    if nsight_meta.get("requested"):
        print(f"Nsight active: {nsight_meta.get('active')}")
        print(f"Nsight report: {nsight_meta.get('expected_report_file')}")
    print("Subtract resource baseline: False (disabled; GPU/memory are Nsight-derived, CPU is process.cpu_times() interval utilization)")
    print(f"Resource baseline duration: {args.resource_baseline_duration_ms} ms")
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

    print("Loading policy...")
    train_config, policy = pi05._load_policy(args)
    model = policy._model
    action_horizon = model.config.action_horizon
    denoising_steps = args.num_steps
    print(f"Action Horizon: {action_horizon}")
    print(f"Denoising Steps: {denoising_steps}")

    observation = pi05.make_observation(args)

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
    if res.should_measure_data_processing(args):
        print("\n" + "-" * 50)
        print("Benchmarking Data Processing Resources...")
        print("-" * 50)
        data_processing_metrics, warnings = benchmark_data_processing_resource(
            policy,
            observation,
            data_processing_iterations,
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
    if res.should_measure_e2e(args):
        print("\n" + "-" * 50)
        print("Benchmarking E2E Inference Resources...")
        print("-" * 50)
        e2e_metrics, warnings = benchmark_e2e_resource(
            policy,
            observation,
            e2e_iterations,
            warmup=e2e_warmup,
            sample_interval_ms=args.resource_sample_interval_ms,
        )
        resource_warnings.extend(warnings)
        e2e_metrics = res.subtract_baseline_from_metrics(e2e_metrics, resource_baseline)
        res.print_resource_summary("E2E", e2e_metrics, include_p95=args.include_p95)

    component_metrics = None
    if res.should_measure_components(args):
        print("\n" + "-" * 50)
        print("Benchmarking Phase Resource Breakdown...")
        print("-" * 50)
        component_metrics = benchmark_components_resource(
            policy,
            observation,
            component_iterations,
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
            ("dit", "DiT-only"),
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
        f"pi0.5 Inference Resource Usage ({denoising_steps} denoising steps):",
        include_p95=args.include_p95,
    )

    print("\n" + "=" * 100)
    print("DETAILED RESOURCE SUMMARY")
    print("=" * 100)
    print(f"\nHardware: {torch.cuda.get_device_name(0)}")
    print(f"Train config: {args.train_config}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Action Horizon: {action_horizon}")
    print(f"Denoising Steps: {denoising_steps}")

    for key, label in [
        ("data_processing", "Data Processing"),
        ("e2e", "E2E"),
        ("vision_encoder", "Vision Encoder"),
        ("llm_backbone", "LLM Backbone"),
        ("action_expert", "Action Expert"),
        ("dit", "DiT-only"),
    ]:
        if key in components:
            res.print_resource_summary(label, components[key], include_p95=args.include_p95)

    if res.should_measure_components(args):
        import numpy as np

        print(
            "  Calls:           "
            f"vision={np.mean(components['vision_encoder_calls']):.1f}, "
            f"llm={np.mean(components['llm_backbone_calls']):.1f}, "
            f"action_expert={np.mean(components['action_expert_calls']):.1f}, "
            f"dit={np.mean(components['dit_calls']):.1f}"
        )
    if components.get("_stage_hook_modules"):
        print("  Stage hook modules:")
        for stage, module_path in components["_stage_hook_modules"].items():
            print(f"    - {stage}: {module_path}")

    if args.output_json:
        res.write_resource_results_json(
            args.output_json,
            model_name="pi05",
            device_name=device_name,
            hardware_name=torch.cuda.get_device_name(0),
            model_info={
                "model_path": None,
                "checkpoint_dir": args.checkpoint_dir,
                "train_config": args.train_config,
                "dataset_path": args.dataset_repo_id if args.example_source == "libero" else None,
                "dataset_root": args.dataset_root if args.example_source == "libero" else None,
                "embodiment_tag": None,
                "example_source": args.example_source,
                "action_horizon": int(action_horizon),
                "denoising_steps": int(denoising_steps),
                "unnorm_key": None,
            },
            args=args,
            components=components,
            sample_interval_ms=args.resource_sample_interval_ms,
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    config = tyro.cli(ResourceBenchmarkConfig)
    main(config)
