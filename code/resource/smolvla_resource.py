#!/usr/bin/env python3

"""
Resource benchmark script for LeRobot SmolVLA inference.

This stays separate from smolvla.py's latency benchmark. It mirrors the same
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

import resource_benchmarking as res
import smolvla


def install_vision_encoder_resource_timer(policy, recorder):
    vlm_with_expert = smolvla._get_nested_attr(policy, "model.vlm_with_expert")
    if vlm_with_expert is None or not hasattr(vlm_with_expert, "embed_image"):
        return None, (
            "Could not find policy.model.vlm_with_expert.embed_image; "
            "vision_encoder resource measurement is unavailable."
        )

    original = vlm_with_expert.embed_image

    def measured_embed_image(*args, **kwargs):
        if not recorder.active:
            return original(*args, **kwargs)
        with recorder.measure("vision_encoder"):
            return original(*args, **kwargs)

    vlm_with_expert.embed_image = measured_embed_image
    return original, None


def install_vlm_forward_resource_timer(policy, recorder):
    vlm_with_expert = smolvla._get_nested_attr(policy, "model.vlm_with_expert")
    if vlm_with_expert is None or not hasattr(vlm_with_expert, "forward"):
        return None, (
            "Could not find policy.model.vlm_with_expert.forward; "
            "llm_backbone/dit resource measurement is unavailable."
        )

    original = vlm_with_expert.forward

    def measured_forward(*args, **kwargs):
        stage = smolvla._classify_smolvlm_forward(args, kwargs) if recorder.active else None
        if stage is None:
            return original(*args, **kwargs)
        with recorder.measure(stage):
            return original(*args, **kwargs)

    vlm_with_expert.forward = measured_forward
    return original, None


def install_action_expert_resource_timer(policy, recorder):
    flow_model = smolvla._get_nested_attr(policy, "model")
    if flow_model is None or not all(
        hasattr(flow_model, name)
        for name in (
            "sample_actions",
            "sample_noise",
            "embed_prefix",
            "denoise_step",
            "vlm_with_expert",
            "_rtc_enabled",
        )
    ):
        return None, (
            "Could not find the expected VLAFlowMatching sample_actions internals; "
            "action_expert resource measurement is unavailable."
        )

    if not hasattr(flow_model, "config") or not hasattr(flow_model.config, "chunk_size"):
        return None, "VLAFlowMatching config.chunk_size is unavailable."

    original = flow_model.sample_actions

    @torch.no_grad()
    def measured_sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs,
    ):
        if not recorder.active:
            return original(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                noise=noise,
                **kwargs,
            )

        bsize = state.shape[0]
        device = state.device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
        )

        with recorder.measure("action_expert"):
            if noise is None:
                actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
                noise = self.sample_noise(actions_shape, device)

            num_steps = self.config.num_steps
            dt = -1.0 / num_steps

            x_t = noise
            for step in range(num_steps):
                denoise_time = 1.0 + step * dt
                time_tensor = torch.tensor(
                    denoise_time, dtype=torch.float32, device=device
                ).expand(bsize)

                def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                    return self.denoise_step(
                        x_t=input_x_t,
                        prefix_pad_masks=prefix_pad_masks,
                        past_key_values=past_key_values,
                        timestep=current_timestep,
                    )

                if self._rtc_enabled():
                    inference_delay = kwargs.get("inference_delay")
                    prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                    execution_horizon = kwargs.get("execution_horizon")

                    v_t = self.rtc_processor.denoise_step(
                        x_t=x_t,
                        prev_chunk_left_over=prev_chunk_left_over,
                        inference_delay=inference_delay,
                        time=denoise_time,
                        original_denoise_step_partial=denoise_step_partial_call,
                        execution_horizon=execution_horizon,
                    )
                else:
                    v_t = denoise_step_partial_call(x_t)

                x_t = x_t + dt * v_t

                if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                    self.rtc_processor.track(time=denoise_time, x_t=x_t, v_t=v_t)

        return x_t

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    flow_model.sample_actions = types.MethodType(measured_sample_actions, flow_model)
    return original, None


def benchmark_data_processing_resource(
    preprocess,
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
            obs = smolvla.clone_observation(observations[i % num_obs])
            _ = smolvla.prepare_model_inputs(preprocess, obs)
        res.cuda_synchronize()
        gc.collect()

    samples = []
    for i in range(num_iterations):
        obs = smolvla.clone_observation(observations[i % num_obs])
        _, sample = res.measure_callable(
            monitor,
            lambda obs=obs: smolvla.prepare_model_inputs(preprocess, obs),
            sync_cuda_before=True,
            sync_cuda_after=True,
            stage_name="data_processing",
        )
        samples.append(sample)

    return res.stage_samples_to_metrics(samples), monitor.warnings


def benchmark_e2e_resource(
    policy,
    preprocess,
    postprocess,
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
            obs = smolvla.clone_observation(observations[i % num_obs])
            policy.reset()
            _ = smolvla.run_public_policy_inference(policy, preprocess, postprocess, obs)
        res.cuda_synchronize()
        gc.collect()

    samples = []
    for i in range(num_iterations):
        obs = smolvla.clone_observation(observations[i % num_obs])
        policy.reset()
        _, sample = res.measure_callable(
            monitor,
            lambda obs=obs: smolvla.run_public_policy_inference(
                policy, preprocess, postprocess, obs
            ),
            sync_cuda_before=True,
            sync_cuda_after=True,
            stage_name="e2e",
        )
        samples.append(sample)

    return res.stage_samples_to_metrics(samples), monitor.warnings


def benchmark_components_resource(
    policy,
    preprocess,
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
        hook_modules["vision_encoder"] = "model.vlm_with_expert.embed_image"

    original_vlm_forward_method, vlm_forward_warning = install_vlm_forward_resource_timer(
        policy, recorder
    )
    if vlm_forward_warning:
        hook_warnings.append(vlm_forward_warning)
    else:
        hook_modules["llm_backbone"] = "model.vlm_with_expert.forward:prefix_cache"
        hook_modules["dit"] = "model.vlm_with_expert.forward:denoise_suffix"

    original_action_expert_method, action_expert_warning = install_action_expert_resource_timer(
        policy, recorder
    )
    if action_expert_warning:
        hook_warnings.append(action_expert_warning)
    else:
        hook_modules["action_expert"] = "model.sample_actions:denoising_loop"

    try:
        for i in range(warmup):
            obs = smolvla.clone_observation(observations[i % num_obs])
            batch = smolvla.prepare_model_inputs(preprocess, obs)
            with torch.inference_mode():
                _ = smolvla.run_model_action_chunk(policy, batch)
        res.cuda_synchronize()
        gc.collect()

        records = []
        for i in range(num_iterations):
            obs = smolvla.clone_observation(observations[i % num_obs])
            batch = smolvla.prepare_model_inputs(preprocess, obs)

            recorder.start_iteration()
            with torch.inference_mode():
                _ = smolvla.run_model_action_chunk(policy, batch)
            res.cuda_synchronize()
            records.append(recorder.finish_iteration())

        components = res.records_to_component_metrics(records)
        for stage in res.COMPONENT_STAGE_KEYS:
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
        smolvla.restore_vision_encoder_method_timer(policy, original_vision_encoder_method)
        smolvla.restore_vlm_forward_stage_timer(policy, original_vlm_forward_method)
        smolvla.restore_action_expert_method_timer(policy, original_action_expert_method)


@dataclass
class ResourceBenchmarkConfig(smolvla.BenchmarkConfig):
    """Configuration for SmolVLA inference resource benchmarking."""

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

    nsight_warnings = res.maybe_reexec_under_nsight(args, "smolvla_resource")

    smolvla.set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: No CUDA GPU detected. Benchmarking requires a GPU.")
        sys.exit(1)
    if args.compile_model and args.measure_components:
        raise ValueError("Component resource measurement requires --compile-model False.")

    device_name = smolvla.get_device_name()

    if args.example_source not in {"synthetic", "libero", "dataset"}:
        raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))
    if args.example_source in {"libero", "dataset"} and args.dataset_repo_id is None:
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
    print("SMOLVLA INFERENCE RESOURCE BENCHMARK")
    print("=" * 100)
    print(f"Device: {device_name} ({torch.cuda.get_device_name(0)})")
    print(f"Model: {args.model_id}")
    print(f"Example source: {args.example_source}")
    print(f"Dataset repo id: {args.dataset_repo_id}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset index: {args.dataset_index}")
    print(f"Compile model: {args.compile_model}")
    print(f"Compile mode: {args.compile_mode if args.compile_model else None}")
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

    print("Loading policy and processors...")
    policy, preprocess, postprocess = smolvla._load_policy_and_processors(args, device)
    action_horizon = policy.config.chunk_size
    denoising_steps = policy.config.num_steps
    print(f"Action Horizon: {action_horizon}")
    print(f"Denoising Steps: {denoising_steps}")
    print(f"Input features: {list((policy.config.input_features or {}).keys())}")

    observation = smolvla.make_observation(policy, args)

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
            preprocess,
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
    if args.measure_e2e:
        print("\n" + "-" * 50)
        print("Benchmarking E2E Inference Resources...")
        print("-" * 50)
        e2e_metrics, warnings = benchmark_e2e_resource(
            policy,
            preprocess,
            postprocess,
            observation,
            e2e_iterations,
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
            policy,
            preprocess,
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
        f"SmolVLA Inference Resource Usage ({denoising_steps} denoising steps):",
        include_p95=args.include_p95,
    )

    print("\n" + "=" * 100)
    print("DETAILED RESOURCE SUMMARY")
    print("=" * 100)
    print(f"\nHardware: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_id}")
    print(f"Example Source: {args.example_source}")
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

    if args.measure_components:
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
            model_name="smolvla",
            device_name=device_name,
            hardware_name=torch.cuda.get_device_name(0),
            model_info={
                "model_path": args.model_id,
                "checkpoint_dir": args.model_id,
                "train_config": None,
                "dataset_path": args.dataset_repo_id if args.example_source in {"libero", "dataset"} else None,
                "dataset_root": args.dataset_root if args.example_source in {"libero", "dataset"} else None,
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
