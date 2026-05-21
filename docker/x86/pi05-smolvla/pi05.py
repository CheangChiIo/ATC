#!/usr/bin/env python3

"""
Benchmark script for OpenPI pi0.5 inference timing.

This script mirrors benchmark_inference.py's latency contract for GR00T:

- Data Processing: policy input transforms, torch tensor conversion, Observation creation
- Vision Encoder: PaliGemma vision tower forward pass
- LLM Backbone: PaliGemma language model prefix forward pass that produces hidden states/KV cache
- Action Expert: complete iterative denoising action generation after prefix VLM encoding
- E2E: full public policy.infer(...) path

Only the PyTorch OpenPI pi0.5 path is supported for component timing because the timing
method relies on Python method wrappers / torch hooks and torch.cuda.synchronize(),
in the same spirit as the GR00T and SmolVLA benchmarks.

The pi0.5 paper describes a hierarchical inference mode where the VLM first
predicts a high-level semantic subtask and then conditions low-level action
generation on that subtask. The public openpi repository currently exposes the
flow-matching action path for pi0.5 inference, and PI0Pytorch.sample_actions(...)
does not run an autoregressive text/subtask generation step. Accordingly, this
benchmark measures the exact public openpi pi0.5 PyTorch policy path: the LLM
Backbone stage is the prefix VLM forward that builds the KV cache for the action
expert. If openpi later adds public subtask generation to sample_actions(...),
that generation should be included in this stage before the action_expert timer.
"""

from __future__ import annotations

from dataclasses import dataclass
import dataclasses
import gc
import inspect
import json
import math
import os
import random
import sys
import time
import types
from typing import Any

import numpy as np
import torch
import tyro


DEFAULT_LIBERO_DATASET_REPO_ID = "physical-intelligence/libero"
DEFAULT_LIBERO_DATASET_ROOT = "/datasets/physical-intelligence/libero"


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_name():
    """Get short device name for table."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        if "H100" in name:
            return "H100"
        if "A100" in name:
            return "A100"
        if "RTX 5090" in name:
            return "RTX 5090"
        if "RTX 4090" in name:
            return "RTX 4090"
        if "RTX 3090" in name:
            return "RTX 3090"
        if "Orin" in name:
            return "Jetson Orin"
        return name.split()[1] if len(name.split()) > 1 else name
    return "CPU"


def cuda_synchronize():
    """Synchronize CUDA so CPU wall-clock includes queued GPU work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimingHooks:
    """
    CUDA-synchronized CPU wall-clock timing for nested pi0.5 model submodules.

    The hook timer records selected internal modules as comparable stages:

    - vision_encoder: full PaliGemma image feature embedding inside PI0Pytorch.embed_prefix(...)
    - llm_backbone: PaliGemma language model prefix forward inside sample_actions(...)
    - action_expert: full iterative denoising loop after prefix VLM encoding
    """

    STAGE_KEYS = ("vision_encoder", "llm_backbone", "action_expert")

    def __init__(self):
        self.active = False
        self.current = None
        self._starts = {key: [] for key in self.STAGE_KEYS}
        self.records = []

    def start_iteration(self):
        self.active = True
        self.current = {key: 0.0 for key in self.STAGE_KEYS}
        self.current.update({f"{key}_calls": 0 for key in self.STAGE_KEYS})
        self._starts = {key: [] for key in self.STAGE_KEYS}

    def finish_iteration(self):
        record = dict(self.current)
        self.records.append(record)
        self.active = False
        self.current = None
        self._starts = {key: [] for key in self.STAGE_KEYS}
        return record

    def pre_hook(self, stage):
        def _pre_hook(_module, _inputs):
            if not self.active:
                return
            cuda_synchronize()
            self._starts[stage].append(time.perf_counter())

        return _pre_hook

    def post_hook(self, stage):
        def _post_hook(_module, _inputs, _output):
            if not self.active or not self._starts[stage]:
                return
            cuda_synchronize()
            elapsed_ms = (time.perf_counter() - self._starts[stage].pop()) * 1000
            self.current[stage] += elapsed_ms
            self.current[f"{stage}_calls"] += 1

        return _post_hook


def _get_nested_attr(obj, attr_path):
    cur = obj
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _first_existing_attr(obj, attr_paths):
    for attr_path in attr_paths:
        value = _get_nested_attr(obj, attr_path)
        if value is not None:
            return value, attr_path
    return None, None


def install_vision_encoder_method_timer(policy, timer):
    """
    Time complete pi0.5 image embedding.

    The OpenPI path calls PaliGemmaWithExpertModel.embed_image(...), which in
    turn calls Hugging Face PaliGemma get_image_features(...). Wrapping this
    method measures the full image feature path rather than only the inner
    SigLIP vision_tower module, so patch embedding and the PaliGemma visual
    projection are included under the same "Vision Encoder" convention used by
    the GR00T benchmark.
    """

    model = getattr(policy, "_model", None)
    paligemma_with_expert = _get_nested_attr(model, "paligemma_with_expert") if model else None
    if paligemma_with_expert is None or not hasattr(paligemma_with_expert, "embed_image"):
        return None, (
            "Could not find _model.paligemma_with_expert.embed_image; "
            "vision_encoder timing is unavailable."
        )

    original = paligemma_with_expert.embed_image

    def timed_embed_image(*args, **kwargs):
        if not timer.active:
            return original(*args, **kwargs)
        cuda_synchronize()
        start = time.perf_counter()
        output = original(*args, **kwargs)
        cuda_synchronize()
        timer.current["vision_encoder"] += (time.perf_counter() - start) * 1000
        timer.current["vision_encoder_calls"] += 1
        return output

    paligemma_with_expert.embed_image = timed_embed_image
    return original, None


def restore_vision_encoder_method_timer(policy, original_method):
    if original_method is None:
        return
    model = getattr(policy, "_model", None)
    paligemma_with_expert = _get_nested_attr(model, "paligemma_with_expert") if model else None
    if paligemma_with_expert is not None:
        paligemma_with_expert.embed_image = original_method


def _get_call_argument(args, kwargs, name, position):
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return None


def _classify_pi05_vlm_forward(args, kwargs):
    """
    Classify PaliGemmaWithExpertModel.forward(...) calls by inference phase.

    OpenPI's PyTorch pi0.5 path runs through paligemma_with_expert.forward(...)
    with prepared inputs_embeds instead of necessarily calling the nested
    language_model.forward(...) or gemma_expert.model.forward(...) modules.
    Classifying this active forward path avoids zero-call LLM timers.
    """

    inputs_embeds = _get_call_argument(args, kwargs, "inputs_embeds", 3)
    if not isinstance(inputs_embeds, (list, tuple)) or len(inputs_embeds) < 2:
        return None

    prefix_embs, suffix_embs = inputs_embeds[0], inputs_embeds[1]
    if prefix_embs is not None and suffix_embs is None:
        return "llm_backbone"
    return None


def install_vlm_forward_stage_timer(policy, timer):
    """
    Time pi0.5's active PaliGemmaWithExpertModel.forward(...) path.

    The prefix call is reported as llm_backbone, matching the GR00T/SmolVLA
    hidden-state/KV-cache VLM pass.
    """

    model = getattr(policy, "_model", None)
    paligemma_with_expert = _get_nested_attr(model, "paligemma_with_expert") if model else None
    if paligemma_with_expert is None or not hasattr(paligemma_with_expert, "forward"):
        return None, (
            "Could not find _model.paligemma_with_expert.forward; "
            "llm_backbone timing is unavailable."
        )

    original = paligemma_with_expert.forward

    def timed_forward(*args, **kwargs):
        stage = _classify_pi05_vlm_forward(args, kwargs) if timer.active else None
        if stage is None:
            return original(*args, **kwargs)
        cuda_synchronize()
        start = time.perf_counter()
        output = original(*args, **kwargs)
        cuda_synchronize()
        timer.current[stage] += (time.perf_counter() - start) * 1000
        timer.current[f"{stage}_calls"] += 1
        return output

    paligemma_with_expert.forward = timed_forward
    return original, None


def restore_vlm_forward_stage_timer(policy, original_method):
    if original_method is None:
        return
    model = getattr(policy, "_model", None)
    paligemma_with_expert = _get_nested_attr(model, "paligemma_with_expert") if model else None
    if paligemma_with_expert is not None:
        paligemma_with_expert.forward = original_method


def _get_policy_sample_kwargs(policy) -> dict[str, Any]:
    return dict(getattr(policy, "_sample_kwargs", {}) or {})


def _get_policy_device(policy):
    device = getattr(policy, "_pytorch_device", None)
    if device is not None:
        return device
    try:
        return next(policy._model.parameters()).device
    except StopIteration:
        return "cuda" if torch.cuda.is_available() else "cpu"


def _embed_prefix_for_inference(model, images, img_masks, lang_tokens, lang_masks):
    """
    Match OpenPI's inference-time prefix embedding call.

    Recent OpenPI PyTorch checkpoints support optional DivPrune through
    embed_prefix(..., apply_divprune=True). Passing the flag keeps the instrumented
    timing path semantically aligned with PI0Pytorch.sample_actions(...), while the
    signature check keeps older OpenPI installs working.
    """

    kwargs = {}
    if "apply_divprune" in inspect.signature(model.embed_prefix).parameters:
        kwargs["apply_divprune"] = True
    return model.embed_prefix(images, img_masks, lang_tokens, lang_masks, **kwargs)


def install_action_expert_method_timer(policy, timer):
    """
    Time pi0.5's complete iterative action generation after prefix VLM encoding.

    OpenPI PI0Pytorch.sample_actions(...) first preprocesses the Observation,
    embeds image/language inputs, and runs the PaliGemma language model once to
    build the prefix KV cache. It then runs the flow-matching denoising loop:

    sample action noise -> embed_suffix(state, action, time) -> action expert
    transformer -> action_out_proj -> Euler update / scheduler step.

    This wrapper copies the official sample_actions structure and starts the
    action_expert timer immediately after the prefix KV cache is available. That
    makes the stage analogous to GR00T's get_action_with_features timing: it
    includes noise initialization, suffix embedding, action expert transformer,
    action decoder/output projection, scheduler updates, and loop overhead.
    """

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
            "action_expert timing is unavailable."
        )

    if not hasattr(model, "config") or not hasattr(model.config, "action_horizon"):
        return None, None, "PI0Pytorch config.action_horizon is unavailable."

    original_model_sample_actions = model.sample_actions
    original_policy_sample_actions = getattr(policy, "_sample_actions", None)

    @torch.no_grad()
    def timed_sample_actions(self, device, observation, noise=None, num_steps=10):
        bsize = observation.state.shape[0]
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = _embed_prefix_for_inference(
            self, images, img_masks, lang_tokens, lang_masks
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

        if timer.active:
            cuda_synchronize()
            start = time.perf_counter()

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

        if timer.active:
            cuda_synchronize()
            timer.current["action_expert"] += (time.perf_counter() - start) * 1000
            timer.current["action_expert_calls"] += 1

        return x_t

    # Reuse OpenPI's helper from the installed package at runtime.
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    bound = types.MethodType(timed_sample_actions, model)
    model.sample_actions = bound
    policy._sample_actions = bound
    return original_model_sample_actions, original_policy_sample_actions, None


def restore_action_expert_method_timer(
    policy, original_model_sample_actions, original_policy_sample_actions
):
    model = getattr(policy, "_model", None)
    if model is not None and original_model_sample_actions is not None:
        model.sample_actions = original_model_sample_actions
    if original_policy_sample_actions is not None:
        policy._sample_actions = original_policy_sample_actions


def remove_hooks(handles):
    for handle in handles:
        handle.remove()


def _tree_to_torch_batch(inputs, device):
    import jax

    def convert_leaf(x):
        if isinstance(x, torch.Tensor):
            tensor = x.to(device)
        else:
            arr = np.asarray(x)
            if arr.dtype.kind in {"O", "U", "S"}:
                raise TypeError(
                    "Input transforms left a non-numeric leaf in the model input tree. "
                    f"Leaf value type={type(x)!r}, dtype={arr.dtype!r}."
                )
            tensor = torch.from_numpy(arr).to(device)
        return tensor[None, ...]

    return jax.tree.map(convert_leaf, inputs)


def prepare_model_inputs(policy, observation):
    """
    Prepare pi0.5 model inputs, mirroring the PyTorch branch of Policy.infer(...).

    Returns (torch_input_dict, Observation). The measured data-processing stage
    includes input transforms, tokenization, normalization, torch tensor
    conversion/device transfer, image layout conversion, and Observation creation.
    """

    import jax
    from openpi.models import model as _model

    inputs = jax.tree.map(lambda x: x, observation)
    inputs = policy._input_transform(inputs)
    inputs = _tree_to_torch_batch(inputs, _get_policy_device(policy))
    model_observation = _model.Observation.from_dict(inputs)
    return inputs, model_observation


def run_model_sample(policy, model_observation):
    sample_kwargs = _get_policy_sample_kwargs(policy)
    return policy._sample_actions(_get_policy_device(policy), model_observation, **sample_kwargs)


def benchmark_data_processing(policy, observation, num_iterations=100, warmup=20):
    """
    Benchmark pi0.5 data processing separately with the same warmup style as GR00T.
    If observation is a list, cycles through observations during benchmarking.
    """

    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    gc.collect()

    if warmup > 0:
        for i in range(warmup):
            obs = observations[i % num_obs]
            _ = prepare_model_inputs(policy, obs)
        cuda_synchronize()
        gc.collect()

    times = []
    for i in range(num_iterations):
        obs = observations[i % num_obs]
        cuda_synchronize()
        start = time.perf_counter()
        _ = prepare_model_inputs(policy, obs)
        cuda_synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times) * 1000


def benchmark_e2e(policy, observation, num_iterations=100, warmup=20):
    """
    Benchmark true end-to-end policy latency.

    This measures the public policy.infer(...) path, including input transforms,
    tensor conversion, model inference, output transforms, and validation/copy-out
    work done by the OpenPI policy. CUDA is synchronized around each timed call so
    CPU wall-clock includes queued GPU work.
    """

    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if warmup > 0:
        for i in range(warmup):
            obs = observations[i % num_obs]
            _ = policy.infer(obs)
        cuda_synchronize()
        gc.collect()

    times = []
    for i in range(num_iterations):
        obs = observations[i % num_obs]
        cuda_synchronize()
        start = time.perf_counter()
        _ = policy.infer(obs)
        cuda_synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times) * 1000


def benchmark_components(policy, observation, num_iterations=100, warmup=20):
    """
    Benchmark component-wise pi0.5 timing.

    Returns dict with the same stage keys used by the GR00T benchmark:
    vision_encoder, llm_backbone, action_expert, and *_calls.
    """

    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    stage_timer = StageTimingHooks()
    hook_handles = []
    hook_warnings = []
    hook_modules = {}
    original_vision_encoder_method, vision_encoder_warning = install_vision_encoder_method_timer(
        policy, stage_timer
    )
    if vision_encoder_warning:
        hook_warnings.append(vision_encoder_warning)
    else:
        hook_modules["vision_encoder"] = "_model.paligemma_with_expert.embed_image"
    original_vlm_forward_method, vlm_forward_warning = install_vlm_forward_stage_timer(
        policy, stage_timer
    )
    if vlm_forward_warning:
        hook_warnings.append(vlm_forward_warning)
    else:
        hook_modules["llm_backbone"] = "_model.paligemma_with_expert.forward:prefix_cache"
    (
        original_model_sample_actions,
        original_policy_sample_actions,
        action_expert_warning,
    ) = install_action_expert_method_timer(policy, stage_timer)
    if action_expert_warning:
        hook_warnings.append(action_expert_warning)
    else:
        hook_modules["action_expert"] = "_model.sample_actions:denoising_loop"

    try:
        for i in range(warmup):
            obs = observations[i % num_obs]
            _, model_observation = prepare_model_inputs(policy, obs)
            with torch.inference_mode():
                _ = run_model_sample(policy, model_observation)
        cuda_synchronize()

        gc.collect()

        vision_encoder_times = []
        llm_backbone_times = []
        action_expert_times = []
        vision_encoder_calls = []
        llm_backbone_calls = []
        action_expert_calls = []

        for i in range(num_iterations):
            obs = observations[i % num_obs]
            _, model_observation = prepare_model_inputs(policy, obs)

            stage_timer.start_iteration()

            with torch.inference_mode():
                _ = run_model_sample(policy, model_observation)
            cuda_synchronize()

            stage_record = stage_timer.finish_iteration()
            vision_encoder_times.append(stage_record["vision_encoder"])
            llm_backbone_times.append(stage_record["llm_backbone"])
            action_expert_times.append(stage_record["action_expert"])
            vision_encoder_calls.append(stage_record["vision_encoder_calls"])
            llm_backbone_calls.append(stage_record["llm_backbone_calls"])
            action_expert_calls.append(stage_record["action_expert_calls"])

        call_arrays = {
            "vision_encoder": np.array(vision_encoder_calls),
            "llm_backbone": np.array(llm_backbone_calls),
            "action_expert": np.array(action_expert_calls),
        }
        for stage, calls in call_arrays.items():
            if stage in hook_modules and calls.size > 0 and np.max(calls) == 0:
                hook_warnings.append(
                    f"Stage '{stage}' timer was installed at '{hook_modules[stage]}', "
                    "but observed zero calls. The active inference path may bypass this module."
                )

        return {
            "vision_encoder": np.array(vision_encoder_times),
            "llm_backbone": np.array(llm_backbone_times),
            "action_expert": np.array(action_expert_times),
            "vision_encoder_calls": call_arrays["vision_encoder"],
            "llm_backbone_calls": call_arrays["llm_backbone"],
            "action_expert_calls": call_arrays["action_expert"],
            "_stage_hook_warnings": hook_warnings,
            "_stage_hook_modules": hook_modules,
        }
    finally:
        restore_vision_encoder_method_timer(policy, original_vision_encoder_method)
        restore_vlm_forward_stage_timer(policy, original_vlm_forward_method)
        restore_action_expert_method_timer(
            policy, original_model_sample_actions, original_policy_sample_actions
        )
        remove_hooks(hook_handles)


def build_components(data_processing_times=None, times_components=None, e2e_times=None):
    """Merge shared data-processing timings with component timings."""
    components = {
        key: value
        for key, value in (times_components or {}).items()
        if not key.startswith("_")
    }
    if data_processing_times is not None:
        components["data_processing"] = data_processing_times
    if e2e_times is not None:
        components["e2e"] = e2e_times
        components["_e2e_note"] = "measured_policy_infer"
    if times_components and "_stage_hook_warnings" in times_components:
        components["_stage_hook_warnings"] = times_components["_stage_hook_warnings"]
    if times_components and "_stage_hook_modules" in times_components:
        components["_stage_hook_modules"] = times_components["_stage_hook_modules"]
    return components


def resolve_count(override, default):
    """Use a per-measurement count override when provided, otherwise use the global default."""
    return default if override is None else override


def latency_mean(values):
    return float(np.mean(values))


def latency_p5(values):
    return float(np.percentile(values, 5))


def latency_p95(values):
    return float(np.percentile(values, 95))


def format_latency(values, include_p95=True):
    """Format latency as mean or p5 / mean / p95."""
    mean_ms = latency_mean(values)
    if include_p95:
        return f"{latency_p5(values):.1f} / {mean_ms:.1f} / {latency_p95(values):.1f} ms"
    return f"{mean_ms:.1f} ms"


def print_latency_summary(label, values, include_p95=True, indent="  "):
    parts = [
        f"mean={latency_mean(values):.2f} ms",
        f"std={np.std(values):.2f} ms",
        f"min={np.min(values):.2f} ms",
        f"max={np.max(values):.2f} ms",
        f"n={len(values)}",
    ]
    if include_p95:
        parts.insert(1, f"p5={latency_p5(values):.2f} ms")
        parts.insert(2, f"p95={latency_p95(values):.2f} ms")
    print(f"{indent}{label}: " + ", ".join(parts))


def print_markdown_table(data, device_name, denoising_steps, include_p95=True):
    """Print results as a markdown table using mean latency, optionally with p95."""
    print("\n" + "=" * 100)
    print("MARKDOWN TABLE (copy/paste into README)")
    print("=" * 100)
    print(f"\npi0.5 Inference Timing ({denoising_steps} denoising steps):\n")

    metric_defs = [
        ("data_processing", "Data Processing"),
        ("vision_encoder", "Vision Encoder"),
        ("llm_backbone", "LLM Backbone"),
        ("action_expert", "Action Expert"),
        ("e2e", "E2E"),
    ]
    measured_metrics = [(key, label) for key, label in metric_defs if key in data]

    latency_unit = "p5 / mean / p95" if include_p95 else "mean"
    print(f"### Component-wise Breakdown ({latency_unit})\n")
    headers = ["Device"] + [label for _, label in measured_metrics]
    if "e2e" in data:
        headers.append("Frequency")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")

    row = [device_name] + [format_latency(data[key], include_p95) for key, _ in measured_metrics]
    if "e2e" in data:
        row.append(f"{1000 / latency_mean(data['e2e']):.2f} Hz")
    print("| " + " | ".join(row) + " |")

    print("\n" + "=" * 100)


LATENCY_KEYS = (
    "data_processing",
    "vision_encoder",
    "llm_backbone",
    "action_expert",
    "e2e",
)

CALL_KEYS = (
    "vision_encoder_calls",
    "llm_backbone_calls",
    "action_expert_calls",
)


def _series_summary(values, include_samples=True):
    arr = np.asarray(values, dtype=float)
    result = {
        "mean": float(np.mean(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }
    if include_samples:
        result["samples"] = [float(x) for x in arr.tolist()]
    return result




PLOT_STAGE_LABELS = {
    "data_processing": "Data Processing",
    "vision_encoder": "Vision Encoder",
    "llm_backbone": "LLM Backbone",
    "action_expert": "Action Expert",
    "e2e": "E2E",
}


def _json_safe_plot_value(value):
    """Return JSON-safe scalar/list values for the shared plotting records."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe_plot_value(item) for item in value]
    return value


def _copy_summary_fields(stats: dict[str, Any], *, prefix: str = "") -> dict[str, Any]:
    """Normalize p5/mean/p95 summaries into stable column names."""
    return {
        f"{prefix}mean": _json_safe_plot_value(stats.get("mean")),
        f"{prefix}p5": _json_safe_plot_value(stats.get("p5")),
        f"{prefix}p95": _json_safe_plot_value(stats.get("p95")),
        f"{prefix}std": _json_safe_plot_value(stats.get("std")),
        f"{prefix}min": _json_safe_plot_value(stats.get("min")),
        f"{prefix}max": _json_safe_plot_value(stats.get("max")),
        f"{prefix}n": _json_safe_plot_value(stats.get("n")),
    }


def _make_plot_base_record(
    *,
    model_name: str,
    device_name: str,
    hardware_name: str,
    model_info: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    """Fields intentionally repeated per record so records can be concatenated directly."""
    return {
        "plot_schema_version": "vla_plot_v1",
        "model": model_name,
        "device_name": device_name,
        "torch_cuda_device_name": hardware_name,
        "model_path": model_info.get("model_path"),
        "checkpoint_dir": model_info.get("checkpoint_dir"),
        "train_config": model_info.get("train_config"),
        "dataset_path": model_info.get("dataset_path"),
        "dataset_root": model_info.get("dataset_root"),
        "example_source": model_info.get("example_source"),
        "action_horizon": model_info.get("action_horizon"),
        "denoising_steps": model_info.get("denoising_steps"),
        "num_steps": benchmark.get("num_steps"),
        "default_measured_iterations": benchmark.get("default_measured_iterations"),
        "default_warmup": benchmark.get("default_warmup"),
    }


def _make_latency_plot_records(
    *,
    model_name: str,
    device_name: str,
    hardware_name: str,
    model_info: dict[str, Any],
    benchmark: dict[str, Any],
    latency_ms: dict[str, Any],
    calls: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Flatten latency output into a stable table-like schema.

    A plotting script can read every JSON file with:
        records.extend(payload["plot_records"])
    and then filter by metric_name/stage/model without model-specific branches.
    """
    base = _make_plot_base_record(
        model_name=model_name,
        device_name=device_name,
        hardware_name=hardware_name,
        model_info=model_info,
        benchmark=benchmark,
    )
    records: list[dict[str, Any]] = []
    for stage in LATENCY_KEYS:
        stats = latency_ms.get(stage)
        if not stats:
            continue
        record = dict(base)
        record.update(
            {
                "metric_family": "latency",
                "metric_name": "latency_ms",
                "unit": "ms",
                "stage": stage,
                "stage_label": PLOT_STAGE_LABELS.get(stage, stage),
            }
        )
        record.update(_copy_summary_fields(stats, prefix="value_"))
        if "samples" in stats:
            record["samples"] = _json_safe_plot_value(stats["samples"])

        call_stats = calls.get(f"{stage}_calls")
        if call_stats:
            record.update(_copy_summary_fields(call_stats, prefix="calls_"))
        else:
            record.update(
                {
                    "calls_mean": None,
                    "calls_p5": None,
                    "calls_p95": None,
                    "calls_std": None,
                    "calls_min": None,
                    "calls_max": None,
                    "calls_n": None,
                }
            )
        records.append(record)
    return records

def write_results_json(
    path,
    *,
    model_name,
    device_name,
    hardware_name,
    train_config_name,
    checkpoint_dir,
    action_horizon,
    denoising_steps,
    args,
    components,
):
    """Write a cross-model JSON schema suitable for plotting."""

    latency_ms = {
        key: _series_summary(components[key], include_samples=args.include_samples_in_json)
        for key in LATENCY_KEYS
        if key in components
    }
    calls = {
        key: _series_summary(components[key], include_samples=args.include_samples_in_json)
        for key in CALL_KEYS
        if key in components
    }
    payload = {
        "schema_version": "vla_latency_v1",
        "model": model_name,
        "hardware": {
            "device_name": device_name,
            "torch_cuda_device_name": hardware_name,
        },
        "model_info": {
            "model_path": None,
            "checkpoint_dir": checkpoint_dir,
            "train_config": train_config_name,
            "dataset_path": args.dataset_repo_id if args.example_source == "libero" else None,
            "dataset_root": args.dataset_root if args.example_source == "libero" else None,
            "embodiment_tag": None,
            "example_source": args.example_source,
            "action_horizon": int(action_horizon),
            "denoising_steps": int(denoising_steps),
        },
        "benchmark": {
            "default_measured_iterations": int(args.num_iterations),
            "default_warmup": int(args.warmup),
            "include_p95": bool(args.include_p95),
            "include_p5": True,
            "measure_data_processing": bool(args.measure_data_processing),
            "measure_e2e": bool(args.measure_e2e),
            "measure_components": bool(args.measure_components),
            "data_processing_iterations": resolve_count(
                args.data_processing_iterations, args.num_iterations
            ),
            "data_processing_warmup": resolve_count(args.data_processing_warmup, args.warmup),
            "e2e_iterations": resolve_count(args.e2e_iterations, args.num_iterations),
            "e2e_warmup": resolve_count(args.e2e_warmup, args.warmup),
            "component_iterations": resolve_count(args.component_iterations, args.num_iterations),
            "component_warmup": resolve_count(args.component_warmup, args.warmup),
            "use_trajectory": False,
            "example_source": args.example_source,
            "pytorch_compile_mode": args.pytorch_compile_mode,
            "num_steps": int(denoising_steps),
        },
        "latency_ms": latency_ms,
        "calls": calls,
        "frequency_hz": {
            "e2e_mean": 1000 / latency_ms["e2e"]["mean"] if "e2e" in latency_ms else None,
        },
        "stage_hook_modules": components.get("_stage_hook_modules", {}),
        "warnings": components.get("_stage_hook_warnings", []),
        "notes": {
            "e2e": components.get("_e2e_note"),
            "action_expert": "Timed around PI0Pytorch.sample_actions denoising loop after prefix VLM cache.",
        },
    }

    payload["plot_schema_version"] = "vla_plot_v1"
    payload["plot_stage_order"] = list(LATENCY_KEYS)
    payload["plot_stage_labels"] = PLOT_STAGE_LABELS
    payload["plot_records"] = _make_latency_plot_records(
        model_name=model_name,
        device_name=device_name,
        hardware_name=hardware_name,
        model_info=payload["model_info"],
        benchmark=payload["benchmark"],
        latency_ms=latency_ms,
        calls=calls,
    )

    output_path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote JSON results to: {output_path}")


def make_synthetic_observation(seed: int = 42, prompt: str = "do something"):
    """Create a deterministic random LIBERO-shaped pi0.5 inference input."""

    rng = np.random.default_rng(seed)
    return {
        "observation/state": rng.uniform(-1.0, 1.0, size=(8,)).astype(np.float32),
        "observation/image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": rng.integers(0, 256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": prompt,
    }


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _first_present(frame, keys):
    for key in keys:
        if key in frame:
            return frame[key]
    return None


def make_libero_observation(
    dataset_repo_id: str,
    dataset_root: str | None,
    dataset_index: int,
    prompt: str | None,
):
    """Load one LIBERO frame and map it to OpenPI's policy inference keys."""

    from libero_dataset_utils import image_to_hwc_uint8, load_lerobot_frame

    frame = load_lerobot_frame(dataset_repo_id, dataset_root, dataset_index)

    state = _first_present(frame, ["observation/state", "state", "observation.state"])
    base_image = _first_present(
        frame,
        [
            "observation/image",
            "image",
            "observation.image",
            "observation.images.image",
            "observation.images.camera1",
            "observation.images.agentview_image",
        ],
    )
    wrist_image = _first_present(
        frame,
        [
            "observation/wrist_image",
            "wrist_image",
            "observation.wrist_image",
            "observation.images.wrist_image",
            "observation.images.camera2",
            "observation.images.robot0_eye_in_hand_image",
        ],
    )
    if state is None or base_image is None or wrist_image is None:
        raise KeyError(
            "Could not map LIBERO frame to pi0.5 inference keys. "
            f"Available keys: {sorted(frame.keys())}"
        )

    frame_prompt = prompt
    if frame_prompt is None:
        frame_prompt = frame.get("prompt", frame.get("task", ""))

    return {
        "observation/state": _to_numpy(state).astype(np.float32),
        "observation/image": image_to_hwc_uint8(base_image),
        "observation/wrist_image": image_to_hwc_uint8(wrist_image),
        "prompt": str(frame_prompt),
    }


def make_observation(args: "BenchmarkConfig"):
    if args.example_source == "synthetic":
        return make_synthetic_observation(seed=args.seed, prompt=args.prompt)
    if args.example_source == "libero":
        return make_libero_observation(
            args.dataset_repo_id,
            args.dataset_root,
            args.dataset_index,
            args.prompt,
        )
    raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))


@dataclass
class BenchmarkConfig:
    """Configuration for OpenPI pi0.5 inference benchmarking."""

    train_config: str = "pi05_libero"
    """OpenPI training config name, e.g. pi05_libero, pi05_droid, pi05_aloha."""

    checkpoint_dir: str = "/checkpoints/pi05_libero"
    """Path or gs:// URI to an OpenPI checkpoint directory."""

    example_source: str = "libero"
    """Input source to use: synthetic or libero."""

    dataset_repo_id: str | None = DEFAULT_LIBERO_DATASET_REPO_ID
    """LeRobot LIBERO dataset repo id when example_source=libero."""

    dataset_root: str | None = DEFAULT_LIBERO_DATASET_ROOT
    """Local LeRobot dataset root used with dataset_repo_id when example_source=libero."""

    dataset_index: int = 0
    """Frame index to load when example_source=libero."""

    prompt: str = "pick up the object"
    """Language instruction used for synthetic inputs or to override a LIBERO frame prompt."""

    num_steps: int = 10
    """Number of pi0.5 denoising steps passed to sample_actions(...)."""

    pytorch_compile_mode: str | None = None
    """Override OpenPI PyTorch compile mode. None disables compile for hookable raw PyTorch timing."""

    num_iterations: int = 100
    """Default number of measured iterations for enabled benchmarks."""

    warmup: int = 20
    """Default number of warmup iterations for enabled benchmarks."""

    data_processing_iterations: int | None = None
    """Override measured iterations for data-processing latency. Defaults to num_iterations."""

    data_processing_warmup: int | None = None
    """Override warmup iterations for data-processing latency. Defaults to warmup."""

    e2e_iterations: int | None = None
    """Override measured iterations for E2E latency. Defaults to num_iterations."""

    e2e_warmup: int | None = None
    """Override warmup iterations for E2E latency. Defaults to warmup."""

    component_iterations: int | None = None
    """Override measured iterations for phase/component latency. Defaults to num_iterations."""

    component_warmup: int | None = None
    """Override warmup iterations for phase/component latency. Defaults to warmup."""

    measure_data_processing: bool = True
    """Whether to measure Data Processing latency."""

    measure_e2e: bool = True
    """Whether to measure standalone E2E policy.infer latency."""

    measure_components: bool = True
    """Whether to measure Vision/LLM/Action Expert phase latencies."""

    include_p95: bool = True
    """Whether to include p95 latency in reports."""

    output_json: str | None = None
    """Optional path to write machine-readable benchmark results."""

    include_samples_in_json: bool = True
    """Whether JSON output includes raw per-iteration samples."""

    seed: int = 42
    """Random seed for reproducibility."""


def _load_policy(args: BenchmarkConfig):
    from openpi.policies import policy_config
    from openpi.training import config as openpi_config

    train_config = openpi_config.get_config(args.train_config)
    if not getattr(train_config.model, "pi05", False):
        raise ValueError(
            f"Train config {args.train_config!r} is not a pi0.5 config "
            "(expected train_config.model.pi05 == True)."
        )

    if hasattr(train_config.model, "pytorch_compile_mode"):
        train_config = dataclasses.replace(
            train_config,
            model=dataclasses.replace(
                train_config.model, pytorch_compile_mode=args.pytorch_compile_mode
            ),
        )

    original_torch_compile = None
    if args.pytorch_compile_mode is None and hasattr(torch, "compile"):
        original_torch_compile = torch.compile

        def identity_compile(fn=None, *compile_args, **compile_kwargs):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        torch.compile = identity_compile

    try:
        policy = policy_config.create_trained_policy(
            train_config,
            args.checkpoint_dir,
            sample_kwargs={"num_steps": args.num_steps},
            pytorch_device="cuda" if torch.cuda.is_available() else "cpu",
        )
    finally:
        if original_torch_compile is not None:
            torch.compile = original_torch_compile

    if not getattr(policy, "_is_pytorch_model", False):
        raise RuntimeError(
            "This benchmark requires an OpenPI PyTorch checkpoint containing model.safetensors. "
            "The loaded checkpoint was detected as a JAX checkpoint, whose internals cannot be "
            "timed with the GR00T-style torch hook method."
        )

    return train_config, policy


def main(args: BenchmarkConfig | None = None):
    if args is None:
        args = tyro.cli(BenchmarkConfig)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: No CUDA GPU detected. Benchmarking requires a GPU.")
        sys.exit(1)
    device_name = get_device_name()

    if args.example_source not in {"synthetic", "libero"}:
        raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))
    if args.example_source == "libero" and args.dataset_repo_id is None:
        raise ValueError("--dataset-repo-id is required when --example-source libero")

    if not (args.measure_data_processing or args.measure_e2e or args.measure_components):
        raise ValueError("At least one measurement group must be enabled.")

    data_processing_iterations = resolve_count(
        args.data_processing_iterations, args.num_iterations
    )
    data_processing_warmup = resolve_count(args.data_processing_warmup, args.warmup)
    e2e_iterations = resolve_count(args.e2e_iterations, args.num_iterations)
    e2e_warmup = resolve_count(args.e2e_warmup, args.warmup)
    component_iterations = resolve_count(args.component_iterations, args.num_iterations)
    component_warmup = resolve_count(args.component_warmup, args.warmup)

    print("=" * 100)
    print("PI0.5 INFERENCE BENCHMARK")
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

    print("Loading policy...")
    train_config, policy = _load_policy(args)
    model = policy._model
    action_horizon = model.config.action_horizon
    denoising_steps = args.num_steps
    print(f"Action Horizon: {action_horizon}")
    print(f"Denoising Steps: {denoising_steps}")

    observation = make_observation(args)

    data_processing_times = None
    if args.measure_data_processing:
        print("\n" + "-" * 50)
        print("Benchmarking Data Processing...")
        print("-" * 50)
        data_processing_times = benchmark_data_processing(
            policy, observation, data_processing_iterations, warmup=data_processing_warmup
        )
        print_latency_summary(
            "Data Processing", data_processing_times, include_p95=args.include_p95
        )

    e2e_times = None
    if args.measure_e2e:
        print("\n" + "-" * 50)
        print("Benchmarking E2E Inference...")
        print("-" * 50)
        e2e_times = benchmark_e2e(policy, observation, e2e_iterations, warmup=e2e_warmup)
        print_latency_summary("E2E", e2e_times, include_p95=args.include_p95)
        print(f"  Frequency: {1000 / latency_mean(e2e_times):.2f} Hz")

    times_components = None
    if args.measure_components:
        print("\n" + "-" * 50)
        print("Benchmarking Phase Breakdown...")
        print("-" * 50)
        times_components = benchmark_components(
            policy, observation, component_iterations, warmup=component_warmup
        )
        print_latency_summary(
            "Vision Encoder", times_components["vision_encoder"], include_p95=args.include_p95
        )
        print_latency_summary(
            "LLM Backbone", times_components["llm_backbone"], include_p95=args.include_p95
        )
        print_latency_summary(
            "Action Expert", times_components["action_expert"], include_p95=args.include_p95
        )

    components = build_components(data_processing_times, times_components, e2e_times)
    if components.get("_stage_hook_warnings"):
        print("  Stage hook warnings:")
        for warning in components["_stage_hook_warnings"]:
            print(f"    - {warning}")

    print_markdown_table(components, device_name, denoising_steps, include_p95=args.include_p95)

    print("\n" + "=" * 100)
    print("DETAILED SUMMARY")
    print("=" * 100)
    print(f"\nHardware: {torch.cuda.get_device_name(0)}")
    print(f"Train config: {args.train_config}")
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Action Horizon: {action_horizon}")
    print(f"Denoising Steps: {denoising_steps}")

    metric_defs = [
        ("data_processing", "Data Processing"),
        ("e2e", "E2E"),
        ("vision_encoder", "Vision Encoder"),
        ("llm_backbone", "LLM Backbone"),
        ("action_expert", "Action Expert"),
    ]
    for key, label in metric_defs:
        if key in components:
            print_latency_summary(label, components[key], include_p95=args.include_p95)
    if "e2e" in components:
        print(f"  Frequency: {1000 / latency_mean(components['e2e']):.2f} Hz")
    if args.measure_components:
        print(
            "  Calls:           "
            f"vision={np.mean(components['vision_encoder_calls']):.1f}, "
            f"llm={np.mean(components['llm_backbone_calls']):.1f}, "
            f"action_expert={np.mean(components['action_expert_calls']):.1f}"
        )
    if components.get("_stage_hook_modules"):
        print("  Stage hook modules:")
        for stage, module_path in components["_stage_hook_modules"].items():
            print(f"    - {stage}: {module_path}")

    if args.output_json:
        write_results_json(
            args.output_json,
            model_name="pi05",
            device_name=device_name,
            hardware_name=torch.cuda.get_device_name(0),
            train_config_name=args.train_config,
            checkpoint_dir=args.checkpoint_dir,
            action_horizon=action_horizon,
            denoising_steps=denoising_steps,
            args=args,
            components=components,
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    config = tyro.cli(BenchmarkConfig)
    main(config)
