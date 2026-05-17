#!/usr/bin/env python3

"""
Benchmark script for LeRobot SmolVLA inference timing.

This mirrors the latency contract used by benchmark_gr00t_inference.py and
benchmark_pi0_inference.py:

- Data Processing: LeRobot policy preprocessor on a raw observation/frame
- Vision Encoder: complete SmolVLM image embedding path, including vision tower
  and connector/projection
- LLM Backbone: SmolVLM prefix VLM forward that produces hidden states/KV cache
- Action Expert: complete iterative denoising action generation after prefix VLM
  encoding has produced the KV cache
- E2E: raw observation -> preprocessor -> public policy.select_action(...) ->
  postprocessor

Only the default PyTorch path is supported for component timing because the
timing method relies on Python method wrappers and torch.cuda.synchronize(), in
the same spirit as the GR00T/pi0 benchmarks.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
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
DEFAULT_LIBERO_DATASET_ROOT = "/home/dell/ATC/datasets/physical-intelligence/libero"


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
    CUDA-synchronized CPU wall-clock timing for nested SmolVLA stages.

    The timer records stages that match the GR00T/pi0 benchmark schema:

    - vision_encoder: full image embedding path inside VLAFlowMatching.embed_prefix(...)
    - llm_backbone: prefix SmolVLM text transformer pass that builds the KV cache
    - action_expert: full denoising loop after prefix VLM cache is prepared
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
    Time complete SmolVLA image embedding.

    SmolVLA calls VLAFlowMatching.vlm_with_expert.embed_image(...) once per
    present camera inside embed_prefix(...). Wrapping this method measures the
    full image feature path rather than only the inner vision tower, so the
    vision tower and SmolVLM connector/projection are both included.
    """

    vlm_with_expert = _get_nested_attr(policy, "model.vlm_with_expert")
    if vlm_with_expert is None or not hasattr(vlm_with_expert, "embed_image"):
        return None, (
            "Could not find policy.model.vlm_with_expert.embed_image; "
            "vision_encoder timing is unavailable."
        )

    original = vlm_with_expert.embed_image

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

    vlm_with_expert.embed_image = timed_embed_image
    return original, None


def restore_vision_encoder_method_timer(policy, original_method):
    if original_method is None:
        return
    vlm_with_expert = _get_nested_attr(policy, "model.vlm_with_expert")
    if vlm_with_expert is not None:
        vlm_with_expert.embed_image = original_method


def _get_call_argument(args, kwargs, name, position):
    if name in kwargs:
        return kwargs[name]
    if len(args) > position:
        return args[position]
    return None


def _classify_smolvlm_forward(args, kwargs):
    """
    Classify SmolVLMWithExpertModel.forward(...) calls by inference phase.

    SmolVLA implements the VLM and action expert in a custom Python loop rather
    than by calling text_model.forward(...). Therefore a direct text_model hook
    would miss the active path. The prefix call has inputs_embeds=[prefix, None]
    and fill_kv_cache=True.
    """

    inputs_embeds = _get_call_argument(args, kwargs, "inputs_embeds", 3)
    fill_kv_cache = _get_call_argument(args, kwargs, "fill_kv_cache", 5)
    if not isinstance(inputs_embeds, (list, tuple)) or len(inputs_embeds) < 2:
        return None

    prefix_embs, suffix_embs = inputs_embeds[0], inputs_embeds[1]
    if fill_kv_cache is True and prefix_embs is not None and suffix_embs is None:
        return "llm_backbone"
    return None


def install_vlm_forward_stage_timer(policy, timer):
    """
    Time SmolVLA's prefix VLM forward.

    The prefix forward is reported as llm_backbone, matching GR00T/pi0's
    hidden-state/KV-cache VLM pass.
    """

    vlm_with_expert = _get_nested_attr(policy, "model.vlm_with_expert")
    if vlm_with_expert is None or not hasattr(vlm_with_expert, "forward"):
        return None, (
            "Could not find policy.model.vlm_with_expert.forward; "
            "llm_backbone timing is unavailable."
        )

    original = vlm_with_expert.forward

    def timed_forward(*args, **kwargs):
        stage = _classify_smolvlm_forward(args, kwargs) if timer.active else None
        if stage is None:
            return original(*args, **kwargs)
        cuda_synchronize()
        start = time.perf_counter()
        output = original(*args, **kwargs)
        cuda_synchronize()
        timer.current[stage] += (time.perf_counter() - start) * 1000
        timer.current[f"{stage}_calls"] += 1
        return output

    vlm_with_expert.forward = timed_forward
    return original, None


def restore_vlm_forward_stage_timer(policy, original_method):
    if original_method is None:
        return
    vlm_with_expert = _get_nested_attr(policy, "model.vlm_with_expert")
    if vlm_with_expert is not None:
        vlm_with_expert.forward = original_method


def install_action_expert_method_timer(policy, timer):
    """
    Time SmolVLA's complete iterative action generation after prefix VLM encoding.

    The official VLAFlowMatching.sample_actions(...) first prepares prefix
    features, then runs SmolVLMWithExpertModel.forward(...) to build the prefix
    KV cache. This wrapper starts action_expert timing immediately after that
    cache is available, so the stage is analogous to:

    - GR00T action_head.get_action_with_features(...)
    - pi0's denoising loop after prefix VLM cache

    It includes action noise initialization, suffix embedding, action expert
    transformer calls, action_out_proj, Euler/flow updates, and loop overhead.
    """

    flow_model = _get_nested_attr(policy, "model")
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
            "action_expert timing is unavailable."
        )

    if not hasattr(flow_model, "config") or not hasattr(flow_model.config, "chunk_size"):
        return None, "VLAFlowMatching config.chunk_size is unavailable."

    original = flow_model.sample_actions

    @torch.no_grad()
    def timed_sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        **kwargs,
    ):
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

        if timer.active:
            cuda_synchronize()
            start = time.perf_counter()

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            denoise_time = 1.0 + step * dt
            time_tensor = torch.tensor(denoise_time, dtype=torch.float32, device=device).expand(bsize)

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

        if timer.active:
            cuda_synchronize()
            timer.current["action_expert"] += (time.perf_counter() - start) * 1000
            timer.current["action_expert_calls"] += 1

        return x_t

    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    flow_model.sample_actions = types.MethodType(timed_sample_actions, flow_model)
    return original, None


def restore_action_expert_method_timer(policy, original_method):
    if original_method is None:
        return
    flow_model = _get_nested_attr(policy, "model")
    if flow_model is not None:
        flow_model.sample_actions = original_method


def clone_observation(observation):
    """Clone a raw observation so preprocessing mutations do not leak across iterations."""
    if isinstance(observation, torch.Tensor):
        return observation.clone()
    if isinstance(observation, np.ndarray):
        return observation.copy()
    if isinstance(observation, dict):
        return {key: clone_observation(value) for key, value in observation.items()}
    if isinstance(observation, list):
        return [clone_observation(value) for value in observation]
    if isinstance(observation, tuple):
        return tuple(clone_observation(value) for value in observation)
    return observation


def prepare_model_inputs(preprocess, observation):
    """
    Prepare SmolVLA model inputs, mirroring the official LeRobot inference path.

    The preprocessor generally adds a batch dimension, appends a newline to the
    task prompt, tokenizes language, moves tensors to the target device, and
    normalizes configured numeric features.
    """

    return preprocess(observation)


def run_public_policy_inference(policy, preprocess, postprocess, observation):
    """
    Run the public SmolVLA inference path on a raw observation.

    The benchmark resets the policy before entering the timed window, because
    policy.select_action(...) queues the remaining actions from a generated
    chunk and otherwise subsequent iterations would time cheap queue pops.
    """

    batch = prepare_model_inputs(preprocess, observation)
    with torch.inference_mode():
        action = policy.select_action(batch)
        action = postprocess(action)
    return action


def run_model_action_chunk(policy, batch):
    """Run one full action-chunk inference on an already preprocessed batch."""
    policy.reset()
    return policy.predict_action_chunk(batch)


def _feature_type_name(feature) -> str:
    value = getattr(feature, "type", None)
    return getattr(value, "value", str(value))


def make_synthetic_observation(config, task: str, seed: int = 42):
    """
    Create a deterministic raw SmolVLA frame from policy.config.input_features.

    This avoids requiring a dataset download for latency smoke tests while still
    exercising the same input feature keys and tensor shapes as the checkpoint.
    Values are unbatched because the LeRobot preprocessor adds batch dimension.
    """

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    observation: dict[str, Any] = {}
    for key, feature in (config.input_features or {}).items():
        shape = tuple(feature.shape)
        feature_type = _feature_type_name(feature)
        if feature_type == "VISUAL":
            observation[key] = torch.rand(shape, generator=generator, dtype=torch.float32)
        elif feature_type == "STATE":
            observation[key] = torch.empty(shape, dtype=torch.float32).uniform_(
                -1.0, 1.0, generator=generator
            )
        else:
            observation[key] = torch.empty(shape, dtype=torch.float32).uniform_(
                -1.0, 1.0, generator=generator
            )

    if "task" not in observation:
        observation["task"] = task
    return observation


def _first_present(frame, keys):
    for key in keys:
        if key in frame:
            return frame[key]
    return None


def _vector_feature(value, shape):
    target_dim = int(shape[0]) if shape else 0
    tensor = value.detach().cpu().to(dtype=torch.float32).flatten() if isinstance(value, torch.Tensor) else torch.as_tensor(value, dtype=torch.float32).flatten()
    if tensor.numel() >= target_dim:
        return tensor[:target_dim]
    padded = torch.zeros(target_dim, dtype=torch.float32)
    padded[: tensor.numel()] = tensor
    return padded


def _map_libero_frame_to_smolvla_observation(frame, config, task: str | None):
    from libero_dataset_utils import image_to_chw_float_tensor

    state = _first_present(frame, ["observation.state", "observation/state", "state"])
    base_image = _first_present(
        frame,
        [
            "observation.images.camera1",
            "observation.image",
            "observation/image",
            "image",
        ],
    )
    wrist_image = _first_present(
        frame,
        [
            "observation.images.camera2",
            "observation.wrist_image",
            "observation/wrist_image",
            "wrist_image",
        ],
    )
    if state is None or base_image is None:
        raise KeyError(
            "Could not map LIBERO frame to SmolVLA inference keys. "
            f"Available keys: {sorted(frame.keys())}"
        )
    if wrist_image is None:
        wrist_image = base_image

    observation: dict[str, Any] = {}
    for key, feature in (config.input_features or {}).items():
        shape = tuple(feature.shape)
        feature_type = _feature_type_name(feature)
        if feature_type == "STATE":
            observation[key] = _vector_feature(state, shape)
        elif feature_type == "VISUAL":
            source_image = wrist_image if ("camera2" in key or "wrist" in key or key.endswith("image2")) else base_image
            observation[key] = image_to_chw_float_tensor(source_image)

    frame_task = task
    if frame_task is None:
        frame_task = frame.get("task", frame.get("prompt", ""))
    observation["task"] = str(frame_task)
    return observation


def load_libero_observation(
    dataset_repo_id: str,
    dataset_root: str | None,
    frame_index: int,
    task: str | None,
    config=None,
):
    """Load one raw LIBERO frame from a LeRobotDataset repo id or local dataset path."""
    from libero_dataset_utils import load_lerobot_frame

    frame = load_lerobot_frame(dataset_repo_id, dataset_root, frame_index)
    if config is not None:
        return _map_libero_frame_to_smolvla_observation(frame, config, task)
    if task is not None:
        frame["task"] = task
    elif "task" not in frame:
        frame["task"] = ""
    return frame


def make_observation(policy, args):
    if args.example_source == "synthetic":
        return make_synthetic_observation(policy.config, args.task, seed=args.seed)
    if args.example_source in {"libero", "dataset"}:
        if args.dataset_repo_id is None:
            raise ValueError("--dataset-repo-id is required when --example-source libero")
        return load_libero_observation(
            args.dataset_repo_id,
            args.dataset_root,
            args.dataset_index,
            args.task,
            policy.config,
        )
    raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))


def benchmark_data_processing(preprocess, observation, num_iterations=100, warmup=20):
    """
    Benchmark SmolVLA preprocessing separately with the same warmup style as
    GR00T/pi0. If observation is a list, cycles through observations.
    """

    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    gc.collect()

    if warmup > 0:
        for i in range(warmup):
            obs = clone_observation(observations[i % num_obs])
            _ = prepare_model_inputs(preprocess, obs)
        cuda_synchronize()
        gc.collect()

    times = []
    for i in range(num_iterations):
        obs = clone_observation(observations[i % num_obs])
        cuda_synchronize()
        start = time.perf_counter()
        _ = prepare_model_inputs(preprocess, obs)
        cuda_synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times) * 1000


def benchmark_e2e(policy, preprocess, postprocess, observation, num_iterations=100, warmup=20):
    """
    Benchmark true end-to-end SmolVLA policy latency.

    This measures raw observation preprocessing, public policy.select_action(...),
    postprocessing, and queue reset needed to force a full action-chunk inference
    each iteration. CUDA is synchronized around each timed call so CPU wall-clock
    includes queued GPU work.
    """

    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if warmup > 0:
        for i in range(warmup):
            obs = clone_observation(observations[i % num_obs])
            policy.reset()
            _ = run_public_policy_inference(policy, preprocess, postprocess, obs)
        cuda_synchronize()
        gc.collect()

    times = []
    for i in range(num_iterations):
        obs = clone_observation(observations[i % num_obs])
        policy.reset()
        cuda_synchronize()
        start = time.perf_counter()
        _ = run_public_policy_inference(policy, preprocess, postprocess, obs)
        cuda_synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times) * 1000


def benchmark_components(policy, preprocess, observation, num_iterations=100, warmup=20):
    """
    Benchmark component-wise SmolVLA timing.

    Returns dict with the same stage keys used by the GR00T/pi0 benchmark:
    vision_encoder, llm_backbone, action_expert, and *_calls.
    """

    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    stage_timer = StageTimingHooks()
    hook_warnings = []
    hook_modules = {}

    original_vision_encoder_method, vision_encoder_warning = install_vision_encoder_method_timer(
        policy, stage_timer
    )
    if vision_encoder_warning:
        hook_warnings.append(vision_encoder_warning)
    else:
        hook_modules["vision_encoder"] = "model.vlm_with_expert.embed_image"

    original_vlm_forward_method, vlm_forward_warning = install_vlm_forward_stage_timer(
        policy, stage_timer
    )
    if vlm_forward_warning:
        hook_warnings.append(vlm_forward_warning)
    else:
        hook_modules["llm_backbone"] = "model.vlm_with_expert.forward:prefix_cache"

    original_action_expert_method, action_expert_warning = install_action_expert_method_timer(
        policy, stage_timer
    )
    if action_expert_warning:
        hook_warnings.append(action_expert_warning)
    else:
        hook_modules["action_expert"] = "model.sample_actions:denoising_loop"

    try:
        for i in range(warmup):
            obs = clone_observation(observations[i % num_obs])
            batch = prepare_model_inputs(preprocess, obs)
            with torch.inference_mode():
                _ = run_model_action_chunk(policy, batch)
        cuda_synchronize()

        gc.collect()

        vision_encoder_times = []
        llm_backbone_times = []
        action_expert_times = []
        vision_encoder_calls = []
        llm_backbone_calls = []
        action_expert_calls = []

        for i in range(num_iterations):
            obs = clone_observation(observations[i % num_obs])
            batch = prepare_model_inputs(preprocess, obs)

            stage_timer.start_iteration()

            with torch.inference_mode():
                _ = run_model_action_chunk(policy, batch)
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
        restore_action_expert_method_timer(policy, original_action_expert_method)


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
        components["_e2e_note"] = "measured_preprocess_select_action_postprocess"
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
    print(f"\nSmolVLA Inference Timing ({denoising_steps} denoising steps):\n")

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
    model_id,
    dataset_repo_id,
    example_source,
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
    compile_mode = args.compile_mode if args.compile_model else None
    payload = {
        "schema_version": "vla_latency_v1",
        "model": model_name,
        "hardware": {
            "device_name": device_name,
            "torch_cuda_device_name": hardware_name,
        },
        "model_info": {
            "model_path": model_id,
            "checkpoint_dir": model_id,
            "train_config": None,
            "dataset_path": dataset_repo_id,
            "dataset_root": args.dataset_root if example_source in {"libero", "dataset"} else None,
            "embodiment_tag": None,
            "example_source": example_source,
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
            "example_source": example_source,
            "pytorch_compile_mode": compile_mode,
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
            "action_expert": "Timed around VLAFlowMatching.sample_actions denoising loop after prefix VLM cache.",
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
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote JSON results to: {output_path}")


@dataclass
class BenchmarkConfig:
    """Configuration for SmolVLA inference benchmarking."""

    model_id: str = "lerobot/smolvla_base"
    """Hugging Face model id or local checkpoint directory."""

    example_source: str = "libero"
    """Input source to use: synthetic or libero."""

    dataset_repo_id: str | None = DEFAULT_LIBERO_DATASET_REPO_ID
    """LeRobot LIBERO dataset repo id when example_source=libero."""

    dataset_root: str | None = DEFAULT_LIBERO_DATASET_ROOT
    """Local LeRobot dataset root used with dataset_repo_id when example_source=libero."""

    dataset_index: int = 0
    """Frame index to load when example_source=libero."""

    task: str = "pick up the object"
    """Language instruction used for synthetic inputs or to override a dataset frame task."""

    compile_model: bool = False
    """Whether to enable SmolVLA torch.compile in the loaded config. Component timing requires False."""

    compile_mode: str = "max-autotune"
    """Torch compile mode used when compile_model=True."""

    strict: bool = False
    """Whether checkpoint loading should require exact key matches."""

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
    """Whether to measure standalone E2E preprocess/select_action/postprocess latency."""

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


def _load_policy_and_processors(args: BenchmarkConfig, device: str):
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(args.model_id)
    if not isinstance(config, SmolVLAConfig):
        raise TypeError(f"Expected SmolVLAConfig, got {type(config).__name__}")
    config.device = device
    config.compile_model = bool(args.compile_model)
    config.compile_mode = args.compile_mode

    policy = SmolVLAPolicy.from_pretrained(args.model_id, config=config, strict=args.strict)
    policy.to(device)
    policy.eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        args.model_id,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    return policy, preprocess, postprocess


def main(args: BenchmarkConfig | None = None):
    if args is None:
        args = tyro.cli(BenchmarkConfig)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: No CUDA GPU detected. Benchmarking requires a GPU.")
        sys.exit(1)
    if args.compile_model and args.measure_components:
        raise ValueError("Component timing requires --compile-model False so Python timers are hookable.")

    device_name = get_device_name()

    if args.example_source not in {"synthetic", "libero", "dataset"}:
        raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))
    if args.example_source in {"libero", "dataset"} and args.dataset_repo_id is None:
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
    print("SMOLVLA INFERENCE BENCHMARK")
    print("=" * 100)
    print(f"Device: {device_name} ({torch.cuda.get_device_name(0)})")
    print(f"Model: {args.model_id}")
    print(f"Example source: {args.example_source}")
    print(f"Dataset repo id: {args.dataset_repo_id}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset index: {args.dataset_index}")
    print(f"Compile model: {args.compile_model}")
    print(f"Compile mode: {args.compile_mode if args.compile_model else None}")
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

    print("Loading policy and processors...")
    policy, preprocess, postprocess = _load_policy_and_processors(args, device)
    action_horizon = policy.config.chunk_size
    denoising_steps = policy.config.num_steps
    print(f"Action Horizon: {action_horizon}")
    print(f"Denoising Steps: {denoising_steps}")
    print(f"Input features: {list((policy.config.input_features or {}).keys())}")

    observation = make_observation(policy, args)

    data_processing_times = None
    if args.measure_data_processing:
        print("\n" + "-" * 50)
        print("Benchmarking Data Processing...")
        print("-" * 50)
        data_processing_times = benchmark_data_processing(
            preprocess, observation, data_processing_iterations, warmup=data_processing_warmup
        )
        print_latency_summary(
            "Data Processing", data_processing_times, include_p95=args.include_p95
        )

    e2e_times = None
    if args.measure_e2e:
        print("\n" + "-" * 50)
        print("Benchmarking E2E Inference...")
        print("-" * 50)
        e2e_times = benchmark_e2e(
            policy, preprocess, postprocess, observation, e2e_iterations, warmup=e2e_warmup
        )
        print_latency_summary("E2E", e2e_times, include_p95=args.include_p95)
        print(f"  Frequency: {1000 / latency_mean(e2e_times):.2f} Hz")

    times_components = None
    if args.measure_components:
        print("\n" + "-" * 50)
        print("Benchmarking Phase Breakdown...")
        print("-" * 50)
        times_components = benchmark_components(
            policy, preprocess, observation, component_iterations, warmup=component_warmup
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
    print(f"Model: {args.model_id}")
    print(f"Example Source: {args.example_source}")
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
            model_name="smolvla",
            device_name=device_name,
            hardware_name=torch.cuda.get_device_name(0),
            model_id=args.model_id,
            dataset_repo_id=args.dataset_repo_id,
            example_source=args.example_source,
            action_horizon=action_horizon,
            denoising_steps=denoising_steps,
            args=args,
            components=components,
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    config = tyro.cli(BenchmarkConfig)
    main(config)
