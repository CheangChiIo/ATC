#!/usr/bin/env python3

"""
Benchmark script for OpenVLA inference timing.

This mirrors the cross-model latency contract used by the GR00T, pi0, pi0.5,
and SmolVLA benchmark scripts:

- Data Processing: prompt construction, image/PIL transforms, processor tokenization,
  image preprocessing, tensor conversion, and device transfer
- Vision Encoder: Prismatic vision backbone plus visual projector, including image
  patch extraction and projected image embeddings
- LLM Backbone: multimodal prefill through the language decoder up to the final
  hidden state, excluding the LM head
- Action Expert: starts exactly when the final hidden state is prepared for the
  LM head to generate the first action token, and includes autoregressive
  action-token decoding plus token-to-normalized-continuous-action conversion
- E2E: processor(...) -> manual core action generation by default, with an
  optional public processor(...) -> vla.predict_action(...) path

Runtime setup for comparable default runs:
- Keep BenchmarkConfig.attn_implementation at its default "flash_attention_2".
  Only use "--attn-implementation sdpa" as an explicit fallback/debug run, since
  SDPA numbers are not directly comparable to the default FA2 OpenVLA runs.
- Install flash-attn in the same Python environment as torch, with nvcc matching
  torch.version.cuda.  On devices such as RTX 5090/sm_120 with torch cu128, set
  CUDA_HOME to a CUDA 12.8 toolkit before installing flash-attn.
- Keep the default "--torch-num-threads 1 --torch-num-interop-threads 1" for
  single-observation latency runs.  OpenVLA's Prismatic image processor uses
  small CPU torch ops, and the default many-thread CPU pool can dominate
  data-processing latency with scheduling spikes that are not present in the
  pi0.5/SmolVLA single-sample preprocessing path.
- Prefer local checkpoint directories via "--model-id /path/to/openvla-checkpoint"
  when available.  The default Hugging Face id is kept portable for new devices.
- For resource occupancy runs, prefer Nsight profiling commands outside this
  latency script.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import gc
import json
import math
import os
import random
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import tyro


OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

DEFAULT_LIBERO_DATASET_REPO_ID = "physical-intelligence/libero"
DEFAULT_LIBERO_DATASET_ROOT = "/home/dell/ATC/datasets/physical-intelligence/libero"
E2E_INFERENCE_PATHS = {"manual_core", "public_predict_action"}


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


def configure_torch_cpu_threads(num_threads: int | None, num_interop_threads: int | None):
    """
    Configure torch CPU threading before benchmark work starts.

    OpenVLA's native Prismatic image processor applies several small torch CPU
    tensor ops to a single image.  Letting torch fan those ops out across a large
    CPU pool makes data-processing latency mostly measure thread scheduling
    overhead.  The default of one thread keeps the measurement closer to the
    per-observation preprocessing measured by the pi0.5 and SmolVLA scripts.
    """
    if num_threads is not None:
        if num_threads < 1:
            raise ValueError("--torch-num-threads must be >= 1 or None.")
        torch.set_num_threads(num_threads)

    if num_interop_threads is not None:
        if num_interop_threads < 1:
            raise ValueError("--torch-num-interop-threads must be >= 1 or None.")
        try:
            torch.set_num_interop_threads(num_interop_threads)
        except RuntimeError as exc:
            print(
                "WARNING: Could not set torch interop threads. "
                f"This must happen before parallel work starts. Details: {exc}"
            )


class StageTimingHooks:
    """
    CUDA-synchronized CPU wall-clock timing for OpenVLA stages.

    The timer records stages that match the shared benchmark schema:

    - vision_encoder: Prismatic vision_backbone + projector
    - llm_backbone: language decoder prefill to final hidden state, no LM head
    - action_expert: first LM-head action-token generation through final action
    """

    STAGE_KEYS = ("vision_encoder", "llm_backbone", "action_expert")

    def __init__(self):
        self.active = False
        self.current = None
        self.records = []

    def start_iteration(self):
        self.active = True
        self.current = {key: 0.0 for key in self.STAGE_KEYS}
        self.current.update({f"{key}_calls": 0 for key in self.STAGE_KEYS})

    def finish_iteration(self):
        record = dict(self.current)
        self.records.append(record)
        self.active = False
        self.current = None
        return record

    def start(self):
        if not self.active:
            return None
        cuda_synchronize()
        return time.perf_counter()

    def stop(self, stage: str, start_time):
        if not self.active or start_time is None:
            return
        cuda_synchronize()
        self.current[stage] += (time.perf_counter() - start_time) * 1000
        self.current[f"{stage}_calls"] += 1


class _LegacyTimerStage:
    def __init__(self, timer, stage: str):
        self.timer = timer
        self.stage = stage
        self.start_time = None

    def __enter__(self):
        self.start_time = self.timer.start()
        return None

    def __exit__(self, exc_type, exc, tb):
        self.timer.stop(self.stage, self.start_time)
        return False


def _measure_timer_stage(timer, stage: str):
    if timer is None:
        return nullcontext()
    measure = getattr(timer, "measure", None)
    if callable(measure):
        return measure(stage)
    return _LegacyTimerStage(timer, stage)


def _to_torch_dtype(name: str | None):
    if name is None or name == "auto":
        return "auto"
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype {name!r}; use auto, bfloat16, float16, or float32.")


def _input_tensor_dtype(name: str):
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported input dtype {name!r}; use bfloat16, float16, or float32.")


def _get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _move_inputs_to_device(inputs, device, dtype):
    """Move a processor BatchFeature/dict to device, casting only floating tensors."""
    moved = {}
    for key, value in dict(inputs).items():
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                moved[key] = value.to(device=device, dtype=dtype)
            else:
                moved[key] = value.to(device=device)
        else:
            moved[key] = value
    return moved


def _get_nested_attr(obj, attr_path):
    cur = obj
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _get_decoder(model):
    """Return the decoder-only module that produces hidden states but not logits."""
    decoder = _get_nested_attr(model, "language_model.model")
    if decoder is not None:
        return decoder
    if hasattr(model, "get_decoder"):
        return model.get_decoder()
    if hasattr(model, "language_model") and hasattr(model.language_model, "get_decoder"):
        return model.language_model.get_decoder()
    raise AttributeError("Could not resolve OpenVLA language decoder module.")


def _get_lm_head(model):
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if lm_head is None:
        lm_head = _get_nested_attr(model, "language_model.lm_head")
    if lm_head is None:
        raise AttributeError("Could not resolve OpenVLA LM head/output embeddings.")
    return lm_head


def _append_empty_token_if_needed(input_ids, attention_mask=None, empty_token_id=29871):
    """
    Match OpenVLAForActionPrediction.predict_action(...): append the Llama empty
    token if the prompt does not already end with it.
    """
    if torch.all(input_ids[:, -1] == empty_token_id):
        return input_ids, attention_mask

    token = torch.full(
        (input_ids.shape[0], 1),
        fill_value=empty_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    input_ids = torch.cat((input_ids, token), dim=1)
    if attention_mask is not None:
        mask = torch.ones(
            (attention_mask.shape[0], 1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat((attention_mask, mask), dim=1)
    return input_ids, attention_mask


def _select_next_token(logits, *, do_sample: bool = False, temperature: float = 1.0):
    logits = logits[:, -1, :]
    if do_sample:
        if temperature <= 0:
            raise ValueError("temperature must be positive when do_sample=True.")
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    return torch.argmax(logits, dim=-1, keepdim=True)


def _decode_token_ids_to_normalized_actions(model, generated_token_ids, unnorm_key):
    """Mirror OpenVLA's token-to-normalized-continuous-action decode."""
    predicted_action_token_ids = generated_token_ids[0, -model.get_action_dim(unnorm_key) :].detach().cpu().numpy()
    discretized_actions = model.vocab_size - predicted_action_token_ids
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=model.bin_centers.shape[0] - 1)
    return model.bin_centers[discretized_actions]


def _unnormalize_actions(model, normalized_actions, unnorm_key):
    """Apply OpenVLA dataset-stat unnormalization outside component timing."""
    action_norm_stats = model.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
    return np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )


@torch.no_grad()
def run_manual_action_generation(
    model,
    inputs,
    *,
    unnorm_key: str | None,
    timer: StageTimingHooks | None = None,
    do_sample: bool = False,
    temperature: float = 1.0,
):
    """
    Run OpenVLA action generation with explicit timing boundaries.

    This intentionally does not call model.generate(...), because the default
    Hugging Face path computes the first LM-head logits inside the first
    model.forward(...). The benchmark boundary requested here places that first
    LM-head call inside action_expert, after the LLM prefill hidden state exists.
    """
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    pixel_values = inputs["pixel_values"]
    input_ids, attention_mask = _append_empty_token_if_needed(input_ids, attention_mask)

    action_dim = model.get_action_dim(unnorm_key)
    decoder = _get_decoder(model)
    lm_head = _get_lm_head(model)

    # Vision encoder: patch extraction plus projected image embeddings.
    with _measure_timer_stage(timer, "vision_encoder"):
        patch_features = model.vision_backbone(pixel_values)
        projected_patch_embeddings = model.projector(patch_features)

    projected_patch_attention_mask = None
    if attention_mask is not None:
        projected_patch_attention_mask = torch.full(
            (projected_patch_embeddings.shape[0], projected_patch_embeddings.shape[1]),
            fill_value=True,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

    input_embeddings = model.get_input_embeddings()(input_ids)
    multimodal_embeddings = torch.cat(
        [input_embeddings[:, :1, :], projected_patch_embeddings, input_embeddings[:, 1:, :]],
        dim=1,
    )
    multimodal_attention_mask = None
    if attention_mask is not None:
        multimodal_attention_mask = torch.cat(
            [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]],
            dim=1,
        )

    # LLM backbone: decoder prefill to final hidden state; no LM head here.
    with _measure_timer_stage(timer, "llm_backbone"):
        decoder_outputs = decoder(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
    hidden_states = decoder_outputs.last_hidden_state
    past_key_values = decoder_outputs.past_key_values

    # Action expert begins exactly at preparing last hidden state for the LM head.
    with _measure_timer_stage(timer, "action_expert"):
        generated_tokens = []

        logits = lm_head(hidden_states[:, -1:, :])
        next_token = _select_next_token(logits, do_sample=do_sample, temperature=temperature)
        generated_tokens.append(next_token)

        for _ in range(1, action_dim):
            decoder_outputs = decoder(
                input_ids=next_token,
                attention_mask=None,
                position_ids=None,
                past_key_values=past_key_values,
                inputs_embeds=None,
                use_cache=True,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden_states = decoder_outputs.last_hidden_state
            past_key_values = decoder_outputs.past_key_values

            logits = lm_head(hidden_states)
            next_token = _select_next_token(logits, do_sample=do_sample, temperature=temperature)
            generated_tokens.append(next_token)

        generated_token_ids = torch.cat(generated_tokens, dim=1)
        normalized_actions = _decode_token_ids_to_normalized_actions(model, generated_token_ids, unnorm_key)

    return _unnormalize_actions(model, normalized_actions, unnorm_key)


def build_prompt(model_id: str, task: str):
    task = str(task).strip()
    if "openvla-v01" in str(model_id).lower():
        return (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: "
            f"What action should the robot take to {task.lower()}? ASSISTANT:"
        )
    return f"In: What action should the robot take to {task.lower()}?\nOut:"


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_pil_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    image = _to_numpy(image)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    return Image.fromarray(image).convert("RGB")


def center_crop_image(image: Image.Image, crop_scale: float = 0.9):
    """Center crop with area=crop_scale*area, then resize back to original size."""
    if crop_scale >= 1.0:
        return image
    if crop_scale <= 0.0:
        raise ValueError("crop_scale must be in (0, 1].")

    width, height = image.size
    ratio = math.sqrt(crop_scale)
    crop_w = max(1, int(round(width * ratio)))
    crop_h = max(1, int(round(height * ratio)))
    left = max(0, (width - crop_w) // 2)
    upper = max(0, (height - crop_h) // 2)
    cropped = image.crop((left, upper, left + crop_w, upper + crop_h))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def clone_observation(observation):
    if isinstance(observation, torch.Tensor):
        return observation.clone()
    if isinstance(observation, np.ndarray):
        return observation.copy()
    if isinstance(observation, Image.Image):
        return observation.copy()
    if isinstance(observation, dict):
        return {key: clone_observation(value) for key, value in observation.items()}
    if isinstance(observation, list):
        return [clone_observation(value) for value in observation]
    if isinstance(observation, tuple):
        return tuple(clone_observation(value) for value in observation)
    return observation


def prepare_model_inputs(model_id, processor, observation, *, device, dtype, center_crop=False, crop_scale=0.9):
    """
    Prepare OpenVLA model inputs, mirroring the public HF inference path.

    The measured data-processing stage includes prompt construction, PIL image
    conversion, optional OpenVLA LIBERO center crop, processor image/text
    preprocessing, tensor conversion, and device transfer.
    """
    image = observation.get("image", observation.get("full_image"))
    if image is None:
        raise KeyError("OpenVLA observation must contain 'image' or 'full_image'.")
    task = observation.get("task", observation.get("prompt", "do something"))

    image = _to_pil_image(image)
    if center_crop:
        image = center_crop_image(image, crop_scale=crop_scale)

    prompt = build_prompt(model_id, task)
    inputs = processor(prompt, image, return_tensors="pt")
    return _move_inputs_to_device(inputs, device, dtype)


def run_public_policy_inference(
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
):
    inputs = prepare_model_inputs(
        model_id,
        processor,
        observation,
        device=device,
        dtype=dtype,
        center_crop=center_crop,
        crop_scale=crop_scale,
    )
    with torch.inference_mode():
        return model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=do_sample)


def run_core_policy_inference(
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
):
    inputs = prepare_model_inputs(
        model_id,
        processor,
        observation,
        device=device,
        dtype=dtype,
        center_crop=center_crop,
        crop_scale=crop_scale,
    )
    with torch.inference_mode():
        return run_manual_action_generation(
            model,
            inputs,
            unnorm_key=unnorm_key,
            do_sample=do_sample,
            temperature=temperature,
        )


def run_e2e_policy_inference(
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
):
    if e2e_inference_path == "manual_core":
        return run_core_policy_inference(
            model,
            model_id,
            processor,
            observation,
            device=device,
            dtype=dtype,
            unnorm_key=unnorm_key,
            center_crop=center_crop,
            crop_scale=crop_scale,
            do_sample=do_sample,
            temperature=temperature,
        )
    if e2e_inference_path == "public_predict_action":
        return run_public_policy_inference(
            model,
            model_id,
            processor,
            observation,
            device=device,
            dtype=dtype,
            unnorm_key=unnorm_key,
            center_crop=center_crop,
            crop_scale=crop_scale,
            do_sample=do_sample,
        )
    raise ValueError(
        f"Unsupported e2e_inference_path={e2e_inference_path!r}; "
        "use manual_core or public_predict_action."
    )


def benchmark_data_processing(
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
):
    """Benchmark OpenVLA preprocessing separately with the same warmup style as the other scripts."""
    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    gc.collect()

    if warmup > 0:
        for i in range(warmup):
            obs = clone_observation(observations[i % num_obs])
            _ = prepare_model_inputs(
                model_id,
                processor,
                obs,
                device=device,
                dtype=dtype,
                center_crop=center_crop,
                crop_scale=crop_scale,
            )
        cuda_synchronize()
        gc.collect()

    times = []
    for i in range(num_iterations):
        obs = clone_observation(observations[i % num_obs])
        cuda_synchronize()
        start = time.perf_counter()
        _ = prepare_model_inputs(
            model_id,
            processor,
            obs,
            device=device,
            dtype=dtype,
            center_crop=center_crop,
            crop_scale=crop_scale,
        )
        cuda_synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times) * 1000


def benchmark_e2e(
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
):
    """
    Benchmark end-to-end OpenVLA policy latency.

    The default manual_core path measures raw observation preprocessing plus the
    same manual core action-generation path used by component timing. The
    public_predict_action path is kept for measuring the public HF/OpenVLA API.
    CUDA is synchronized around each timed call so CPU wall-clock includes
    queued GPU work.
    """
    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if warmup > 0:
        for i in range(warmup):
            obs = clone_observation(observations[i % num_obs])
            _ = run_e2e_policy_inference(
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
        cuda_synchronize()
        gc.collect()

    times = []
    for i in range(num_iterations):
        obs = clone_observation(observations[i % num_obs])
        cuda_synchronize()
        start = time.perf_counter()
        _ = run_e2e_policy_inference(
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
        cuda_synchronize()
        end = time.perf_counter()
        times.append(end - start)

    return np.array(times) * 1000


def benchmark_components(
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
):
    """
    Benchmark component-wise OpenVLA timing.

    Returns dict with the same stage keys used by the other model benchmarks:
    vision_encoder, llm_backbone, action_expert, and *_calls.
    """
    observations = observation if isinstance(observation, list) else [observation]
    num_obs = len(observations)
    timer = StageTimingHooks()

    hook_warnings = []
    hook_modules = {
        "vision_encoder": "vision_backbone + projector",
        "llm_backbone": "language_model decoder prefill to final hidden state, excluding lm_head",
        "action_expert": "manual action-token decode starting at first lm_head(last_hidden)",
    }

    for i in range(warmup):
        obs = clone_observation(observations[i % num_obs])
        inputs = prepare_model_inputs(
            model_id,
            processor,
            obs,
            device=device,
            dtype=dtype,
            center_crop=center_crop,
            crop_scale=crop_scale,
        )
        with torch.inference_mode():
            _ = run_manual_action_generation(
                model,
                inputs,
                unnorm_key=unnorm_key,
                do_sample=do_sample,
                temperature=temperature,
            )
    cuda_synchronize()
    gc.collect()

    records = []
    for i in range(num_iterations):
        obs = clone_observation(observations[i % num_obs])
        inputs = prepare_model_inputs(
            model_id,
            processor,
            obs,
            device=device,
            dtype=dtype,
            center_crop=center_crop,
            crop_scale=crop_scale,
        )

        timer.start_iteration()
        with torch.inference_mode():
            _ = run_manual_action_generation(
                model,
                inputs,
                unnorm_key=unnorm_key,
                timer=timer,
                do_sample=do_sample,
                temperature=temperature,
            )
        cuda_synchronize()
        records.append(timer.finish_iteration())

    components = {
        "vision_encoder": np.array([record["vision_encoder"] for record in records]),
        "llm_backbone": np.array([record["llm_backbone"] for record in records]),
        "action_expert": np.array([record["action_expert"] for record in records]),
        "vision_encoder_calls": np.array([record["vision_encoder_calls"] for record in records]),
        "llm_backbone_calls": np.array([record["llm_backbone_calls"] for record in records]),
        "action_expert_calls": np.array([record["action_expert_calls"] for record in records]),
        "_stage_hook_warnings": hook_warnings,
        "_stage_hook_modules": hook_modules,
    }

    for stage in ("vision_encoder", "llm_backbone", "action_expert"):
        calls = components[f"{stage}_calls"]
        if calls.size > 0 and np.max(calls) == 0:
            hook_warnings.append(
                f"Stage '{stage}' timer was installed at '{hook_modules[stage]}', but observed zero calls."
            )

    return components


def build_components(
    data_processing_times=None,
    times_components=None,
    e2e_times=None,
    e2e_note=None,
):
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
        components["_e2e_note"] = e2e_note or "measured_processor_manual_core"
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


def print_markdown_table(data, device_name, action_tokens, include_p95=True):
    """Print results as a markdown table using mean latency, optionally with p95."""
    print("\n" + "=" * 100)
    print("MARKDOWN TABLE (copy/paste into README)")
    print("=" * 100)
    print(f"\nOpenVLA Inference Timing ({action_tokens} action tokens):\n")

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
        "action_tokens": model_info.get("action_tokens"),
        "generation_steps": model_info.get("generation_steps"),
        "step_semantics": model_info.get("step_semantics"),
        "num_steps": benchmark.get("num_steps"),
        "num_steps_semantics": benchmark.get("num_steps_semantics"),
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
    unnorm_key,
    action_horizon,
    action_tokens,
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
            "model_path": model_id,
            "checkpoint_dir": model_id,
            "train_config": None,
            "dataset_path": dataset_repo_id if example_source == "libero" else None,
            "dataset_root": args.dataset_root if example_source == "libero" else None,
            "embodiment_tag": None,
            "example_source": example_source,
            "action_horizon": int(action_horizon),
            "denoising_steps": 0,
            "action_tokens": int(action_tokens),
            "generation_steps": int(action_tokens),
            "step_semantics": "autoregressive_action_tokens_no_diffusion",
            "unnorm_key": unnorm_key,
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
            "pytorch_compile_mode": None,
            "num_steps": int(action_tokens),
            "num_steps_semantics": "autoregressive_action_tokens",
            "attn_implementation": args.attn_implementation,
            "torch_dtype": args.torch_dtype,
            "input_dtype": args.input_dtype,
            "torch_num_threads": args.torch_num_threads,
            "torch_num_interop_threads": args.torch_num_interop_threads,
            "center_crop": bool(args.center_crop),
            "crop_scale": float(args.crop_scale),
            "e2e_inference_path": args.e2e_inference_path,
            "load_in_8bit": bool(args.load_in_8bit),
            "load_in_4bit": bool(args.load_in_4bit),
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
            "data_processing": "Uses the native OpenVLA/Prismatic processor path with default torch CPU thread counts pinned to 1 for stable single-observation preprocessing latency.",
            "llm_backbone": "Timed through decoder prefill final hidden state; LM head is excluded.",
            "action_expert": "Starts at lm_head(last_hidden) for the first action token and includes autoregressive action-token decode plus token-to-normalized-continuous-action conversion; dataset-stat unnormalization is outside component timing and only included in E2E.",
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


def make_synthetic_observation(seed: int = 42, task: str = "pick up the object"):
    """Create a deterministic random OpenVLA image/task input."""
    rng = np.random.default_rng(seed)
    return {
        "image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
        "task": task,
    }


def _first_present(frame, keys):
    for key in keys:
        if key in frame:
            return frame[key]
    return None


def make_libero_observation(
    dataset_repo_id: str,
    dataset_root: str | None,
    dataset_index: int,
    task: str | None,
):
    """Load one LIBERO frame and map it to OpenVLA's image/task inference keys."""
    from libero_dataset_utils import image_to_hwc_uint8, load_lerobot_frame

    frame = load_lerobot_frame(dataset_repo_id, dataset_root, dataset_index)

    image = _first_present(
        frame,
        [
            "observation/image",
            "image",
            "observation.image",
            "observation.images.image",
            "observation.images.agentview_image",
            "observation.images.camera1",
        ],
    )
    if image is None:
        raise KeyError(
            "Could not map LIBERO frame to OpenVLA image input. "
            f"Available keys: {sorted(frame.keys())}"
        )

    frame_task = task
    if frame_task is None:
        frame_task = frame.get("task", frame.get("prompt", ""))

    return {
        "image": image_to_hwc_uint8(image),
        "task": str(frame_task),
    }


def make_observation(args: "BenchmarkConfig"):
    if args.example_source == "synthetic":
        return make_synthetic_observation(seed=args.seed, task=args.task)
    if args.example_source == "libero":
        return make_libero_observation(
            args.dataset_repo_id,
            args.dataset_root,
            args.dataset_index,
            args.task,
        )
    raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))


@dataclass
class BenchmarkConfig:
    """Configuration for OpenVLA inference benchmarking."""

    model_id: str = "/home/dell/桌面/STJ/openvla-main/checkpoints/openvla-7b-finetuned-libero-spatial"
    """Hugging Face model id or local OpenVLA checkpoint directory."""

    unnorm_key: str | None = "libero_spatial"
    """Dataset key used by OpenVLA to unnormalize action tokens."""

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

    center_crop: bool = True
    """Whether to apply OpenVLA's LIBERO center crop before processor preprocessing."""

    crop_scale: float = 0.9
    """Area ratio for OpenVLA center crop when center_crop=True."""

    torch_dtype: str | None = "bfloat16"
    """Model dtype passed to from_pretrained: auto, bfloat16, float16, or float32."""

    input_dtype: str = "bfloat16"
    """Floating input tensor dtype: bfloat16, float16, or float32."""

    torch_num_threads: int | None = 1
    """Torch CPU intra-op threads for single-observation preprocessing latency."""

    torch_num_interop_threads: int | None = 1
    """Torch CPU inter-op threads for single-observation preprocessing latency."""

    attn_implementation: str | None = "flash_attention_2"
    """Attention implementation passed to from_pretrained.

    For default cross-model comparisons keep "flash_attention_2".  Configure
    flash-attn/nvcc for the local torch CUDA version instead of silently changing
    this to "sdpa"; use sdpa/eager only for explicit fallback diagnostics.
    """

    load_in_8bit: bool = False
    """Whether to load the model with bitsandbytes 8-bit quantization."""

    load_in_4bit: bool = False
    """Whether to load the model with bitsandbytes 4-bit quantization."""

    do_sample: bool = False
    """Whether action-token generation samples instead of greedy decoding."""

    temperature: float = 1.0
    """Sampling temperature used when do_sample=True for manual-core timing."""

    e2e_inference_path: str = "manual_core"
    """E2E path: manual_core or public_predict_action."""

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
    """Whether to measure standalone E2E latency."""

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


def _register_openvla_auto_classes():
    """
    Register OpenVLA classes when the local openvla/prismatic package is installed.
    Remote-code checkpoints can still load without this registration.
    """
    try:
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

        def register_once(fn, *register_args):
            try:
                fn(*register_args, exist_ok=True)
            except TypeError:
                try:
                    fn(*register_args)
                except ValueError:
                    pass
            except ValueError:
                pass

        register_once(AutoConfig.register, "openvla", OpenVLAConfig)
        register_once(AutoImageProcessor.register, OpenVLAConfig, PrismaticImageProcessor)
        register_once(AutoProcessor.register, OpenVLAConfig, PrismaticProcessor)
        register_once(AutoModelForVision2Seq.register, OpenVLAConfig, OpenVLAForActionPrediction)
    except Exception:
        return


def _load_model_and_processor(args: BenchmarkConfig, device: str):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    _register_openvla_auto_classes()
    if args.load_in_8bit and args.load_in_4bit:
        raise ValueError("Cannot use both load_in_8bit and load_in_4bit.")

    model_kwargs: dict[str, Any] = {
        "torch_dtype": _to_torch_dtype(args.torch_dtype),
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
        "load_in_8bit": args.load_in_8bit,
        "load_in_4bit": args.load_in_4bit,
    }
    if args.attn_implementation is not None:
        model_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForVision2Seq.from_pretrained(args.model_id, **model_kwargs)
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(device)
    model.eval()

    # Local converted checkpoints sometimes store stats beside the model files.
    stats_path = os.path.join(os.path.abspath(os.path.expanduser(args.model_id)), "dataset_statistics.json")
    if os.path.isfile(stats_path):
        with open(stats_path, "r", encoding="utf-8") as f:
            model.norm_stats = json.load(f)

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    return model, processor


def main(args: BenchmarkConfig | None = None):
    if args is None:
        args = tyro.cli(BenchmarkConfig)

    configure_torch_cpu_threads(args.torch_num_threads, args.torch_num_interop_threads)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: No CUDA GPU detected. Benchmarking requires a GPU.")
        sys.exit(1)
    device_name = get_device_name()
    input_dtype = _input_tensor_dtype(args.input_dtype)

    if args.example_source not in {"synthetic", "libero"}:
        raise ValueError("Unsupported example_source={!r}; use synthetic or libero.".format(args.example_source))
    if args.example_source == "libero" and args.dataset_repo_id is None:
        raise ValueError("--dataset-repo-id is required when --example-source libero")
    if not (args.measure_data_processing or args.measure_e2e or args.measure_components):
        raise ValueError("At least one measurement group must be enabled.")
    if args.do_sample and args.temperature <= 0:
        raise ValueError("--temperature must be positive when --do-sample=True.")
    if args.e2e_inference_path not in E2E_INFERENCE_PATHS:
        raise ValueError(
            f"Unsupported e2e_inference_path={args.e2e_inference_path!r}; "
            "use manual_core or public_predict_action."
        )

    data_processing_iterations = resolve_count(
        args.data_processing_iterations, args.num_iterations
    )
    data_processing_warmup = resolve_count(args.data_processing_warmup, args.warmup)
    e2e_iterations = resolve_count(args.e2e_iterations, args.num_iterations)
    e2e_warmup = resolve_count(args.e2e_warmup, args.warmup)
    component_iterations = resolve_count(args.component_iterations, args.num_iterations)
    component_warmup = resolve_count(args.component_warmup, args.warmup)

    print("=" * 100)
    print("OPENVLA INFERENCE BENCHMARK")
    print("=" * 100)
    print(f"Device: {device_name} ({torch.cuda.get_device_name(0)})")
    print(f"Model: {args.model_id}")
    print(f"Unnorm key: {args.unnorm_key}")
    print(f"Example source: {args.example_source}")
    print(f"Dataset repo/path: {args.dataset_repo_id}")
    print(f"Dataset root: {args.dataset_root}")
    print(f"Dataset index: {args.dataset_index}")
    print(f"Center crop: {args.center_crop} (scale={args.crop_scale})")
    print(f"Attention implementation: {args.attn_implementation}")
    print(f"Torch dtype: {args.torch_dtype}")
    print(f"Input dtype: {args.input_dtype}")
    print(
        "Torch CPU threads: "
        f"intra-op={torch.get_num_threads()}, inter-op={torch.get_num_interop_threads()}"
    )
    print(f"Quantization: 8bit={args.load_in_8bit}, 4bit={args.load_in_4bit}")
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

    print("Loading model and processor...")
    model, processor = _load_model_and_processor(args, device)
    model_device = _get_model_device(model)
    action_tokens = model.get_action_dim(args.unnorm_key)
    action_horizon = 1
    print(f"Action Horizon: {action_horizon}")
    print(f"Action Tokens: {action_tokens}")
    print(f"Model device: {model_device}")

    observation = make_observation(args)

    data_processing_times = None
    if args.measure_data_processing:
        print("\n" + "-" * 50)
        print("Benchmarking Data Processing...")
        print("-" * 50)
        data_processing_times = benchmark_data_processing(
            args.model_id,
            processor,
            observation,
            device=model_device,
            dtype=input_dtype,
            center_crop=args.center_crop,
            crop_scale=args.crop_scale,
            num_iterations=data_processing_iterations,
            warmup=data_processing_warmup,
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
        )
        print_latency_summary("E2E", e2e_times, include_p95=args.include_p95)
        print(f"  Frequency: {1000 / latency_mean(e2e_times):.2f} Hz")

    times_components = None
    if args.measure_components:
        print("\n" + "-" * 50)
        print("Benchmarking Phase Breakdown...")
        print("-" * 50)
        times_components = benchmark_components(
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

    e2e_note = (
        "measured_processor_manual_core"
        if args.e2e_inference_path == "manual_core"
        else "measured_processor_predict_action"
    )
    components = build_components(
        data_processing_times,
        times_components,
        e2e_times,
        e2e_note=e2e_note,
    )
    if components.get("_stage_hook_warnings"):
        print("  Stage hook warnings:")
        for warning in components["_stage_hook_warnings"]:
            print(f"    - {warning}")

    print_markdown_table(components, device_name, action_tokens, include_p95=args.include_p95)

    print("\n" + "=" * 100)
    print("DETAILED SUMMARY")
    print("=" * 100)
    print(f"\nHardware: {torch.cuda.get_device_name(0)}")
    print(f"Model: {args.model_id}")
    print(f"Unnorm key: {args.unnorm_key}")
    print(f"Action Horizon: {action_horizon}")
    print(f"Action Tokens: {action_tokens}")

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
            model_name="openvla",
            device_name=device_name,
            hardware_name=torch.cuda.get_device_name(0),
            model_id=args.model_id,
            dataset_repo_id=args.dataset_repo_id,
            example_source=args.example_source,
            unnorm_key=args.unnorm_key,
            action_horizon=action_horizon,
            action_tokens=action_tokens,
            args=args,
            components=components,
        )

    print("\n" + "=" * 100)


if __name__ == "__main__":
    config = tyro.cli(BenchmarkConfig)
    main(config)
