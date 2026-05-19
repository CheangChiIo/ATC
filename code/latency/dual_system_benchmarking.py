#!/usr/bin/env python3
"""Shared latency/resource benchmark utilities for dual-system VLA/VLM-action models.

This file intentionally keeps the same measurement philosophy as openvla.py,
pi05.py and smolvla.py in this directory, but exposes the new dual-system stage
schema used by Hume, RoboDual and OpenHelix:

    data_processing
    system2_vision_encoder
    system2_inference
    system_bridge
    system1_vision_encoder
    system1_action_expert
    e2e

Boundary contract:
- data_processing: raw sample -> model-ready tensors/prompts on target device.
- system2_vision_encoder: System-2 image tensor -> System-2 visual/projected tokens.
- system2_inference: System-2 transformer/VLA/VLM starts -> System-2 high-level output ready.
- system_bridge: System-2 output -> System-1 condition tensor/latent/action-condition ready.
- system1_vision_encoder: System-1 observation image/RGB-D/scene input -> System-1 visual/scene tokens.
- system1_action_expert: before noisy action/trajectory initialization -> final scheduler/Euler
  update finished and clean internal-space action chunk is produced. Dataset-stat
  unnormalization and robot command postprocessing are intentionally outside this component.
- e2e: full public inference path, including preprocessing and any public postprocessing.

For real repositories, pass --model-loader package.module:function. The loader should
return an adapter object implementing the methods defined by DualSystemAdapter.
A deterministic DummyDualSystemAdapter is provided for smoke tests and CI without a model repo.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from contextlib import contextmanager
import argparse
import gc
import importlib
import json
import math
import os
import random
import time
from typing import Any, Callable

import numpy as np
import torch

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None

try:
    import libero_dataset_utils
except Exception:  # pragma: no cover
    libero_dataset_utils = None

try:
    import calvin_dataset_utils
except Exception:  # pragma: no cover
    calvin_dataset_utils = None

DEFAULT_LIBERO_DATASET_REPO_ID = "physical-intelligence/libero"
DEFAULT_LIBERO_DATASET_ROOT = "/home/dell/ATC/datasets/physical-intelligence/libero"
DEFAULT_CALVIN_DATASET_ROOT = "/home/dell/ATC/datasets/calvin/calvin_debug_dataset"
DEFAULT_CALVIN_SPLIT = "validation"

DUAL_SYSTEM_LATENCY_KEYS = (
    "data_processing",
    "system2_vision_encoder",
    "system2_inference",
    "system_bridge",
    "system1_vision_encoder",
    "system1_action_expert",
    "e2e",
)

DUAL_SYSTEM_COMPONENT_KEYS = (
    "system2_vision_encoder",
    "system2_inference",
    "system_bridge",
    "system1_vision_encoder",
    "system1_action_expert",
)

DUAL_SYSTEM_CALL_KEYS = tuple(f"{stage}_calls" for stage in DUAL_SYSTEM_COMPONENT_KEYS)

DUAL_SYSTEM_STAGE_LABELS = {
    "data_processing": "Data Processing",
    "system2_vision_encoder": "System2 Vision Encoder",
    "system2_inference": "System2 Inference",
    "system_bridge": "System Bridge / Projector",
    "system1_vision_encoder": "System1 Vision Encoder",
    "system1_action_expert": "System1 Action Expert",
    "e2e": "E2E",
}


@dataclass
class DualSystemModelSpec:
    model_name: str
    default_model_id: str | None = None
    default_checkpoint_dir: str | None = None
    default_train_config: str | None = None
    default_example_source: str = "libero"
    default_calvin_dataset_root: str = DEFAULT_CALVIN_DATASET_ROOT
    default_calvin_split: str = DEFAULT_CALVIN_SPLIT
    system2_output_name: str = "system2_output"
    bridge_name: str = "projector_adapter"
    system1_action_expert_name: str = "system1_action_expert"
    action_horizon: int = 8
    action_dim: int = 7
    # None means "runtime adapter must report the true iterative step count".
    # Individual model wrappers should set this only when the value is truly used
    # by the execution path, such as RoboDual's specialist diffusion steps.
    denoising_steps: int | None = None
    default_task: str = "pick up the object"
    notes: dict[str, str] | None = None


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device_name():
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
        return name
    return "CPU"


def cuda_synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def measure_latency_stage(timer: "DualSystemStageTimer", stage: str):
    if not timer.active:
        yield
        return
    cuda_synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        cuda_synchronize()
        timer.current[stage] += (time.perf_counter() - start) * 1000.0
        timer.current[f"{stage}_calls"] += 1


class DualSystemStageTimer:
    def __init__(self):
        self.active = False
        self.current: dict[str, float | int] = {}
        self.records: list[dict[str, float | int]] = []

    def start_iteration(self):
        self.active = True
        self.current = {stage: 0.0 for stage in DUAL_SYSTEM_COMPONENT_KEYS}
        self.current.update({f"{stage}_calls": 0 for stage in DUAL_SYSTEM_COMPONENT_KEYS})

    def finish_iteration(self) -> dict[str, float | int]:
        record = dict(self.current)
        self.records.append(record)
        self.active = False
        self.current = {}
        return record


class DualSystemAdapter:
    """Adapter interface used by the shared benchmark loop.

    Real model loaders should return an instance with these methods. The methods
    should NOT do timing themselves; this benchmark wraps them with identical
    synchronized boundaries for latency and with identical NVTX/resource ranges
    for resource scripts.
    """

    model_info: dict[str, Any] = {}

    def prepare_inputs(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def run_system2_vision_encoder(self, inputs: dict[str, Any]) -> Any:
        raise NotImplementedError

    def run_system2_inference(self, inputs: dict[str, Any], system2_visual: Any) -> Any:
        raise NotImplementedError

    def run_system_bridge(self, inputs: dict[str, Any], system2_output: Any) -> Any:
        raise NotImplementedError

    def run_system1_vision_encoder(self, inputs: dict[str, Any]) -> Any:
        raise NotImplementedError

    def run_system1_action_expert(self, inputs: dict[str, Any], bridge_output: Any, system1_visual: Any) -> Any:
        raise NotImplementedError

    def infer(self, observation: dict[str, Any]) -> Any:
        inputs = self.prepare_inputs(observation)
        return run_component_path(self, inputs)


class TinyBlock(torch.nn.Module):
    def __init__(self, dim: int, layers: int = 2):
        super().__init__()
        mods = []
        for _ in range(layers):
            mods.extend([torch.nn.LayerNorm(dim), torch.nn.Linear(dim, dim), torch.nn.GELU()])
        self.net = torch.nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DummyDualSystemAdapter(DualSystemAdapter):
    """Deterministic stand-in that exercises all benchmark stages.

    It is not a scientific model. It exists only so every script can be run on a
    server before the actual Hume/RoboDual/OpenHelix repository adapter is wired.
    """

    def __init__(
        self,
        spec: DualSystemModelSpec,
        *,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float32,
        image_size: int = 224,
        system2_dim: int = 512,
        system1_dim: int = 256,
    ):
        self.spec = spec
        self.device = torch.device(device)
        self.dtype = dtype
        self.image_size = int(image_size)
        self.system2_dim = int(system2_dim)
        self.system1_dim = int(system1_dim)
        self.action_horizon = int(spec.action_horizon)
        self.action_dim = int(spec.action_dim)
        # Dummy-only fallback. Real adapters must report runtime step counts.
        # If a spec omits denoising_steps, dummy keeps a deterministic smoke-test
        # loop count without leaking that fallback into real-model metadata.
        self.denoising_steps = int(spec.denoising_steps) if spec.denoising_steps is not None and int(spec.denoising_steps) > 0 else 10

        g = torch.Generator(device="cpu")
        g.manual_seed(1234)
        self.s2_patch = torch.nn.Conv2d(3, system2_dim, kernel_size=16, stride=16)
        self.s2_proj = torch.nn.Linear(system2_dim, system2_dim)
        self.s2_transformer = TinyBlock(system2_dim, layers=4)
        self.s2_action_head = torch.nn.Linear(system2_dim, self.action_horizon * self.action_dim)

        self.bridge = torch.nn.Linear(self.action_horizon * self.action_dim, system1_dim)

        self.s1_patch = torch.nn.Conv2d(3, system1_dim, kernel_size=16, stride=16)
        self.s1_proj = torch.nn.Linear(system1_dim, system1_dim)
        self.state_mlp = torch.nn.Sequential(torch.nn.Linear(16, system1_dim), torch.nn.GELU(), torch.nn.Linear(system1_dim, system1_dim))
        self.action_embed = torch.nn.Linear(self.action_dim, system1_dim)
        self.timestep_embed = torch.nn.Linear(1, system1_dim)
        self.s1_denoiser = TinyBlock(system1_dim, layers=3)
        self.s1_out = torch.nn.Linear(system1_dim, self.action_dim)
        self.to(device=self.device, dtype=self.dtype)
        self.eval()
        self.model_info = {
            "adapter": "DummyDualSystemAdapter",
            "dummy": True,
            "image_size": self.image_size,
            "dummy_denoising_steps": self.denoising_steps,
        }

    def to(self, *args, **kwargs):
        for m in [self.s2_patch, self.s2_proj, self.s2_transformer, self.s2_action_head, self.bridge, self.s1_patch, self.s1_proj, self.state_mlp, self.action_embed, self.timestep_embed, self.s1_denoiser, self.s1_out]:
            m.to(*args, **kwargs)
        return self

    def eval(self):
        for m in [self.s2_patch, self.s2_proj, self.s2_transformer, self.s2_action_head, self.bridge, self.s1_patch, self.s1_proj, self.state_mlp, self.action_embed, self.timestep_embed, self.s1_denoiser, self.s1_out]:
            m.eval()
        return self

    def _image_to_tensor(self, image: Any) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            tensor = image.detach().clone()
            if tensor.ndim == 3 and tensor.shape[0] not in {1, 3} and tensor.shape[-1] in {1, 3}:
                tensor = tensor.permute(2, 0, 1)
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.float()
            if tensor.max().item() > 2.0:
                tensor = tensor / 255.0
        else:
            arr = np.asarray(image, dtype=np.uint8).copy()
            if arr.ndim == 3 and arr.shape[-1] in {1, 3}:
                tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            else:
                raise ValueError(f"Unsupported image shape for dummy adapter: {arr.shape}")
        tensor = torch.nn.functional.interpolate(tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        return tensor.to(self.device, dtype=self.dtype)

    def prepare_inputs(self, observation: dict[str, Any]) -> dict[str, Any]:
        image = observation.get("image")
        if image is None:
            image = observation.get("third_view_image")
        if image is None:
            image = observation.get("wrist_image")
        system1_image = observation.get("wrist_image")
        if system1_image is None:
            system1_image = image
        state_value = observation.get("state")
        if state_value is None:
            state = torch.zeros(1, 16, device=self.device, dtype=self.dtype)
        else:
            state = torch.as_tensor(state_value, dtype=self.dtype, device=self.device).flatten()
            if state.numel() < 16:
                state = torch.nn.functional.pad(state, (0, 16 - state.numel()))
            state = state[:16].reshape(1, 16)
        return {
            "system2_image": self._image_to_tensor(image),
            "system1_image": self._image_to_tensor(system1_image),
            "state": state,
            "task": observation.get("task", self.spec.default_task),
        }

    @torch.inference_mode()
    def run_system2_vision_encoder(self, inputs: dict[str, Any]) -> torch.Tensor:
        x = self.s2_patch(inputs["system2_image"]).flatten(2).transpose(1, 2)
        return self.s2_proj(x)

    @torch.inference_mode()
    def run_system2_inference(self, inputs: dict[str, Any], system2_visual: torch.Tensor) -> torch.Tensor:
        h = self.s2_transformer(system2_visual).mean(dim=1)
        # Simulate full System-2 completion: high-level action/latent ready.
        return self.s2_action_head(h).reshape(1, self.action_horizon, self.action_dim)

    @torch.inference_mode()
    def run_system_bridge(self, inputs: dict[str, Any], system2_output: torch.Tensor) -> torch.Tensor:
        return self.bridge(system2_output.flatten(1))

    @torch.inference_mode()
    def run_system1_vision_encoder(self, inputs: dict[str, Any]) -> torch.Tensor:
        x = self.s1_patch(inputs["system1_image"]).flatten(2).transpose(1, 2)
        return self.s1_proj(x)

    @torch.inference_mode()
    def run_system1_action_expert(self, inputs: dict[str, Any], bridge_output: torch.Tensor, system1_visual: torch.Tensor) -> torch.Tensor:
        # Boundary starts before noisy action initialization and ends after final scheduler update.
        bsz = bridge_output.shape[0]
        x_t = torch.randn(bsz, self.action_horizon, self.action_dim, device=self.device, dtype=self.dtype)
        state_cond = self.state_mlp(inputs["state"])
        visual_cond = system1_visual.mean(dim=1)
        bridge_cond = bridge_output
        dt = -1.0 / float(self.denoising_steps)
        denoise_time = torch.ones(bsz, 1, device=self.device, dtype=self.dtype)
        for _ in range(self.denoising_steps):
            h = self.action_embed(x_t) + state_cond[:, None, :] + visual_cond[:, None, :] + bridge_cond[:, None, :]
            h = h + self.timestep_embed(denoise_time)[:, None, :]
            v_t = self.s1_out(self.s1_denoiser(h))
            x_t = x_t + dt * v_t
            denoise_time = denoise_time + dt
        return x_t


def _callable_from_string(path: str) -> Callable[..., Any]:
    if ":" not in path:
        raise ValueError("--model-loader must have form package.module:function")
    module_name, fn_name = path.split(":", 1)
    module = importlib.import_module(module_name)
    fn = getattr(module, fn_name)
    if not callable(fn):
        raise TypeError(f"{path!r} is not callable")
    return fn


def load_adapter_from_args(args: argparse.Namespace, spec: DualSystemModelSpec) -> DualSystemAdapter:
    dtype = dtype_from_string(args.torch_dtype)
    if args.model_loader:
        fn = _callable_from_string(args.model_loader)
        adapter = fn(
            model_id=args.model_id,
            checkpoint_dir=args.checkpoint_dir,
            train_config=args.train_config,
            device=args.device,
            dtype=dtype,
            spec=spec,
            args=args,
        )
        required = [
            "prepare_inputs",
            "run_system2_vision_encoder",
            "run_system2_inference",
            "run_system_bridge",
            "run_system1_vision_encoder",
            "run_system1_action_expert",
        ]
        missing = [name for name in required if not hasattr(adapter, name)]
        if missing:
            raise TypeError(f"Adapter returned by {args.model_loader!r} is missing methods: {missing}")
        return adapter
    return DummyDualSystemAdapter(
        spec,
        device=args.device,
        dtype=dtype,
        image_size=args.image_size,
        system2_dim=args.dummy_system2_dim,
        system1_dim=args.dummy_system1_dim,
    )


def dtype_from_string(name: str) -> torch.dtype:
    value = str(name).lower().replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if value not in mapping:
        raise ValueError(f"Unsupported torch_dtype={name!r}; expected one of {sorted(mapping)}")
    return mapping[value]


def make_synthetic_observation(seed: int = 42, task: str = "pick up the object") -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
        "wrist_image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
        "third_view_image": rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8),
        "state": rng.normal(size=(16,)).astype(np.float32),
        "task": task,
    }


def _first_present(frame: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        if key in frame and frame[key] is not None:
            return frame[key]
    return None


def make_libero_observation(dataset_repo_id: str, dataset_root: str | None, dataset_index: int, task: str | None) -> dict[str, Any]:
    if libero_dataset_utils is None:
        raise RuntimeError("libero_dataset_utils.py could not be imported from this directory.")
    frame = libero_dataset_utils.load_lerobot_frame(dataset_repo_id, dataset_root, dataset_index)
    image = _first_present(frame, [
        "image",
        "observation.image",
        "observation.images.image",
        "observation.images.front",
        "observation.images.agentview_rgb",
        "observation.images.primary",
    ])
    wrist = _first_present(frame, [
        "wrist_image",
        "observation.wrist_image",
        "observation.images.wrist",
        "observation.images.wrist_rgb",
        "observation.images.hand",
    ])
    state = _first_present(frame, ["state", "observation.state", "robot_state", "proprio", "observation.proprio"])
    if image is None:
        image = make_synthetic_observation(dataset_index)["image"]
    if wrist is None:
        wrist = image
    if state is None:
        state = np.zeros((16,), dtype=np.float32)
    return {
        "image": _to_hwc_uint8(image),
        "wrist_image": _to_hwc_uint8(wrist),
        "third_view_image": _to_hwc_uint8(image),
        "state": np.asarray(state, dtype=np.float32).reshape(-1),
        "task": task or frame.get("task") or "pick up the object",
    }


def _to_hwc_uint8(value: Any) -> np.ndarray:
    if libero_dataset_utils is not None:
        try:
            return libero_dataset_utils.image_to_hwc_uint8(value)
        except Exception:
            pass
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def make_calvin_observation(dataset_root: str, split: str, dataset_index: int, task: str | None) -> dict[str, Any]:
    if calvin_dataset_utils is None:
        raise RuntimeError("calvin_dataset_utils.py could not be imported from this directory.")
    frame = calvin_dataset_utils.load_calvin_frame(dataset_root, split, dataset_index)
    image = frame.get("image")
    wrist = frame.get("wrist_image")
    third = frame.get("third_view_image")
    state = frame.get("state")
    if image is None:
        image = make_synthetic_observation(dataset_index)["image"]
    if wrist is None:
        wrist = image
    if third is None:
        third = image
    if state is None:
        state = np.zeros((16,), dtype=np.float32)
    return {
        "image": _to_hwc_uint8(image),
        "wrist_image": _to_hwc_uint8(wrist),
        "third_view_image": _to_hwc_uint8(third),
        "depth_static": frame.get("depth_static"),
        "depth_gripper": frame.get("depth_gripper"),
        "state": np.asarray(state, dtype=np.float32).reshape(-1),
        "task": task or frame.get("task") or "perform a manipulation task",
    }


def make_observations(args: argparse.Namespace, spec: DualSystemModelSpec) -> list[dict[str, Any]]:
    observations = []
    count = max(1, int(args.dataset_samples))
    if args.example_source == "libero":
        for i in range(count):
            observations.append(
                make_libero_observation(
                    args.dataset_repo_id,
                    args.dataset_root,
                    int(args.dataset_index) + i,
                    args.task or spec.default_task,
                )
            )
    elif args.example_source == "calvin":
        for i in range(count):
            observations.append(
                make_calvin_observation(
                    args.calvin_dataset_root,
                    args.calvin_split,
                    int(args.dataset_index) + i,
                    args.task or spec.default_task,
                )
            )
    else:
        for i in range(count):
            observations.append(make_synthetic_observation(seed=int(args.seed) + i, task=args.task or spec.default_task))
    return observations


def run_component_path(adapter: DualSystemAdapter, prepared_inputs: dict[str, Any], timer: DualSystemStageTimer | None = None) -> Any:
    if timer is None:
        system2_visual = adapter.run_system2_vision_encoder(prepared_inputs)
        system2_output = adapter.run_system2_inference(prepared_inputs, system2_visual)
        bridge_output = adapter.run_system_bridge(prepared_inputs, system2_output)
        system1_visual = adapter.run_system1_vision_encoder(prepared_inputs)
        return adapter.run_system1_action_expert(prepared_inputs, bridge_output, system1_visual)
    with measure_latency_stage(timer, "system2_vision_encoder"):
        system2_visual = adapter.run_system2_vision_encoder(prepared_inputs)
    with measure_latency_stage(timer, "system2_inference"):
        system2_output = adapter.run_system2_inference(prepared_inputs, system2_visual)
    with measure_latency_stage(timer, "system_bridge"):
        bridge_output = adapter.run_system_bridge(prepared_inputs, system2_output)
    with measure_latency_stage(timer, "system1_vision_encoder"):
        system1_visual = adapter.run_system1_vision_encoder(prepared_inputs)
    with measure_latency_stage(timer, "system1_action_expert"):
        output = adapter.run_system1_action_expert(prepared_inputs, bridge_output, system1_visual)
    return output


def run_public_e2e(adapter: DualSystemAdapter, observation: dict[str, Any]) -> Any:
    if hasattr(adapter, "infer"):
        return adapter.infer(observation)
    inputs = adapter.prepare_inputs(observation)
    return run_component_path(adapter, inputs)


def benchmark_data_processing(adapter: DualSystemAdapter, observations: list[dict[str, Any]], *, num_iterations: int, warmup: int) -> np.ndarray:
    num_obs = len(observations)
    gc.collect()
    for i in range(warmup):
        _ = adapter.prepare_inputs(observations[i % num_obs])
    cuda_synchronize()
    times = []
    for i in range(num_iterations):
        cuda_synchronize()
        start = time.perf_counter()
        _ = adapter.prepare_inputs(observations[i % num_obs])
        cuda_synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(times, dtype=float)


def benchmark_e2e(adapter: DualSystemAdapter, observations: list[dict[str, Any]], *, num_iterations: int, warmup: int) -> np.ndarray:
    num_obs = len(observations)
    gc.collect()
    for i in range(warmup):
        with torch.inference_mode():
            _ = run_public_e2e(adapter, observations[i % num_obs])
    cuda_synchronize()
    times = []
    for i in range(num_iterations):
        cuda_synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            _ = run_public_e2e(adapter, observations[i % num_obs])
        cuda_synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return np.asarray(times, dtype=float)


def benchmark_components(adapter: DualSystemAdapter, observations: list[dict[str, Any]], *, num_iterations: int, warmup: int) -> dict[str, Any]:
    num_obs = len(observations)
    prepared = [adapter.prepare_inputs(obs) for obs in observations]
    timer = DualSystemStageTimer()
    gc.collect()
    for i in range(warmup):
        with torch.inference_mode():
            _ = run_component_path(adapter, prepared[i % num_obs])
    cuda_synchronize()
    records = []
    for i in range(num_iterations):
        timer.start_iteration()
        with torch.inference_mode():
            _ = run_component_path(adapter, prepared[i % num_obs], timer=timer)
        cuda_synchronize()
        records.append(timer.finish_iteration())
    result: dict[str, Any] = {}
    for stage in DUAL_SYSTEM_COMPONENT_KEYS:
        result[stage] = np.asarray([r[stage] for r in records], dtype=float)
        result[f"{stage}_calls"] = np.asarray([r[f"{stage}_calls"] for r in records], dtype=float)
    result["_stage_hook_modules"] = {stage: "explicit_adapter_boundary" for stage in DUAL_SYSTEM_COMPONENT_KEYS}
    result["_stage_hook_warnings"] = []
    return result


def resolve_count(value: int | None, default: int) -> int:
    return int(default if value is None else value)


def build_components(data_processing_times=None, component_times=None, e2e_times=None) -> dict[str, Any]:
    components: dict[str, Any] = {}
    if component_times:
        components.update(component_times)
    if data_processing_times is not None:
        components["data_processing"] = data_processing_times
    if e2e_times is not None:
        components["e2e"] = e2e_times
    return components


def latency_mean(values):
    return float(np.mean(values))


def _series_summary(values, include_samples: bool = True) -> dict[str, Any]:
    arr = np.asarray(values, dtype=float)
    out = {
        "mean": float(np.mean(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": int(arr.size),
    }
    if include_samples:
        out["samples"] = [float(x) for x in arr.tolist()]
    return out


def _json_safe(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in list(value)]
    return value


def _copy_summary_fields(stats: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    return {
        f"{prefix}mean": _json_safe(stats.get("mean")),
        f"{prefix}p5": _json_safe(stats.get("p5")),
        f"{prefix}p95": _json_safe(stats.get("p95")),
        f"{prefix}std": _json_safe(stats.get("std")),
        f"{prefix}min": _json_safe(stats.get("min")),
        f"{prefix}max": _json_safe(stats.get("max")),
        f"{prefix}n": _json_safe(stats.get("n")),
    }


def _safe_int(value: Any, fallback: int | None = None) -> int | None:
    try:
        if value is None:
            return fallback
        return int(value)
    except Exception:
        return fallback


def _adapter_model_info(adapter: Any) -> dict[str, Any]:
    info = getattr(adapter, "model_info", {}) or {}
    return dict(info) if isinstance(info, dict) else {}


def resolve_runtime_model_info(
    *,
    spec: DualSystemModelSpec,
    args: argparse.Namespace,
    adapter: Any,
) -> dict[str, Any]:
    """Build model_info using runtime adapter metadata when available.

    Some real dual-system adapters know their actual iterative denoising
    settings only after the checkpoint/config is loaded.  Prefer those runtime
    values over script-level defaults so JSON/plot metadata never emits stale
    hand-entered denoising step annotations.
    """
    adapter_info = _adapter_model_info(adapter)
    runtime_action_horizon = _safe_int(adapter_info.get("action_horizon"), int(spec.action_horizon))
    runtime_action_dim = _safe_int(adapter_info.get("action_dim"), int(spec.action_dim))
    runtime_step_value = adapter_info.get("denoising_steps", adapter_info.get("system1_denoising_steps"))
    used_runtime_denoising_steps = runtime_step_value is not None
    if not used_runtime_denoising_steps and adapter_info.get("dummy") is False:
        raise ValueError(
            f"{spec.model_name} real adapter did not report denoising_steps/system1_denoising_steps; "
            "refusing to emit a hand-entered fallback value."
        )
    spec_denoising_steps = _safe_int(spec.denoising_steps, None)
    runtime_denoising_steps = _safe_int(runtime_step_value, spec_denoising_steps)
    reported_denoising_steps = runtime_denoising_steps if runtime_denoising_steps is not None else spec_denoising_steps
    return {
        "model_path": args.model_id,
        "checkpoint_dir": args.checkpoint_dir,
        "train_config": args.train_config,
        "dataset_path": args.dataset_repo_id if args.example_source == "libero" else (args.calvin_dataset_root if args.example_source == "calvin" else None),
        "dataset_root": args.dataset_root if args.example_source == "libero" else (args.calvin_dataset_root if args.example_source == "calvin" else None),
        "embodiment_tag": None,
        "example_source": args.example_source,
        "action_horizon": int(runtime_action_horizon or spec.action_horizon),
        "denoising_steps": int(reported_denoising_steps) if reported_denoising_steps is not None else None,
        "action_dim": int(runtime_action_dim or spec.action_dim),
        "adapter_info": adapter_info,
        "denoising_steps_source": "adapter_runtime" if used_runtime_denoising_steps else "spec_fallback",
        # Corrected spec values used for reporting/plotting.  Stale script-level
        # hand annotations are intentionally not emitted anywhere in the JSON.
        "manual_spec": {
            "action_horizon": int(runtime_action_horizon or spec.action_horizon),
            "denoising_steps": int(reported_denoising_steps) if reported_denoising_steps is not None else None,
            "action_dim": int(runtime_action_dim or spec.action_dim),
        },
    }



def make_plot_records(model_name: str, device_name: str, hardware_name: str, model_info: dict[str, Any], benchmark: dict[str, Any], latency_ms: dict[str, Any], calls: dict[str, Any]) -> list[dict[str, Any]]:
    base = {
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
    records = []
    for stage in DUAL_SYSTEM_LATENCY_KEYS:
        stats = latency_ms.get(stage)
        if not stats:
            continue
        rec = dict(base)
        rec.update({
            "metric_family": "latency",
            "metric_name": "latency_ms",
            "unit": "ms",
            "stage": stage,
            "stage_label": DUAL_SYSTEM_STAGE_LABELS.get(stage, stage),
        })
        rec.update(_copy_summary_fields(stats, "value_"))
        if "samples" in stats:
            rec["samples"] = _json_safe(stats["samples"])
        call_stats = calls.get(f"{stage}_calls")
        if call_stats:
            rec.update(_copy_summary_fields(call_stats, "calls_"))
        else:
            rec.update({f"calls_{k}": None for k in ["mean", "p5", "p95", "std", "min", "max", "n"]})
        records.append(rec)
    return records


def write_latency_json(path: str, *, spec: DualSystemModelSpec, args: argparse.Namespace, adapter: DualSystemAdapter, components: dict[str, Any], device_name: str, hardware_name: str) -> None:
    latency_ms = {
        stage: _series_summary(components[stage], include_samples=args.include_samples_in_json)
        for stage in DUAL_SYSTEM_LATENCY_KEYS
        if stage in components
    }
    calls = {
        key: _series_summary(components[key], include_samples=args.include_samples_in_json)
        for key in DUAL_SYSTEM_CALL_KEYS
        if key in components
    }
    model_info = resolve_runtime_model_info(spec=spec, args=args, adapter=adapter)
    benchmark = {
        "default_measured_iterations": int(args.num_iterations),
        "default_warmup": int(args.warmup),
        "include_p95": True,
        "include_p5": True,
        "measure_data_processing": bool(args.measure_data_processing),
        "measure_e2e": bool(args.measure_e2e),
        "measure_components": bool(args.measure_components),
        "data_processing_iterations": resolve_count(args.data_processing_iterations, args.num_iterations),
        "data_processing_warmup": resolve_count(args.data_processing_warmup, args.warmup),
        "e2e_iterations": resolve_count(args.e2e_iterations, args.num_iterations),
        "e2e_warmup": resolve_count(args.e2e_warmup, args.warmup),
        "component_iterations": resolve_count(args.component_iterations, args.num_iterations),
        "component_warmup": resolve_count(args.component_warmup, args.warmup),
        "example_source": args.example_source,
        "torch_dtype": args.torch_dtype,
        "num_steps": int(model_info["denoising_steps"]) if model_info.get("denoising_steps") is not None else None,
        "schema_family": "dual_system",
    }
    payload = {
        "schema_version": "vla_latency_v2_dual_system",
        "plot_schema_version": "vla_plot_v1",
        "model": spec.model_name,
        "hardware": {"device_name": device_name, "torch_cuda_device_name": hardware_name},
        "model_info": model_info,
        "benchmark": benchmark,
        "latency_ms": latency_ms,
        "calls": calls,
        "frequency_hz": {"e2e_mean": 1000 / latency_ms["e2e"]["mean"] if "e2e" in latency_ms else None},
        "stage_hook_modules": components.get("_stage_hook_modules", {}),
        "warnings": components.get("_stage_hook_warnings", []),
        "notes": make_stage_notes(spec),
        "plot_stage_order": list(DUAL_SYSTEM_LATENCY_KEYS),
        "plot_stage_labels": DUAL_SYSTEM_STAGE_LABELS,
    }
    payload["plot_records"] = make_plot_records(spec.model_name, device_name, hardware_name, model_info, benchmark, latency_ms, calls)
    output = os.path.abspath(os.path.expanduser(path))
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"\nWrote JSON results to: {output}")


def make_stage_notes(spec: DualSystemModelSpec) -> dict[str, str]:
    notes = {
        "data_processing": "Raw observation/language/state to model-ready tensors/prompts on the target device.",
        "system2_vision_encoder": "System-2 image tensor to visual/projected tokens; includes image patch/embedding and visual projection when present.",
        "system2_inference": f"System-2 transformer/VLM/VLA starts and ends when {spec.system2_output_name} is fully available. Text token embedding is not included when the adapter follows the original benchmark's transformer-input boundary.",
        "system_bridge": f"{spec.bridge_name}: System-2 output to System-1 condition tensor/latent/action-condition ready.",
        "system1_vision_encoder": "System-1 observation image/RGB-D/scene input to System-1 visual/sensory/scene tokens.",
        "system1_action_expert": "Starts before noisy action/trajectory initialization; includes state/proprio encoder, timestep/action embedding, iterative denoising, output projection/decoder and final scheduler/Euler update. Dataset-stat unnormalization and robot API postprocessing are excluded from this component.",
        "e2e": "Full public inference path, including data processing and public postprocessing when present.",
    }
    if spec.notes:
        notes.update(spec.notes)
    return notes


def print_latency_summary(label: str, values: np.ndarray):
    print(
        f"  {label}: mean={np.mean(values):.2f} ms, "
        f"p5={np.percentile(values, 5):.2f} ms, "
        f"p95={np.percentile(values, 95):.2f} ms, "
        f"std={np.std(values):.2f} ms, n={len(values)}"
    )


def print_markdown_table(components: dict[str, Any], device_name: str, title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)
    headers = ["Device"] + [DUAL_SYSTEM_STAGE_LABELS[s] for s in DUAL_SYSTEM_LATENCY_KEYS if s in components]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    row = [device_name]
    for stage in DUAL_SYSTEM_LATENCY_KEYS:
        if stage in components:
            v = components[stage]
            row.append(f"{np.percentile(v,5):.1f} / {np.mean(v):.1f} / {np.percentile(v,95):.1f} ms")
    print("| " + " | ".join(row) + " |")


def add_common_latency_args(parser: argparse.ArgumentParser, spec: DualSystemModelSpec) -> argparse.ArgumentParser:
    parser.add_argument("--model-id", default=spec.default_model_id)
    parser.add_argument("--checkpoint-dir", default=spec.default_checkpoint_dir)
    parser.add_argument("--train-config", default=spec.default_train_config)
    parser.add_argument("--model-loader", default=None, help="Optional package.module:function returning a DualSystemAdapter-compatible object.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--torch-dtype", default="float32", choices=["float32", "fp32", "bfloat16", "bf16", "float16", "fp16"])
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--torch-num-interop-threads", type=int, default=1)
    parser.add_argument("--example-source", choices=["synthetic", "libero", "calvin"], default=spec.default_example_source)
    parser.add_argument("--dataset-repo-id", default=DEFAULT_LIBERO_DATASET_REPO_ID)
    parser.add_argument("--dataset-root", default=DEFAULT_LIBERO_DATASET_ROOT)
    parser.add_argument("--calvin-dataset-root", default=spec.default_calvin_dataset_root)
    parser.add_argument("--calvin-split", default=spec.default_calvin_split, choices=["training", "validation"])
    parser.add_argument("--dataset-index", type=int, default=0)
    parser.add_argument("--dataset-samples", type=int, default=1)
    parser.add_argument("--task", default=spec.default_task)
    parser.add_argument("--num-iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--data-processing-iterations", type=int, default=None)
    parser.add_argument("--data-processing-warmup", type=int, default=None)
    parser.add_argument("--e2e-iterations", type=int, default=None)
    parser.add_argument("--e2e-warmup", type=int, default=None)
    parser.add_argument("--component-iterations", type=int, default=None)
    parser.add_argument("--component-warmup", type=int, default=None)
    parser.add_argument("--measure-data-processing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measure-e2e", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--measure-components", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", default=f"{spec.model_name.lower()}_latency.json")
    parser.add_argument("--include-samples-in-json", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=64, help="Dummy adapter image size only; real adapters can ignore this.")
    parser.add_argument("--dummy-system2-dim", type=int, default=64)
    parser.add_argument("--dummy-system1-dim", type=int, default=32)
    return parser


def run_latency_main(spec: DualSystemModelSpec) -> None:
    parser = argparse.ArgumentParser(description=f"{spec.model_name} dual-system latency benchmark")
    add_common_latency_args(parser, spec)
    args = parser.parse_args()
    if args.torch_num_threads is not None:
        torch.set_num_threads(int(args.torch_num_threads))
    if args.torch_num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(args.torch_num_interop_threads))
        except RuntimeError:
            pass
    set_seed(args.seed)
    device_name = get_device_name()
    hardware_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    adapter = load_adapter_from_args(args, spec)
    observations = make_observations(args, spec)

    print(f"Model: {spec.model_name}")
    print(f"Device: {device_name}")
    print(f"Example source: {args.example_source}")
    print(f"Adapter: {getattr(adapter, 'model_info', {}).get('adapter', type(adapter).__name__)}")
    print(f"Iterations: n={args.num_iterations}, warmup={args.warmup}")

    data_processing = None
    e2e = None
    comps = None
    if args.measure_data_processing:
        data_processing = benchmark_data_processing(
            adapter,
            observations,
            num_iterations=resolve_count(args.data_processing_iterations, args.num_iterations),
            warmup=resolve_count(args.data_processing_warmup, args.warmup),
        )
        print_latency_summary("Data Processing", data_processing)
    if args.measure_e2e:
        e2e = benchmark_e2e(
            adapter,
            observations,
            num_iterations=resolve_count(args.e2e_iterations, args.num_iterations),
            warmup=resolve_count(args.e2e_warmup, args.warmup),
        )
        print_latency_summary("E2E", e2e)
        print(f"  Frequency: {1000 / np.mean(e2e):.2f} Hz")
    if args.measure_components:
        comps = benchmark_components(
            adapter,
            observations,
            num_iterations=resolve_count(args.component_iterations, args.num_iterations),
            warmup=resolve_count(args.component_warmup, args.warmup),
        )
        for stage in DUAL_SYSTEM_COMPONENT_KEYS:
            print_latency_summary(DUAL_SYSTEM_STAGE_LABELS[stage], comps[stage])

    components = build_components(data_processing, comps, e2e)
    print_markdown_table(components, device_name, f"{spec.model_name} Dual-System Inference Timing")
    write_latency_json(
        args.output_json,
        spec=spec,
        args=args,
        adapter=adapter,
        components=components,
        device_name=device_name,
        hardware_name=hardware_name,
    )
