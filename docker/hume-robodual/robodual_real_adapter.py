from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


ROBODUAL_REPO = Path(os.environ.get("ROBODUAL_REPO", "/repos/RoboDual_AGX"))
DEFAULT_GENERALIST = Path("/checkpoints/robodual-openvla-generalist")
DEFAULT_SPECIALIST = ROBODUAL_REPO / "RoboDual-OpenVLA-Speciallist" / "Specialist+Depth+Gripper.pt"
OPENVLA_EMPTY_TOKEN_ID = 29871


def _make_shim_module(name: str, *, is_package: bool = False) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    mod.__package__ = name if is_package else name.rpartition(".")[0]
    if is_package:
        mod.__path__ = []
    return mod


def _install_fsdp_import_shim() -> None:
    # Jetson PyTorch builds can omit torch.distributed.fsdp; RoboDual inference
    # only needs prismatic modules to import, not FSDP training behavior.
    try:
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy as _unused  # noqa: F401
        return
    except Exception:
        pass

    def _never_wrap(*args: Any, **kwargs: Any) -> bool:
        return False

    def _or_policy(*args: Any, policies: Any = None, **kwargs: Any) -> bool:
        if not policies:
            return False
        return any(bool(policy(*args, **kwargs)) for policy in policies)

    class _NoopFSDP(torch.nn.Module):
        def __init__(self, module: Any = None, *args: Any, **kwargs: Any) -> None:
            super().__init__()
            self.module = module

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            if self.module is None:
                raise RuntimeError("FSDP shim cannot run without a wrapped module")
            return self.module(*args, **kwargs)

        @staticmethod
        @contextmanager
        def state_dict_type(*args: Any, **kwargs: Any):
            yield

    class _MixedPrecision:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.args = args
            self.kwargs = kwargs

    class _StateDictType:
        FULL_STATE_DICT = "FULL_STATE_DICT"

    class _ShardingStrategy:
        HYBRID_SHARD = "HYBRID_SHARD"
        _HYBRID_SHARD_ZERO2 = "_HYBRID_SHARD_ZERO2"

    fsdp_mod = _make_shim_module("torch.distributed.fsdp", is_package=True)
    wrap_mod = _make_shim_module("torch.distributed.fsdp.wrap")
    wrap_mod.transformer_auto_wrap_policy = _never_wrap
    wrap_mod._module_wrap_policy = _never_wrap
    wrap_mod._or_policy = _or_policy
    fsdp_mod.wrap = wrap_mod
    fsdp_mod.FullyShardedDataParallel = _NoopFSDP
    fsdp_mod.MixedPrecision = _MixedPrecision
    fsdp_mod.ShardingStrategy = _ShardingStrategy
    fsdp_mod.StateDictType = _StateDictType
    sys.modules["torch.distributed.fsdp"] = fsdp_mod
    sys.modules["torch.distributed.fsdp.wrap"] = wrap_mod
    dist_mod = sys.modules.get("torch.distributed")
    if dist_mod is not None:
        setattr(dist_mod, "fsdp", fsdp_mod)


def _install_training_import_shims() -> None:
    if "torchtune" not in sys.modules:
        torchtune_mod = _make_shim_module("torchtune", is_package=True)
        modules_mod = _make_shim_module("torchtune.modules")

        def _scheduler_unavailable(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("torchtune scheduler is unavailable in the Jetson inference image")

        modules_mod.get_cosine_schedule_with_warmup = _scheduler_unavailable
        torchtune_mod.modules = modules_mod
        sys.modules["torchtune"] = torchtune_mod
        sys.modules["torchtune.modules"] = modules_mod

    if "wandb" not in sys.modules:
        wandb_mod = _make_shim_module("wandb")

        def _wandb_noop(*args: Any, **kwargs: Any) -> Any:
            return None

        wandb_mod.init = _wandb_noop
        wandb_mod.log = _wandb_noop
        wandb_mod.finish = _wandb_noop
        sys.modules["wandb"] = wandb_mod

    package_names = [
        "prismatic.vla.datasets",
        "prismatic.vla.datasets.rlds",
        "prismatic.vla.datasets.rlds.utils",
    ]
    for name in package_names:
        if name not in sys.modules:
            mod = _make_shim_module(name, is_package=True)
            sys.modules[name] = mod
    data_utils_name = "prismatic.vla.datasets.rlds.utils.data_utils"
    if data_utils_name not in sys.modules:
        data_utils_mod = _make_shim_module(data_utils_name)

        def _save_dataset_statistics_unavailable(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("RLDS dataset utilities are unavailable in the Jetson inference image")

        data_utils_mod.save_dataset_statistics = _save_dataset_statistics_unavailable
        sys.modules[data_utils_name] = data_utils_mod


def _install_prismatic_inference_import_shims() -> None:
    vla_mod = sys.modules.get("prismatic.vla")
    if vla_mod is None or not hasattr(vla_mod, "__path__"):
        vla_mod = _make_shim_module("prismatic.vla", is_package=True)
        vla_mod.__path__ = [str(ROBODUAL_REPO / "prismatic" / "vla")]
        sys.modules["prismatic.vla"] = vla_mod


def _ensure_robodual_imports() -> None:
    _install_fsdp_import_shim()
    _install_prismatic_inference_import_shims()
    _install_training_import_shims()
    for path in (ROBODUAL_REPO, ROBODUAL_REPO / "vla-scripts"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _nested_attr(obj: Any, attr_path: str) -> Any | None:
    cur = obj
    for part in attr_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _get_decoder(model: Any) -> torch.nn.Module:
    decoder = _nested_attr(model, "language_model.model")
    if decoder is not None:
        return decoder
    if hasattr(model, "get_decoder"):
        return model.get_decoder()
    if hasattr(model, "language_model") and hasattr(model.language_model, "get_decoder"):
        return model.language_model.get_decoder()
    raise AttributeError("Could not resolve RoboDual/OpenVLA language decoder module.")


def _get_lm_head(model: Any) -> torch.nn.Module:
    lm_head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    if lm_head is None:
        lm_head = _nested_attr(model, "language_model.lm_head")
    if lm_head is None:
        raise AttributeError("Could not resolve RoboDual/OpenVLA LM head.")
    return lm_head


def _append_empty_token_if_needed(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if torch.all(input_ids[:, -1] == OPENVLA_EMPTY_TOKEN_ID):
        return input_ids, attention_mask
    token = torch.full(
        (input_ids.shape[0], 1),
        fill_value=OPENVLA_EMPTY_TOKEN_ID,
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


class RoboDualRealAdapter:
    def __init__(
        self,
        model_id: str | None,
        checkpoint_dir: str | None,
        train_config: str | None,
        device: str,
        dtype: torch.dtype,
        spec: Any,
        args: Any,
    ):
        _ensure_robodual_imports()

        from diffusers.schedulers.scheduling_ddim import DDIMScheduler
        from prismatic.models.policy.diffusion_policy import DiffusionDiTImagePolicy
        from prismatic.vla.action_tokenizer import ActionTokenizer
        from train_spacialist_calvin import DualSystem
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.spec = spec
        self.device = torch.device(device)
        self.dtype = torch.bfloat16 if dtype == torch.float32 else dtype
        self.generalist_path = Path(model_id or DEFAULT_GENERALIST).expanduser().resolve()
        self.specialist_path = Path(checkpoint_dir or DEFAULT_SPECIALIST).expanduser().resolve()
        if not self.generalist_path.is_dir():
            raise FileNotFoundError(f"RoboDual generalist checkpoint not found: {self.generalist_path}")
        if not self.specialist_path.is_file():
            raise FileNotFoundError(f"RoboDual specialist checkpoint not found: {self.specialist_path}")

        self.processor = AutoProcessor.from_pretrained(str(self.generalist_path), trust_remote_code=True)
        self.generalist = AutoModelForVision2Seq.from_pretrained(
            str(self.generalist_path),
            torch_dtype=self.dtype,
            low_cpu_mem_usage=False,
            trust_remote_code=True,
        ).to(self.device).eval()

        scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon",
        )
        self.specialist = DiffusionDiTImagePolicy(
            shape_meta={"action": {"shape": [spec.action_dim]}},
            noise_scheduler=scheduler,
            n_action_steps=spec.action_horizon,
            num_inference_steps=spec.denoising_steps,
            vision_encoder="DINO",
            with_depth=True,
            progressive_noise=False,
            with_gripper=True,
            with_tactile=False,
        ).to(self.device).eval()
        self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.dual_system = DualSystem(self.generalist, self.specialist, self.action_tokenizer).to(self.device).eval()
        state = torch.load(str(self.specialist_path), map_location=self.device)
        self.dual_system.ema_fast_system.load_state_dict(state, strict=False)
        actual_steps = getattr(self.specialist, "num_inference_steps", None)
        if actual_steps is None:
            actual_steps = getattr(getattr(self.specialist, "config", None), "num_inference_steps", spec.denoising_steps)
        self.model_info = {
            "adapter": type(self).__name__,
            "dummy": False,
            "model_id": str(self.generalist_path),
            "checkpoint_dir": str(self.specialist_path),
            # This value is not only metadata for RoboDual: it is passed into
            # DiffusionDiTImagePolicy(num_inference_steps=...).
            "denoising_steps": int(actual_steps),
            "system1_denoising_steps": int(actual_steps),
            "system2_autoregressive_action_tokens": int((spec.action_dim + 1) * spec.action_horizon),
            "stage_split_note": "RoboDual is measured through explicit staged boundaries: S2 vision is OpenVLA vision_backbone+projector; S2 inference is decoder/action-token generation; S1 vision is specialist RGB-D/gripper embedding; S1 action expert is diffusion sampling.",
        }

        self.depth_min = 3.5
        self.depth_max = 6.2
        self.gripper_depth_min = 0.0
        self.gripper_depth_max = 2.0

    def _prompt(self, task: str) -> str:
        return f"In: What action should the robot take to {task.lower()}?\nOut:"

    def _image(self, value: Any, fallback_shape: tuple[int, int, int]) -> np.ndarray:
        if value is None:
            return np.zeros(fallback_shape, dtype=np.uint8)
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.uint8)

    def _depth(self, value: Any, fallback_shape: tuple[int, int], min_value: float, max_value: float) -> torch.Tensor:
        if value is None:
            arr = np.zeros(fallback_shape, dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32)
        # Match the upstream RoboDual evaluation preprocessing expression.
        return torch.from_numpy(arr).unsqueeze(0).to(self.device) - min_value / (max_value - min_value)

    def prepare_inputs(self, observation: dict[str, Any]) -> dict[str, Any]:
        task = str(observation.get("task") or self.spec.default_task)
        image_value = observation.get("image")
        if image_value is None:
            image_value = observation.get("third_view_image")
        image = self._image(image_value, (200, 200, 3))
        gripper_image_np = self._image(observation.get("wrist_image"), (84, 84, 3))
        prompt_inputs = self.processor(self._prompt(task), Image.fromarray(image)).to(self.device, dtype=self.dtype)
        gripper_image = (
            self.processor.image_processor.apply_transform(Image.fromarray(gripper_image_np))[:3]
            .unsqueeze(0)
            .to(self.device)
        )
        state = np.asarray(observation.get("state", np.zeros((15,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if state.shape[0] < 15:
            state = np.pad(state, (0, 15 - state.shape[0]))
        proprio = torch.from_numpy(np.concatenate([state[:6], state[-1:]])).to(self.device, dtype=torch.float32).unsqueeze(0)
        hist_action = torch.zeros((1, 4, self.spec.action_dim), device=self.device)
        prev_img = (
            self.processor.image_processor.apply_transform(Image.fromarray(image))[:3]
            .unsqueeze(0)
            .to(self.device)
        )
        return {
            "task": task,
            "prompt_inputs": prompt_inputs,
            "ref_image": image,
            "dp_obs": (prompt_inputs["pixel_values"][:, :3].to(torch.float32), prev_img.to(torch.float32)),
            "depth_obs": self._depth(observation.get("depth_static"), (200, 200), self.depth_min, self.depth_max),
            "gripper_obs": (
                gripper_image,
                self._depth(observation.get("depth_gripper"), (84, 84), self.gripper_depth_min, self.gripper_depth_max),
            ),
            "proprio": proprio,
            "hist_action": hist_action,
        }

    @torch.inference_mode()
    def run_system2_vision_encoder(self, inputs: dict[str, Any]) -> torch.Tensor:
        pixel_values = inputs["prompt_inputs"]["pixel_values"]
        patch_features = self.generalist.vision_backbone(pixel_values)
        return self.generalist.projector(patch_features)

    @torch.inference_mode()
    def run_system2_inference(self, inputs: dict[str, Any], system2_visual: torch.Tensor) -> dict[str, torch.Tensor]:
        prompt_inputs = inputs["prompt_inputs"]
        input_ids, attention_mask = _append_empty_token_if_needed(
            prompt_inputs["input_ids"],
            prompt_inputs.get("attention_mask"),
        )
        decoder = _get_decoder(self.generalist)
        lm_head = _get_lm_head(self.generalist)

        projected_patch_attention_mask = None
        if attention_mask is not None:
            projected_patch_attention_mask = torch.full(
                (system2_visual.shape[0], system2_visual.shape[1]),
                fill_value=True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        input_embeddings = self.generalist.get_input_embeddings()(input_ids)
        multimodal_embeddings = torch.cat(
            [input_embeddings[:, :1, :], system2_visual, input_embeddings[:, 1:, :]],
            dim=1,
        )
        multimodal_attention_mask = None
        if attention_mask is not None:
            multimodal_attention_mask = torch.cat(
                [attention_mask[:, :1], projected_patch_attention_mask, attention_mask[:, 1:]],
                dim=1,
            )

        actions_len = (self.spec.action_dim + 1) * self.spec.action_horizon
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
        collected_hidden_states = [hidden_states]
        generated_tokens = []

        logits = lm_head(hidden_states[:, -1:, :])
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)

        for _ in range(1, actions_len):
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
            collected_hidden_states.append(hidden_states)

            logits = lm_head(hidden_states)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)

        generated_ids = torch.cat(generated_tokens, dim=1)
        predicted_action_token_ids = generated_ids[0, -actions_len:].detach().cpu().numpy()
        discretized_actions = self.generalist.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.generalist.bin_centers.shape[0] - 1)
        normalized_actions = self.generalist.bin_centers[discretized_actions]

        hidden_states = torch.cat(collected_hidden_states, dim=1)[:, system2_visual.shape[1] :]
        action = torch.tensor(normalized_actions, device=hidden_states.device, dtype=torch.float32).unsqueeze(0)
        action = action.reshape(1, self.spec.action_horizon, -1)[:, :, : self.spec.action_dim]
        return {"action": action, "hidden_states": hidden_states}

    @torch.inference_mode()
    def run_system_bridge(self, inputs: dict[str, Any], system2_output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return system2_output

    @torch.inference_mode()
    def run_system1_vision_encoder(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        policy = self.dual_system.ema_fast_system.ema_model
        obs = inputs["dp_obs"]
        if policy.encoder_type == "Theia":
            if isinstance(obs, tuple):
                visual_embedding = torch.stack(
                    [policy.vision_encoder.forward_feature(image.permute(0, 2, 3, 1) * 0.5 + 0.5) for image in obs],
                    dim=1,
                )
            else:
                visual_embedding = policy.vision_encoder.forward_feature(obs.permute(0, 2, 3, 1) * 0.5 + 0.5)
        elif policy.encoder_type == "DINO":
            if isinstance(obs, tuple):
                visual_embedding = torch.stack([policy.vision_encoder.forward_features(image) for image in obs], dim=1)
            else:
                visual_embedding = policy.vision_encoder.forward_features(obs)
        else:
            raise ValueError(f"Unsupported specialist encoder type: {policy.encoder_type}")

        depth_embedding = None
        if policy.with_depth:
            depth_obs = policy.depth_resize(inputs["depth_obs"].to(torch.float32).unsqueeze(1))
            depth_embedding = policy.depth_encoder(depth_obs)

        visual_embedding_gripper = None
        depth_embedding_gripper = None
        if policy.with_gripper:
            gripper_rgb = inputs["gripper_obs"][0].to(torch.float32)
            gripper_depth = inputs["gripper_obs"][1].to(torch.float32)
            visual_embedding_gripper = policy.vision_encoder.forward_features(gripper_rgb)
            gripper_depth_obs = policy.depth_resize(gripper_depth.unsqueeze(1))
            depth_embedding_gripper = policy.depth_encoder(gripper_depth_obs)

        return {
            "visual_embedding": visual_embedding,
            "depth_embedding": depth_embedding,
            "visual_embedding_gripper": visual_embedding_gripper,
            "depth_embedding_gripper": depth_embedding_gripper,
            "tactile_embedding": None,
        }

    @torch.inference_mode()
    def run_system1_action_expert(
        self,
        inputs: dict[str, Any],
        bridge_output: dict[str, torch.Tensor],
        system1_visual: dict[str, torch.Tensor | None],
    ) -> torch.Tensor:
        policy = self.dual_system.ema_fast_system.ema_model
        cond_data = torch.zeros(
            size=(inputs["proprio"].shape[0], policy.n_action_steps, policy.action_dim),
            device=policy.device,
            dtype=policy.dtype,
        )
        action_pred = policy.conditional_sample(
            cond_data,
            local_cond=bridge_output["action"].to(torch.float32),
            global_cond=(
                bridge_output["hidden_states"].to(torch.float32),
                system1_visual["visual_embedding"],
                system1_visual["depth_embedding"],
                system1_visual["visual_embedding_gripper"],
                system1_visual["depth_embedding_gripper"],
                system1_visual["tactile_embedding"],
            ),
            hist_action=inputs["hist_action"],
            lang=inputs["task"],
            proprio=inputs["proprio"],
            **policy.kwargs,
        )
        start = policy.n_obs_steps - 1
        end = start + policy.n_action_steps
        return action_pred[:, start:end]

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        inputs = self.prepare_inputs(observation)
        system2_visual = self.run_system2_vision_encoder(inputs)
        system2_output = self.run_system2_inference(inputs, system2_visual)
        bridge_output = self.run_system_bridge(inputs, system2_output)
        system1_visual = self.run_system1_vision_encoder(inputs)
        return self.run_system1_action_expert(inputs, bridge_output, system1_visual)


def load_adapter(model_id=None, checkpoint_dir=None, train_config=None, device="cuda", dtype=torch.float32, spec=None, args=None):
    return RoboDualRealAdapter(model_id, checkpoint_dir, train_config, device, dtype, spec, args)
