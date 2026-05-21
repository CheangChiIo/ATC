from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


HUME_REPO_SRC = Path(os.environ.get("HUME_REPO_SRC", "/repos/hume_repo_check/src"))
DEFAULT_HUME_CHECKPOINT = Path("/checkpoints/hume-libero-spatial-1")


def _ensure_hume_imports() -> None:
    value = str(HUME_REPO_SRC)
    if value not in sys.path:
        sys.path.insert(0, value)
    loaded = sys.modules.get("hume")
    if loaded is not None and not hasattr(loaded, "__path__"):
        sys.modules.pop("hume", None)


class HumeRealAdapter:
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
        _ensure_hume_imports()
        from hume import HumeConfig, HumePolicy
        from hume.models.modeling_hume import make_att_2d_masks
        from transformers import AutoTokenizer
        import safetensors.torch

        self.spec = spec
        self.device = torch.device(device)
        self.dtype = dtype
        self.checkpoint_path = Path(model_id or checkpoint_dir or DEFAULT_HUME_CHECKPOINT).expanduser().resolve()
        if not (self.checkpoint_path / "model.safetensors").is_file():
            raise FileNotFoundError(f"Hume checkpoint is incomplete: {self.checkpoint_path}")

        config = HumeConfig.from_pretrained(str(self.checkpoint_path))
        config.device = str(self.device)
        self.policy = HumePolicy(config)
        safetensors.torch.load_model(
            self.policy,
            str(self.checkpoint_path / "model.safetensors"),
            strict=False,
            device=str(self.device),
        )
        self.policy.language_tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
        self.policy.to(self.device).eval()
        self._make_att_2d_masks = make_att_2d_masks
        self.policy.init_infer(
            dict(
                replan_steps=spec.action_horizon,
                s2_replan_steps=16,
                s2_candidates_num=5,
                noise_temp_lower_bound=1.0,
                noise_temp_upper_bound=1.0,
                time_temp_lower_bound=0.9,
                time_temp_upper_bound=1.0,
                post_process_action=True,
                device=str(self.device),
            )
        )
        def _required_int(obj, attr: str, label: str) -> int:
            if not hasattr(obj, attr):
                raise AttributeError(
                    f"Hume adapter could not read {label}.{attr}; refusing to fall back to "
                    "the script-level DualSystemModelSpec.denoising_steps because that value is only a fallback, "
                    "not guaranteed to match the checkpoint's true iterative schedule."
                )
            return int(getattr(obj, attr))

        s1_steps = _required_int(self.policy.s1_model.config, "s1_num_steps", "policy.s1_model.config")
        s2_base_steps = _required_int(self.policy.s2_model.config, "num_steps", "policy.s2_model.config")
        self.model_info = {
            "adapter": type(self).__name__,
            "dummy": False,
            "model_id": str(self.checkpoint_path),
            # Runtime values from the loaded checkpoint/config.  These override
            # the manually-entered DualSystemModelSpec values in JSON/plot metadata.
            "denoising_steps": s1_steps,
            "system1_denoising_steps": s1_steps,
            "system2_base_denoising_steps": s2_base_steps,
            "system2_candidate_count": int(self.policy.infer_cfg.s2_candidates_num),
            "system2_effective_steps_note": "Hume System-2 samples several candidate action chunks; each candidate uses model.config.num_steps as the base flow-matching grid, but the effective loop count also depends on time_temp and theta2.",
            "stage_split_note": "Hume is measured through explicit staged boundaries: image resize/normalization/tokenization are data processing; S2 vision is SigLIP/PaliGemma image embedding including the visual projection path when present; S2 inference reuses those visual tokens for candidate generation and value-query scoring; S1 vision is DINOv2 image embedding including projector/adapter/resampler when present.",
        }

    def _image(self, value: Any, fallback_shape: tuple[int, int, int]) -> np.ndarray:
        if value is None:
            return np.zeros(fallback_shape, dtype=np.uint8)
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
            arr = np.transpose(arr, (1, 2, 0))
        return arr.astype(np.uint8)

    def _state(self, value: Any) -> np.ndarray:
        state_feature = getattr(self.policy.config, "state_feature", None)
        if state_feature is None:
            state_feature = self.policy.config.input_features["observation.state"]
        state_dim = int(state_feature.shape[0])
        arr = np.asarray(value if value is not None else np.zeros((state_dim,), dtype=np.float32), dtype=np.float32).reshape(-1)
        if arr.shape[0] < state_dim:
            arr = np.pad(arr, (0, state_dim - arr.shape[0]))
        return arr[:state_dim]

    def prepare_inputs(self, observation: dict[str, Any]) -> dict[str, Any]:
        image_value = observation.get("image")
        if image_value is None:
            image_value = observation.get("third_view_image")
        image = self._image(image_value, (224, 224, 3))
        wrist = self._image(observation.get("wrist_image"), image.shape)
        state = self._state(observation.get("state"))
        state_hist = np.expand_dims(state, 0).repeat(int(self.policy.config.s1_his_state_size), axis=0)
        batch = {
            "observation.images.image": torch.tensor(image[None] / 255).permute(0, 3, 1, 2).to(self.device).float(),
            "observation.images.wrist_image": torch.tensor(wrist[None] / 255).permute(0, 3, 1, 2).to(self.device).float(),
            "observation.state": torch.tensor(state_hist[None]).to(self.device).float(),
            "task": [str(observation.get("task") or self.spec.default_task)],
        }
        batch = self.policy.normalize_inputs(batch)
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        lang_tokens, lang_masks = self.policy.prepare_language(batch)
        stamp = torch.zeros(1, device=self.device, dtype=torch.float32)
        return {
            "batch": batch,
            "images": images,
            "img_masks": img_masks,
            "state": state,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "stamp": stamp,
        }

    def _embed_s2_visual_prefix(self, images: list[torch.Tensor], img_masks: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        embs = []
        pad_masks = []
        att_mask_values = []
        bsize = None
        for img, img_mask in zip(images, img_masks, strict=False):
            img_emb = self.policy.s2_model.paligemma_with_expert.embed_image(img).to(dtype=torch.bfloat16)
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_mask_values += [0] * num_img_embs
        if bsize is None:
            raise ValueError("Hume requires at least one prepared image for S2 vision encoding.")
        visual_embs = torch.cat(embs, dim=1)
        visual_pad_masks = torch.cat(pad_masks, dim=1)
        visual_att_masks = torch.tensor(att_mask_values, dtype=torch.bool, device=visual_pad_masks.device)
        visual_att_masks = visual_att_masks[None, :].expand(bsize, len(att_mask_values))
        return {
            "visual_embs": visual_embs,
            "visual_pad_masks": visual_pad_masks,
            "visual_att_masks": visual_att_masks,
        }

    def _append_language_to_s2_visual(
        self,
        visual: dict[str, torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        *,
        detach_language: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lang_emb = self.policy.s2_model.paligemma_with_expert.embed_language_tokens(lang_tokens)
        if detach_language:
            lang_emb = lang_emb.detach()
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        lang_att_masks = torch.zeros(
            (lang_emb.shape[0], lang_emb.shape[1]),
            dtype=visual["visual_att_masks"].dtype,
            device=lang_emb.device,
        )
        embs = torch.cat([visual["visual_embs"], lang_emb], dim=1)
        pad_masks = torch.cat([visual["visual_pad_masks"], lang_masks], dim=1)
        att_masks = torch.cat([visual["visual_att_masks"], lang_att_masks], dim=1)
        return embs, pad_masks, att_masks

    def _append_vqh_query(
        self,
        embs: torch.Tensor,
        pad_masks: torch.Tensor,
        att_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsize = embs.shape[0]
        seq_lengths = pad_masks.sum(dim=1).long()
        seq_len = embs.shape[1]
        new_seq_len = seq_len + 1
        new_embs = torch.zeros((bsize, new_seq_len, embs.shape[-1]), dtype=embs.dtype, device=embs.device)
        new_pad_masks = torch.zeros((bsize, new_seq_len), dtype=pad_masks.dtype, device=pad_masks.device)
        new_att_masks = torch.zeros((bsize, new_seq_len), dtype=att_masks.dtype, device=att_masks.device)

        batch_idx = torch.arange(bsize, device=embs.device).view(-1, 1)
        seq_idx = torch.arange(seq_len, device=embs.device).view(1, -1).expand(bsize, -1)
        mask = seq_idx >= seq_lengths.unsqueeze(1)
        new_seq_idx = seq_idx + mask.long()

        new_embs[batch_idx, new_seq_idx] = embs
        new_pad_masks[batch_idx, new_seq_idx] = pad_masks
        new_att_masks[batch_idx, new_seq_idx] = att_masks
        new_embs[torch.arange(bsize, device=embs.device), seq_lengths] = self.policy.value_query_head.query_embedding.unsqueeze(
            0
        ).expand(bsize, -1)
        new_pad_masks[torch.arange(bsize, device=embs.device), seq_lengths] = True
        new_att_masks[torch.arange(bsize, device=embs.device), seq_lengths] = False
        return new_embs, new_pad_masks, new_att_masks

    def _sample_s2_actions_from_prefix(
        self,
        state: torch.Tensor,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
        *,
        time_temp: float,
        noise_temp: float,
        noise: torch.Tensor | None = None,
        past_key_values: Any | None = None,
    ) -> torch.Tensor:
        model = self.policy.s2_model
        bsize = state.shape[0]
        device = state.device
        if noise is None:
            noise = model.sample_noise(
                (bsize, model.config.n_action_steps, model.config.max_action_dim),
                device,
            )

        prefix_att_2d_masks = self._make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        if past_key_values is None:
            _, past_key_values = model.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=model.config.use_cache,
                fill_kv_cache=True,
            )

        dt = torch.tensor(-1.0 / model.config.num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(time_temp, dtype=torch.float32, device=device)
        while time >= -dt / 2 + (1 - model.config.theta2):
            expanded_time = time.expand(bsize)
            v_t = model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )
            x_t += dt * v_t * noise_temp
            time += dt
        return x_t

    def _select_q_actions_from_visual(
        self,
        system2_visual: dict[str, torch.Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        noise_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embs, pad_masks, att_masks = self._append_language_to_s2_visual(
            system2_visual,
            lang_tokens,
            lang_masks,
            detach_language=True,
        )
        embs, pad_masks, att_masks = self._append_vqh_query(embs, pad_masks, att_masks)
        att_2d_masks = self._make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        suffix_out = self.policy.value_query_head.vqh_backbone.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            inputs_embeds=embs,
        )
        batch_indices = torch.arange(suffix_out.shape[0], device=suffix_out.device)
        query_embedding_idx = pad_masks.sum(-1).long() - 1
        query_embedding = suffix_out[batch_indices, query_embedding_idx]

        batch_size, s2_candidates_num = noise_actions.shape[:2]
        q_values = self.policy.value_query_head.calql.get_q_values(
            query_embedding,
            noise_actions.reshape(batch_size, s2_candidates_num, -1),
        )
        return torch.argmax(q_values, dim=1), q_values

    def _sample_s1_actions_from_prefix(
        self,
        state: torch.Tensor,
        noise: torch.Tensor,
        stamp: torch.Tensor,
        prefix_embs: torch.Tensor,
        prefix_pad_masks: torch.Tensor,
        prefix_att_masks: torch.Tensor,
    ) -> torch.Tensor:
        model = self.policy.s1_model
        bsize = state.shape[0]
        device = state.device
        dt = torch.tensor(-model.config.theta1 / model.config.s1_num_steps, dtype=torch.float32, device=device)
        x_t = noise
        time = torch.tensor(model.config.theta1, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = model.denoise_step(
                state,
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                x_t,
                expanded_time,
                stamp,
            )
            x_t += dt * v_t
            time += dt
        return x_t

    @torch.inference_mode()
    def run_system2_vision_encoder(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return self._embed_s2_visual_prefix(inputs["images"], inputs["img_masks"])

    @torch.inference_mode()
    def run_system2_inference(self, inputs: dict[str, Any], system2_visual: dict[str, Any]) -> dict[str, torch.Tensor]:
        cfg = self.policy.config
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._append_language_to_s2_visual(
            system2_visual,
            inputs["lang_tokens"],
            inputs["lang_masks"],
        )
        candidates = []
        n = int(self.policy.infer_cfg.s2_candidates_num)
        for i in range(n):
            candidates.append(
                self._sample_s2_actions_from_prefix(
                    inputs["state"][:, -1, :],
                    prefix_embs,
                    prefix_pad_masks,
                    prefix_att_masks,
                    time_temp=(i / n)
                    * (self.policy.infer_cfg.time_temp_upper_bound - self.policy.infer_cfg.time_temp_lower_bound)
                    + self.policy.infer_cfg.time_temp_lower_bound,
                    noise_temp=(i / n)
                    * (self.policy.infer_cfg.noise_temp_upper_bound - self.policy.infer_cfg.noise_temp_lower_bound)
                    + self.policy.infer_cfg.noise_temp_lower_bound,
                )
            )
        noise_actions = torch.stack(candidates, dim=1)
        original_action_dim = int(cfg.action_feature.shape[0])
        noise_actions_wo_pad = noise_actions[:, :, : cfg.vqh_chunk_size, :original_action_dim]
        action_index, _ = self._select_q_actions_from_visual(
            system2_visual,
            inputs["lang_tokens"],
            inputs["lang_masks"],
            noise_actions_wo_pad,
        )
        batch_idx = torch.arange(noise_actions.shape[0], device=noise_actions.device)
        return {"noise_action": noise_actions[batch_idx, action_index]}

    @torch.inference_mode()
    def run_system_bridge(self, inputs: dict[str, Any], system2_output: dict[str, torch.Tensor]) -> dict[str, Any]:
        cfg = self.policy.config
        noise_action = system2_output["noise_action"]
        idcs = (inputs["stamp"] * cfg.s2_chunk_size).long().unsqueeze(1) + torch.arange(
            cfg.s1_chunk_size, device=noise_action.device
        )
        batch_idcs = torch.arange(noise_action.shape[0], device=noise_action.device).unsqueeze(1)
        return {
            "noise_action_slides": noise_action[batch_idcs, idcs],
        }

    @torch.inference_mode()
    def run_system1_vision_encoder(self, inputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.policy.s1_model.embed_prefix(
            inputs["images"],
            inputs["img_masks"],
        )
        return {
            "prefix_embs": prefix_embs,
            "prefix_pad_masks": prefix_pad_masks,
            "prefix_att_masks": prefix_att_masks,
        }

    @torch.inference_mode()
    def run_system1_action_expert(self, inputs: dict[str, Any], bridge_output: dict[str, Any], system1_visual: Any) -> torch.Tensor:
        actions = self._sample_s1_actions_from_prefix(
            inputs["state"],
            bridge_output["noise_action_slides"],
            stamp=inputs["stamp"],
            prefix_embs=system1_visual["prefix_embs"],
            prefix_pad_masks=system1_visual["prefix_pad_masks"],
            prefix_att_masks=system1_visual["prefix_att_masks"],
        )
        original_action_dim = int(self.policy.config.action_feature.shape[0])
        return actions[:, :, :original_action_dim]

    @torch.inference_mode()
    def postprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.policy.unnormalize_outputs({"action": actions})["action"]

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        inputs = self.prepare_inputs(observation)
        visual = self.run_system2_vision_encoder(inputs)
        system2 = self.run_system2_inference(inputs, visual)
        bridge = self.run_system_bridge(inputs, system2)
        system1_visual = self.run_system1_vision_encoder(inputs)
        actions = self.run_system1_action_expert(inputs, bridge, system1_visual)
        return self.postprocess_actions(actions)


def load_adapter(model_id=None, checkpoint_dir=None, train_config=None, device="cuda", dtype=torch.float32, spec=None, args=None):
    return HumeRealAdapter(model_id, checkpoint_dir, train_config, device, dtype, spec, args)
