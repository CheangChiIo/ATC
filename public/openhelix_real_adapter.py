from __future__ import annotations

import os
import random
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


OPENHELIX_REPO = Path(os.environ.get("OPENHELIX_REPO", "/tmp/openhelix_repo_check"))
DEFAULT_OPENHELIX_ROOT = Path("/home/dell/ATC/checkpoints/openhelix/prompt_tuning_aux")
DEFAULT_CLIP_ROOT = Path("/home/dell/ATC/checkpoints/openhelix/clip-vit-large-patch14")


def _ensure_openhelix_imports() -> None:
    os.environ.setdefault("DGLBACKEND", "pytorch")
    value = str(OPENHELIX_REPO)
    if value not in sys.path:
        sys.path.insert(0, value)
    _ensure_peft_importable()


def _ensure_peft_importable() -> None:
    try:
        import peft  # noqa: F401

        return
    except Exception:
        for name in list(sys.modules):
            if name == "peft" or name.startswith("peft."):
                sys.modules.pop(name, None)

    peft_stub = ModuleType("peft")
    peft_stub.__spec__ = ModuleSpec("peft", loader=None)

    class LoraConfig:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    def get_peft_model(model, *_args, **_kwargs):
        return model

    peft_stub.LoraConfig = LoraConfig
    peft_stub.get_peft_model = get_peft_model
    sys.modules["peft"] = peft_stub


def _as_path(value: str | None, default: Path) -> Path:
    return Path(value).expanduser().resolve() if value else default


def _depth_to_pcd(depth: Any, shape: tuple[int, int]) -> np.ndarray:
    depth_arr = np.asarray(depth, dtype=np.float32)
    if depth_arr.ndim != 2:
        depth_arr = np.ones(shape, dtype=np.float32)
    if depth_arr.shape != shape:
        depth_arr = cv2.resize(depth_arr, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, shape[0], dtype=np.float32),
        np.linspace(-1.0, 1.0, shape[1], dtype=np.float32),
        indexing="ij",
    )
    return np.stack([xx * depth_arr, yy * depth_arr, depth_arr], axis=-1)


def _to_float_rgb(value: Any, fallback_shape: tuple[int, int, int] = (200, 200, 3)) -> np.ndarray:
    if value is None:
        return np.zeros(fallback_shape, dtype=np.float32)
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    arr = arr.astype(np.float32)
    if arr.max(initial=0) > 2.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _resize_hwc(value: np.ndarray, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    if value.shape[:2] == shape:
        return value
    return cv2.resize(value, (shape[1], shape[0]), interpolation=interpolation)


def _state_to_proprio(value: Any) -> np.ndarray:
    state = np.asarray(value if value is not None else np.zeros(15), dtype=np.float32).reshape(-1)
    if state.shape[0] < 15:
        state = np.pad(state, (0, 15 - state.shape[0]))
    quat_xyzw = None
    try:
        from scipy.spatial.transform import Rotation

        quat_xyzw = Rotation.from_euler("XYZ", state[3:6]).as_quat()
    except Exception:
        quat_xyzw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    quat_wxyz = np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
    proprio = np.concatenate(
        [
            state[:3],
            quat_wxyz,
            (state[[-1]] + 1.0) / 2.0,
        ],
        axis=-1,
    ).astype(np.float32)
    return proprio[None]


def _strip_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stripped = {}
    for key, value in state_dict.items():
        if key.startswith("base_model.model."):
            key = key.replace("base_model.model.", "", 1)
        if key.startswith("module."):
            key = key[len("module.") :]
        stripped[key] = value
    return stripped


class OpenHelixRealAdapter:
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
        _ensure_openhelix_imports()

        import transformers
        from model.llava.mm_utils import tokenizer_image_token
        from model.llava import conversation as conversation_lib
        from model.llava.constants import DEFAULT_IMAGE_TOKEN, IGNORE_INDEX, IMAGE_TOKEN_INDEX
        from model.llava.model.language_model.llava_llama import LlavaConfig
        from planer import LISAForCausalLM
        from transformers import CLIPImageProcessor
        from diffuser_actor import DiffuserActorACTS
        from utils.common_utils import get_gripper_loc_bounds

        self.spec = spec
        self.device = torch.device(device)
        self.dtype = dtype if dtype in {torch.float16, torch.bfloat16} else torch.bfloat16
        self.root = _as_path(model_id, DEFAULT_OPENHELIX_ROOT)
        ckpt_path = _as_path(checkpoint_dir, self.root / "policy.pth")
        if ckpt_path.is_dir():
            ckpt_path = ckpt_path / "policy.pth"
        self.policy_path = ckpt_path
        self.llm_state_path = self.root / "pytorch_model.bin"
        self.vision_tower = Path(train_config).expanduser().resolve() if train_config else DEFAULT_CLIP_ROOT

        if not self.llm_state_path.is_file():
            raise FileNotFoundError(f"OpenHelix LLM checkpoint is incomplete: {self.llm_state_path}")
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"OpenHelix policy checkpoint is missing: {self.policy_path}")
        if not (self.vision_tower / "config.json").is_file():
            raise FileNotFoundError(f"CLIP vision tower checkpoint is missing: {self.vision_tower}")
        if not (self.root / "config.json").is_file():
            raise FileNotFoundError(f"LLaVA config/tokenizer files are missing from: {self.root}")

        self._tokenizer_image_token = tokenizer_image_token
        self._conversation_lib = conversation_lib
        self._default_image_token = DEFAULT_IMAGE_TOKEN
        self._ignore_index = IGNORE_INDEX
        self._image_token_index = IMAGE_TOKEN_INDEX

        self.clip_image_processor = CLIPImageProcessor.from_pretrained(str(self.vision_tower))
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            str(self.root),
            cache_dir=None,
            model_max_length=512,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens("<ACT>")
        self.tokenizer.add_tokens(["<im_start>", "<im_end>"], special_tokens=True)

        config = LlavaConfig.from_pretrained(str(self.root))
        self.llm = LISAForCausalLM(
            config,
            out_dim=512,
            vision_tower=str(self.vision_tower),
            use_mm_start_end=True,
        )
        self.llm.resize_token_embeddings(len(self.tokenizer))
        self.llm.config.eos_token_id = self.tokenizer.eos_token_id
        self.llm.config.bos_token_id = self.tokenizer.bos_token_id
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        self.llm.get_model().initialize_vision_modules(self.llm.get_model().config)
        self.llm.to(device=self.device, dtype=self.dtype)
        self.llm.model.text_hidden_fcs.float()

        llm_state = torch.load(self.llm_state_path, map_location="cpu")
        self.llm.load_state_dict(_strip_state_dict(llm_state), strict=False)
        del llm_state
        self.llm.eval()

        gripper_bounds = get_gripper_loc_bounds(
            str(OPENHELIX_REPO / "tasks/calvin_rel_traj_location_bounds_task_ABC_D.json"),
            buffer=0.01,
        )
        self.policy_args = SimpleNamespace(
            action_dim=7,
            interpolation_length=20,
            num_history=1,
        )
        self.policy = DiffuserActorACTS(
            backbone="clip",
            image_size=(256, 256),
            embedding_dim=192,
            num_vis_ins_attn_layers=2,
            use_instruction=True,
            fps_subsampling_factor=3,
            gripper_loc_bounds=gripper_bounds,
            rotation_parametrization="6D",
            quaternion_format="wxyz",
            diffusion_timesteps=25,
            nhist=1,
            relative=True,
            lang_enhanced=True,
        )
        policy_state = torch.load(self.policy_path, map_location="cpu")["weight"]
        self.policy.load_state_dict(_strip_state_dict(policy_state), strict=True)
        del policy_state
        self.policy.to(self.device).eval()

        self.model_info = {
            "adapter": type(self).__name__,
            "dummy": False,
            "model_id": str(self.root),
            "checkpoint_dir": str(self.policy_path),
            "vision_tower": str(self.vision_tower),
            "stage_split_note": "System-1 visual encoding is split from the official DiffuserActor compute_trajectory path. CALVIN debug depth is converted to an approximate point cloud for latency only.",
        }

    def _conversation(self, task: str) -> str:
        from datasets.utils_llcb import ANSWER_LIST, LONG_QUESTION_LIST

        random.seed(0)
        conv = self._conversation_lib.conv_llava_v1.copy()
        question = LONG_QUESTION_LIST[0].format(sent=task)
        answer = ANSWER_LIST[0].format(sent=task)
        conv.messages = []
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], answer)
        return conv.get_prompt()

    def _llm_inputs(self, image: np.ndarray, task: str) -> tuple[torch.Tensor, torch.Tensor]:
        pil_image = Image.fromarray((np.clip(image, 0.0, 1.0) * 255).astype(np.uint8))
        image_clip = self.clip_image_processor.preprocess([pil_image], return_tensors="pt")["pixel_values"]
        image_clip = image_clip.to(self.device, dtype=self.dtype)

        self._conversation_lib.default_conversation = self._conversation_lib.conv_templates["llava_v1"]
        conv = self._conversation_lib.default_conversation.copy()
        sep = conv.sep + conv.roles[1] + ":"
        short_input_ids = []
        for round_text in self._conversation(task).split(conv.sep2):
            if round_text == "":
                break
            parts = round_text.split(sep)
            parts[0] += sep + " " + "<ACT>"
            if self._default_image_token in parts[0]:
                ids = self._tokenizer_image_token(parts[0], self.tokenizer, return_tensors="pt")
            else:
                ids = torch.tensor(self.tokenizer(parts[0]).input_ids, dtype=torch.long)
            short_input_ids.append(ids)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            short_input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        ).to(self.device)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id)
        truncate_len = self.tokenizer.model_max_length - 255
        if input_ids.shape[1] > truncate_len:
            input_ids = input_ids[:, :truncate_len]
            attention_masks = attention_masks[:, :truncate_len]
        return image_clip, input_ids, attention_masks

    def prepare_inputs(self, observation: dict[str, Any]) -> dict[str, Any]:
        static = _to_float_rgb(observation.get("image"))
        gripper = _to_float_rgb(observation.get("wrist_image"), static.shape)
        shape = static.shape[:2]
        gripper = _resize_hwc(gripper, shape, cv2.INTER_LINEAR)
        static_pcd = _depth_to_pcd(observation.get("depth_static"), shape)
        gripper_pcd = _depth_to_pcd(observation.get("depth_gripper"), gripper.shape[:2])
        gripper_pcd = _resize_hwc(gripper_pcd, shape, cv2.INTER_NEAREST)

        image_clip, input_ids, attention_masks = self._llm_inputs(
            static,
            str(observation.get("task") or self.spec.default_task),
        )
        rgbs = np.stack([static, gripper], axis=0).transpose(0, 3, 1, 2)
        pcds = np.stack([static_pcd, gripper_pcd], axis=0).transpose(0, 3, 1, 2)
        proprio = _state_to_proprio(observation.get("state"))
        return {
            "image_clip": image_clip,
            "input_ids": input_ids,
            "attention_masks": attention_masks,
            "rgbs": torch.as_tensor(rgbs, device=self.device).float().unsqueeze(0),
            "pcds": torch.as_tensor(pcds, device=self.device).float().unsqueeze(0),
            "curr_gripper": torch.as_tensor(proprio, device=self.device).float().unsqueeze(0),
            "trajectory_mask": torch.full([1, self.policy_args.interpolation_length - 1], False, device=self.device),
        }

    @torch.inference_mode()
    def run_system2_vision_encoder(self, inputs: dict[str, Any]) -> torch.Tensor:
        return self.llm.encode_images(inputs["image_clip"])

    def _prepare_llm_embeds(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cur_input_ids = input_ids[0]
        image_token_indices = torch.where(cur_input_ids == self._image_token_index)[0]
        pieces = []
        cur_image_idx = 0
        while image_token_indices.numel() > 0:
            image_token_start = image_token_indices[0]
            pieces.append(self.llm.get_model().embed_tokens(cur_input_ids[:image_token_start]))
            pieces.append(image_features[cur_image_idx])
            pieces.append(self.llm.get_model().embed_tokens(cur_input_ids[image_token_start + 1 : image_token_start + 2]))
            cur_image_idx += 1
            cur_input_ids = cur_input_ids[image_token_start + 2 :]
            image_token_indices = torch.where(cur_input_ids == self._image_token_index)[0]
        if cur_input_ids.numel() > 0:
            pieces.append(self.llm.get_model().embed_tokens(cur_input_ids))
        embeds = torch.cat([piece.to(self.device) for piece in pieces], dim=0).unsqueeze(0)
        pad_left = embeds.shape[1] - input_ids.shape[1]
        if pad_left > 0:
            image_mask = torch.full(
                (attention_mask.shape[0], pad_left),
                True,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((image_mask, attention_mask), dim=1)
        return embeds, attention_mask

    @torch.inference_mode()
    def run_system2_inference(self, inputs: dict[str, Any], system2_visual: torch.Tensor) -> torch.Tensor:
        embeds, attention_mask = self._prepare_llm_embeds(
            inputs["input_ids"],
            inputs["attention_masks"],
            system2_visual,
        )
        output = self.llm.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        projected = self.llm.model.text_hidden_fcs[0](hidden.float())
        seg_token_mask = inputs["input_ids"][:, 1:] == self.llm.seg_token_idx
        prefix = torch.zeros(
            (seg_token_mask.shape[0], system2_visual.shape[1]),
            dtype=torch.bool,
            device=seg_token_mask.device,
        )
        seg_token_mask = torch.cat([prefix, seg_token_mask], dim=1)
        return projected[seg_token_mask]

    @torch.inference_mode()
    def run_system_bridge(self, inputs: dict[str, Any], system2_output: torch.Tensor) -> torch.Tensor:
        lang_embedding = system2_output.unsqueeze(0)
        inputs["lang_embedding"] = lang_embedding
        return lang_embedding

    @torch.inference_mode()
    def run_system1_vision_encoder(self, inputs: dict[str, Any]) -> dict[str, Any]:
        rgb_obs = inputs["rgbs"]
        pcd_obs = inputs["pcds"].clone()
        curr_gripper = inputs["curr_gripper"].clone()
        if self.policy._relative:
            pcd_obs, curr_gripper = self.policy.convert2rel(pcd_obs, curr_gripper)
        curr_gripper = curr_gripper[..., :7]
        pcd_obs = torch.permute(
            self.policy.normalize_pos(torch.permute(pcd_obs, [0, 1, 3, 4, 2])),
            [0, 1, 4, 2, 3],
        )
        curr_gripper[..., :3] = self.policy.normalize_pos(curr_gripper[..., :3])
        curr_gripper = self.policy.convert_rot(curr_gripper)
        fixed_inputs = self.policy.encode_inputs(
            rgb_obs,
            pcd_obs,
            inputs["lang_embedding"],
            curr_gripper,
        )
        return {"fixed_inputs": fixed_inputs, "curr_gripper": curr_gripper}

    @torch.inference_mode()
    def run_system1_action_expert(
        self,
        inputs: dict[str, Any],
        bridge_output: torch.Tensor,
        system1_visual: dict[str, Any],
    ) -> torch.Tensor:
        curr_gripper = system1_visual["curr_gripper"]
        batch_size, _, dim = curr_gripper.shape
        cond_data = torch.zeros(
            (batch_size, inputs["trajectory_mask"].size(1), dim),
            device=self.device,
        )
        cond_mask = torch.zeros_like(cond_data).bool()
        trajectory = self.policy.conditional_sample(
            cond_data,
            cond_mask,
            system1_visual["fixed_inputs"],
        )
        if self.policy._rotation_parametrization != "6D":
            from diffuser_actor.utils.utils import normalise_quat

            trajectory[:, :, 3:7] = normalise_quat(trajectory[:, :, 3:7])
        trajectory = self.policy.unconvert_rot(trajectory)
        trajectory[:, :, :3] = self.policy.unnormalize_pos(trajectory[:, :, :3])
        if trajectory.shape[-1] > 7:
            trajectory[..., 7] = trajectory[..., 7].sigmoid()
        return trajectory

    @torch.inference_mode()
    def infer(self, observation: dict[str, Any]) -> torch.Tensor:
        inputs = self.prepare_inputs(observation)
        visual = self.run_system2_vision_encoder(inputs)
        system2 = self.run_system2_inference(inputs, visual)
        bridge = self.run_system_bridge(inputs, system2)
        system1_visual = self.run_system1_vision_encoder(inputs)
        return self.run_system1_action_expert(inputs, bridge, system1_visual)


def load_adapter(model_id=None, checkpoint_dir=None, train_config=None, device="cuda", dtype=torch.float32, spec=None, args=None):
    return OpenHelixRealAdapter(model_id, checkpoint_dir, train_config, device, dtype, spec, args)
