"""Template for connecting a real Hume/RoboDual/OpenHelix repository.

Use:
    python hume.py --model-loader dual_system_adapter_template:load_adapter --model-id /path/to/ckpt

Replace every TODO with the real repository calls. Do not add timing code here;
the benchmark runner wraps each method with identical latency/resource boundaries.
"""

from __future__ import annotations
from typing import Any

import torch


class RealDualSystemAdapterTemplate:
    def __init__(self, model_id: str | None, checkpoint_dir: str | None, train_config: str | None, device: str, dtype: torch.dtype, spec: Any, args: Any):
        self.device = torch.device(device)
        self.dtype = dtype
        self.spec = spec
        self.model_info = {
            "adapter": type(self).__name__,
            "dummy": False,
            "model_id": model_id,
            "checkpoint_dir": checkpoint_dir,
            "train_config": train_config,
        }
        # TODO: load the real repository model/policy/processor here.
        # self.policy = ...

    def prepare_inputs(self, observation: dict[str, Any]) -> dict[str, Any]:
        # TODO: raw dataset frame -> model-ready tensors/prompts.
        # Keep text tokenization here if the original benchmarks prepared text before transformer input.
        raise NotImplementedError

    def run_system2_vision_encoder(self, inputs: dict[str, Any]) -> Any:
        # TODO: image tensor -> System-2 visual/projected tokens.
        # Include patch embedding and visual projector/connector.
        raise NotImplementedError

    def run_system2_inference(self, inputs: dict[str, Any], system2_visual: Any) -> Any:
        # TODO: System-2 transformer/VLM/VLA starts here.
        # Hume: end after selected high-value action chunk.
        # RoboDual: end after generalist action tokens + VLA latents.
        # OpenHelix: end after final-layer <ACT> hidden embedding.
        raise NotImplementedError

    def run_system_bridge(self, inputs: dict[str, Any], system2_output: Any) -> Any:
        # TODO: System-2 output -> System-1 condition ready.
        # Hume: chunk slicing / queue / condition formatting.
        # RoboDual: latent/action projection and alignment.
        # OpenHelix: <ACT> projector, e.g. 4096 -> 512.
        raise NotImplementedError

    def run_system1_vision_encoder(self, inputs: dict[str, Any]) -> Any:
        # TODO: current low-level observation -> System-1 visual/sensory/scene tokens.
        raise NotImplementedError

    def run_system1_action_expert(self, inputs: dict[str, Any], bridge_output: Any, system1_visual: Any) -> Any:
        # TODO: boundary starts before noisy action/trajectory initialization.
        # Include state/proprio encoder, timestep/action embedding, denoising network,
        # output head/decoder and final scheduler/Euler update.
        # Exclude dataset-stat unnormalization and robot API postprocessing.
        raise NotImplementedError

    def infer(self, observation: dict[str, Any]) -> Any:
        # Optional: public model path. If unavailable, delete this method; benchmark will use manual path.
        inputs = self.prepare_inputs(observation)
        s2v = self.run_system2_vision_encoder(inputs)
        s2o = self.run_system2_inference(inputs, s2v)
        cond = self.run_system_bridge(inputs, s2o)
        s1v = self.run_system1_vision_encoder(inputs)
        return self.run_system1_action_expert(inputs, cond, s1v)


def load_adapter(model_id=None, checkpoint_dir=None, train_config=None, device="cuda", dtype=torch.float32, spec=None, args=None):
    return RealDualSystemAdapterTemplate(model_id, checkpoint_dir, train_config, device, dtype, spec, args)
