#!/usr/bin/env python3
from dual_system_benchmarking import DualSystemModelSpec, run_latency_main

SPEC = DualSystemModelSpec(
    model_name="OpenHelix",
    default_model_id="/home/dell/ATC/checkpoints/openhelix/prompt_tuning_aux",
    default_example_source="calvin",
    default_calvin_dataset_root="/home/dell/ATC/datasets/calvin/calvin_debug_dataset",
    default_calvin_split="validation",
    default_task="perform a manipulation task",
    system2_output_name="final-layer <ACT> hidden embedding",
    bridge_name="<ACT> latent projector / 4096-to-512 adapter",
    system1_action_expert_name="3D Diffuser Actor low-level diffusion policy",
    action_horizon=8,
    action_dim=7,
    denoising_steps=25,
    notes={
        "system2_vision_encoder": "OpenHelix: LLaVA vision tower plus mm projector/connector when present, including patch/image embedding.",
        "system2_inference": "OpenHelix: LLaVA LLM transformer from prepared multimodal embeddings to final-layer <ACT> hidden embedding. Text token embedding is outside this stage when prepared before transformer input.",
        "system_bridge": "OpenHelix: <ACT> hidden embedding to low-level latent goal, including linear projector such as 4096->512 and condition formatting.",
        "system1_vision_encoder": "OpenHelix: low-level 3DDA RGB-D/point-cloud/scene encoder that produces scene tokens.",
        "system1_action_expert": "OpenHelix: noisy trajectory initialization, proprio encoder, diffusion timestep embedding, 3DDA transformer/trajectory decoder and final scheduler update; excludes unnormalization/postprocessing.",
    },
)

if __name__ == "__main__":
    run_latency_main(SPEC)
