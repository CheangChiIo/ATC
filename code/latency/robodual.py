#!/usr/bin/env python3
from dual_system_benchmarking import DualSystemModelSpec, run_latency_main

SPEC = DualSystemModelSpec(
    model_name="RoboDual",
    default_model_id="/home/dell/ATC/checkpoints/robodual-openvla-generalist",
    default_example_source="calvin",
    default_calvin_dataset_root="/home/dell/ATC/datasets/calvin/calvin_debug_dataset",
    default_calvin_split="validation",
    default_task="perform a manipulation task",
    system2_output_name="generalist action tokens plus VLA latent representations",
    bridge_name="generalist latent/action projection and specialist condition formatting",
    system1_action_expert_name="DiT specialist diffusion action expert",
    action_horizon=8,
    action_dim=7,
    denoising_steps=10,
    notes={
        "system2_vision_encoder": "RoboDual: OpenVLA/Prismatic-style System-2 visual backbone plus visual projector, including patch/image embedding.",
        "system2_inference": "RoboDual: OpenVLA generalist transformer/prefill plus autoregressive action-token generation and VLA latent extraction. Reuses the measured System-2 visual tokens; text token embedding is outside the vision stage.",
        "system_bridge": "RoboDual: projected generalist latents, discretized action condition formatting and specialist hidden-space alignment.",
        "system1_vision_encoder": "RoboDual: specialist sensory encoders for RGB/depth/tactile/multiview observations plus Perceiver/resampler when present.",
        "system1_action_expert": "RoboDual: specialist noisy action initialization, proprio MLP, timestep/action embedding, DiT denoising blocks, noise head and final scheduler update; reuses measured System-1 sensory tokens and excludes unnormalization/postprocessing.",
    },
)

if __name__ == "__main__":
    run_latency_main(SPEC)
