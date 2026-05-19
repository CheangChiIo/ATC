#!/usr/bin/env python3
from dual_system_benchmarking import DualSystemModelSpec, run_latency_main

SPEC = DualSystemModelSpec(
    model_name="Hume",
    default_model_id="/home/dell/ATC/checkpoints/hume-libero-spatial-1",
    system2_output_name="selected high-value System-2 action chunk",
    bridge_name="selected-action chunk slicing / System-1 condition formatting",
    system1_action_expert_name="System-1 cascaded action denoiser",
    action_horizon=8,
    action_dim=7,
    # Runtime-only for the real Hume adapter: the actual step counts are read
    # from checkpoint config (s1_model.config.s1_num_steps and s2_model.config.num_steps).
    # Do not set denoising_steps here, because any hand-entered value would be stale metadata.
    notes={
        "system2_vision_encoder": "Hume: System-2 SigLIP/PaliGemma image tower plus multimodal projector, producing reusable visual tokens for the current observation.",
        "system2_inference": "Hume: language embedding, System-2 PaliGemma/Gemma prefix cache, flow/action candidate generation, value-query scoring and Best-of-N selection. Reuses the measured System-2 visual tokens.",
        "system_bridge": "Hume: selected long-horizon action chunk to System-1 sub-action condition / queue entry.",
        "system1_vision_encoder": "Hume: DINOv2-small/current-image encoder path, including patch/image embedding.",
        "system1_action_expert": "Hume: current visual/state/sub-action condition to refined action chunk; includes noisy-action handling and all cascaded denoising updates, reusing measured System-1 visual tokens and excluding unnormalization/postprocessing.",
    },
)

if __name__ == "__main__":
    run_latency_main(SPEC)
