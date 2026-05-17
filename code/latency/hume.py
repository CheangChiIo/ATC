#!/usr/bin/env python3
from dual_system_benchmarking import DualSystemModelSpec, run_latency_main

SPEC = DualSystemModelSpec(
    model_name="Hume",
    default_model_id="hume",
    system2_output_name="selected high-value System-2 action chunk",
    bridge_name="selected-action chunk slicing / System-1 condition formatting",
    system1_action_expert_name="System-1 cascaded action denoiser",
    action_horizon=8,
    action_dim=7,
    denoising_steps=10,
    notes={
        "system2_inference": "Hume: includes System-2 VLA/VLM transformer, System-2 flow/action candidate generation, value-query scoring and Best-of-N selection. Text token embedding is outside this stage when prepared before transformer input.",
        "system_bridge": "Hume: selected long-horizon action chunk to System-1 sub-action condition / queue entry.",
        "system1_vision_encoder": "Hume: DINOv2-small/current-image encoder path, including patch/image embedding.",
        "system1_action_expert": "Hume: current visual/state/sub-action condition to refined action chunk; includes noisy-action handling and all cascaded denoising updates, excludes unnormalization/postprocessing.",
    },
)

if __name__ == "__main__":
    run_latency_main(SPEC)
