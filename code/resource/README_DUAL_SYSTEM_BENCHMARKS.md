# Dual-system benchmark scripts: Hume / RoboDual / OpenHelix

新增脚本：

- `hume.py`, `robodual.py`, `openhelix.py`: latency benchmark
- `hume_resource.py`, `robodual_resource.py`, `openhelix_resource.py`: resource benchmark
- `dual_system_benchmarking.py`: shared latency/data/JSON logic
- `dual_system_resource.py`: shared Nsight + psutil CPU-time resource logic

阶段边界固定为：

1. `data_processing`
2. `system2_vision_encoder`
3. `system2_inference`
4. `system_bridge`
5. `system1_vision_encoder`
6. `system1_action_expert`
7. `e2e`

## 边界定义

### data_processing
Raw observation / language / state 到 model-ready tensors/prompts。包含 resize、crop、normalization、tokenization、tensor conversion、device transfer 等准备工作。

### system2_vision_encoder
System-2 图像 tensor 到 System-2 visual/projected tokens。包含 image patch / image embedding / visual backbone / visual projector 或 connector。

### system2_inference
从准备输入 System-2 transformer / VLM / VLA 主干开始，到 System-2 高层输出完全可用为止。

- Hume: selected high-value action chunk 完成；包含 candidate action generation、value-query scoring、Best-of-N。
- RoboDual: generalist action tokens + VLA latent representations 完成。
- OpenHelix: final-layer `<ACT>` hidden embedding 完成。

沿用原有代码口径：如果文本 token embedding 在 transformer 输入准备前已经完成，则不计入这一段。

### system_bridge
System-2 output 到 System-1 condition ready。

- Hume: selected chunk slicing / queue / condition formatting。
- RoboDual: generalist latent projection、discretized action condition formatting、hidden-space alignment。
- OpenHelix: `<ACT>` hidden embedding projector，例如 4096 -> 512。

### system1_vision_encoder
System-1 当前观测到 low-level visual/sensory/scene tokens。

- Hume: DINOv2-small current-image encoder。
- RoboDual: specialist sensory encoders + Perceiver/resampler。
- OpenHelix: RGB-D / point-cloud / 3D scene encoder。

### system1_action_expert
从 System-1 condition 已准备好、但尚未初始化 noisy action / noisy trajectory 开始，到最后一步 scheduler / Euler update 完成、生成 clean internal-space action chunk 为止。

包含：noisy action 初始化、state/proprio encoder、timestep embedding、action embedding、DiT/Transformer/denoiser、output projection/decoder、scheduler update。

不包含：dataset-stat unnormalization、action clipping、robot API command conversion、env.step。

### e2e
完整 public inference path。若公开推理函数包含 postprocess / unnormalization，则 E2E 包含它们。

## Smoke test

不接真实模型时，默认使用 deterministic dummy adapter，只用于检查统计边界和 JSON schema：

```bash
python hume.py --num-iterations 2 --warmup 1 --output-json hume_dummy_latency.json
python robodual.py --num-iterations 2 --warmup 1 --output-json robodual_dummy_latency.json
python openhelix.py --num-iterations 2 --warmup 1 --output-json openhelix_dummy_latency.json
```

资源测试可在没有 Nsight 的环境下用 `none` 做边界/JSON smoke test：

```bash
python hume_resource.py --resource-sampler-backend none --num-iterations 1 --warmup 0 --output-json hume_dummy_resource.json
```

实际 GPU/显存资源测试默认使用 Nsight：

```bash
python hume_resource.py --num-iterations 100 --warmup 20 --output-json hume_resource.json
```

## 接真实模型

传入 `--model-loader package.module:function`。该 function 必须返回一个 adapter object，实现：

```python
prepare_inputs(observation) -> dict
run_system2_vision_encoder(inputs) -> Any
run_system2_inference(inputs, system2_visual) -> Any
run_system_bridge(inputs, system2_output) -> Any
run_system1_vision_encoder(inputs) -> Any
run_system1_action_expert(inputs, bridge_output, system1_visual) -> Any
infer(observation) -> Any  # optional; missing时用上述手动路径作为E2E
```

loader 签名建议为：

```python
def load_adapter(model_id, checkpoint_dir, train_config, device, dtype, spec, args):
    return MyAdapter(...)
```

这样所有真实仓库只改 adapter，不改计时器、资源边界、JSON schema 和绘图脚本。

## 绘图

`plot_vla_json_records.py` 已支持新旧两套 stage，可直接混合读 OpenVLA/pi0.5/SmolVLA 和 Hume/RoboDual/OpenHelix 的 JSON：

```bash
python plot_vla_json_records.py hume_latency.json robodual_latency.json openhelix_latency.json \
  --stages data_processing system2_vision_encoder system2_inference system_bridge system1_vision_encoder system1_action_expert e2e
```


## Defaults

The dual-system scripts use the same benchmark defaults as the existing scripts: `--example-source libero`, `--num-iterations 100`, `--warmup 20`, and per-section overrides default to those values. Use `--example-source synthetic` only for smoke tests without the LIBERO dataset.
