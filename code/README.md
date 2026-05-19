# VLA 模型基准测试套件

本目录整理了 Vision-Language-Action (VLA) 模型的 **推理延迟 latency** 与
**资源占用 resource** 基准测试代码，覆盖两类模型：

- **单系统/传统 VLA**：OpenVLA、pi0.5、SmolVLA
- **双系统架构**：Hume、RoboDual、OpenHelix，按 System-2 + System-1 阶段拆分

建议所有实验从 `warehouse/code` 目录启动，并显式设置 `PYTHONPATH`：

```bash
cd /home/dell/ATC/warehouse/code
export PYTHONPATH="$PWD:$PWD/latency:$PWD/resource:$PWD/public:${PYTHONPATH:-}"
```

如果把 `code/` 单独复制到另一台机器，也需要保持同样的目录结构，或者相应修改
`PYTHONPATH`。

## 目录结构

```text
code/
├── latency/                         # 推理延迟测试
│   ├── openvla.py                   # OpenVLA latency
│   ├── pi05.py                      # pi0.5 latency
│   ├── smolvla.py                   # SmolVLA latency
│   ├── hume.py                      # Hume latency, dual-system
│   ├── robodual.py                  # RoboDual latency, dual-system
│   ├── openhelix.py                 # OpenHelix latency, dual-system
│   └── dual_system_benchmarking.py  # 双系统 latency 共享框架和 dummy adapter
│
├── resource/                        # GPU/CPU/显存资源占用测试
│   ├── openvla_resource.py
│   ├── pi05_resource.py
│   ├── smolvla_resource.py
│   ├── hume_resource.py
│   ├── robodual_resource.py
│   ├── openhelix_resource.py
│   ├── dual_system_resource.py      # 双系统 resource 共享框架
│   └── resource_benchmarking.py     # Nsight + CPU/PyTorch 三趟协议核心
│
└── public/                          # 公用依赖、数据集工具和真实模型适配器
    ├── dual_system_adapter_template.py
    ├── hume_real_adapter.py
    ├── robodual_real_adapter.py
    ├── openhelix_real_adapter.py
    ├── calvin_dataset_utils.py
    ├── libero_dataset_utils.py
    ├── sitecustomize.py
    └── plot_vla_json_records.py
```

> 注意：当前 `warehouse/code` 下没有 `tools/` 目录。实验报告脚本位于 ATC 仓库根目录：
> `../../tools/generate_mentor_report.py`。

## 环境要求

| 依赖 | 说明 |
| --- | --- |
| Python | 建议 3.10 或 3.11；具体以各模型仓库环境为准 |
| CUDA + PyTorch | PyTorch 需带 CUDA 支持；OpenVLA 默认 FA2 路径要求 CUDA toolkit / `nvcc` 与 PyTorch 匹配 |
| `tyro` | 单系统脚本使用的命令行参数解析 |
| `numpy`, `Pillow` | 数据处理 |
| `psutil` | resource 后两趟 CPU 利用率测量 |
| NVIDIA Nsight Systems (`nsys`) | `resource-sampler-backend=full/nsight` 时需要 |
| `pandas`, `matplotlib` | 仅 `public/plot_vla_json_records.py` 需要 |
| `flash-attn` | OpenVLA 默认 `flash_attention_2` 需要；SDPA 只能作为回退/调试结果 |

各模型还需要自己的仓库或包：

| 模型 | 额外依赖 |
| --- | --- |
| OpenVLA | `prismatic` / OpenVLA remote-code 分支、`transformers` |
| pi0.5 | `openpi`、JAX/OpenPI 相关依赖 |
| SmolVLA | `lerobot` |
| Hume | Hume 仓库源码和 Hume checkpoint |
| RoboDual | RoboDual 仓库、OpenVLA generalist checkpoint、specialist checkpoint |
| OpenHelix | OpenHelix 仓库、policy checkpoint、CLIP vision tower、`peft`、`dgl`、`diffuser_actor` 等 |

## 快速冒烟测试

真实模型和数据集接好前，建议先用双系统 dummy adapter 验证测试框架能否跑通：

```bash
cd /home/dell/ATC/warehouse/code
export PYTHONPATH="$PWD:$PWD/latency:$PWD/resource:$PWD/public:${PYTHONPATH:-}"

# 延迟冒烟测试，不依赖真实模型
python latency/hume.py \
    --num-iterations 2 --warmup 1 \
    --output-json results/hume_dummy_latency.json

# 资源冒烟测试，不启动 Nsight
python resource/hume_resource.py \
    --resource-sampler-backend none \
    --num-iterations 1 --warmup 0 \
    --output-json results/hume_dummy_resource.json
```

## Latency 基准测试

Latency 脚本只用于正式速度结论。所有计时窗口都会在测量前后进行 CUDA 同步，
并在 warmup 后统计 mean / P5 / P95 / std / min / max / n。

### 单系统模型

```bash
# OpenVLA
python latency/openvla.py \
    --model-id /path/to/openvla-checkpoint \
    --num-iterations 100 --warmup 20 \
    --output-json results/openvla_latency.json

# pi0.5: 注意参数名是 --checkpoint-dir，不是 --model-id
python latency/pi05.py \
    --checkpoint-dir /path/to/pi05-checkpoint \
    --num-iterations 100 --warmup 20 \
    --output-json results/pi05_latency.json

# SmolVLA
python latency/smolvla.py \
    --model-id /path/to/smolvla-checkpoint \
    --num-iterations 100 --warmup 20 \
    --output-json results/smolvla_latency.json
```

OpenVLA 默认使用 `--attn-implementation flash_attention_2`。如果为了排错切到
`--attn-implementation sdpa`，这组数据不要和默认 FA2 数据直接横向比较。

### 双系统模型

双系统脚本通过 `--model-loader package.module:function` 加载真实模型适配器。
适配器负责封装真实模型调用，测试框架负责统一计时；不要在适配器内部再写额外计时代码。

```bash
# Hume: --model-id 指向 Hume checkpoint
python latency/hume.py \
    --model-loader public.hume_real_adapter:load_adapter \
    --model-id /path/to/hume-checkpoint \
    --example-source libero \
    --num-iterations 100 --warmup 20 \
    --output-json results/hume_latency.json

# RoboDual:
#   --model-id       指向 OpenVLA generalist checkpoint
#   --checkpoint-dir 指向 specialist checkpoint 文件
python latency/robodual.py \
    --model-loader public.robodual_real_adapter:load_adapter \
    --model-id /path/to/robodual-openvla-generalist \
    --checkpoint-dir /path/to/Specialist+Depth+Gripper.pt \
    --example-source calvin \
    --num-iterations 100 --warmup 20 \
    --output-json results/robodual_latency.json

# OpenHelix:
#   --model-id       指向 OpenHelix root，例如 prompt_tuning_aux
#   --checkpoint-dir 可省略；默认使用 --model-id/policy.pth
#   --train-config   在该适配器中用于 CLIP vision tower 路径
python latency/openhelix.py \
    --model-loader public.openhelix_real_adapter:load_adapter \
    --model-id /path/to/openhelix/prompt_tuning_aux \
    --checkpoint-dir /path/to/openhelix/prompt_tuning_aux/policy.pth \
    --train-config /path/to/clip-vit-large-patch14 \
    --example-source calvin \
    --num-iterations 100 --warmup 20 \
    --output-json results/openhelix_latency.json
```

## 阶段划分

单系统模型使用兼容的阶段集合：

```text
data_processing -> vision_encoder -> llm_backbone -> action_expert -> dit -> e2e
```

具体模型可能不输出其中某些阶段。例如 OpenVLA 主要输出
`vision_encoder / llm_backbone / action_expert`，pi0.5 和 SmolVLA 的阶段由各自脚本定义。

双系统模型统一使用：

```text
data_processing
-> system2_vision_encoder
-> system2_inference
-> system_bridge
-> system1_vision_encoder
-> system1_action_expert
-> e2e
```

双系统阶段边界详见 `latency/dual_system_benchmarking.py` 文件头部注释。

## Resource 基准测试

资源测量默认使用 `resource-sampler-backend=full`，即三趟合并协议：

1. **Nsight GPU pass**：采 CUDA timeline、NVTX、GPU busy ratio、SM utilization、CUDA memory usage。
2. **非 Nsight E2E CPU/PyTorch pass**：用 `psutil` 测 CPU，用 PyTorch allocator 测显存。
3. **非 Nsight components CPU/PyTorch pass**：按模块重复第 2 类测量。

因此，resource JSON 里的 duration 只是资源窗口耗时，不作为正式 latency 结论。
正式速度结论应使用 `*_latency.json`。

```bash
# 完整三趟资源测试
python resource/openvla_resource.py \
    --model-id /path/to/openvla-checkpoint \
    --num-iterations 100 --warmup 20 \
    --output-json results/openvla_resource.json

# 仅 Nsight GPU pass
python resource/openvla_resource.py \
    --model-id /path/to/openvla-checkpoint \
    --resource-sampler-backend nsight \
    --num-iterations 100 --warmup 20 \
    --output-json results/openvla_resource.json

# 仅 CPU/PyTorch pass，不采 Nsight GPU 指标
python resource/openvla_resource.py \
    --model-id /path/to/openvla-checkpoint \
    --resource-sampler-backend cpu_pytorch \
    --num-iterations 100 --warmup 20 \
    --output-json results/openvla_resource.json
```

双系统 resource 命令与 latency 参数一致，只是脚本换到 `resource/*_resource.py`：

```bash
python resource/robodual_resource.py \
    --model-loader public.robodual_real_adapter:load_adapter \
    --model-id /path/to/robodual-openvla-generalist \
    --checkpoint-dir /path/to/Specialist+Depth+Gripper.pt \
    --example-source calvin \
    --num-iterations 100 --warmup 20 \
    --output-json results/robodual_resource.json
```

### Nsight 配置与权限

```bash
# 自定义 Nsight 报告输出目录
python resource/openvla_resource.py --nsight-output-dir /workspace/nsight_reports ...

# 禁用 GPU metrics，只保留 CUDA/NVTX 等 timeline；不适合“全指标”实验
python resource/openvla_resource.py --nsight-gpu-metrics-device none ...

# 附加 nsys 参数。参数值本身以 -- 开头时，建议用等号形式
python resource/openvla_resource.py --nsight-extra-args=--stats=true ...

# 指定 nsys 路径
python resource/openvla_resource.py --nsight-nsys-path /usr/local/cuda/bin/nsys ...
```

资源脚本内部调用的 Nsight 关键参数包括：

- `--trace=cuda,nvtx,osrt`
- `--sample=process-tree`
- `--cpuctxsw=process-tree`
- `--cuda-memory-usage=true`
- `--gpu-metrics-devices=all`，除非显式设置为 `none`

如果系统设置了 `RmProfilingAdminOnly=1` 或 `perf_event_paranoid` 较严格，普通用户可能无法采到
GPU metrics 或 CPU 采样。需要使用有权限的 root/sudo 运行，或者调整驱动/内核权限。使用
`--nsight-gpu-metrics-device none` 只能作为降级方案，不能代表“全指标”结果。

可用环境变量：

| 变量 | 说明 |
| --- | --- |
| `VLA_NSIGHT_OUTPUT_DIR` | Nsight 报告目录，默认 `nsight_reports` |
| `VLA_NSIGHT_GPU_METRICS_DEVICE` | GPU metrics 设备过滤，默认 `all` |
| `VLA_NSIGHT_TRACE` | Nsight trace 域，默认 `cuda,nvtx,osrt` |
| `VLA_NSIGHT_EXTRA_ARGS` | 额外 `nsys profile` 参数 |
| `VLA_NSIGHT_NSYS_PATH` | `nsys` 可执行文件路径 |
| `VLA_RESOURCE_BACKEND` | 默认资源后端：`full`、`nsight`、`cpu_pytorch`、`none` |

## 结果指标怎么理解

### Latency JSON

- `latency_ms.<stage>.mean`：该阶段平均耗时，单位 ms。
- `p5` / `p95`：5 分位和 95 分位，用于描述稳定性与尾延迟。
- `frequency_hz.e2e_mean`：`1000 / E2E mean(ms)`，理论 E2E 吞吐。
- `plot_records`：统一绘图记录，供 `plot_vla_json_records.py` 或报告脚本读取。

### Resource JSON

- `gpu_busy_ratio_percent_mean`：NVTX 窗口内 CUDA kernel 忙碌时间比例。
- `sm_util_percent_mean`：Nsight GPU metrics 中 SM Active/SM utilization 类指标的窗口平均。
- `nsight_cuda_memory_peak_mb`：Nsight CUDA memory usage 事件重建出的 CUDA 显存驻留峰值。
- `torch_memory_allocated_mb_peak`：PyTorch active tensor memory 峰值。
- `torch_memory_reserved_mb_peak`：PyTorch caching allocator 保留显存峰值。
- `cpu_process_percent_mean`：进程 CPU 利用率，按可用逻辑核数归一到整机 0-100%。
- `cpu_process_core_percent_mean`：不按逻辑核数归一的 core-equivalent CPU 利用率；100% 约等于占满 1 个逻辑核。

## 需要重点修改的路径

换机运行时，优先检查以下路径。大部分都可以通过命令行参数覆盖，只有真实适配器里的外部仓库路径需要改源码或设环境变量。

### 单系统模型与数据集默认路径

| 文件 | 默认路径/参数 |
| --- | --- |
| `latency/openvla.py` | `model_id=/home/dell/桌面/STJ/openvla-main/checkpoints/openvla-7b-finetuned-libero-spatial` |
| `latency/pi05.py` | `checkpoint_dir=/home/dell/桌面/STJ/openpi/checkpoints/pi05_libero_pt` |
| `latency/smolvla.py` | `model_id=/home/dell/ATC/checkpoints/smolvla_base` |
| `latency/openvla.py`, `latency/pi05.py`, `latency/smolvla.py` | `DEFAULT_LIBERO_DATASET_ROOT=/home/dell/ATC/datasets/physical-intelligence/libero` |
| `latency/dual_system_benchmarking.py` | LIBERO/CALVIN 默认数据集路径 |

对应的 resource 脚本通常继承 latency 脚本的配置。

### 双系统真实适配器路径

| 文件 | 路径含义 |
| --- | --- |
| `public/hume_real_adapter.py` | `HUME_REPO_SRC=/tmp/hume_repo_check/src`，`DEFAULT_HUME_CHECKPOINT=/home/dell/ATC/checkpoints/hume-libero-spatial-1` |
| `public/robodual_real_adapter.py` | `ROBODUAL_REPO` 可用环境变量覆盖；`DEFAULT_GENERALIST=/home/dell/ATC/checkpoints/robodual-openvla-generalist`；specialist 默认在 RoboDual 仓库下 |
| `public/openhelix_real_adapter.py` | `OPENHELIX_REPO` 可用环境变量覆盖；`DEFAULT_OPENHELIX_ROOT=/home/dell/ATC/checkpoints/openhelix/prompt_tuning_aux`；`DEFAULT_CLIP_ROOT=/home/dell/ATC/checkpoints/openhelix/clip-vit-large-patch14` |

示例：

```bash
export ROBODUAL_REPO=/your/path/to/RoboDual_AGX
export OPENHELIX_REPO=/your/path/to/OpenHelix
```

### HuggingFace 缓存

`public/sitecustomize.py` 只在 `public/.cache/huggingface` 已存在时设置：

```python
HF_HOME=public/.cache/huggingface
HUGGINGFACE_HUB_CACHE=public/.cache/huggingface/hub
TRANSFORMERS_CACHE=public/.cache/huggingface/hub
```

如果不希望使用该缓存，直接在 shell 中提前设置 `HF_HOME`、`HUGGINGFACE_HUB_CACHE`
或删除/不创建这个 `.cache` 目录即可。

## 常用命令行参数

| 参数 | 适用范围 | 默认/说明 |
| --- | --- | --- |
| `--model-id` | OpenVLA、SmolVLA、双系统 general/root checkpoint | HuggingFace ID 或本地路径 |
| `--checkpoint-dir` | pi0.5、RoboDual specialist、OpenHelix policy | pi0.5 是 checkpoint 目录；RoboDual 是 specialist 文件；OpenHelix 可省略 |
| `--train-config` | 双系统适配器 | OpenHelix 适配器中用于 CLIP vision tower 路径 |
| `--model-loader` | 双系统真实模型 | 形如 `public.hume_real_adapter:load_adapter`；不传则使用 dummy adapter |
| `--example-source` | 全部 | `synthetic`、`libero` 或 `calvin`，具体选择由脚本限制 |
| `--dataset-root` | LIBERO | 本地 LIBERO 数据路径 |
| `--calvin-dataset-root` | 双系统 CALVIN | 本地 CALVIN 数据路径 |
| `--num-iterations` | 全部 | 默认 100，有效统计次数 |
| `--warmup` | 全部 | 默认 20，不计入结果 |
| `--data-processing-iterations` / `--e2e-iterations` / `--component-iterations` | 全部 | 分别覆盖各测量组次数 |
| `--output-json` | 全部 | JSON 结果输出路径 |
| `--torch-dtype` | 大多数脚本 | `float32`、`bfloat16`、`float16` 等，具体由脚本限制 |
| `--torch-num-threads` / `--torch-num-interop-threads` | 大多数脚本 | 默认 1，减少 CPU 线程池抖动 |
| `--attn-implementation` | OpenVLA | 默认 `flash_attention_2` |
| `--pytorch-compile-mode` | pi0.5 | PyTorch compile 模式，默认 `None` |
| `--compile-model` / `--compile-mode` | SmolVLA | 是否启用 `torch.compile` |
| `--resource-sampler-backend` | resource 脚本 | `full`、`nsight`、`cpu_pytorch`、`none` |

## 接入新模型

1. 复制 `public/dual_system_adapter_template.py`，实现真实模型适配器。
2. 参照 `latency/hume.py` 创建 latency 入口，并定义 `DualSystemModelSpec`。
3. 参照 `resource/hume_resource.py` 创建 resource 入口。
4. 确保适配器方法只负责模型调用，不在适配器内部再添加计时代码。

双系统适配器至少需要提供：

- `prepare_inputs`
- `run_e2e`
- `run_system2_vision_encoder`
- `run_system2_inference`
- `run_system_bridge`
- `run_system1_vision_encoder`
- `run_system1_action_expert`

## 输出文件

所有脚本都会写统一 schema 的 JSON：

- **Latency JSON**：`latency_ms`、`frequency_hz`、`calls`、`plot_records`
- **Resource JSON**：`resource_metrics`、`gpu_profile_metrics`、`e2e_cpu_pytorch_metrics`、`stage_cpu_pytorch_metrics`、`plot_records`
- **Nsight 报告**：默认在 `nsight_reports/<report_prefix>_<timestamp>_<pid>.nsys-rep`
- **Nsight SQLite**：resource 后处理会导出同名 `.sqlite`，用于解析 GPU metrics/NVTX/kernel/memory 事件

## 实验报告生成

当前报告脚本在 ATC 仓库根目录，不在 `warehouse/code` 内：

```bash
cd /home/dell/ATC
python tools/generate_mentor_report.py
```

脚本顶部路径常量需要指向实际结果目录：

```python
ROOT = Path("/home/dell/ATC")
LATENCY_DIR = ROOT / "results/local_rerun_20260518_124717"
RESOURCE_DIR = ROOT / "results/nsight_full_metrics_20260518_153748"
OUT_DIR = ROOT / "results/mentor_report_20260518"
```

当前版本会生成：

```text
{OUT_DIR}/
├── mentor_experiment_report.md
├── mentor_experiment_report.html
├── latency_report.md
├── resource_report.md
├── tables/
│   ├── latency_summary.csv
│   ├── latency_components.csv
│   ├── resource_summary.csv
│   ├── resource_components.csv
│   └── nsight_integrity.csv
└── figures/
    ├── latency_e2e.svg
    ├── throughput.svg
    ├── resource_gpu_util.svg
    ├── resource_memory.svg
    └── resource_cpu.svg
```

## 快速绘图

如果只需要快速查看一组 JSON 的某个指标，可以使用：

```bash
python public/plot_vla_json_records.py results/*.json \
    --metric-family latency \
    --metric-name latency_ms \
    --stages data_processing system2_vision_encoder system2_inference \
             system_bridge system1_vision_encoder system1_action_expert e2e \
    --csv results/plot_records.csv \
    --png results/latency_plot.png
```

该脚本读取 JSON 中的 `plot_records`，可混合读取单系统和双系统结果；需要
`pandas` 与 `matplotlib`。
