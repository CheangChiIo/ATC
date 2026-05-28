## 1. 核心结论一：VLA 跨平台性能不是单一算力缩放

### 1.1 现象

从 RTX5090 到 Jetson AGX Orin，E2E latency 的放大倍数看起来相对集中：

| 模型 | RTX5090 E2E | AGX Orin E2E | AGX/RTX |
|---|---:|---:|---:|
| OpenVLA | 124.2 ms | 863.9 ms | 6.96x |
| pi0.5 | 113.6 ms | 899.7 ms | 7.92x |
| SmolVLA | 90.7 ms | 599.9 ms | 6.61x |
| Hume | 575.1 ms | 4070.2 ms | 7.08x |
| OpenHelix | 174.6 ms | 1158.4 ms | 6.63x |
| RoboDual | 905.6 ms | 6472.8 ms | 7.15x |

但分阶段看，倍率差异非常大：

| 阶段案例 | RTX5090 | AGX Orin | AGX/RTX |
|---|---:|---:|---:|
| SmolVLA / Data | 0.5 ms | 1.2 ms | 2.40x |
| OpenVLA / Data | 2.2 ms | 8.1 ms | 3.68x |
| SmolVLA / Vision | 9.4 ms | 51.8 ms | 5.51x |
| OpenVLA / LLM | 38.4 ms | 257.9 ms | 6.72x |
| SmolVLA / Action | 71.1 ms | 487.3 ms | 6.85x |
| pi0.5 / LLM | 28.2 ms | 283.8 ms | 10.06x |

### 1.2 为什么这说明不是单一瓶颈

如果瓶颈只有一个，例如“纯 GPU 算力不足”，那么所有 GPU-heavy 阶段在 AGX/RTX 之间应该表现出相近的放大倍率。可是结果中既有 2.40x 的阶段，也有 10.06x 的阶段，并且这些差异不只出现在 CPU preprocessing 这类非 GPU 阶段，也出现在 LLM、Vision、Action、S2_Inference 等核心模块。

这说明不同阶段受到的限制不同：

- Data processing 多数时候 GPU 活动极低，更接近 CPU/Python/HuggingFace processor 开销。
- LLM 或 S2_Inference 中部分模块 GR/SM 接近饱和，更接近 dense compute-bound。
- Action 或 S1_Action 中很多模块 latency 长但 GR/DRAM 很低，更接近 kernel launch、runtime orchestration、Python loop 或同步开销。
- RoboDual S2_Inference 同时有较高 GR/SM 与 DRAM，属于 GPU compute 和内存流量共同限制。

### 1.3 原因

以 Jetson AGX Orin 为例：

| 模型 / 模块 | Latency | GR Active | SM Util | DRAM | CPU | 瓶颈判断 |
|---|---:|---:|---:|---:|---:|---|
| OpenVLA / Data | 8.1 ms | 1.3% | 0.0% | 1.1% | 32.4% | host/CPU preprocessing |
| pi0.5 / LLM | 283.8 ms | 98.9% | 97.5% | 39.6% | 3.0% | dense compute-bound |
| SmolVLA / Action | 487.3 ms | 8.8% | 34.7% | 3.4% | 8.0% | launch/orchestration-bound |
| OpenHelix / S2_Inference | 239.8 ms | 96.8% | 94.9% | 47.4% | 5.1% | high-utilization compute-bound |
| Hume / S2_Inference | 3608.6 ms | 22.7% | 53.3% | 10.8% | 7.7% | fragmented/sequential execution |
| RoboDual / S2_Inference | 6061.4 ms | 73.9% | 76.5% | 55.2% | 7.8% | compute + DRAM mixed bottleneck |


### 1.4 意义

这条结论的普适意义是：**VLA benchmark 不能只报告 E2E latency，也不能只报告模型参数量或显存占用。** 对 VLA 来说，阶段级 latency 与阶段级资源指标是必要的，否则会把不同瓶颈混成一个平均数。

## 2. 核心结论二：模型排名不是平台不变属性

### 2.1 现象

RTX5090 上 E2E 排名：

1. SmolVLA：90.7 ms
2. pi0.5：113.6 ms
3. OpenVLA：124.2 ms
4. OpenHelix：174.6 ms
5. Hume：575.1 ms
6. RoboDual：905.6 ms

Jetson AGX Orin 上 E2E 排名：

1. SmolVLA：599.9 ms
2. OpenVLA：863.9 ms
3. pi0.5：899.7 ms
4. OpenHelix：1158.4 ms
5. Hume：4070.2 ms
6. RoboDual：6472.8 ms

最重要的变化是 **OpenVLA 与 pi0.5 的排名反转**。RTX5090 上 pi0.5 比 OpenVLA 快；AGX Orin 上 OpenVLA 反而比 pi0.5 快。

### 2.2 反转发生在哪里

反转主要来自 LLM 阶段：

| 阶段 | RTX5090 | AGX Orin | AGX/RTX |
|---|---:|---:|---:|
| OpenVLA / LLM | 38.4 ms | 257.9 ms | 6.72x |
| pi0.5 / LLM | 28.2 ms | 283.8 ms | 10.06x |

在 RTX5090 上，pi0.5 LLM 更快；但在 AGX 上，pi0.5 LLM 被放大得更厉害，最终比 OpenVLA 更慢。


### 2.3 反转的 Nsight 证据

这个反转不是因为 pi0.5 在 AGX 上突然需要更多显存，也不是因为 CPU 侧变成主瓶颈。Nsight SQLite 显示，两者的 LLM 阶段在 RTX5090 和 AGX 上都具有很高的 GPU duty，但二者的 kernel 形态对 AGX 的适配程度不同。

| LLM 阶段 | RTX latency | AGX latency | kernel/range RTX | kernel/range AGX | kernel duty RTX | kernel duty AGX | median kernel RTX | median kernel AGX |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenVLA | 38.4 ms | 257.9 ms | 1483 | 1532 | 92.2% | 97.5% | 2.98 us | 55.6 us |
| pi0.5 | 28.2 ms | 283.8 ms | 884 | 848 | 93.3% | 98.6% | 4.38 us | 90.4 us |

RTX5090 上，pi0.5 的优势主要来自 **kernel 数量更少、GEMM 总时间更短**。按 Nsight 中 LLM NVTX range 内的 kernel 类型粗分：

| 平台 / 模型 | GEMM/TensorCore time per range | elementwise/copy time per range | 解释 |
|---|---:|---:|---|
| RTX5090 / OpenVLA | 32.2 ms | 2.7 ms | GEMM 时间较长，因此 LLM 慢于 pi0.5 |
| RTX5090 / pi0.5 | 23.2 ms | 2.9 ms | kernel 更少，GEMM 总时间更短，因此 RTX 上更快 |
| AGX Orin / OpenVLA | 196.4 ms | 60.0 ms | GEMM 和 elementwise 都被放大，但仍低于 pi0.5 |
| AGX Orin / pi0.5 | 215.7 ms | 68.4 ms | 主 GEMM 与大 elementwise 在 AGX 上放大更严重，导致排名反转 |

也就是说，pi0.5 在 RTX5090 上赢，是因为它的 LLM 阶段用更少 kernel 完成了更短的 GEMM 工作；但到了 AGX Orin，pi0.5 的主导 BF16 GEMM 和大 elementwise/copy kernel 的有效执行时间被放大得更厉害，GEMM per range 反而比 OpenVLA 多约 19 ms，elementwise/copy 也多约 8 ms。

同时，pi0.5 在 AGX LLM 中的主导 kernel 为 `ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn` 这一类 BF16 GEMM。它每个 LLM range 大约调用 90 次，总计约 206.6 ms，单 kernel median 约 2.29 ms，p90 可到约 6.38 ms。OpenVLA 的主导 GEMM 更分散，单个主导 GEMM median 约 388 us，GEMM 总时间约 196.4 ms。这个差异说明 pi0.5 的 LLM 不是输在 kernel 数量，而是输在 AGX 对其 GEMM shape、Tensor Core occupancy、memory hierarchy 和调度形态的适配上。

资源指标也排除了几个常见但错误的解释。pi0.5 的显存只有约 7.4 GB，明显低于 OpenVLA 的约 14.6 GB，但 AGX LLM 反而更慢，因此这不是显存容量瓶颈。pi0.5 LLM 的 CPU 只有 3.0%，OpenVLA 为 5.2%，因此也不是 CPU 主导。pi0.5 的 DRAM bandwidth 为 39.6%，低于 OpenVLA 的 49.1%，因此也不能简单归因于更高 DRAM 带宽压力。最合理的解释是：**RTX5090 奖励 pi0.5 少而大的 LLM kernel 结构；AGX Orin 对这类 kernel shape 的有效吞吐更低，导致原本的桌面 GPU 优势在端侧平台上失效。**

### 2.4 意义

这说明高端 GPU 上的 VLA 排名不能直接外推到边缘 GPU。如果只给 RTX5090 上的排序，很可能对 Jetson 部署形成错误结论。


## 3. 核心结论三：小显存占用不等于边缘高效

### 3.1 现象

SmolVLA 的显存占用约 0.96 GB，是所有模型中非常轻量的一个。但它在 Jetson 上并没有因为显存小而自动高效。AGX Orin 上 SmolVLA 的 E2E 为 599.9 ms，其中 Action 为 487.3 ms，占 E2E 的 81.2%；Orin Nano 上 E2E 为 781.7 ms，其中 Action 为 586.7 ms，占 75.1%。

| SmolVLA / Action | RTX5090 | AGX Orin | Orin Nano |
|---|---:|---:|---:|
| Latency | 71.1 ms | 487.3 ms | 586.7 ms |
| GR Active | 35.7% | 8.8% | 10.9% |
| SM Util | 16.4% | 34.7% | 28.7% |
| DRAM | 2.3% | 3.4% | 4.4% |
| Nsight Memory | 960 MB | 966 MB | 966 MB |

### 3.2 为什么不是显存容量瓶颈

如果 SmolVLA 的主要问题是显存容量或显存不足，那么应该看到显存占用接近设备上限，或者 DRAM 带宽压力明显上升。但实际不是：显存约 966 MB，DRAM 只有 3.4%-4.4%，GR Active 也只有 8.8%-10.9%。这说明 Action 阶段慢，不是因为模型放不下，也不是因为持续读写大张量耗尽带宽。

### 3.3 真正原因：执行粒度过碎

kernel summary 显示，SmolVLA Action 每个 range 包含约 11.4k-11.8k 个 kernel，且 kernel duty 很低。这个现象需要和其他模型横向比较才有诊断意义：**SmolVLA Action 的 kernel 数量不是所有阶段中绝对最高的，但它属于 action-generation 模块中 kernel 数量最高的一档，并且在 Jetson 类设备上的 kernel duty 和 GR Active 明显偏低。** 这说明问题不只是“kernel 多”，而是“kernel 多、每个 kernel 很短、阶段窗口里 GPU 有效执行占比低”。

| 平台 | kernels/range | kernel duty | kernel median |
|---|---:|---:|---:|
| RTX5090 | 11751 | 21.2% | 1.15 us |
| AGX Orin | 11421 | 6.8% | 3.58 us |
| Orin Nano | 11401 | 8.7% | 5.34 us |

AGX Orin 上的 action 或 S1_Action 横向对比如下：

| 模型 / 模块 | Latency | kernels/range | kernel duty | GR Active | 解释 |
|---|---:|---:|---:|---:|---|
| OpenVLA / Action | 553.1 ms | 8991 | 72.1% | 73.3% | kernel 数量不少，但 GPU 有效执行占比较高，更像 GPU/DRAM 混合压力 |
| pi0.5 / Action | 480.4 ms | 11484 | 13.7% | 15.7% | kernel 数量与 SmolVLA 接近，明显 launch/orchestration-bound |
| SmolVLA / Action | 487.3 ms | 11421 | 6.8% | 8.8% | 与 pi0.5 同量级 kernel 数，但 duty 更低，碎片化更明显 |
| Hume / S1_Action | 398.6 ms | 10723 | 24.1% | 26.0% | 也有较多小 kernel，但利用率高于 SmolVLA |
| OpenHelix / S1_Action | 756.3 ms | 15552 | 8.8% | 11.8% | action loop 更重，和 SmolVLA 同属低 duty 类型 |
| RoboDual / S1_Action | 310.8 ms | 5923 | 4.4% | 7.4% | kernel 数较少但 duty 极低，host/orchestration 占比更高 |

这个对比带来三个判断。

第一，SmolVLA Action 的 kernel 数量和 pi0.5 Action、Hume S1_Action 属于同一量级，明显高于 Vision/LLM 这类阶段。例如 AGX 上 SmolVLA Vision 只有 237 kernels/range，而 Action 有 11421 kernels/range，相差约 48 倍。因此 Action 的执行图确实更碎。

第二，kernel 数量本身还不足以判断瓶颈。OpenVLA Action 也有 8991 kernels/range，但 kernel duty 为 72.1%、GR Active 为 73.3%，说明 GPU 大部分时间仍在有效执行；而 SmolVLA Action 的 kernel duty 只有 6.8%、GR Active 只有 8.8%。因此 SmolVLA 的问题不是“有一万个 kernel”本身，而是这些 kernel 太短、太分散，导致调度和同步开销吞掉了窗口时间。

第三，SmolVLA 和 OpenHelix S1_Action 更像同一类问题：二者在 AGX 上都有万级 kernel 数量、低 GR Active 和低 kernel duty。这类阶段即使显存不高、DRAM 不高，也会因为执行粒度过细而成为 E2E 主瓶颈。

这意味着 GPU 并没有持续执行几个大的高效 kernel，而是在大量非常短的 kernel 之间反复调度、同步、等待。边缘 GPU 上 launch 和同步开销被放大，因此显存很小的模型也可能慢。

### 3.4 意义

很多边缘部署工作会把“模型小、显存低”当成可部署性的主要指标。SmolVLA 说明这不充分：**edge-efficient 不仅要求模型小，还要求执行图对边缘 GPU 友好。**


## 4. 核心结论四：同一模型内部也会发生模块级瓶颈切换

### 4.1 SmolVLA 的例子

在 Orin Nano 上，SmolVLA 的 Vision 与 Action 表现完全不同：

| SmolVLA / Orin Nano | Latency | GR Active | SM Util | DRAM | 判断 |
|---|---:|---:|---:|---:|---|
| Vision | 123.4 ms | 92.7% | 91.2% | 35.7% | compute-bound |
| Action | 586.7 ms | 10.9% | 28.7% | 4.4% | launch/orchestration-bound |

同一模型、同一设备上，Vision 阶段几乎把 GPU 吃满，Action 阶段却长时间低利用率。这说明给整个模型贴一个“compute-bound”或“memory-bound”标签是不准确的。

### 4.2 pi0.5 的例子

pi0.5 在 AGX Orin 上也呈现模块级瓶颈切换：

| pi0.5 / AGX | Latency | GR Active | SM Util | DRAM | 判断 |
|---|---:|---:|---:|---:|---|
| LLM | 283.8 ms | 98.9% | 97.5% | 39.6% | compute-bound |
| Action | 480.4 ms | 15.7% | 31.6% | 7.7% | launch/orchestration-bound |

一个模型内部，LLM 需要 dense compute 优化，而 Action 需要减少 kernel 数量、减少 Python loop 和同步点。这两个优化方向完全不同。

### 4.3 OpenHelix 的例子

OpenHelix 的 S2_Inference 是高效 GPU compute 阶段，但 S1_Action 是低利用率阶段：

| OpenHelix / AGX | Latency | GR Active | SM Util | DRAM | kernel duty | 判断 |
|---|---:|---:|---:|---:|---:|---|
| S2_Inference | 239.8 ms | 96.8% | 94.9% | 47.4% | 96.3% | high-utilization compute |
| S1_Action | 756.3 ms | 11.8% | 31.6% | 4.1% | 8.8% | fragmented action loop |

### 4.4 意义

说明 VLA 优化必须是模块级的。对 Vision/LLM/S2 compute 阶段，TensorRT、FlashAttention、低精度和 GEMM/attention fusion 可能有效；对 Action/S1_Action 这类低利用率阶段，CUDA Graph、kernel fusion、减少 denoising loop 同步、减少 Python 调度更重要。


## 5. 核心结论五：双系统 VLA 的 S2 瓶颈至少存在两种机制

### 5.1 Hume：S2 很慢，但不是 GPU 持续饱和

AGX Orin 上 Hume 的 E2E 为 4070.2 ms，其中 S2_Inference 为 3608.6 ms，占 E2E 的 88.7%。但资源利用率不高：

| Hume / AGX S2_Inference | 数值 |
|---|---:|
| Latency | 3608.6 ms |
| E2E 占比 | 88.7% |
| GR Active | 22.7% |
| SM Util | 53.3% |
| DRAM | 10.8% |
| kernel duty | 20.9% |
| kernels/range | ~77k |

如果只是算力不足，GR/SM 应该更接近饱和。现在 latency 极长但 GR/DRAM 很低，说明更多是顺序化、小 kernel、同步或 host-device orchestration 导致的长尾。

### 5.2 RoboDual：S2 是真实 GPU + DRAM 压力

RoboDual 的 S2_Inference 更像真实硬件压力型瓶颈：

| RoboDual / AGX S2_Inference | 数值 |
|---|---:|
| Latency | 6061.4 ms |
| E2E 占比 | 93.6% |
| GR Active | 73.9% |
| SM Util | 76.5% |
| DRAM | 55.2% |
| kernel duty | 72.8% |
| kernels/range | ~95.9k |

这里 GPU 和 DRAM 都有明显压力，所以优化方向包括减少 S2 计算量、减少中间张量读写、内存 layout 优化、算子融合和量化。

### 5.3 OpenHelix：S2 不是 E2E 最大瓶颈

OpenHelix 的 S2_Inference 在 AGX 上只有 239.8 ms，但它非常高效地使用 GPU：GR Active 96.8%，SM Util 94.9%，kernel duty 96.3%。真正主导 E2E 的是 S1_Action，756.3 ms，且 GR Active 只有 11.8%。

这说明即使都是双系统模型，也不能简单说“S2 一定是瓶颈”。Hume、RoboDual、OpenHelix 的 S2 表现机制完全不同。



## 6. 核心结论六：kernel 粒度应该成为边缘 VLA 论文的一等指标

### 6.1 现象

很多慢阶段并不是 GPU 利用率高，而是 kernel 数量大、kernel duty 低。例如：

| 模型 / 阶段 | 平台 | Latency/resource duration | kernels/range | kernel duty | GR Active | 判断 |
|---|---|---:|---:|---:|---:|---|
| SmolVLA / Action | AGX | 487.3 ms latency | ~11.4k | 6.8% | 8.8% | launch-bound |
| SmolVLA / Action | Nano | 586.7 ms latency | ~11.4k | 8.7% | 10.9% | launch-bound |
| pi0.5 / Action | AGX | 480.4 ms latency | ~11.5k | 13.7% | 15.7% | launch-bound |
| OpenHelix / S1_Action | AGX | 756.3 ms latency | ~15.6k | 8.8% | 11.8% | launch-bound |
| RoboDual / S1_Action | AGX | 310.8 ms latency | ~5.9k | 4.4% | 7.4% | launch-bound |

这些阶段的共同点是：latency 长，但 GR Active、DRAM、kernel duty 都不高。GPU 不是一直在计算，而是在大量短 kernel 之间被调度开销拖住。

### 6.2 为什么这很重要

传统模型部署分析常看参数量、FLOPs、显存、GPU utilization。但 VLA 的 action generation 往往包含迭代式 denoising、控制循环、动态 shape、host-device 同步和框架层调度。FLOPs 不高的阶段也可能因为 kernel 颗粒度太碎而成为 E2E 主瓶颈。

因此论文中应该把 kernel count、kernel duty、kernel median duration 作为重要指标。否则会误判优化方向：继续压缩权重并不能解决 launch-bound 阶段。

同时也要强调：**kernel count 不能单独解释瓶颈，必须和 kernel duty、GR Active、SM Util、DRAM 一起看。** 例如 AGX Orin 上的 S2_Inference 阶段也可能有远高于 SmolVLA Action 的 kernel 数量：

| 模型 / 模块 | kernels/range | kernel duty | GR Active | DRAM | 判断 |
|---|---:|---:|---:|---:|---|
| Hume / S2_Inference | 77002 | 20.9% | 22.7% | 10.8% | kernel 很多但利用率低，偏 fragmented/sequential |
| RoboDual / S2_Inference | 95919 | 72.8% | 73.9% | 55.2% | kernel 很多且硬件压力高，偏 compute + DRAM |
| OpenHelix / S2_Inference | 1266 | 96.3% | 96.8% | 47.4% | kernel 数少但 duty 高，高效 dense compute |
| SmolVLA / Action | 11421 | 6.8% | 8.8% | 3.4% | kernel 数高且 duty 低，典型 launch/orchestration |

这个对照说明，VLA 中至少有三种不同的 kernel 形态：

1. **高 kernel 数 + 高 duty**：例如 RoboDual S2，说明 GPU 长时间有工作，瓶颈更接近计算和 DRAM。
2. **高 kernel 数 + 低 duty**：例如 SmolVLA Action、OpenHelix S1_Action、Hume S2，说明调度、同步、顺序化或 host orchestration 很可能占主导。
3. **低 kernel 数 + 高 duty**：例如 OpenHelix S2，说明算子比较集中，硬件利用效率高。

所以文中关于 SmolVLA Action 的表述应该是：它不是 kernel count 最高的阶段，但它在 action-generation 模块中呈现出“万级 kernel + 极低 duty + 低 GR/DRAM”的组合，这是 launch-bound 的关键证据。

### 6.3 优化启示

对这类阶段，优先考虑：

- CUDA Graph 捕获稳定的 action loop。
- 合并 denoising step 中的小算子。
- 减少 Python 循环和同步点。
- 用 TensorRT plugin 或自定义 fused kernel 重写碎片化路径。
- 尽量 batch 化 action generation，让每次 GPU 调用更大。



## 7. 模型-模块-设备瓶颈

| 模型 / 模块 | 主要受影响设备 | 资源证据 | 瓶颈类型 | 优化方向 |
|---|---|---|---|---|
| OpenVLA / Data | RTX5090, AGX | CPU 高，GPU/DRAM 低 | CPU/host preprocessing | 缓存 processor，减少 Python/HF 开销 |
| OpenVLA / LLM | AGX | GR 98.2%，SM 95.2%，DRAM 49.1% | dense compute + memory traffic | TensorRT, FlashAttention, quantization |
| OpenVLA / Action | AGX | GR 73.3%，SM 76.5%，DRAM 55.7% | compute + DRAM mixed | layout 优化、减少中间张量、融合 |
| pi0.5 / LLM | AGX | GR 98.9%，SM 97.5%，最大放大 10.06x | 强 compute-bound | dense kernel 优化、低精度 |
| pi0.5 / Action | AGX | GR 15.7%，DRAM 7.7%，kernel duty 13.7% | launch/orchestration-bound | CUDA Graph, kernel fusion |
| SmolVLA / Vision | Orin Nano | GR 92.7%，SM 91.2% | compute-bound | TensorRT/算子优化 |
| SmolVLA / Action | AGX, Nano | 显存低但 kernel 多、duty 低 | execution granularity bottleneck | 合并 action loop、小 kernel fusion |
| Hume / S2_Inference | AGX | latency 极长但 GR 22.7% | fragmented/sequential S2 | 重构执行流，减少同步 |
| OpenHelix / S2_Inference | AGX | GR 96.8%，SM 94.9% | high-utilization compute | 低精度/算子优化 |
| OpenHelix / S1_Action | AGX | GR 11.8%，kernel duty 8.8% | launch-bound action loop | CUDA Graph/fusion |
| RoboDual / S2_Inference | AGX | GR 73.9%，DRAM 55.2% | compute + DRAM pressure | 算子与内存访问优化 |
| RoboDual / S1_Action | AGX | GR 7.4%，kernel duty 4.4% | orchestration-bound | 减少小 kernel 与同步 |

