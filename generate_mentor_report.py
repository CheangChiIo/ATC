from __future__ import annotations

import csv
import html
import math
import sqlite3
from pathlib import Path
from typing import Any

try:
    from report_schema_adapter import BenchmarkSchemaAdapter
except ImportError:  # pragma: no cover - supports importing as tools.generate_mentor_report
    from tools.report_schema_adapter import BenchmarkSchemaAdapter

ROOT = Path("/home/dell/ATC")
LATENCY_DIR = ROOT / "results/local_rerun_20260518_124717"
RESOURCE_DIR = ROOT / "results/nsight_full_metrics_20260518_153748"
OUT_DIR = ROOT / "results/mentor_report_20260518"

MODEL_LABELS = {
    "openvla": "OpenVLA",
    "smolvla": "SmolVLA",
    "pi05": "pi0.5",
    "Hume": "Hume",
    "OpenHelix": "OpenHelix",
    "RoboDual": "RoboDual",
}

MODEL_ORDER = ["SmolVLA", "pi0.5", "OpenVLA", "OpenHelix", "Hume", "RoboDual"]

def model_label(raw: str) -> str:
    return MODEL_LABELS.get(raw, raw)


def fmt(value: float | None, digits: int = 1) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def fmt_int(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "-"
    return str(int(round(value)))


def sort_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {name: idx for idx, name in enumerate(MODEL_ORDER)}
    return sorted(rows, key=lambda row: order.get(row["model"], 999))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def svg_bar_chart(
    path: Path,
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    y_label: str,
    color: str,
    digits: int = 1,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 430
    margin_left, margin_right, margin_top, margin_bottom = 92, 34, 58, 82
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    values = [float(row[value_key]) for row in rows if row.get(value_key) is not None]
    max_value = max(values) if values else 1.0
    y_max = max_value * 1.18 if max_value > 0 else 1.0
    bar_gap = 18
    bar_w = (chart_w - bar_gap * (len(rows) - 1)) / max(len(rows), 1)

    def sx(i: int) -> float:
        return margin_left + i * (bar_w + bar_gap)

    def sy(v: float) -> float:
        return margin_top + chart_h - (v / y_max * chart_h)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="22" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="13" fill="#374151" transform="rotate(-90 22 {height / 2})">{html.escape(y_label)}</text>',
    ]
    for tick in range(5):
        value = y_max * tick / 4
        y = sy(value)
        parts.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#6b7280">{fmt(value, 0)}</text>')
    parts.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{margin_top + chart_h}" y2="{margin_top + chart_h}" stroke="#9ca3af"/>')
    for i, row in enumerate(rows):
        value = float(row[value_key])
        x = sx(i)
        y = sy(value)
        h = margin_top + chart_h - y
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">{fmt(value, digits)}</text>')
        parts.append(f'<text x="{x + bar_w / 2:.1f}" y="{height - 42}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{html.escape(row["model"])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def svg_grouped_bar_chart(
    path: Path,
    rows: list[dict[str, Any]],
    series: list[tuple[str, str, str]],
    title: str,
    y_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 980, 430
    margin_left, margin_right, margin_top, margin_bottom = 92, 150, 58, 82
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    max_value = max(float(row[key]) for row in rows for key, _, _ in series if row.get(key) is not None)
    y_max = max(100.0, max_value * 1.12)
    group_gap = 24
    group_w = (chart_w - group_gap * (len(rows) - 1)) / max(len(rows), 1)
    bar_gap = 4
    bar_w = (group_w - bar_gap * (len(series) - 1)) / len(series)

    def sy(v: float) -> float:
        return margin_top + chart_h - (v / y_max * chart_h)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>',
        f'<text x="22" y="{height / 2}" text-anchor="middle" font-family="Arial" font-size="13" fill="#374151" transform="rotate(-90 22 {height / 2})">{html.escape(y_label)}</text>',
    ]
    for tick in range(5):
        value = y_max * tick / 4
        y = sy(value)
        parts.append(f'<line x1="{margin_left}" x2="{width - margin_right}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#6b7280">{fmt(value, 0)}</text>')
    for i, row in enumerate(rows):
        gx = margin_left + i * (group_w + group_gap)
        for j, (key, label, color) in enumerate(series):
            value = float(row[key])
            x = gx + j * (bar_w + bar_gap)
            y = sy(value)
            h = margin_top + chart_h - y
            parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{gx + group_w / 2:.1f}" y="{height - 42}" text-anchor="middle" font-family="Arial" font-size="13" fill="#111827">{html.escape(row["model"])}</text>')
    legend_x = width - margin_right + 18
    for j, (_, label, color) in enumerate(series):
        y = margin_top + j * 24
        parts.append(f'<rect x="{legend_x}" y="{y - 12}" width="14" height="14" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{legend_x + 22}" y="{y}" font-family="Arial" font-size="12" fill="#111827">{html.escape(label)}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def collect_latency() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    for path in sorted(LATENCY_DIR.glob("*_latency.json")):
        adapter = BenchmarkSchemaAdapter.from_file(path)
        model = model_label(adapter.model)
        mean = adapter.latency_stat("e2e", "mean")
        throughput = adapter.throughput_hz()
        benchmark = adapter.benchmark
        row = {
            "model": model,
            "source_file": str(path),
            "e2e_mean_ms": mean,
            "e2e_p5_ms": adapter.latency_stat("e2e", "p5"),
            "e2e_p95_ms": adapter.latency_stat("e2e", "p95"),
            "e2e_std_ms": adapter.latency_stat("e2e", "std"),
            "n": adapter.latency_n("e2e"),
            "throughput_hz": float(throughput) if throughput is not None else None,
            "example_source": benchmark.get("example_source"),
            "default_iterations": benchmark.get("e2e_iterations") or benchmark.get("default_measured_iterations"),
            "warmup": benchmark.get("e2e_warmup") or benchmark.get("default_warmup"),
        }
        summary.append(row)

        for stage in adapter.latency_stage_order():
            components.append(
                {
                    "model": model,
                    "stage": stage,
                    "stage_label": adapter.stage_label(stage),
                    "mean_ms": adapter.latency_stat(stage, "mean"),
                    "p5_ms": adapter.latency_stat(stage, "p5"),
                    "p95_ms": adapter.latency_stat(stage, "p95"),
                    "std_ms": adapter.latency_stat(stage, "std"),
                    "n": adapter.latency_n(stage),
                }
            )
    return sort_models(summary), sort_models(components)


def collect_resource() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary: list[dict[str, Any]] = []
    components: list[dict[str, Any]] = []
    integrity: list[dict[str, Any]] = []
    for path in sorted(RESOURCE_DIR.glob("*_resource.json")):
        adapter = BenchmarkSchemaAdapter.from_file(path)
        model = model_label(adapter.model)
        nsight = adapter.nsight
        report_file = Path(nsight.get("expected_report_file", ""))
        sqlite_file = Path(nsight.get("sqlite_export_file", ""))
        report_size_mb = report_file.stat().st_size / 1024**2 if report_file.exists() else None
        sqlite_size_mb = sqlite_file.stat().st_size / 1024**2 if sqlite_file.exists() else None
        gpu_rows = target_rows = nvtx_rows = None
        if sqlite_file.exists():
            conn = sqlite3.connect(sqlite_file)
            try:
                gpu_rows = conn.execute("SELECT COUNT(*) FROM GPU_METRICS").fetchone()[0]
                target_rows = conn.execute("SELECT COUNT(*) FROM TARGET_INFO_GPU_METRICS").fetchone()[0]
                nvtx_rows = conn.execute("SELECT COUNT(*) FROM NVTX_EVENTS").fetchone()[0]
            finally:
                conn.close()
        summary.append(
            {
                "model": model,
                "source_file": str(path),
                "gpu_profile_duration_mean_ms": adapter.resource_metric_mean("e2e", "gpu_profile_duration_ms"),
                "cpu_pytorch_duration_mean_ms": adapter.resource_metric_mean("e2e", "duration_ms"),
                "gpu_busy_mean_percent": adapter.resource_metric_mean("e2e", "gpu_busy_ratio_percent_mean"),
                "gpu_busy_p95_percent": adapter.resource_metric_stat("e2e", "gpu_busy_ratio_percent_mean", "p95"),
                "sm_mean_percent": adapter.resource_metric_mean("e2e", "sm_util_percent_mean"),
                "sm_p95_percent": adapter.resource_metric_stat("e2e", "sm_util_percent_mean", "p95"),
                "nsight_cuda_memory_peak_mb": adapter.resource_metric_mean("e2e", "nsight_cuda_memory_peak_mb"),
                "torch_allocated_peak_mb": adapter.resource_metric_mean("e2e", "torch_memory_allocated_mb_peak"),
                "torch_reserved_peak_mb": adapter.resource_metric_mean("e2e", "torch_memory_reserved_mb_peak"),
                "cpu_process_mean_percent": adapter.resource_metric_mean("e2e", "cpu_process_percent_mean"),
                "cpu_process_core_mean_percent": adapter.resource_metric_mean("e2e", "cpu_process_core_percent_mean"),
                "n": adapter.resource_n("e2e", "duration_ms", "gpu_profile_duration_ms"),
                "nsys_rep_size_mb": report_size_mb,
                "sqlite_size_mb": sqlite_size_mb,
                "gpu_metrics_rows": gpu_rows,
                "nvtx_events_rows": nvtx_rows,
            }
        )
        integrity.append(
            {
                "model": model,
                "nsys_rep": str(report_file),
                "sqlite": str(sqlite_file),
                "nsys_rep_size_mb": report_size_mb,
                "sqlite_size_mb": sqlite_size_mb,
                "GPU_METRICS_rows": gpu_rows,
                "TARGET_INFO_GPU_METRICS_rows": target_rows,
                "NVTX_EVENTS_rows": nvtx_rows,
            }
        )

        for stage in adapter.resource_stage_order():
            components.append(
                {
                    "model": model,
                    "stage": stage,
                    "stage_label": adapter.stage_label(stage),
                    "duration_mean_ms": adapter.resource_metric_mean(stage, "duration_ms"),
                    "gpu_profile_duration_mean_ms": adapter.resource_metric_mean(stage, "gpu_profile_duration_ms"),
                    "gpu_busy_mean_percent": adapter.resource_metric_mean(stage, "gpu_busy_ratio_percent_mean"),
                    "sm_mean_percent": adapter.resource_metric_mean(stage, "sm_util_percent_mean"),
                    "nsight_cuda_memory_peak_mb": adapter.resource_metric_mean(stage, "nsight_cuda_memory_peak_mb"),
                    "torch_allocated_peak_mb": adapter.resource_metric_mean(stage, "torch_memory_allocated_mb_peak"),
                    "cpu_process_mean_percent": adapter.resource_metric_mean(stage, "cpu_process_percent_mean"),
                }
            )
    return sort_models(summary), sort_models(components), sort_models(integrity)


def dominant_stage(rows: list[dict[str, Any]], model: str, value_key: str) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row["model"] == model and row["stage"] not in {"e2e", "data_processing"} and row.get(value_key) is not None
    ]
    return max(candidates, key=lambda row: row[value_key])


def report_text(
    latency_summary: list[dict[str, Any]],
    latency_components: list[dict[str, Any]],
    resource_summary: list[dict[str, Any]],
    resource_components: list[dict[str, Any]],
    integrity: list[dict[str, Any]],
) -> str:
    latency_rank = sorted(latency_summary, key=lambda row: row["e2e_mean_ms"])
    resource_by_mem = sorted(resource_summary, key=lambda row: row["nsight_cuda_memory_peak_mb"])
    resource_by_sm = sorted(resource_summary, key=lambda row: row["sm_mean_percent"], reverse=True)
    fastest = latency_rank[0]
    slowest = latency_rank[-1]
    lightest = resource_by_mem[0]
    heaviest = resource_by_mem[-1]
    highest_sm = resource_by_sm[0]

    latency_table = markdown_table(
        ["Rank", "Model", "E2E mean (ms)", "P5-P95 (ms)", "Throughput (Hz)", "n"],
        [
            [
                str(idx + 1),
                row["model"],
                fmt(row["e2e_mean_ms"], 1),
                f'{fmt(row["e2e_p5_ms"], 1)}-{fmt(row["e2e_p95_ms"], 1)}',
                fmt(row["throughput_hz"], 2),
                str(row["n"]),
            ]
            for idx, row in enumerate(latency_rank)
        ],
    )
    resource_table = markdown_table(
        [
            "Model",
            "GPU busy mean (%)",
            "SM mean (%)",
            "CUDA peak (GB)",
            "Torch alloc peak (GB)",
            "CPU mean (%)",
            ".nsys-rep",
        ],
        [
            [
                row["model"],
                fmt(row["gpu_busy_mean_percent"], 1),
                fmt(row["sm_mean_percent"], 1),
                fmt(row["nsight_cuda_memory_peak_mb"] / 1024 if row["nsight_cuda_memory_peak_mb"] else None, 2),
                fmt(row["torch_allocated_peak_mb"] / 1024 if row["torch_allocated_peak_mb"] else None, 2),
                fmt(row["cpu_process_mean_percent"], 1),
                f'{fmt(row["nsys_rep_size_mb"], 0)} MB',
            ]
            for row in sort_models(resource_summary)
        ],
    )
    bottleneck_rows = []
    for row in sort_models(latency_summary):
        model = row["model"]
        lat_stage = dominant_stage(latency_components, model, "mean_ms")
        res_stage = dominant_stage(resource_components, model, "duration_mean_ms")
        bottleneck_rows.append(
            [
                model,
                lat_stage["stage_label"],
                fmt(lat_stage["mean_ms"], 1),
                res_stage["stage_label"],
                fmt(res_stage["duration_mean_ms"], 1),
                fmt(res_stage["sm_mean_percent"], 1),
            ]
        )
    bottleneck_table = markdown_table(
        [
            "Model",
            "Latency bottleneck",
            "Latency mean (ms)",
            "Resource longest stage",
            "Resource duration (ms)",
            "Stage SM mean (%)",
        ],
        bottleneck_rows,
    )
    integrity_table = markdown_table(
        ["Model", "GPU_METRICS rows", "TARGET_INFO rows", "NVTX_EVENTS rows", "SQLite size"],
        [
            [
                row["model"],
                fmt_int(row["GPU_METRICS_rows"]),
                fmt_int(row["TARGET_INFO_GPU_METRICS_rows"]),
                fmt_int(row["NVTX_EVENTS_rows"]),
                f'{fmt(row["sqlite_size_mb"] / 1024 if row["sqlite_size_mb"] else None, 2)} GB',
            ]
            for row in sort_models(integrity)
        ],
    )
    return f"""# VLA 模型 Latency 与资源占用实验报告

## 1. 汇报摘要

- **速度最快**：{fastest["model"]}，E2E latency mean = **{fmt(fastest["e2e_mean_ms"], 1)} ms**，约 **{fmt(fastest["throughput_hz"], 2)} Hz**。
- **速度最慢**：{slowest["model"]}，E2E latency mean = **{fmt(slowest["e2e_mean_ms"], 1)} ms**。主要瓶颈见第 5 节。
- **显存占用最低**：{lightest["model"]}，Nsight CUDA peak = **{fmt(lightest["nsight_cuda_memory_peak_mb"] / 1024, 2)} GB**。
- **显存占用最高**：{heaviest["model"]}，Nsight CUDA peak = **{fmt(heaviest["nsight_cuda_memory_peak_mb"] / 1024, 2)} GB**。
- **GPU/SM 压力最高**：{highest_sm["model"]}，E2E SM mean = **{fmt(highest_sm["sm_mean_percent"], 1)}%**。

## 2. 数据来源与口径

- Latency 来源：`{LATENCY_DIR}`。
- 资源占用来源：`{RESOURCE_DIR}`。
- Latency 与资源占用是**分开测量**的：正式速度结论只使用 latency JSON；资源报告中的 duration 只作为资源窗口，不作为官方 latency。
- 资源占用采用三轮合并协议：Nsight GPU pass 采 CUDA/NVTX/GPU metrics/CUDA memory；后两轮非 Nsight pass 采 CPU 与 PyTorch allocator 显存。
- 统计默认配置保持原脚本默认值，E2E latency 表中的 `n` 为有效统计次数。

## 3. 指标定义与测量方法

### 3.1 Latency 指标

| 指标 | 含义 | 测量方法 | 汇报时怎么解释 |
| --- | --- | --- | --- |
| E2E latency mean / P5 / P95 | 单次完整推理端到端耗时的均值、5 分位、95 分位 | latency 脚本在 warmup 后循环测量，使用 CUDA synchronize 包住计时窗口，单位 ms | mean 看平均速度，P95 看尾延迟稳定性 |
| Throughput Hz | 理论每秒可完成的 E2E 推理次数 | `1000 / E2E latency mean(ms)` | 越高越快，是 latency 的倒数表达 |
| Stage latency | 模块级耗时，例如 Vision Encoder、LLM Backbone、Action Expert、System2 Inference | latency 脚本对模块或阶段加计时窗口，warmup 后统计 mean/P5/P95 | 用来定位速度瓶颈，不等同于简单相加后的 E2E |
| n | 有效统计次数 | JSON 中每个统计项的样本数 | 本次多数为默认 100 次 |

### 3.2 资源占用指标

| 指标 | 含义 | 测量方法 | 注意事项 |
| --- | --- | --- | --- |
| GPU busy mean (%) | NVTX 窗口内 CUDA kernel 执行时间占窗口墙钟时间的比例 | Nsight SQLite 中 CUDA kernel interval 与 NVTX range 求交集并合并重叠区间 | 这是“时间忙碌比例”，不是 nvidia-smi 那种瞬时利用率 |
| SM mean (%) | SM Active / SM utilization 相关 GPU metrics 的窗口平均值 | Nsight Systems `GPU_METRICS` 表中 GB20x 指标集，按 NVTX 时间窗口聚合 | 更接近 GPU 计算单元活跃程度 |
| Nsight CUDA memory peak | CUDA device memory residency 峰值 | Nsight 的 CUDA memory usage 事件重建窗口内显存驻留峰值 | 来自 CUDA 事件，不是整卡 nvidia-smi used memory |
| Torch allocated peak | PyTorch allocator active tensor memory 峰值 | 后两轮非 Nsight pass 中用 PyTorch allocator API 记录 peak allocated | 反映 PyTorch 张量实际分配峰值 |
| Torch reserved peak | PyTorch caching allocator 向 CUDA 保留的显存峰值 | 后两轮非 Nsight pass 中用 PyTorch allocator API 记录 peak reserved | 通常大于或等于 allocated，代表缓存池保留量 |
| CPU mean (%) | 进程 CPU 时间增量 / 墙钟时间 / 可用逻辑核数 | 后两轮非 Nsight pass 中用 psutil `process.cpu_times()` 计算 | 归一到整机 0-100%，便于跨模型对比 |
| CPU core mean (%) | 进程 CPU 时间增量 / 墙钟时间，不除以逻辑核数 | 同上 | 100% 约等于占满 1 个逻辑核，主要用于诊断 |
| Resource duration | 资源测量窗口耗时 | 资源脚本中的窗口计时；Nsight pass 与非 Nsight pass 分开记录 | 不作为正式 latency 结论，只用于资源窗口对齐 |

### 3.3 Nsight 完整性指标

| 字段 | 含义 | 用途 |
| --- | --- | --- |
| GPU_METRICS rows | Nsight SQLite 中 GPU metrics 采样行数 | 验证确实采到了 GPU 全指标 |
| TARGET_INFO_GPU_METRICS rows | Nsight 记录的可用 GPU metric 定义数 | 验证指标名/metricId 映射存在 |
| NVTX_EVENTS rows | Nsight 记录的 NVTX 事件数 | 验证 E2E 与模块级时间线都被捕获 |
| `.nsys-rep` size / SQLite size | Nsight 原始报告和导出数据库大小 | 报告大小差异通常来自运行时长、kernel 数、GPU metrics 采样数和 NVTX 事件数差异 |

## 4. E2E Latency 排名

![E2E latency](figures/latency_e2e.svg)

{latency_table}

![Throughput](figures/throughput.svg)

## 5. 模块级瓶颈

{bottleneck_table}

解释：单系统模型主要瓶颈集中在 Action Expert 或 LLM Backbone；双系统模型中，Hume 与 RoboDual 的 System2 Inference 是主耗时段，OpenHelix 的 System1 Action Expert 更突出。

## 6. 资源占用对比

![GPU utilization](figures/resource_gpu_util.svg)

![Memory](figures/resource_memory.svg)

![CPU](figures/resource_cpu.svg)

{resource_table}

## 7. Nsight 报告完整性检查

以下表格用于说明这次 `.nsys-rep/.sqlite` 确实包含完整 GPU metrics 和 NVTX 事件，而不是只记录 e2e 或空指标。

{integrity_table}

## 8. 汇报建议

1. 先讲速度排序：SmolVLA、pi0.5、OpenVLA 是第一梯队，OpenHelix 居中，Hume 和 RoboDual 明显更慢。
2. 再讲资源代价：SmolVLA 显存占用最低；RoboDual 的显存、GPU busy 和 SM 压力最高；OpenVLA 显存也偏高且 CPU 占用明显高于其他模型。
3. 最后讲瓶颈定位：RoboDual/Hume 的瓶颈主要来自 System2 Inference；OpenVLA/pi0.5/SmolVLA 的主要耗时来自 Action Expert；OpenHelix 的瓶颈更偏 System1 Action Expert。

## 9. 附件

- `tables/latency_summary.csv`
- `tables/latency_components.csv`
- `tables/resource_summary.csv`
- `tables/resource_components.csv`
- `tables/nsight_integrity.csv`
"""


def latency_report_text(latency_summary: list[dict[str, Any]], latency_components: list[dict[str, Any]]) -> str:
    latency_rank = sorted(latency_summary, key=lambda row: row["e2e_mean_ms"])
    latency_table = markdown_table(
        ["Rank", "Model", "E2E mean (ms)", "P5-P95 (ms)", "Throughput (Hz)", "n"],
        [
            [
                str(idx + 1),
                row["model"],
                fmt(row["e2e_mean_ms"], 1),
                f'{fmt(row["e2e_p5_ms"], 1)}-{fmt(row["e2e_p95_ms"], 1)}',
                fmt(row["throughput_hz"], 2),
                str(row["n"]),
            ]
            for idx, row in enumerate(latency_rank)
        ],
    )
    bottleneck_rows = []
    for row in sort_models(latency_summary):
        stage = dominant_stage(latency_components, row["model"], "mean_ms")
        bottleneck_rows.append([row["model"], stage["stage_label"], fmt(stage["mean_ms"], 1), fmt(stage["p95_ms"], 1)])
    bottleneck_table = markdown_table(
        ["Model", "Main latency bottleneck", "Mean (ms)", "P95 (ms)"],
        bottleneck_rows,
    )
    return f"""# Latency 测量结果报告

## 指标定义与测量方法

| 指标 | 含义 | 测量方法 | 怎么读 |
| --- | --- | --- | --- |
| E2E latency mean | 完整推理端到端平均耗时 | latency 脚本 warmup 后循环测量，CUDA synchronize 包住窗口 | 越低越快 |
| P5 / P95 | 5 分位和 95 分位耗时 | 对所有有效样本做分位数统计 | P95 反映尾延迟和稳定性 |
| Throughput Hz | 每秒可完成的推理次数 | `1000 / E2E mean(ms)` | 越高越快 |
| Stage latency | 模块级耗时 | 对各模块计时窗口分别统计 | 用于定位瓶颈 |
| n | 有效样本数 | JSON 中统计项的样本数量 | 本次 E2E 均为 100 |

## E2E 排名

![E2E latency](figures/latency_e2e.svg)

{latency_table}

![Throughput](figures/throughput.svg)

## 模块级瓶颈

{bottleneck_table}

## 结论

- SmolVLA、pi0.5、OpenVLA 是速度第一梯队。
- OpenHelix 居中。
- Hume 与 RoboDual 明显较慢，其中主要瓶颈分别集中在 System2 Inference。
"""


def resource_report_text(
    resource_summary: list[dict[str, Any]],
    resource_components: list[dict[str, Any]],
    integrity: list[dict[str, Any]],
) -> str:
    resource_table = markdown_table(
        [
            "Model",
            "GPU busy mean (%)",
            "SM mean (%)",
            "CUDA peak (GB)",
            "Torch alloc peak (GB)",
            "CPU mean (%)",
        ],
        [
            [
                row["model"],
                fmt(row["gpu_busy_mean_percent"], 1),
                fmt(row["sm_mean_percent"], 1),
                fmt(row["nsight_cuda_memory_peak_mb"] / 1024 if row["nsight_cuda_memory_peak_mb"] else None, 2),
                fmt(row["torch_allocated_peak_mb"] / 1024 if row["torch_allocated_peak_mb"] else None, 2),
                fmt(row["cpu_process_mean_percent"], 1),
            ]
            for row in sort_models(resource_summary)
        ],
    )
    bottleneck_rows = []
    for row in sort_models(resource_summary):
        stage = dominant_stage(resource_components, row["model"], "duration_mean_ms")
        bottleneck_rows.append(
            [row["model"], stage["stage_label"], fmt(stage["duration_mean_ms"], 1), fmt(stage["gpu_busy_mean_percent"], 1), fmt(stage["sm_mean_percent"], 1)]
        )
    bottleneck_table = markdown_table(
        ["Model", "Longest resource stage", "Duration (ms)", "GPU busy (%)", "SM mean (%)"],
        bottleneck_rows,
    )
    integrity_table = markdown_table(
        ["Model", "GPU_METRICS rows", "NVTX_EVENTS rows", ".nsys-rep size", "SQLite size"],
        [
            [
                row["model"],
                fmt_int(row["GPU_METRICS_rows"]),
                fmt_int(row["NVTX_EVENTS_rows"]),
                f'{fmt(row["nsys_rep_size_mb"], 0)} MB',
                f'{fmt(row["sqlite_size_mb"] / 1024 if row["sqlite_size_mb"] else None, 2)} GB',
            ]
            for row in sort_models(integrity)
        ],
    )
    return f"""# 资源占用结果报告

## 指标定义与测量方法

| 指标 | 含义 | 测量方法 | 注意事项 |
| --- | --- | --- | --- |
| GPU busy mean (%) | CUDA kernel 忙碌时间占 NVTX 窗口时间比例 | Nsight SQLite 中 kernel interval 与 NVTX range 求交集 | 时间忙碌比例，不是 nvidia-smi 瞬时利用率 |
| SM mean (%) | SM Active / SM utilization 平均值 | Nsight `GPU_METRICS` 表按 NVTX 窗口聚合 | 反映计算单元活跃程度 |
| Nsight CUDA memory peak | CUDA device memory 驻留峰值 | Nsight CUDA memory usage 事件重建 | 不是整卡 used memory |
| Torch allocated peak | PyTorch active tensor memory 峰值 | 非 Nsight pass 中 PyTorch allocator API | 反映张量实际分配峰值 |
| CPU mean (%) | 进程 CPU 利用率 | 非 Nsight pass 中 psutil CPU time delta / wall time / 逻辑核数 | 归一到整机 0-100% |
| Resource duration | 资源测量窗口耗时 | 资源脚本窗口计时 | 不作为正式 latency 结论 |

## 总体资源占用

![GPU utilization](figures/resource_gpu_util.svg)

![Memory](figures/resource_memory.svg)

![CPU](figures/resource_cpu.svg)

{resource_table}

## 模块级资源瓶颈

{bottleneck_table}

## Nsight 完整性

{integrity_table}

## 结论

- SmolVLA 资源占用最低，显存峰值远低于其他模型。
- RoboDual 的 GPU busy、SM mean 和显存峰值最高，是资源压力最大的模型。
- OpenVLA 显存接近 RoboDual，但 CPU mean 明显更高。
- Hume 和 RoboDual 的长耗时主要来自 System2 Inference。
"""


def _rows_for_model(rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["model"] == model]


def _summary_by_model(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["model"]: row for row in rows}


def _non_e2e_stage_rows(rows: list[dict[str, Any]], value_key: str) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["stage"] not in {"e2e", "data_processing"} and row.get(value_key) is not None
    ]


def stage_breakdown_report_text(
    latency_summary: list[dict[str, Any]],
    latency_components: list[dict[str, Any]],
    resource_summary: list[dict[str, Any]],
    resource_components: list[dict[str, Any]],
) -> str:
    lat_summary = _summary_by_model(latency_summary)
    res_summary = _summary_by_model(resource_summary)
    models = [model for model in MODEL_ORDER if model in lat_summary or model in res_summary]

    top_latency = sorted(
        _non_e2e_stage_rows(latency_components, "mean_ms"),
        key=lambda row: row["mean_ms"],
        reverse=True,
    )[:12]
    top_resource_duration = sorted(
        _non_e2e_stage_rows(resource_components, "duration_mean_ms"),
        key=lambda row: row["duration_mean_ms"],
        reverse=True,
    )[:12]
    top_resource_sm = sorted(
        _non_e2e_stage_rows(resource_components, "sm_mean_percent"),
        key=lambda row: row["sm_mean_percent"],
        reverse=True,
    )[:12]

    top_latency_table = markdown_table(
        ["Rank", "Model", "Stage", "Mean (ms)", "P5-P95 (ms)", "% of E2E"],
        [
            [
                str(idx + 1),
                row["model"],
                row["stage_label"],
                fmt(row["mean_ms"], 1),
                f'{fmt(row["p5_ms"], 1)}-{fmt(row["p95_ms"], 1)}',
                fmt(row["mean_ms"] / lat_summary[row["model"]]["e2e_mean_ms"] * 100, 1),
            ]
            for idx, row in enumerate(top_latency)
        ],
    )
    top_resource_duration_table = markdown_table(
        ["Rank", "Model", "Stage", "Duration (ms)", "GPU busy (%)", "SM mean (%)", "% of resource E2E"],
        [
            [
                str(idx + 1),
                row["model"],
                row["stage_label"],
                fmt(row["duration_mean_ms"], 1),
                fmt(row["gpu_busy_mean_percent"], 1),
                fmt(row["sm_mean_percent"], 1),
                fmt(
                    row["duration_mean_ms"]
                    / res_summary[row["model"]]["cpu_pytorch_duration_mean_ms"]
                    * 100,
                    1,
                ),
            ]
            for idx, row in enumerate(top_resource_duration)
        ],
    )
    top_resource_sm_table = markdown_table(
        ["Rank", "Model", "Stage", "SM mean (%)", "GPU busy (%)", "Duration (ms)"],
        [
            [
                str(idx + 1),
                row["model"],
                row["stage_label"],
                fmt(row["sm_mean_percent"], 1),
                fmt(row["gpu_busy_mean_percent"], 1),
                fmt(row["duration_mean_ms"], 1),
            ]
            for idx, row in enumerate(top_resource_sm)
        ],
    )

    sections = [
        "# Latency 与资源占用分阶段报告",
        "",
        "## 阅读说明",
        "",
        "- 本报告按模型展开，列出每个阶段的 latency 和 resource 指标。",
        "- Latency 阶段耗时来自 `*_latency.json`，是正式速度测量结果。",
        "- Resource 阶段指标来自 `*_resource.json`，其中 GPU 指标来自 Nsight pass，CPU/PyTorch 显存来自后两轮非 Nsight pass。",
        "- 阶段耗时与 E2E 是分开测的，百分比只用于定位瓶颈，不应强行相加成 E2E。",
        "",
        "## 全局阶段瓶颈 Top 表",
        "",
        "### Latency 最耗时阶段",
        "",
        top_latency_table,
        "",
        "### Resource 最长阶段",
        "",
        top_resource_duration_table,
        "",
        "### SM 利用率最高阶段",
        "",
        top_resource_sm_table,
        "",
        "## 按模型分阶段明细",
        "",
    ]

    for model in models:
        lat_e2e = lat_summary.get(model, {})
        res_e2e = res_summary.get(model, {})
        lat_rows = []
        for row in _rows_for_model(latency_components, model):
            pct = None
            if row.get("mean_ms") is not None and lat_e2e.get("e2e_mean_ms"):
                pct = row["mean_ms"] / lat_e2e["e2e_mean_ms"] * 100
            lat_rows.append(
                [
                    row["stage_label"],
                    fmt(row.get("mean_ms"), 1),
                    f'{fmt(row.get("p5_ms"), 1)}-{fmt(row.get("p95_ms"), 1)}',
                    fmt(row.get("std_ms"), 2),
                    fmt(pct, 1),
                    str(row.get("n") or "-"),
                ]
            )
        res_rows = []
        for row in _rows_for_model(resource_components, model):
            pct = None
            if row.get("duration_mean_ms") is not None and res_e2e.get("cpu_pytorch_duration_mean_ms"):
                pct = row["duration_mean_ms"] / res_e2e["cpu_pytorch_duration_mean_ms"] * 100
            res_rows.append(
                [
                    row["stage_label"],
                    fmt(row.get("duration_mean_ms"), 1),
                    fmt(pct, 1),
                    fmt(row.get("gpu_busy_mean_percent"), 1),
                    fmt(row.get("sm_mean_percent"), 1),
                    fmt(row["nsight_cuda_memory_peak_mb"] / 1024 if row.get("nsight_cuda_memory_peak_mb") else None, 2),
                    fmt(row["torch_allocated_peak_mb"] / 1024 if row.get("torch_allocated_peak_mb") else None, 2),
                    fmt(row.get("cpu_process_mean_percent"), 1),
                ]
            )

        lat_stage = dominant_stage(latency_components, model, "mean_ms") if model in lat_summary else None
        res_stage = dominant_stage(resource_components, model, "duration_mean_ms") if model in res_summary else None
        headline = []
        if lat_stage:
            headline.append(f"Latency 主瓶颈：**{lat_stage['stage_label']}**（{fmt(lat_stage['mean_ms'], 1)} ms）")
        if res_stage:
            headline.append(
                f"资源最长阶段：**{res_stage['stage_label']}**（{fmt(res_stage['duration_mean_ms'], 1)} ms，SM {fmt(res_stage['sm_mean_percent'], 1)}%）"
            )

        sections.extend(
            [
                f"### {model}",
                "",
                "；".join(headline) + "。",
                "",
                "**Latency 分阶段**",
                "",
                markdown_table(
                    ["Stage", "Mean (ms)", "P5-P95 (ms)", "Std (ms)", "% of E2E", "n"],
                    lat_rows,
                ),
                "",
                "**Resource 分阶段**",
                "",
                markdown_table(
                    [
                        "Stage",
                        "Duration (ms)",
                        "% of resource E2E",
                        "GPU busy (%)",
                        "SM mean (%)",
                        "CUDA peak (GB)",
                        "Torch alloc peak (GB)",
                        "CPU mean (%)",
                    ],
                    res_rows,
                ),
                "",
            ]
        )

    sections.extend(
        [
            "## 汇报时可直接说的结论",
            "",
            "- 单系统模型中，SmolVLA、pi0.5、OpenVLA 的主耗时基本都落在 Action Expert；OpenVLA 的该阶段 GPU busy 和 SM 利用率更高。",
            "- 双系统模型中，Hume 和 RoboDual 的主瓶颈是 System2 Inference；RoboDual 该阶段同时具有最高的 SM 利用率和最长 duration。",
            "- OpenHelix 的瓶颈更偏 System1 Action Expert，而不是 System2 Inference。",
            "- 显存峰值在阶段间变化不大时，说明主要由模型常驻参数/缓存决定；阶段表里的显存更适合作为该窗口内峰值，而不是阶段独占显存。",
        ]
    )
    return "\n".join(sections) + "\n"


def html_report(markdown: str) -> str:
    escaped = html.escape(markdown)
    # Keep the HTML dependency-free. Markdown source is embedded verbatim, while
    # SVG figures are linked explicitly for browser/PDF presentation.
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>VLA 模型 Latency 与资源占用实验报告</title>
  <style>
    body {{ font-family: Arial, "Noto Sans CJK SC", sans-serif; margin: 36px auto; max-width: 1120px; line-height: 1.55; color: #111827; }}
    h1, h2 {{ color: #111827; }}
    pre {{ white-space: pre-wrap; background: #f9fafb; border: 1px solid #e5e7eb; padding: 16px; border-radius: 8px; }}
    img {{ max-width: 100%; margin: 12px 0 24px; border: 1px solid #e5e7eb; border-radius: 8px; }}
  </style>
</head>
<body>
  <h1>VLA 模型 Latency 与资源占用实验报告</h1>
  <p>HTML 版用于快速浏览图表；完整表格和说明以 Markdown 源为准。</p>
  <img src="figures/latency_e2e.svg" alt="E2E latency">
  <img src="figures/throughput.svg" alt="Throughput">
  <img src="figures/resource_gpu_util.svg" alt="GPU utilization">
  <img src="figures/resource_memory.svg" alt="Memory">
  <img src="figures/resource_cpu.svg" alt="CPU">
  <h2>Markdown 源报告</h2>
  <pre>{escaped}</pre>
</body>
</html>
"""


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "tables").mkdir(exist_ok=True)
    (OUT_DIR / "figures").mkdir(exist_ok=True)

    latency_summary, latency_components = collect_latency()
    resource_summary, resource_components, integrity = collect_resource()

    write_csv(
        OUT_DIR / "tables/latency_summary.csv",
        sorted(latency_summary, key=lambda row: row["e2e_mean_ms"]),
        [
            "model",
            "e2e_mean_ms",
            "e2e_p5_ms",
            "e2e_p95_ms",
            "e2e_std_ms",
            "throughput_hz",
            "n",
            "example_source",
            "default_iterations",
            "warmup",
            "source_file",
        ],
    )
    write_csv(
        OUT_DIR / "tables/latency_components.csv",
        latency_components,
        ["model", "stage", "stage_label", "mean_ms", "p5_ms", "p95_ms", "std_ms", "n"],
    )
    write_csv(
        OUT_DIR / "tables/resource_summary.csv",
        resource_summary,
        [
            "model",
            "gpu_profile_duration_mean_ms",
            "cpu_pytorch_duration_mean_ms",
            "gpu_busy_mean_percent",
            "gpu_busy_p95_percent",
            "sm_mean_percent",
            "sm_p95_percent",
            "nsight_cuda_memory_peak_mb",
            "torch_allocated_peak_mb",
            "torch_reserved_peak_mb",
            "cpu_process_mean_percent",
            "cpu_process_core_mean_percent",
            "n",
            "nsys_rep_size_mb",
            "sqlite_size_mb",
            "gpu_metrics_rows",
            "nvtx_events_rows",
            "source_file",
        ],
    )
    write_csv(
        OUT_DIR / "tables/resource_components.csv",
        resource_components,
        [
            "model",
            "stage",
            "stage_label",
            "duration_mean_ms",
            "gpu_profile_duration_mean_ms",
            "gpu_busy_mean_percent",
            "sm_mean_percent",
            "nsight_cuda_memory_peak_mb",
            "torch_allocated_peak_mb",
            "cpu_process_mean_percent",
        ],
    )
    write_csv(
        OUT_DIR / "tables/nsight_integrity.csv",
        integrity,
        [
            "model",
            "GPU_METRICS_rows",
            "TARGET_INFO_GPU_METRICS_rows",
            "NVTX_EVENTS_rows",
            "nsys_rep_size_mb",
            "sqlite_size_mb",
            "nsys_rep",
            "sqlite",
        ],
    )

    latency_rank = sorted(latency_summary, key=lambda row: row["e2e_mean_ms"])
    svg_bar_chart(
        OUT_DIR / "figures/latency_e2e.svg",
        latency_rank,
        "e2e_mean_ms",
        "E2E Latency Mean (lower is better)",
        "ms",
        "#2563eb",
    )
    svg_bar_chart(
        OUT_DIR / "figures/throughput.svg",
        sorted(latency_summary, key=lambda row: row["throughput_hz"], reverse=True),
        "throughput_hz",
        "E2E Throughput",
        "Hz",
        "#059669",
        digits=2,
    )
    svg_grouped_bar_chart(
        OUT_DIR / "figures/resource_gpu_util.svg",
        sort_models(resource_summary),
        [
            ("gpu_busy_mean_percent", "GPU busy", "#f97316"),
            ("sm_mean_percent", "SM mean", "#7c3aed"),
        ],
        "E2E GPU Utilization Metrics",
        "%",
    )
    memory_rows = [
        {
            **row,
            "cuda_gb": row["nsight_cuda_memory_peak_mb"] / 1024,
            "torch_alloc_gb": row["torch_allocated_peak_mb"] / 1024,
        }
        for row in sort_models(resource_summary)
    ]
    svg_grouped_bar_chart(
        OUT_DIR / "figures/resource_memory.svg",
        memory_rows,
        [
            ("cuda_gb", "Nsight CUDA peak", "#dc2626"),
            ("torch_alloc_gb", "Torch alloc peak", "#0891b2"),
        ],
        "E2E Memory Peak",
        "GB",
    )
    svg_bar_chart(
        OUT_DIR / "figures/resource_cpu.svg",
        sort_models(resource_summary),
        "cpu_process_mean_percent",
        "E2E CPU Process Utilization Mean",
        "%",
        "#4b5563",
    )

    md = report_text(latency_summary, latency_components, resource_summary, resource_components, integrity)
    (OUT_DIR / "mentor_experiment_report.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "mentor_experiment_report.html").write_text(html_report(md), encoding="utf-8")
    (OUT_DIR / "latency_report.md").write_text(latency_report_text(latency_summary, latency_components), encoding="utf-8")
    (OUT_DIR / "resource_report.md").write_text(
        resource_report_text(resource_summary, resource_components, integrity),
        encoding="utf-8",
    )
    (OUT_DIR / "stage_breakdown_report.md").write_text(
        stage_breakdown_report_text(
            latency_summary,
            latency_components,
            resource_summary,
            resource_components,
        ),
        encoding="utf-8",
    )
    print(OUT_DIR)


if __name__ == "__main__":
    main()
