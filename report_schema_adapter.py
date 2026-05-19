from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def stat_value(stats: Any, key: str) -> float | None:
    """Read a numeric value from either a stats dict or a scalar legacy field."""
    if isinstance(stats, dict):
        return as_float(stats.get(key))
    if key == "mean":
        return as_float(stats)
    return None


def metric_value(stage_metrics: Any, metric_name: str, key: str = "mean") -> float | None:
    if not isinstance(stage_metrics, dict):
        return None
    return stat_value(stage_metrics.get(metric_name), key)


@dataclass(frozen=True)
class BenchmarkSchemaAdapter:
    """Compatibility layer for benchmark JSON report readers.

    The benchmark files have evolved over time. Most stage metrics are stored as
    `{stage: {mean, p5, p95, ...}}`, while a few derived or legacy fields use
    flatter names such as `frequency_hz.e2e_mean`. Report code should access
    payloads through this adapter so schema quirks stay in one place.
    """

    payload: dict[str, Any]
    path: Path | None = None

    @classmethod
    def from_file(cls, path: Path) -> "BenchmarkSchemaAdapter":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls(payload=payload, path=path)

    @property
    def model(self) -> str:
        return str(self.payload.get("model", ""))

    @property
    def benchmark(self) -> dict[str, Any]:
        value = self.payload.get("benchmark")
        return value if isinstance(value, dict) else {}

    @property
    def model_info(self) -> dict[str, Any]:
        value = self.payload.get("model_info")
        return value if isinstance(value, dict) else {}

    @property
    def nsight(self) -> dict[str, Any]:
        value = self.benchmark.get("nsight")
        return value if isinstance(value, dict) else {}

    def stage_label(self, stage: str) -> str:
        labels = self.payload.get("plot_stage_labels")
        if isinstance(labels, dict):
            label = labels.get(stage)
            if isinstance(label, str) and label:
                return label
        return stage

    def _stage_order(self, root_key: str) -> list[str]:
        metrics = self.payload.get(root_key)
        if not isinstance(metrics, dict):
            return []

        order = self.payload.get("plot_stage_order")
        if isinstance(order, list):
            stages = [str(stage) for stage in order if stage in metrics]
            if stages:
                return stages

        return [str(stage) for stage in metrics.keys()]

    def latency_stage_order(self) -> list[str]:
        return self._stage_order("latency_ms")

    def resource_stage_order(self) -> list[str]:
        return self._stage_order("resource_metrics")

    def latency_stage_stats(self, stage: str) -> dict[str, Any]:
        metrics = self.payload.get("latency_ms")
        if not isinstance(metrics, dict):
            return {}
        stats = metrics.get(stage)
        return stats if isinstance(stats, dict) else {}

    def latency_stat(self, stage: str, key: str) -> float | None:
        return stat_value(self.latency_stage_stats(stage), key)

    def latency_n(self, stage: str) -> int | None:
        value = self.latency_stage_stats(stage).get("n")
        if isinstance(value, bool) or value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def throughput_hz(self) -> float | None:
        freq = self.payload.get("frequency_hz")
        candidates: list[Any] = []
        if isinstance(freq, dict):
            candidates.extend(
                [
                    freq.get("e2e_mean"),
                    stat_value(freq.get("e2e"), "mean"),
                    freq.get("e2e"),
                    freq.get("mean"),
                ]
            )
        else:
            candidates.append(freq)

        for candidate in candidates:
            value = as_float(candidate)
            if value is not None:
                return value

        mean = self.latency_stat("e2e", "mean")
        if mean and mean > 0:
            return 1000.0 / mean
        return None

    def resource_stage_metrics(self, stage: str) -> dict[str, Any]:
        metrics = self.payload.get("resource_metrics")
        if not isinstance(metrics, dict):
            return {}
        stage_metrics = metrics.get(stage)
        return stage_metrics if isinstance(stage_metrics, dict) else {}

    def resource_metric_stat(self, stage: str, metric_name: str, key: str) -> float | None:
        return metric_value(self.resource_stage_metrics(stage), metric_name, key)

    def resource_metric_mean(self, stage: str, metric_name: str) -> float | None:
        return self.resource_metric_stat(stage, metric_name, "mean")

    def resource_n(self, stage: str, *metric_names: str) -> int | None:
        stage_metrics = self.resource_stage_metrics(stage)
        for metric_name in metric_names:
            value = stage_metrics.get(metric_name)
            if isinstance(value, dict):
                n = value.get("n")
                if n is not None:
                    try:
                        return int(n)
                    except (TypeError, ValueError):
                        continue
        return None
