#!/usr/bin/env python3
"""Read latency/resource JSON files produced by the VLA benchmark scripts.

The modified benchmark scripts all write a top-level `plot_records` list using
schema `vla_plot_v1`. This helper demonstrates how a single plotting script can
load OpenVLA, pi0.5, and SmolVLA latency/resource files without model-specific
branches.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt


def load_plot_records(paths: list[str]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if payload.get("plot_schema_version") != "vla_plot_v1":
            raise ValueError(f"{path} does not contain plot_schema_version='vla_plot_v1'.")
        for record in payload.get("plot_records", []):
            item = dict(record)
            item["source_file"] = str(path)
            records.append(item)
    if not records:
        raise ValueError("No plot_records found in the provided JSON files.")
    return pd.DataFrame.from_records(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("json_files", nargs="+", help="Benchmark JSON files to read.")
    parser.add_argument("--metric-family", choices=["latency", "resource"], default="latency")
    parser.add_argument("--metric-name", default="latency_ms", help="e.g. latency_ms, duration_ms, gpu_util_percent_mean")
    parser.add_argument("--metrics-source", default="net", help="resource only: net or raw")
    parser.add_argument("--stages", nargs="*", default=None, help="Optional stage filter.")
    parser.add_argument("--csv", default=None, help="Optional output CSV path.")
    parser.add_argument("--png", default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    df = load_plot_records(args.json_files)
    mask = (df["metric_family"] == args.metric_family) & (df["metric_name"] == args.metric_name)
    if args.metric_family == "resource" and "metrics_source" in df.columns:
        mask &= df["metrics_source"].fillna("net").eq(args.metrics_source)
    if args.stages:
        mask &= df["stage"].isin(args.stages)
    df = df.loc[mask].copy()
    if df.empty:
        raise ValueError("No matching records after filtering.")

    # Stable stage order when present in the JSON records.
    stage_order = [
        "data_processing",
        "vision_encoder",
        "llm_backbone",
        "action_expert",
        "dit",
        "system2_vision_encoder",
        "system2_inference",
        "system_bridge",
        "system1_vision_encoder",
        "system1_action_expert",
        "e2e",
    ]
    df["stage"] = pd.Categorical(df["stage"], categories=stage_order, ordered=True)
    df = df.sort_values(["model", "stage"])

    show_cols = [
        "model",
        "device_name",
        "stage",
        "stage_label",
        "metric_family",
        "metric_name",
        "unit",
        "value_mean",
        "value_p5",
        "value_p95",
        "value_n",
    ]
    print(df[[c for c in show_cols if c in df.columns]].to_string(index=False))

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)

    if args.png:
        pivot = df.pivot_table(index="stage_label", columns="model", values="value_mean", aggfunc="mean")
        pivot = pivot.loc[[label for label in [
            "Data Processing",
            "Vision Encoder",
            "LLM Backbone",
            "Action Expert",
            "DiT/Decode-only",
            "System2 Vision Encoder",
            "System2 Inference",
            "System Bridge / Projector",
            "System1 Vision Encoder",
            "System1 Action Expert",
            "E2E",
        ] if label in pivot.index]]
        ax = pivot.plot(kind="bar", figsize=(max(8, len(pivot) * 1.4), 4.5))
        ax.set_ylabel(f"{args.metric_name} ({df['unit'].dropna().iloc[0] if 'unit' in df and not df['unit'].dropna().empty else ''})")
        ax.set_xlabel("Stage")
        ax.set_title(f"{args.metric_family}: {args.metric_name}")
        ax.figure.tight_layout()
        Path(args.png).parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(args.png, dpi=200)


if __name__ == "__main__":
    main()
