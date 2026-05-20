from __future__ import annotations

import io
import json
import os
from typing import Any

import numpy as np
from PIL import Image
import torch


def _iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _decode_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_path = value.get("path")
        if image_path is not None:
            return Image.open(image_path).convert("RGB")
    array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
    if array.ndim == 3 and array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if np.issubdtype(array.dtype, np.floating):
        if array.max(initial=0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def image_to_hwc_uint8(value: Any) -> np.ndarray:
    return np.asarray(_decode_image(value), dtype=np.uint8)


def image_to_chw_float_tensor(value: Any) -> torch.Tensor:
    array = image_to_hwc_uint8(value).copy()
    return torch.from_numpy(array).permute(2, 0, 1).to(dtype=torch.float32) / 255.0


def _load_local_lerobot_v2_frame(dataset_root: str, dataset_index: int) -> dict[str, Any]:
    root = os.path.abspath(os.path.expanduser(dataset_root))
    info_path = os.path.join(root, "meta", "info.json")
    episodes_path = os.path.join(root, "meta", "episodes.jsonl")
    tasks_path = os.path.join(root, "meta", "tasks.jsonl")
    if not os.path.isfile(info_path) or not os.path.isfile(episodes_path):
        raise FileNotFoundError(
            "Local LeRobot dataset metadata was not found under "
            f"{root!r}; expected meta/info.json and meta/episodes.jsonl."
        )

    with open(info_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    remaining = int(dataset_index)
    episode = None
    frame_offset = None
    for candidate in _iter_jsonl(episodes_path):
        length = int(candidate["length"])
        if remaining < length:
            episode = candidate
            frame_offset = remaining
            break
        remaining -= length
    if episode is None or frame_offset is None:
        raise IndexError(f"Dataset index {dataset_index} is out of range.")

    episode_index = int(episode["episode_index"])
    chunks_size = int(info.get("chunks_size", 1000))
    episode_chunk = episode_index // chunks_size
    rel_data_path = info["data_path"].format(
        episode_chunk=episode_chunk,
        episode_index=episode_index,
    )
    data_path = os.path.join(root, rel_data_path)
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Expected episode parquet file was not found: {data_path}")

    import pyarrow.parquet as pq

    row = pq.read_table(data_path).slice(frame_offset, 1).to_pydict()
    frame = {key: values[0] for key, values in row.items()}

    tasks = {}
    if os.path.isfile(tasks_path):
        tasks = {int(item["task_index"]): item["task"] for item in _iter_jsonl(tasks_path)}
    task = None
    task_index = frame.get("task_index")
    if task_index is not None:
        task = tasks.get(int(task_index))
    if task is None:
        task_list = episode.get("tasks") or []
        task = task_list[0] if task_list else ""

    for key in ("image", "wrist_image"):
        if key in frame:
            frame[key] = image_to_chw_float_tensor(frame[key])
    for key in ("state", "actions"):
        if key in frame:
            frame[key] = torch.tensor(frame[key], dtype=torch.float32)
    for key in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if key in frame:
            frame[key] = torch.tensor(frame[key])
    frame["task"] = task
    return frame


def load_lerobot_frame(
    dataset_repo_id: str,
    dataset_root: str | None,
    dataset_index: int,
) -> dict[str, Any]:
    if dataset_root is not None:
        root = os.path.abspath(os.path.expanduser(dataset_root))
        if os.path.isfile(os.path.join(root, "meta", "info.json")):
            return _load_local_lerobot_v2_frame(root, dataset_index)

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except (ImportError, ModuleNotFoundError):
        if dataset_root is None:
            raise
        return _load_local_lerobot_v2_frame(dataset_root, dataset_index)

    dataset_kwargs = {}
    if dataset_root is not None:
        dataset_kwargs["root"] = os.path.abspath(os.path.expanduser(dataset_root))
    try:
        dataset = LeRobotDataset(dataset_repo_id, download_videos=False, **dataset_kwargs)
    except TypeError:
        dataset = LeRobotDataset(dataset_repo_id, **dataset_kwargs)
    return dict(dataset[dataset_index])
