from __future__ import annotations

import os
import re
from typing import Any

import numpy as np


def _load_lang_annotations(lang_dir: str) -> dict[int, str]:
    """Load auto_lang_ann.npy and return {episode_number: instruction}."""
    lang_path = os.path.join(lang_dir, "auto_lang_ann.npy")
    if not os.path.isfile(lang_path):
        raise FileNotFoundError(f"Language annotations not found at {lang_path}")
    data = np.load(lang_path, allow_pickle=True).item()
    language = data.get("language", {})
    info = data.get("info", {})
    annotations = language.get("ann") or language.get("task") or []
    ranges = info.get("indx") or []

    by_episode: dict[int, str] = {}
    for idx, episode_range in enumerate(ranges):
        if idx >= len(annotations) or len(episode_range) != 2:
            continue
        start, end = int(episode_range[0]), int(episode_range[1])
        instruction = str(annotations[idx])
        for episode_number in range(start, end + 1):
            by_episode[episode_number] = instruction
    return by_episode


def _build_episode_file_list(data_dir: str) -> list[str]:
    """Return sorted list of episode_XXXXXXX.npz file paths."""
    files = sorted(
        f for f in os.listdir(data_dir)
        if f.startswith("episode_") and f.endswith(".npz")
    )
    if not files:
        raise FileNotFoundError(f"No episode_*.npz files found under {data_dir}")
    return [os.path.join(data_dir, f) for f in files]


def _episode_number_from_path(path: str) -> int | None:
    match = re.search(r"episode_(\d+)\.npz$", os.path.basename(path))
    return int(match.group(1)) if match else None


def image_to_hwc_uint8(value: Any) -> np.ndarray:
    """Convert an image-like value to HWC uint8 numpy array."""
    arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[-1] not in {1, 3}:
        arr = np.transpose(arr, (1, 2, 0))
    if np.issubdtype(arr.dtype, np.floating):
        if arr.max(initial=0) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    else:
        arr = arr.astype(np.uint8)
    return arr


def load_calvin_frame(
    dataset_root: str,
    split: str,
    dataset_index: int,
) -> dict[str, Any]:
    """Load a single CALVIN episode (timestep) as a frame dict.

    Args:
        dataset_root: Path to the CALVIN dataset directory (e.g. calvin_debug_dataset/).
        split: "training" or "validation".
        dataset_index: Index into the episode file list.

    Returns a dict with keys matching the dual-system benchmark convention:
        image, wrist_image, third_view_image, state, task
    """
    root = os.path.abspath(os.path.expanduser(dataset_root))
    data_dir = os.path.join(root, split)
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"CALVIN {split} directory not found: {data_dir}")

    episode_files = _build_episode_file_list(data_dir)
    if dataset_index < 0 or dataset_index >= len(episode_files):
        raise IndexError(
            f"CALVIN dataset index {dataset_index} out of range "
            f"[0, {len(episode_files)}) for split {split!r}"
        )

    episode_path = episode_files[dataset_index]
    ep = np.load(episode_path, allow_pickle=True)

    rgb_static = ep["rgb_static"]             # (200, 200, 3) uint8
    rgb_gripper = ep["rgb_gripper"]            # (84, 84, 3) uint8
    robot_obs = np.asarray(ep["robot_obs"], dtype=np.float32)  # (15,) float32

    # Load language annotation
    task = "perform a manipulation task"
    lang_dir = os.path.join(data_dir, "lang_annotations")
    if os.path.isdir(lang_dir):
        try:
            tasks = _load_lang_annotations(lang_dir)
            episode_number = _episode_number_from_path(episode_path)
            if episode_number is not None:
                task = tasks.get(episode_number, task)
        except Exception:
            pass

    return {
        "image": rgb_static,
        "wrist_image": rgb_gripper,
        "third_view_image": rgb_static,
        "depth_static": np.asarray(ep["depth_static"], dtype=np.float32),
        "depth_gripper": np.asarray(ep["depth_gripper"], dtype=np.float32),
        "state": robot_obs,
        "task": task,
    }
