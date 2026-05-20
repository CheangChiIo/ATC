from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"

if PROJECT_HF_HOME.exists():
    os.environ.setdefault("HF_HOME", str(PROJECT_HF_HOME))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(PROJECT_HF_HOME / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_HF_HOME / "hub"))
