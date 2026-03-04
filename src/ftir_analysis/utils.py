"""Utility helpers for reproducible FTIR pipelines."""

from __future__ import annotations

import hashlib
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def resolve_device(override: str | None = None) -> torch.device:
    """Return the best available device, or the requested override."""
    if override is not None:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int = 42) -> None:
    """Set deterministic seeds for Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def labels_to_log(y: torch.Tensor) -> torch.Tensor:
    """Map ppmv labels to log-space used by model training."""
    return torch.log1p(torch.clamp(y, min=0.0))


def labels_from_log(y_log: torch.Tensor) -> torch.Tensor:
    """Invert log-space outputs back to ppmv concentrations."""
    return torch.expm1(torch.clamp(y_log, min=0.0))


def now_utc_iso() -> str:
    """Return a timezone-aware UTC timestamp string."""
    return datetime.now(tz=timezone.utc).isoformat()


def sha1_file(path: Path) -> str:
    """Hash a file content using SHA1 for provenance metadata."""
    h = hashlib.sha1()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha1_text(text: str) -> str:
    """Hash a string using SHA1."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_sample_id(path: str, species: str, concentration_ppmv: float) -> str:
    """Generate a deterministic sample identifier."""
    payload = f"{path}|{species}|{concentration_ppmv:.12g}"
    return sha1_text(payload)


def ensure_dir(path: Path) -> None:
    """Create directory if needed."""
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON with consistent formatting."""
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read JSON object from disk."""
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def set_mpl_config_if_needed() -> None:
    """Avoid matplotlib cache warnings on non-writable $HOME configs."""
    if "MPLCONFIGDIR" in os.environ:
        return
    fallback = Path("/tmp") / "mplconfig_ftir"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback)
