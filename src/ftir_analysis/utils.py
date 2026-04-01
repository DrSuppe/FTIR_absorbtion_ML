"""Utility helpers for reproducible FTIR pipelines."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)


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


class LabelNormalizer:
    """Per-species zero-mean / unit-std normalizer fitted on active (nonzero) training labels.

    Labels are assumed to already be in log1p(ppmv) space when passed to fit().
    Normalization equalizes regression difficulty across species that span very
    different concentration ranges (e.g. H2O ~10000 ppmv vs HCN ~5 ppmv).

    Inactive labels (zero in log space) are excluded from stats computation and
    are mapped to -mean/std after normalization. This is the unique value that
    denormalizes back to 0, and it sits far below the active N(0,1) distribution
    (typically -10 to -20σ), giving the model a clear gradient to learn presence/absence.

    Also computes per-species loss weights: species with lower typical concentrations
    receive higher weight (1 / log1p(median_active_ppmv)) so trace species are not
    swamped by major species in the loss.
    """

    def __init__(
        self,
        means: np.ndarray,
        stds: np.ndarray,
        species_weights: np.ndarray,
    ) -> None:
        self.means = np.asarray(means, dtype=np.float32)
        self.stds = np.asarray(stds, dtype=np.float32)
        self.species_weights = np.asarray(species_weights, dtype=np.float32)
        self.n_species = len(self.means)

    @classmethod
    def fit(cls, y_log: np.ndarray) -> "LabelNormalizer":
        """Fit from a (N, n_species) array of log1p(ppmv) training labels."""
        n_species = y_log.shape[1]
        means = np.zeros(n_species, dtype=np.float32)
        stds = np.ones(n_species, dtype=np.float32)
        raw_weights = np.ones(n_species, dtype=np.float32)

        for i in range(n_species):
            active = y_log[:, i][y_log[:, i] > 0]
            if active.size == 0:
                log.warning("Species index %d has no active training samples — normalizer will use identity.", i)
                continue
            means[i] = float(active.mean())
            stds[i] = float(max(active.std(), 1e-4))

            # Weight = 1 / log1p(median concentration in ppmv)
            median_ppmv = float(np.expm1(np.median(active)))
            raw_weights[i] = 1.0 / max(np.log1p(median_ppmv), 1e-4)

        # Normalise weights so their mean is 1 (keeps overall loss scale stable)
        species_weights = raw_weights / max(float(raw_weights.mean()), 1e-8)
        return cls(means=means, stds=stds, species_weights=species_weights)

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        """Map log1p(ppmv) labels → normalised space.

        Inactive (zero) entries are mapped to -mean/std — the unique value
        that denormalizes back to 0. This keeps inactive species clearly
        separated from active species (which land in ~N(0,1)), giving the
        model a distinguishable gradient for presence/absence.

        Previously inactive entries were mapped to 0.0, which collides with
        the active-at-median distribution and caused the model to learn a
        trivial solution (predict everything at the mean).
        """
        means_t = torch.tensor(self.means, dtype=y.dtype, device=y.device)
        stds_t = torch.tensor(self.stds, dtype=y.dtype, device=y.device)
        active = y > 0
        inactive_sentinel = -means_t / stds_t          # denorm of this → 0
        y_norm = torch.where(active, (y - means_t) / stds_t, inactive_sentinel)
        return y_norm

    def denormalize(self, y_norm: torch.Tensor) -> torch.Tensor:
        """Map normalised predictions back to log1p(ppmv) space."""
        means_t = torch.tensor(self.means, dtype=y_norm.dtype, device=y_norm.device)
        stds_t = torch.tensor(self.stds, dtype=y_norm.dtype, device=y_norm.device)
        return y_norm * stds_t + means_t

    def as_dict(self) -> dict[str, Any]:
        return {
            "means": self.means.tolist(),
            "stds": self.stds.tolist(),
            "species_weights": self.species_weights.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LabelNormalizer":
        return cls(
            means=np.array(d["means"], dtype=np.float32),
            stds=np.array(d["stds"], dtype=np.float32),
            species_weights=np.array(d["species_weights"], dtype=np.float32),
        )


def set_mpl_config_if_needed() -> None:
    """Avoid matplotlib cache warnings on non-writable $HOME configs."""
    if "MPLCONFIGDIR" in os.environ:
        return
    fallback = Path("/tmp") / "mplconfig_ftir"
    fallback.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(fallback)
