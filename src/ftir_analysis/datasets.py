"""Dataset classes for FTIR training data.

Two data sources:
1. Synthetic spectra (pre-generated .npz): multi-gas, 11-species label vectors.
2. Reference spectra (from manifest): single-gas, zero-padded to 11-species.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .constants import DEFAULT_TARGET_SPECIES
from .spectra import SpectrumLoadError, interpolate_to_grid, load_spectrum

log = logging.getLogger(__name__)

TARGET_SPECIES = DEFAULT_TARGET_SPECIES  # 11 species


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _species_index(species: str) -> int | None:
    """Return index of *species* in TARGET_SPECIES, or None if not a target."""
    try:
        return TARGET_SPECIES.index(species)
    except ValueError:
        return None


def validate_target_species(user_species: Sequence[str]) -> None:
    """Raise ValueError if any requested species are not in TARGET_SPECIES."""
    unknown = [s for s in user_species if s not in TARGET_SPECIES]
    if unknown:
        raise ValueError(
            f"Unknown target species: {unknown}. "
            f"Valid choices: {TARGET_SPECIES}"
        )


# ---------------------------------------------------------------------------
# Synthetic dataset (from pre-generated .npz)
# ---------------------------------------------------------------------------

class ArraySpectrumDataset(Dataset):
    """Dataset backed by numpy arrays (X, y) from a synthetic .npz file."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        log_transform: bool = True,
    ) -> None:
        assert X.shape[0] == y.shape[0]
        assert y.shape[1] == len(TARGET_SPECIES), (
            f"Label dim {y.shape[1]} != {len(TARGET_SPECIES)} target species"
        )
        self.X = X  # Keep as native numpy float16 array to save RAM
        if log_transform:
            self.y = torch.log1p(torch.from_numpy(y.astype(np.float32)).clamp(min=0.0))
        else:
            self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        # Cast to float32 tensor only upon batch loading to minimize peak memory
        return torch.tensor(self.X[idx], dtype=torch.float32), self.y[idx]


def load_synthetic_aux_arrays(
    npz_path: Path | str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Load pre-generated synthetic spectra from a .npz file.

    Returns (X, y) arrays or None if the file doesn't exist.
    """
    if npz_path is None:
        from .constants import SYNTHETIC_DIR
        npz_path = Path(SYNTHETIC_DIR) / "spectra.npz"
    npz_path = Path(npz_path)
    if not npz_path.exists():
        log.warning("Synthetic data not found at %s", npz_path)
        return None
    data = np.load(npz_path, allow_pickle=False)
    X = data["X"]  # Don't .astype(np.float32) here! Keep it float16
    y = data["y"].astype(np.float32)
    # Validate label dimension
    if y.shape[1] != len(TARGET_SPECIES):
        # Saved file may be from old 7-species model; reject gracefully
        log.warning(
            "Synthetic .npz has %d label columns, expected %d. Ignoring file.",
            y.shape[1],
            len(TARGET_SPECIES),
        )
        return None
    log.info("Loaded synthetic data: X=%s y=%s from %s", X.shape, y.shape, npz_path)
    return X, y


# ---------------------------------------------------------------------------
# Reference spectra dataset (from manifest)
# ---------------------------------------------------------------------------

class ReferenceSpectraDataset(Dataset):
    """Dataset built from manifest .spc reference spectra.

    Each sample is a single-species measurement. The label vector is zero-
    padded to len(TARGET_SPECIES), with the species column set to its
    concentration in ppmv (log1p transformed).
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        *,
        splits: Sequence[str] = ("train",),
        log_transform: bool = True,
        min_concentration_ppmv: float = 0.0,
    ) -> None:
        self.log_transform = log_transform

        # Filter to requested splits and loadable rows
        df = manifest[manifest["split"].isin(splits)].copy()
        df = df[df["source_format"] == "spc"]

        # Only include species that map to our target head
        df["species_idx"] = df["species"].apply(_species_index)
        df = df[df["species_idx"].notna()].copy()
        df = df[df["concentration_ppmv"] >= min_concentration_ppmv]

        self._records = df[
            ["source_path", "species_idx", "concentration_ppmv"]
        ].reset_index(drop=True)

        # Pre-load all spectra into memory
        self._X: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        self._load_all()

    def _load_all(self) -> None:
        ok, skip = 0, 0
        for _, row in self._records.iterrows():
            try:
                x_raw, y_raw = load_spectrum(row["source_path"])
                arr = interpolate_to_grid(x_raw, y_raw)
            except (SpectrumLoadError, Exception) as e:
                log.debug("Skip %s: %s", row["source_path"], e)
                skip += 1
                continue

            # Build label vector
            label = np.zeros(len(TARGET_SPECIES), dtype=np.float32)
            label[int(row["species_idx"])] = float(row["concentration_ppmv"])

            self._X.append(arr.astype(np.float32))
            self._y.append(label)
            ok += 1

        log.info(
            "ReferenceSpectraDataset: %d loaded (%d target species), %d skipped",
            ok,
            len(self._records["species_idx"].dropna().unique()),
            skip,
        )

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self._X[idx])
        y = torch.from_numpy(self._y[idx])
        if self.log_transform:
            y = torch.log1p(y.clamp(min=0.0))
        return x, y


# ---------------------------------------------------------------------------
# Legacy helper (kept for backward compat with older training.py signatures)
# ---------------------------------------------------------------------------

def build_reference_arrays(
    manifest: pd.DataFrame,
    target_species: Optional[List[str]] = None,
    splits: Sequence[str] = ("train",),
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    """Return (X, y) numpy arrays from reference spectra manifest.

    Deprecated in favour of ReferenceSpectraDataset, but kept so that
    old training code doesn't break.
    """
    if target_species:
        validate_target_species(target_species)
    ds = ReferenceSpectraDataset(manifest, splits=splits, log_transform=False)
    if len(ds) == 0:
        return None, None
    X = np.stack(ds._X)
    y = np.stack(ds._y)
    return X, y
