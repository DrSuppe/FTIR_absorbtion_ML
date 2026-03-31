"""Spectrum IO and preprocessing utilities."""

from __future__ import annotations

import csv
import struct
from pathlib import Path
from typing import Tuple

import numpy as np

from .constants import GRID as _GRID_F64, GRID_NPTS, SATURATION_AU, WAVENUMBER_MIN, WAVENUMBER_MAX, WAVENUMBER_STEP

GRID = _GRID_F64.astype(np.float32)


class SpectrumLoadError(RuntimeError):
    """Raised when a spectrum cannot be loaded or normalized."""


def parse_mks_spc(filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse an MKS Version-M SPC file into native x/y arrays."""
    path = Path(filepath)
    with path.open("rb") as fh:
        raw = fh.read()

    if len(raw) < 548:
        raise SpectrumLoadError(f"SPC file is too small: {path}")

    fexp = struct.unpack("<b", raw[3:4])[0]
    ffirst = struct.unpack("<f", raw[8:12])[0]
    flast = struct.unpack("<f", raw[12:16])[0]

    y_offset = 544
    n_available = (len(raw) - y_offset) // 4
    if n_available <= 0:
        raise SpectrumLoadError(f"No data payload found in SPC: {path}")

    if abs(fexp) < 64 and fexp != -128:
        scale = (2.0 ** fexp) / (2.0 ** 32)
        y_raw = np.frombuffer(raw[y_offset : y_offset + n_available * 4], dtype=np.int32)
        y = y_raw.astype(np.float64) * scale
    else:
        y = np.frombuffer(raw[y_offset : y_offset + n_available * 4], dtype=np.float32).astype(np.float64)

    if y.size == 0:
        raise SpectrumLoadError(f"Decoded zero points from SPC: {path}")

    x = np.linspace(ffirst, flast, y.size, dtype=np.float32)

    mask = x >= WAVENUMBER_MIN
    x = x[mask]
    y = y[mask].astype(np.float32)
    y = np.clip(y, -0.1, SATURATION_AU).astype(np.float32)

    if x.size == 0 or y.size == 0:
        raise SpectrumLoadError(f"No usable points after filtering in SPC: {path}")

    return x, y


def _try_parse_float(token: str) -> float | None:
    try:
        return float(token)
    except Exception:
        return None


def parse_csv_spectrum(filepath: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse exported CSV/TXT spectra into x/y arrays.

    Supported layouts:
    - 2-column numeric data: wavenumber, absorbance
    - 1-column numeric data: absorbance already on target grid
    """
    path = Path(filepath)

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
                continue
            vals = [_try_parse_float(tok.strip()) for tok in row[:2]]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            rows.append(vals)

    if not rows:
        raise SpectrumLoadError(f"No numeric rows found in CSV/TXT: {path}")

    max_cols = max(len(r) for r in rows)
    if max_cols >= 2:
        x = np.array([r[0] for r in rows if len(r) >= 2], dtype=np.float32)
        y = np.array([r[1] for r in rows if len(r) >= 2], dtype=np.float32)
    else:
        y = np.array([r[0] for r in rows], dtype=np.float32)
        x = GRID.copy()

    if x.size != y.size:
        raise SpectrumLoadError(f"CSV/TXT x/y length mismatch in {path}")

    if x.size == GRID_NPTS:
        x0, x1 = float(x[0]), float(x[-1])
        # If grid appears descending, fix orientation for interpolation.
        if x1 < x0:
            x = x[::-1]
            y = y[::-1]

    y = np.clip(y, -0.1, SATURATION_AU).astype(np.float32)
    return x.astype(np.float32), y


def interpolate_to_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate any valid spectrum onto the fixed training grid."""
    if x.size < 2:
        raise SpectrumLoadError("Need at least 2 x-points for interpolation")

    # Ensure monotonic increasing x for np.interp.
    if np.any(np.diff(x) < 0):
        order = np.argsort(x)
        x = x[order]
        y = y[order]

    y_grid = np.interp(GRID, x, y).astype(np.float32)
    if y_grid.size != GRID_NPTS:
        raise SpectrumLoadError(f"Interpolated spectrum has wrong length: {y_grid.size}")
    return y_grid


def load_spectrum(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a spectrum from supported source formats.

    v1 policy:
    - `.spc` is supported directly.
    - `.csv`/`.txt` exported spectra are supported directly.
    - `.lab` is intentionally unsupported and must be exported first.
    """
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".spc":
        return parse_mks_spc(p)

    if suffix in {".csv", ".txt"}:
        return parse_csv_spectrum(p)

    if suffix == ".lab":
        raise SpectrumLoadError(
            f"LAB input is unsupported in v1 ({p}). Export to .spc or .csv first."
        )

    raise SpectrumLoadError(f"Unsupported spectrum format: {p.suffix} ({p})")


def load_on_grid(path: str | Path) -> np.ndarray:
    """Load spectrum and return 1D absorbance over fixed grid."""
    x, y = load_spectrum(path)
    return interpolate_to_grid(x, y)
