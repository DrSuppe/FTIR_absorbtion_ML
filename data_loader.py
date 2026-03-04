"""
data_loader.py — FTIR ML Solver v2
Parses MKS 2030 FTIR .spc files (Galactic Version M / 0x4D)
and wraps synthetic numpy tensors into a PyTorch Dataset.
"""

import os
import glob
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

# ── Instrument Ground Truth ────────────────────────────────────────────────────
WAVENUMBER_MIN   = 800.0    # Exclude noisy below-800 cm-1 region
WAVENUMBER_MAX   = 5000.0   # Instrument upper limit
WAVENUMBER_STEP  = 0.25     # Measured spacing in real .spc files
SATURATION_AU    = 8.0      # MKS 2030 hard absorbance ceiling
TARGET_SPECIES   = ['H2O', 'CO2', 'CO', 'NO', 'NO2', 'NH3', 'CH4']

# Reference grid every spectrum will be interpolated to
GRID = np.arange(WAVENUMBER_MIN, WAVENUMBER_MAX, WAVENUMBER_STEP, dtype=np.float32)
GRID_NPTS = len(GRID)       # ~16,800 points


# ── SPC Parser ─────────────────────────────────────────────────────────────────
def parse_mks_spc(filepath: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse MKS 2030 .spc file (Galactic Version M, fversn=0x4D).

    The main 512-byte header is followed immediately by a 32-byte sub-file header,
    then the Y array stored as signed int32 values scaled by 2^fexp / 2^32.

    Returns
    -------
    x : np.ndarray  – wavenumbers [cm-1]
    y : np.ndarray  – absorbance [AU], clipped to [-0.1, SATURATION_AU]
    """
    with open(filepath, 'rb') as fh:
        raw = fh.read()

    # --- Header decode ---
    fexp  = struct.unpack('<b',  raw[3:4])[0]       # signed byte
    npts  = struct.unpack('<i',  raw[4:8])[0]       # int32 — often garbage in Vers-M

    # For Version M the real ffirst/flast are stored as 32-bit floats at offset 8/12
    ffirst = struct.unpack('<f', raw[8:12])[0]
    flast  = struct.unpack('<f', raw[12:16])[0]

    # Y data starts after 512-byte main header + 32-byte subfile header = offset 544
    Y_OFFSET = 544
    n_available = (len(raw) - Y_OFFSET) // 4

    if abs(fexp) < 64 and fexp != -128:
        # Integer-scaled path (standard for Version M)
        scale = (2.0 ** fexp) / (2 ** 32)
        y_raw = np.frombuffer(raw[Y_OFFSET:Y_OFFSET + n_available * 4], dtype=np.int32)
        y = y_raw.astype(np.float64) * scale
    else:
        # Fallback: treat as float32
        y = np.frombuffer(raw[Y_OFFSET:Y_OFFSET + n_available * 4], dtype=np.float32).astype(np.float64)

    n = len(y)
    x = np.linspace(ffirst, flast, n, dtype=np.float32)

    # --- Trim below-800 cm-1 noise ---
    mask = x >= WAVENUMBER_MIN
    x, y = x[mask], y[mask].astype(np.float32)

    # --- Clip to sensor saturation ---
    y = np.clip(y, -0.1, SATURATION_AU).astype(np.float32)

    return x, y


def interpolate_to_grid(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Resample any spectrum onto the fixed reference grid via linear interpolation."""
    return np.interp(GRID, x, y).astype(np.float32)


# ── PyTorch Datasets ───────────────────────────────────────────────────────────
class SPCSequenceDataset(Dataset):
    """
    Dataset backed by real .spc files, emitting single spectra on the fixed grid.
    Hidden-state continuity is managed externally by inference.py.
    """
    def __init__(self, data_dir: str):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.spc')))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        x, y = parse_mks_spc(self.files[idx])
        y_grid = interpolate_to_grid(x, y)
        return torch.from_numpy(y_grid), os.path.basename(self.files[idx])


class SyntheticFTIRDataset(Dataset):
    """
    Dataset backed by pre-generated synthetic tensors (X.npy / y.npy).
    Wraps each spectrum as (1, GRID_NPTS) channels-first for the 1D CNN.
    """
    def __init__(self, X_path: str, y_path: str):
        self.X = torch.from_numpy(np.load(X_path)).float()   # (N, GRID_NPTS)
        self.y = torch.from_numpy(np.load(y_path)).float()   # (N, 7)
        assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Add channel dim: (1, GRID_NPTS)
        return self.X[idx].unsqueeze(0), self.y[idx]
