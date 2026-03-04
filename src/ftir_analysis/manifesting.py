"""Manifest creation from SPC reference spectra.

Scans data/reference/spc_files/*.spc directly — no .lab files are used.
Parses species, concentration, temperature from the SPC filename.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_TARGET_SPECIES,
    INTERFERENCE_SPECIES,
    MANIFEST_COLUMNS,
    MANIFEST_FILENAME,
    PRIMARY_CLASS_THRESHOLD,
    REFERENCE_ROOT,
)
from .utils import seed_everything, stable_sample_id

# ---------------------------------------------------------------------------
# Filename / species normalisation
# ---------------------------------------------------------------------------

# Map common filename species tokens → canonical internal symbol
_SPECIES_ALIASES: dict[str, str] = {
    # Water
    "water": "H2O",
    "h2o": "H2O",
    # CO2
    "carbon dioxide": "CO2",
    "co2": "CO2",
    # CO
    "carbon monoxide": "CO",
    "co": "CO",
    # NO
    "nitric oxide": "NO",
    "no": "NO",
    # NO2
    "nitrogen dioxide": "NO2",
    "no2": "NO2",
    # NOx
    "noxp": "NOxP",
    # NH3
    "ammonia": "NH3",
    "nh3": "NH3",
    # CH4
    "methane": "CH4",
    "ch4": "CH4",
    # N2O
    "nitrous oxide": "N2O",
    "n2o": "N2O",
    # C2H4 / Ethylene
    "ethylene": "C2H4",
    "c2h4": "C2H4",
    # C2H6 / Ethane
    "ethane": "C2H6",
    "c2h6": "C2H6",
    # C2H6O / Ethanol
    "ethanol": "C2H6O",
    "c2h6o": "C2H6O",
    # HCN
    "hydrogen cyanide": "HCN",
    "hcn": "HCN",
    # HNCO
    "isocyanic acid": "HNCO",
    "hnco": "HNCO",
    # Urea — resolves to HNCO (thermal decomposition product; spectra nearly identical)
    "urea": "HNCO",
    # HNO3
    "nitric acid": "HNO3",
    "hno3": "HNO3",
    # H2SO4
    "sulfuric acid": "H2SO4",
    "h2so4": "H2SO4",
    # Biodiesel / Diesel (mixture proxies)
    "b-99": "Biodiesel",
    "biodiesel": "Biodiesel",
    "diesel": "Diesel",
    # BKG (background / calibration — skip)
    "bkg": "__SKIP__",
}

_ALL_KNOWN_SPECIES = set(DEFAULT_TARGET_SPECIES) | set(INTERFERENCE_SPECIES)


def _normalise_species(token: str) -> str | None:
    """Return canonical species symbol for a filename token, or None to skip."""
    t = token.strip().lower()
    # Try exact alias
    if t in _SPECIES_ALIASES:
        sym = _SPECIES_ALIASES[t]
        return None if sym == "__SKIP__" else sym
    # Try prefix match against aliases (handles things like "ch4%" → "ch4")
    for alias, sym in _SPECIES_ALIASES.items():
        if t.startswith(alias):
            return None if sym == "__SKIP__" else sym
    return None


# ---------------------------------------------------------------------------
# Concentration / temperature parsing helpers
# ---------------------------------------------------------------------------

# Robust number parser — handles tokens like "29p7", "1.5%", "39p7", etc.
_NUM_RE = re.compile(r"(-?\d+(?:[.,]\d*)?(?:[pP]\d+)?)")


def _parse_float_token(token: str) -> float | None:
    """Parse a numeric token that may use 'p' instead of '.' for fractional part."""
    cleaned = token.replace(",", ".").replace("p", ".").replace("P", ".")
    try:
        return float(cleaned)
    except ValueError:
        m = _NUM_RE.search(cleaned)
        if m:
            try:
                return float(m.group(1).replace("p", ".").replace("P", "."))
            except ValueError:
                pass
    return None


# Pattern to capture the content of the first parenthesised block:
#   "NH3 (300.14 ppm 191 c 0.5cm-1 MNB) 0.25mm bc -ice.spc"
#   → "300.14 ppm 191 c 0.5cm-1 MNB"
_PAREN_RE = re.compile(r"\(([^)]+)\)")

# Match something like "300.14 ppm" or "0.02 %" or "1.084 % adj"
_CONC_PPM_RE = re.compile(
    r"(-?\d+(?:[.,]\d*)?(?:[pP]\d+)?)\s*(?:ppm|ppmv)", re.IGNORECASE
)
_CONC_PCT_RE = re.compile(
    r"(-?\d+(?:[.,]\d*)?(?:[pP]\d+)?)\s*(?:%|percent)", re.IGNORECASE
)

# Temperature: "191 c", "191C", "190c"
_TEMP_RE = re.compile(r"(\d+)\s*c\b", re.IGNORECASE)


def _parse_conc_ppmv(paren_content: str) -> float | None:
    """Extract concentration in ppmv from the parenthesised block."""
    m = _CONC_PPM_RE.search(paren_content)
    if m:
        v = _parse_float_token(m.group(1))
        if v is not None:
            return v  # already ppmv
    # Try percent
    m = _CONC_PCT_RE.search(paren_content)
    if m:
        v = _parse_float_token(m.group(1))
        if v is not None:
            return v * 10_000.0  # % → ppmv (1 % = 10 000 ppm)
    # Bare leading number — treat as ppmv if small, % if > 1
    # e.g. "  20.00" in "Ammonia (  20.00ppm, ...)"
    tokens = paren_content.strip().split()
    if tokens:
        v = _parse_float_token(tokens[0])
        if v is not None:
            return v
    return None


def _parse_temp_c(paren_content: str) -> float:
    """Extract temperature in °C from the parenthesised block; default 191."""
    m = _TEMP_RE.search(paren_content)
    return float(m.group(1)) if m else 191.0


def _parse_spc_filename(stem: str) -> tuple[str, float, float] | None:
    """
    Parse `(species, concentration_ppmv, temperature_c)` from a SPC stem.

    Returns None if the file should be skipped (background, unrecognised).
    """
    # Extract everything before the first '(' as the species token
    paren_match = _PAREN_RE.search(stem)
    if not paren_match:
        return None

    species_token = stem[: paren_match.start()].strip()
    paren_content = paren_match.group(1)

    species = _normalise_species(species_token)
    if species is None:
        return None

    conc = _parse_conc_ppmv(paren_content)
    if conc is None or conc < 0:
        return None

    temp = _parse_temp_c(paren_content)
    return species, conc, temp


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _assign_split_labels(
    df: pd.DataFrame,
    *,
    seed: int = 42,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> pd.Series:
    """Deterministic, species-stratified split assignment."""
    rng = np.random.default_rng(seed)
    split = pd.Series(["staging"] * len(df), index=df.index, dtype="object")

    for _species, idxs in df.groupby("species").groups.items():
        idx_arr = np.array(list(idxs), dtype=int)
        rng.shuffle(idx_arr)
        n = len(idx_arr)
        if n == 0:
            continue

        n_train = max(1, round(n * train_frac))
        n_val = max(0, round(n * val_frac))
        # Adjust so we don't exceed n
        while n_train + n_val > n:
            if n_val > 0:
                n_val -= 1
            else:
                n_train -= 1
        n_test = n - n_train - n_val

        split.loc[idx_arr[:n_train]] = "train"
        split.loc[idx_arr[n_train : n_train + n_val]] = "val"
        split.loc[idx_arr[n_train + n_val : n_train + n_val + n_test]] = "test"

    return split


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_manifest(
    *,
    reference_root: Path = REFERENCE_ROOT,
    output_path: Path | None = None,
    seed: int = 42,
    primary_threshold: int = PRIMARY_CLASS_THRESHOLD,
    spc_dir: str = "spc_files",
) -> pd.DataFrame:
    """Build manifest_v1.csv by scanning data/reference/spc_files/*.spc.

    No .lab files, no lab_index.csv — SPC filenames are the sole metadata source.
    """
    reference_root = Path(reference_root)
    output_path = Path(output_path) if output_path else reference_root / MANIFEST_FILENAME
    spc_folder = reference_root / spc_dir

    if not spc_folder.exists():
        raise FileNotFoundError(f"SPC files directory not found: {spc_folder}")

    seed_everything(seed)

    rows: list[dict] = []
    for spc_path in sorted(spc_folder.glob("*.spc")):
        parsed = _parse_spc_filename(spc_path.stem)
        if parsed is None:
            continue
        species, conc_ppmv, temp_c = parsed

        # Path length: all MKS 2030 reference spectra use 5.11 m cell
        # A few H2SO4 entries use 5.60 m — try to detect from filename
        path_length_cm = 511.0
        if "5.60" in spc_path.stem or "5.6m" in spc_path.stem.lower():
            path_length_cm = 560.0

        sample_id = stable_sample_id(str(spc_path.resolve()), species, conc_ppmv)
        rows.append(
            {
                "sample_id": sample_id,
                "source_path": str(spc_path.resolve()),
                "source_format": "spc",
                "species": species,
                "concentration_ppmv": conc_ppmv,
                "temperature_c": temp_c,
                "path_length_cm": path_length_cm,
                "is_sparse_class": False,  # assigned below
                "split": "staging",
                "quality_flags": "",
            }
        )

    if not rows:
        raise RuntimeError(f"No usable SPC files found in {spc_folder}")

    manifest = pd.DataFrame(rows)

    # Mark sparse classes (fewer than primary_threshold samples)
    class_counts = manifest["species"].value_counts()
    sparse_species = set(class_counts[class_counts < primary_threshold].index)
    manifest["is_sparse_class"] = manifest["species"].isin(sparse_species)

    # Assign splits only for primary (non-sparse) species
    primary_mask = ~manifest["is_sparse_class"]
    split_series = _assign_split_labels(manifest.loc[primary_mask], seed=seed)
    manifest.loc[split_series.index, "split"] = split_series
    manifest.loc[~primary_mask, "split"] = "staging"

    # Stable ordering
    manifest = manifest.sort_values(["species", "source_path"]).reset_index(drop=True)
    manifest = manifest[MANIFEST_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest
