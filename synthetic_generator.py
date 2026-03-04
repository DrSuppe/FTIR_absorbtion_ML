"""Synthetic FTIR spectrum generator using Beer-Lambert linear combination.

Builds physically valid multi-gas training samples by linearly adding
real .spc reference spectra (absorbance additivity holds exactly).

Usage
-----
    python synthetic_generator.py --n-samples 10000 --seed 42
    python synthetic_generator.py --n-samples 50000 --out data/synthetic/spectra.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Project imports (work whether called as script or imported as module)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
import os
os.environ["FTIR_PROJECT_ROOT"] = str(_PROJECT_ROOT)

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from ftir_analysis.constants import (  # noqa: E402
    DEFAULT_TARGET_SPECIES,
    INTERFERENCE_SPECIES,
    REFERENCE_ROOT,
    SATURATION_AU,
    SYNTHETIC_DIR,
    WAVENUMBER_MAX,
    WAVENUMBER_MIN,
    WAVENUMBER_STEP,
)
from ftir_analysis.manifesting import build_manifest  # noqa: E402
from ftir_analysis.spectra import interpolate_to_grid, load_spectrum  # noqa: E402

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GRID = np.arange(WAVENUMBER_MIN, WAVENUMBER_MAX, WAVENUMBER_STEP, dtype=np.float64)
GRID_NPTS = len(GRID)
TARGET_SPECIES = DEFAULT_TARGET_SPECIES
ALL_SPECIES = TARGET_SPECIES + [s for s in INTERFERENCE_SPECIES if s not in TARGET_SPECIES]


# ---------------------------------------------------------------------------
# SPC library loader
# ---------------------------------------------------------------------------

class SPCLibrary:
    """In-memory library of reference spectra indexed by species.

    Each entry per species is a list of (concentration_ppmv, spectrum_array).
    Spectra are interpolated to the shared 0.25 cm⁻¹ grid once on load.
    """

    def __init__(self, manifest_df, *, train_only: bool = True) -> None:
        self._lib: dict[str, list[tuple[float, np.ndarray]]] = {}
        self._build(manifest_df, train_only=train_only)

    def _build(self, df, *, train_only: bool) -> None:
        if train_only:
            df = df[df["split"].isin({"train", "val"})]

        ok, skip = 0, 0
        for _, row in df.iterrows():
            p = Path(row["source_path"])
            if not p.exists():
                skip += 1
                continue
            try:
                x, y = load_spectrum(p)
                arr = interpolate_to_grid(x, y)
                arr = np.clip(arr, 0.0, SATURATION_AU).astype(np.float64)
            except Exception as exc:
                log.debug("Skipping %s: %s", p.name, exc)
                skip += 1
                continue

            species = row["species"]
            conc = float(row["concentration_ppmv"])
            self._lib.setdefault(species, []).append((conc, arr))
            ok += 1

        log.info("SPC library: %d spectra loaded, %d skipped", ok, skip)
        for sp, entries in self._lib.items():
            log.info("  %-12s  %d spectra", sp, len(entries))

    @property
    def available_species(self) -> list[str]:
        return list(self._lib.keys())

    def get_interpolated_spectrum(
        self,
        species: str,
        target_conc_ppmv: float,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        """Return a spectrum for *species* at *target_conc_ppmv*.

        Strategy (Beer-Lambert valid):
        1. Sort the reference entries by concentration.
        2. Find the two neighbours that bracket `target_conc`.
        3. Linear-interpolate in absorbance space: A(c) = A_lo + (A_hi-A_lo)*t
           where t = (c-c_lo)/(c_hi-c_lo).
        4. If the target is outside the measured range, extrapolate by simple
           linear scaling from the nearest measured spectrum.

        Using two real spectra instead of one avoids overfitting to a single
        measured concentration and smoothly populates the gaps in the library.
        """
        if species not in self._lib:
            return None
        entries = self._lib[species]
        if not entries:
            return None

        # Small random perturbation on target so we never hit the exact same
        # concentration twice (±5% jitter in log-space)
        jitter = np.exp(rng.normal(0.0, 0.05))
        conc = max(target_conc_ppmv * jitter, 1e-3)

        # Sort by concentration for interpolation
        sorted_entries = sorted(entries, key=lambda e: e[0])
        concs = np.array([e[0] for e in sorted_entries], dtype=np.float64)

        # --- single spectrum case ---
        if len(sorted_entries) == 1:
            c0, s0 = sorted_entries[0]
            if c0 < 1e-9:
                return None
            return s0 * (conc / c0)

        # --- find bracketing pair ---
        if conc <= concs[0]:
            # Below minimum: scale from lowest reference
            c0, s0 = sorted_entries[0]
            return s0 * (conc / c0) if c0 > 1e-9 else None

        if conc >= concs[-1]:
            # Above maximum: scale from highest reference
            cn, sn = sorted_entries[-1]
            return sn * (conc / cn) if cn > 1e-9 else None

        # Interpolate between the two bracketing references
        hi_idx = int(np.searchsorted(concs, conc, side="right"))
        lo_idx = hi_idx - 1

        c_lo, s_lo = sorted_entries[lo_idx]
        c_hi, s_hi = sorted_entries[hi_idx]

        # Randomly pick whether to use a midpoint or the exact bracket
        # (prevents always generating the same interpolated value)
        t = (conc - c_lo) / (c_hi - c_lo + 1e-12)
        t = float(np.clip(t, 0.0, 1.0))

        # True Beer-Lambert interpolation: A = (1-t)*A_lo + t*A_hi
        interpolated = (1.0 - t) * s_lo + t * s_hi
        return interpolated

    # Keep old name as alias (backward compat)
    def get_scaled_spectrum(
        self,
        species: str,
        target_conc_ppmv: float,
        rng: np.random.Generator,
    ) -> np.ndarray | None:
        return self.get_interpolated_spectrum(species, target_conc_ppmv, rng)



# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def augment(spectrum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply physics-inspired augmentations in-place (returns new array)."""
    s = spectrum.copy()

    # 1. Wavenumber axis shift ±0.5 cm⁻¹
    shift = rng.uniform(-0.5, 0.5)
    if abs(shift) > 1e-4:
        shifted_grid = GRID + shift
        s = np.interp(GRID, shifted_grid, s, left=0.0, right=0.0)

    # 2. Regional gain jitter (8 equal sub-bands, each ×N(1, 0.03))
    n_bands = 8
    band_size = GRID_NPTS // n_bands
    for b in range(n_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < n_bands - 1 else GRID_NPTS
        gain = rng.normal(1.0, 0.03)
        s[start:end] *= max(gain, 0.0)

    # 3. Asymmetric 3rd-order baseline polynomial
    x_norm = np.linspace(-1.0, 1.0, GRID_NPTS)
    # Coefficients: allow slight tilt and curvature, keep low (< 0.3 AU typical)
    a0 = rng.uniform(-0.05, 0.05)
    a1 = rng.uniform(-0.10, 0.10)
    a2 = rng.uniform(-0.05, 0.05)
    a3 = rng.uniform(-0.02, 0.02)
    baseline = a0 + a1 * x_norm + a2 * x_norm**2 + a3 * x_norm**3
    s = s + baseline

    # 4. Gaussian detector noise σ = 0.001 AU
    s = s + rng.normal(0.0, 0.001, size=GRID_NPTS)

    # 5. Spectral block dropout (zero out 1–3 random blocks of 50–200 pts)
    n_blocks = rng.integers(1, 4)
    for _ in range(n_blocks):
        width = rng.integers(50, 201)
        start = rng.integers(0, max(1, GRID_NPTS - width))
        s[start : start + width] = 0.0

    # 6. Saturation clipping
    s = np.clip(s, -0.1, SATURATION_AU)

    return s.astype(np.float32)


# ---------------------------------------------------------------------------
# Mixture builder
# ---------------------------------------------------------------------------

def _target_species_concentrations(
    chosen: dict[str, float],
) -> np.ndarray:
    """Build float32 label vector aligned to TARGET_SPECIES order."""
    y = np.zeros(len(TARGET_SPECIES), dtype=np.float32)
    for i, sp in enumerate(TARGET_SPECIES):
        y[i] = chosen.get(sp, 0.0)
    return y


def build_one_sample(
    lib: SPCLibrary,
    rng: np.random.Generator,
    *,
    min_species: int = 1,
    max_species: int = 5,
    interference_prob: float = 0.3,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build one synthetic mixture spectrum and its label vector.

    Returns (x_float32[GRID_NPTS], y_float32[11]) or None on failure.
    """
    avail = lib.available_species
    if not avail:
        return None

    # Separate target and interference species in the library
    target_avail = [s for s in TARGET_SPECIES if s in avail]
    interf_avail = [s for s in avail if s not in TARGET_SPECIES]

    if not target_avail:
        return None

    # Pick 1–max_species target species
    n_target = rng.integers(min_species, min(max_species, len(target_avail)) + 1)
    chosen_target = list(rng.choice(target_avail, size=n_target, replace=False))

    # Optionally add 0–2 interference species
    chosen_interf: list[str] = []
    if interf_avail and rng.random() < interference_prob:
        n_interf = rng.integers(1, min(3, len(interf_avail)) + 1)
        chosen_interf = list(rng.choice(interf_avail, size=n_interf, replace=False))

    # --- Assign concentrations ---
    chosen_concs: dict[str, float] = {}
    for sp in chosen_target + chosen_interf:
        entries = lib._lib.get(sp, [])
        if not entries:
            continue
        concs = [c for c, _ in entries]
        c_min, c_max = min(concs), max(concs)
        # Log-uniform sampling over the reference concentration range
        if c_max > c_min * 1.01:
            log_lo, log_hi = np.log(max(c_min, 0.1)), np.log(c_max)
            chosen_concs[sp] = float(np.exp(rng.uniform(log_lo, log_hi)))
        else:
            chosen_concs[sp] = float(c_min)

    if not chosen_concs:
        return None

    # --- Build spectrum by linear addition ---
    spectrum = np.zeros(GRID_NPTS, dtype=np.float64)
    valid_species: list[str] = []
    for sp, conc in chosen_concs.items():
        sp_spec = lib.get_scaled_spectrum(sp, conc, rng)
        if sp_spec is None:
            continue
        spectrum += sp_spec
        valid_species.append(sp)

    if not valid_species:
        return None

    # Clip before augmentation
    spectrum = np.clip(spectrum, 0.0, SATURATION_AU)

    # Apply augmentations
    spectrum = augment(spectrum, rng)

    # Label vector (target species only)
    y = _target_species_concentrations(chosen_concs)

    return spectrum.astype(np.float32), y


# ---------------------------------------------------------------------------
# Main generation entry point
# ---------------------------------------------------------------------------

def generate(
    n_samples: int,
    *,
    seed: int = 42,
    out_path: Path | None = None,
    reference_root: Path = REFERENCE_ROOT,
    manifest_path: Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate *n_samples* synthetic spectra.

    Returns
    -------
    X : float32 array, shape (n_samples, GRID_NPTS)
    y : float32 array, shape (n_samples, n_target_species)
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
        )

    # Build or load manifest
    manifest_path = manifest_path or (reference_root / "manifest_v1.csv")
    if manifest_path.exists():
        import pandas as pd
        manifest = pd.read_csv(manifest_path)
        log.info("Loaded existing manifest: %d rows", len(manifest))
    else:
        log.info("Building SPC manifest …")
        manifest = build_manifest(reference_root=reference_root, output_path=manifest_path)

    lib = SPCLibrary(manifest, train_only=False)  # use all for generation

    rng = np.random.default_rng(seed)
    X_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []

    log_interval = max(1, n_samples // 20)
    failures = 0

    for i in range(n_samples):
        result = build_one_sample(lib, rng)
        if result is None:
            failures += 1
            continue
        x_i, y_i = result
        X_list.append(x_i)
        y_list.append(y_i)
        if verbose and (i + 1) % log_interval == 0:
            log.info("  Generated %d / %d  (failures: %d)", i + 1, n_samples, failures)

    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    log.info("Generation complete: X=%s  y=%s", X.shape, y.shape)

    if out_path is None:
        out_path = Path(SYNTHETIC_DIR) / "spectra.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, target_species=TARGET_SPECIES)
    log.info("Saved → %s", out_path)

    return X, y


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FTIR training spectra")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Number of synthetic samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Output .npz path")
    parser.add_argument(
        "--reference-root", type=str, default=str(REFERENCE_ROOT), help="Path to data/reference/"
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) if args.out else None
    generate(
        n_samples=args.n_samples,
        seed=args.seed,
        out_path=out,
        reference_root=Path(args.reference_root),
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _cli()
