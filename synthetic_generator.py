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
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

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
    GRID,
    GRID_NPTS,
    INTERFERENCE_SPECIES,
    REFERENCE_ROOT,
    SATURATION_AU,
    SYNTHETIC_DIR,
    WAVENUMBER_MAX,
    WAVENUMBER_MIN,
    WAVENUMBER_STEP,
)
from ftir_analysis.features import smooth_saturate  # noqa: E402
from ftir_analysis.manifesting import build_manifest  # noqa: E402
from ftir_analysis.spectra import interpolate_to_grid, load_spectrum  # noqa: E402

log = logging.getLogger(__name__)
TARGET_SPECIES = DEFAULT_TARGET_SPECIES
DEFAULT_MAJOR_SPECIES = ["H2O", "CO2", "CO", "NO", "NO2", "NH3"]
DEFAULT_TRACE_SPECIES = ["CH4", "N2O", "C2H4", "HCN", "HNCO"]
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
            raw_p = Path(row["source_path"])
            if raw_p.is_absolute():
                p = raw_p
            else:
                p = REFERENCE_ROOT / raw_p

            if not p.exists():
                log.warning("Spectrum file not found: %s (resolved to %s)", row["source_path"], p)
                skip += 1
                continue
            try:
                x, y = load_spectrum(p)
                arr = interpolate_to_grid(x, y)
                arr = np.clip(arr, 0.0, None).astype(np.float64)
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

def _temperature_perturbation(
    spectrum: np.ndarray,
    rng: np.random.Generator,
    *,
    temperature_c: float = 25.0,
    reference_temp_c: float = 25.0,
) -> np.ndarray:
    """Apply temperature-dependent spectral perturbation.

    Real FTIR absorption cross-sections change with temperature due to
    rotational-vibrational population shifts (Boltzmann distribution).
    This approximates the effect as:
      1. A global intensity scaling proportional to ΔT (hotter → weaker absorption
         for most species in the mid-IR, roughly -0.2%/°C).
      2. A line-broadening effect simulated by Gaussian convolution whose width
         scales with |ΔT|.
      3. Small random band-specific perturbations that mimic the fact that
         different vibrational modes have different temperature sensitivities.

    The reference spectra in the SPC library are typically recorded at ~25°C.
    """
    dt = temperature_c - reference_temp_c
    if abs(dt) < 0.5:
        return spectrum.copy()

    s = spectrum.copy()

    # 1. Global intensity scaling (~-0.2%/°C with ±0.05%/°C jitter)
    coeff = -0.002 + rng.normal(0.0, 0.0005)
    s *= (1.0 + coeff * dt)

    # 2. Line broadening — convolve with a narrow Gaussian whose σ scales with |ΔT|
    sigma_pts = max(1.0, 0.15 * abs(dt))  # in grid points (~0.25 cm⁻¹ each)
    if sigma_pts > 1.0:
        from scipy.ndimage import gaussian_filter1d
        s = gaussian_filter1d(s, sigma=sigma_pts).astype(np.float32)

    # 3. Band-specific perturbation (4 random bands with ΔT-proportional gain shifts)
    n_bands = 4
    band_size = len(s) // n_bands
    for b in range(n_bands):
        start = b * band_size
        end = (b + 1) * band_size if b < n_bands - 1 else len(s)
        band_shift = rng.normal(0.0, 0.001 * abs(dt))
        s[start:end] *= (1.0 + band_shift)

    return s.astype(np.float32)


def augment(
    spectrum: np.ndarray,
    rng: np.random.Generator,
    *,
    profile: str = "mild",
) -> np.ndarray:
    """Apply physics-inspired augmentations in-place (returns new array)."""
    s = spectrum.copy()

    profile = profile.lower().strip()
    if profile not in {"mild", "strong"}:
        raise ValueError(f"Unsupported augmentation profile: {profile}")

    if profile == "strong":
        shift = rng.uniform(-0.5, 0.5)
        if abs(shift) > 1e-4:
            shifted_grid = GRID + shift
            s = np.interp(GRID, shifted_grid, s, left=0.0, right=0.0)

        n_bands = 8
        band_size = GRID_NPTS // n_bands
        for b in range(n_bands):
            start = b * band_size
            end = (b + 1) * band_size if b < n_bands - 1 else GRID_NPTS
            gain = rng.normal(1.0, 0.03)
            s[start:end] *= max(gain, 0.0)

    x_norm = np.linspace(-1.0, 1.0, GRID_NPTS)
    if profile == "strong":
        a0 = rng.uniform(-0.05, 0.05)
        a1 = rng.uniform(-0.10, 0.10)
        a2 = rng.uniform(-0.05, 0.05)
        a3 = rng.uniform(-0.02, 0.02)
        noise_sigma = 0.001
    else:
        a0 = rng.uniform(-0.01, 0.01)
        a1 = rng.uniform(-0.02, 0.02)
        a2 = rng.uniform(-0.01, 0.01)
        a3 = rng.uniform(-0.005, 0.005)
        noise_sigma = 2e-4
    baseline = a0 + a1 * x_norm + a2 * x_norm**2 + a3 * x_norm**3
    s = s + baseline

    s = s + rng.normal(0.0, noise_sigma, size=GRID_NPTS)

    if profile == "strong":
        n_blocks = rng.integers(1, 4)
        for _ in range(n_blocks):
            width = rng.integers(50, 201)
            start = rng.integers(0, max(1, GRID_NPTS - width))
            s[start : start + width] = 0.0

    s = np.clip(s, -0.1, None)  # floor only; ceiling via smooth saturation
    s = smooth_saturate(s, ceiling=SATURATION_AU)

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


def _weighted_choice_without_replacement(
    rng: np.random.Generator,
    items: list[str],
    weights: list[float],
    k: int,
) -> list[str]:
    """Sample k unique items using positive weights."""
    if k <= 0 or not items:
        return []
    if k >= len(items):
        return list(items)
    probs = np.array(weights, dtype=np.float64)
    probs = np.clip(probs, 1e-12, None)
    probs = probs / probs.sum()
    idx = rng.choice(len(items), size=k, replace=False, p=probs)
    return [items[int(i)] for i in idx]


def _latin_hypercube(
    rng: np.random.Generator,
    n_samples: int,
    n_dims: int,
) -> np.ndarray:
    """Simple LHS over [0,1]^n_dims."""
    if n_samples <= 0 or n_dims <= 0:
        return np.zeros((0, 0), dtype=np.float64)
    out = np.zeros((n_samples, n_dims), dtype=np.float64)
    for d in range(n_dims):
        bins = (np.arange(n_samples, dtype=np.float64) + rng.random(n_samples)) / n_samples
        rng.shuffle(bins)
        out[:, d] = bins
    return out


def _species_stats_from_library(
    lib: SPCLibrary,
) -> dict[str, dict[str, float]]:
    """Build per-species concentration stats from loaded references."""
    stats: dict[str, dict[str, float]] = {}
    for sp, entries in lib._lib.items():
        if not entries:
            continue
        concs = np.array([float(c) for c, _ in entries], dtype=np.float64)
        concs = concs[concs > 0]
        if concs.size == 0:
            continue
        stats[sp] = {
            "min_positive": float(np.min(concs)),
            "p90": float(np.percentile(concs, 90)),
            "p95": float(np.percentile(concs, 95)),
            "max": float(np.max(concs)),
        }
    return stats


def _log_bin_index(value: float, lo: float, hi: float) -> int:
    """Map concentration to low/med/high bin in log space."""
    lo_eff = max(lo, 1e-6)
    hi_eff = max(hi, lo_eff * 1.0001)
    lv = np.log(max(value, lo_eff))
    l0 = np.log(lo_eff)
    l1 = np.log(hi_eff)
    u = 0.0 if l1 <= l0 else (lv - l0) / (l1 - l0)
    u = float(np.clip(u, 0.0, 1.0 - 1e-12))
    return int(u * 3.0)


def _sample_log_concentration(
    rng: np.random.Generator,
    *,
    lo: float,
    hi: float,
    force_bin: int | None = None,
) -> float:
    """Sample concentration log-uniformly, optionally constrained to a 3-bin range."""
    lo_eff = max(lo, 1e-6)
    hi_eff = max(hi, lo_eff * 1.0001)
    l0 = np.log(lo_eff)
    l1 = np.log(hi_eff)
    if force_bin is not None:
        b = int(np.clip(force_bin, 0, 2))
        seg = (l1 - l0) / 3.0
        l0 = l0 + b * seg
        l1 = l0 + seg
    return float(np.exp(rng.uniform(l0, l1)))


def _sample_temperature(rng: np.random.Generator) -> float:
    """Sample a measurement temperature from a realistic distribution.

    FTIR measurements in emission testing typically span 15-200°C,
    with most near 25°C (lab) or 50-150°C (exhaust stack).
    We use a mixture: 60% near-ambient, 40% elevated.
    """
    if rng.random() < 0.6:
        return float(rng.normal(25.0, 10.0))   # lab-ambient range
    return float(rng.uniform(40.0, 200.0))      # elevated exhaust range


def _build_sample_from_concentrations(
    lib: SPCLibrary,
    rng: np.random.Generator,
    chosen_concs: dict[str, float],
    *,
    augment_profile: str = "mild",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build one spectrum from preselected species concentrations."""
    if not chosen_concs:
        return None

    spectrum = np.zeros(GRID_NPTS, dtype=np.float64)
    valid_species = 0
    for sp, conc in chosen_concs.items():
        sp_spec = lib.get_scaled_spectrum(sp, conc, rng)
        if sp_spec is None:
            continue
        spectrum += sp_spec
        valid_species += 1

    if valid_species == 0:
        return None

    spectrum = np.clip(spectrum, 0.0, None)  # floor only; augment applies smooth saturation
    spectrum = augment(spectrum, rng, profile=augment_profile)
    y = _target_species_concentrations(chosen_concs)
    return spectrum.astype(np.float32), y


class CurriculumSamplerV2:
    """Two-stage sampler with major-species bias and LHS coverage."""

    def __init__(
        self,
        *,
        lib: SPCLibrary,
        rng: np.random.Generator,
        n_samples: int,
        major_species: list[str],
        stage1_cap_policy: str,
        stage1_frac: float,
        lhs_frac: float,
        min_active_species: int,
        max_active_species: int,
    ) -> None:
        self.lib = lib
        self.rng = rng
        self.n_samples = max(1, int(n_samples))
        self.stage1_cap_policy = stage1_cap_policy
        self.min_active_species = max(1, int(min_active_species))
        self.max_active_species = max(self.min_active_species, int(max_active_species))
        self.stats = _species_stats_from_library(lib)

        target_avail = [s for s in TARGET_SPECIES if s in lib.available_species and s in self.stats]
        self.target_avail = target_avail
        self.interf_avail = [s for s in lib.available_species if s not in TARGET_SPECIES and s in self.stats]
        self.major_species = [s for s in major_species if s in self.target_avail]
        self.minor_species = [s for s in self.target_avail if s not in self.major_species]
        if not self.major_species:
            self.major_species = [s for s in self.target_avail]

        # Normalize stage weights so both knobs can be provided safely.
        stage1_w = max(float(stage1_frac), 0.0)
        stage2_w = max(float(lhs_frac), 0.0)
        total = stage1_w + stage2_w
        if total <= 0:
            stage1_w, stage2_w = 0.7, 0.3
            total = 1.0
        stage2_ratio = stage2_w / total
        self.stage2_n = int(round(self.n_samples * stage2_ratio))
        self.stage2_n = min(max(self.stage2_n, 0), self.n_samples)
        self.stage1_n = self.n_samples - self.stage2_n

        self.major_presence_counts = {s: 0 for s in self.major_species}
        self.major_presence_target = 0.65
        self.major_bin_counts = {s: np.zeros(3, dtype=np.int32) for s in self.major_species}
        self.stage_bin_counts = {
            "stage1": {s: np.zeros(3, dtype=np.int32) for s in self.major_species},
            "stage2": {s: np.zeros(3, dtype=np.int32) for s in self.major_species},
        }
        self.stage1_cap_violations = 0

        self._lhs = _latin_hypercube(self.rng, self.stage2_n, len(self.target_avail))
        self._stage2_cursor = 0

    def stage_for_index(self, idx: int) -> str:
        return "stage1" if idx < self.stage1_n else "stage2"

    def _stage1_cap(self, species: str) -> float:
        st = self.stats[species]
        if self.stage1_cap_policy == "p90":
            return st["p90"]
        if self.stage1_cap_policy == "max":
            return st["max"]
        return st["p95"]

    def _sample_species_stage1(self, n_active: int, sample_idx: int) -> list[str]:
        items = self.target_avail
        weights: list[float] = []
        for sp in items:
            base = 4.0 if sp in self.major_species else 1.0
            if sp in self.major_species:
                expected = self.major_presence_target * (sample_idx + 1)
                deficit = max(0.0, expected - self.major_presence_counts[sp])
                base = base * (1.0 + min(deficit, 5.0))
            weights.append(base)
        return _weighted_choice_without_replacement(self.rng, items, weights, n_active)

    def _sample_species_stage2(self, n_active: int) -> tuple[list[str], int]:
        if self.stage2_n <= 0:
            return [], -1
        row_idx = min(self._stage2_cursor, self.stage2_n - 1)
        row = self._lhs[row_idx]
        self._stage2_cursor += 1

        scored: list[tuple[float, str]] = []
        for j, sp in enumerate(self.target_avail):
            score = float(row[j])
            if sp in self.major_species:
                score += 0.35
            scored.append((score, sp))
        scored.sort(reverse=True)
        chosen = [sp for _, sp in scored[:n_active]]
        return chosen, row_idx

    def _sample_target_concentration(self, species: str, stage: str, stage2_u: float | None) -> float:
        st = self.stats[species]
        lo = max(st["min_positive"], 1e-3)
        hi = self._stage1_cap(species) if stage == "stage1" else st["max"]
        hi = max(hi, lo * 1.0001)

        if species in self.major_species:
            if stage == "stage1":
                counts = self.major_bin_counts[species]
                inv = 1.0 / np.maximum(counts.astype(np.float64), 1.0)
                probs = inv / inv.sum()
                forced_bin = int(self.rng.choice([0, 1, 2], p=probs))
                conc = _sample_log_concentration(self.rng, lo=lo, hi=hi, force_bin=forced_bin)
            else:
                # LHS stage: map u in [0,1) to log concentration directly for coverage.
                u = float(np.clip(stage2_u if stage2_u is not None else self.rng.random(), 0.0, 0.999999))
                l0 = np.log(lo)
                l1 = np.log(hi)
                conc = float(np.exp(l0 + u * (l1 - l0)))
        else:
            conc = _sample_log_concentration(self.rng, lo=lo, hi=hi)

        if stage == "stage1" and conc > hi + 1e-6:
            self.stage1_cap_violations += 1
        return float(np.clip(conc, lo, hi))

    def _sample_interference(self, stage: str) -> dict[str, float]:
        if not self.interf_avail:
            return {}
        p = 0.15 if stage == "stage2" else 0.25
        if self.rng.random() >= p:
            return {}
        n_interf = int(self.rng.integers(1, min(3, len(self.interf_avail)) + 1))
        species = list(self.rng.choice(self.interf_avail, size=n_interf, replace=False))
        out: dict[str, float] = {}
        for sp in species:
            st = self.stats[sp]
            lo = max(st["min_positive"], 1e-3)
            hi = st["max"] if stage == "stage2" else self._stage1_cap(sp)
            hi = max(hi, lo * 1.0001)
            out[sp] = _sample_log_concentration(self.rng, lo=lo, hi=hi)
        return out

    def sample(self, sample_idx: int) -> dict[str, float]:
        if not self.target_avail:
            return {}
        stage = self.stage_for_index(sample_idx)
        n_active = int(
            self.rng.integers(
                self.min_active_species,
                min(self.max_active_species, len(self.target_avail)) + 1,
            )
        )

        if stage == "stage1":
            active_species = self._sample_species_stage1(n_active, sample_idx)
            stage2_row = None
        else:
            active_species, row_idx = self._sample_species_stage2(n_active)
            if row_idx >= 0:
                stage2_row = self._lhs[row_idx]
            else:
                stage2_row = None

        chosen: dict[str, float] = {}
        for sp in active_species:
            u = None
            if stage2_row is not None:
                u = float(stage2_row[self.target_avail.index(sp)])
            conc = self._sample_target_concentration(sp, stage, u)
            chosen[sp] = conc

            if sp in self.major_species:
                self.major_presence_counts[sp] += 1
                b = _log_bin_index(conc, self.stats[sp]["min_positive"], self.stats[sp]["max"])
                self.major_bin_counts[sp][b] += 1
                self.stage_bin_counts[stage][sp][b] += 1

        chosen.update(self._sample_interference(stage))
        return chosen

    def diagnostics(self) -> dict:
        return {
            "stage_counts": {"stage1": self.stage1_n, "stage2": self.stage2_n},
            "major_presence_counts": {k: int(v) for k, v in self.major_presence_counts.items()},
            "major_presence_fraction": {
                k: float(v / max(self.n_samples, 1)) for k, v in self.major_presence_counts.items()
            },
            "major_bin_counts": {k: v.astype(int).tolist() for k, v in self.major_bin_counts.items()},
            "stage_bin_counts": {
                stage: {sp: arr.astype(int).tolist() for sp, arr in d.items()}
                for stage, d in self.stage_bin_counts.items()
            },
            "stage1_cap_policy": self.stage1_cap_policy,
            "stage1_cap_violations": int(self.stage1_cap_violations),
        }


def _log_generation_diagnostics(
    y: np.ndarray,
    *,
    species: list[str],
    major_species: list[str],
    stats: dict[str, dict[str, float]],
    extra: dict | None = None,
) -> dict:
    """Compute and emit concise generation diagnostics."""
    eps = 1e-6
    diag: dict[str, object] = {}
    presence: dict[str, float] = {}
    quantiles: dict[str, dict[str, float]] = {}
    major_bins: dict[str, list[int]] = {}

    for i, sp in enumerate(species):
        yi = y[:, i]
        nz = yi[yi > 0]
        presence[sp] = float(nz.size / max(len(yi), 1))
        if nz.size == 0:
            quantiles[sp] = {"p10": 0.0, "p50": 0.0, "p90": 0.0}
            if sp in major_species:
                major_bins[sp] = [0, 0, 0]
            continue
        q10, q50, q90 = np.percentile(nz, [10, 50, 90])
        quantiles[sp] = {"p10": float(q10), "p50": float(q50), "p90": float(q90)}
        if sp in major_species and sp in stats:
            lo = max(stats[sp]["min_positive"], eps)
            hi = max(stats[sp]["max"], lo * 1.0001)
            bins = [0, 0, 0]
            for v in nz:
                bins[_log_bin_index(float(v), lo, hi)] += 1
            major_bins[sp] = bins

    log.info("Generation diagnostics: species presence fractions")
    for sp in species:
        log.info("  %-6s %.3f", sp, presence[sp])

    log.info("Generation diagnostics: major species concentration bins [low,mid,high]")
    for sp in major_species:
        if sp in major_bins:
            log.info("  %-6s %s", sp, major_bins[sp])

    diag["presence_fraction"] = presence
    diag["nonzero_quantiles_ppmv"] = quantiles
    diag["major_bin_counts"] = major_bins
    if extra:
        diag["extra"] = extra
    return diag


def build_one_sample(
    lib: SPCLibrary,
    rng: np.random.Generator,
    *,
    min_species: int = 1,
    max_species: int = 5,
    interference_prob: float = 0.3,
    augment_profile: str = "mild",
    mandatory_target_species: Sequence[str] | None = None,
    target_weights: dict[str, float] | None = None,
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
    required = [sp for sp in (mandatory_target_species or []) if sp in target_avail]
    required = list(dict.fromkeys(required))
    chosen_target = required[:n_target]

    remaining = n_target - len(chosen_target)
    if remaining > 0:
        pool = [sp for sp in target_avail if sp not in chosen_target]
        if target_weights is None:
            extra = list(rng.choice(pool, size=remaining, replace=False))
        else:
            weights = [float(target_weights.get(sp, 1.0)) for sp in pool]
            extra = _weighted_choice_without_replacement(rng, pool, weights, remaining)
        chosen_target.extend(extra)

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

    return _build_sample_from_concentrations(lib, rng, chosen_concs, augment_profile=augment_profile)


class HybridTraceSampler:
    """Mostly default mixtures with a controlled trace-species booster subset."""

    def __init__(
        self,
        *,
        lib: SPCLibrary,
        rng: np.random.Generator,
        n_samples: int,
        trace_species: Sequence[str],
        trace_fraction: float = 0.15,
        min_active_species: int = 1,
        max_active_species: int = 4,
        augment_profile: str = "mild",
    ) -> None:
        self.lib = lib
        self.rng = rng
        self.n_samples = max(1, int(n_samples))
        self.trace_species = [sp for sp in trace_species if sp in lib.available_species]
        self.trace_fraction = float(np.clip(trace_fraction, 0.0, 1.0))
        self.min_active_species = max(1, int(min_active_species))
        self.max_active_species = max(self.min_active_species, int(max_active_species))
        self.augment_profile = augment_profile

        n_trace = int(round(self.n_samples * self.trace_fraction))
        n_trace = min(max(n_trace, 0), self.n_samples)
        flags = np.zeros(self.n_samples, dtype=bool)
        flags[:n_trace] = True
        self.rng.shuffle(flags)
        self._trace_flags = flags
        self._trace_counts = {sp: 0 for sp in self.trace_species}

    def sample(self, sample_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
        if self._trace_flags[sample_idx] and self.trace_species:
            trace_sp = str(self.rng.choice(self.trace_species))
            self._trace_counts[trace_sp] += 1
            return build_one_sample(
                self.lib,
                self.rng,
                min_species=self.min_active_species,
                max_species=self.max_active_species,
                interference_prob=0.20,
                augment_profile=self.augment_profile,
                mandatory_target_species=[trace_sp],
            )

        return build_one_sample(
            self.lib,
            self.rng,
            min_species=self.min_active_species,
            max_species=self.max_active_species,
            interference_prob=0.20,
            augment_profile=self.augment_profile,
        )

    def diagnostics(self) -> dict[str, object]:
        trace_total = int(self._trace_flags.sum())
        return {
            "stage_counts": {
                "default": int(self.n_samples - trace_total),
                "trace_boost": trace_total,
            },
            "trace_species_counts": {sp: int(v) for sp, v in self._trace_counts.items()},
            "trace_fraction": float(trace_total / max(self.n_samples, 1)),
            "min_active_species": int(self.min_active_species),
            "max_active_species": int(self.max_active_species),
        }


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
    sampling_mode: str = "default",
    augmentation_profile: str = "mild",
    curriculum_stage1_frac: float = 0.70,
    major_species: list[str] | None = None,
    stage1_cap_policy: str = "p95",
    lhs_frac: float = 0.30,
    hybrid_trace_fraction: float = 0.15,
    min_active_species: int = 1,
    max_active_species: int = 4,
    diagnostics_json: Path | None = None,
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

    lib = SPCLibrary(manifest, train_only=True)  # only train/val to avoid test-set leakage

    rng = np.random.default_rng(seed)
    stage1_cap_policy = stage1_cap_policy.lower().strip()
    if stage1_cap_policy not in {"p90", "p95", "max"}:
        raise ValueError(f"Invalid stage1_cap_policy={stage1_cap_policy}; use p90|p95|max")
    sampling_mode = sampling_mode.lower().strip()
    augmentation_profile = augmentation_profile.lower().strip()
    if augmentation_profile not in {"mild", "strong"}:
        raise ValueError(f"Invalid augmentation_profile={augmentation_profile}; use mild|strong")
    if sampling_mode not in {"default", "curriculum_v2", "hybrid_v4"}:
        raise ValueError(f"Invalid sampling_mode={sampling_mode}; use default|curriculum_v2|hybrid_v4")

    major_species_cfg = major_species or DEFAULT_MAJOR_SPECIES
    major_species_cfg = [s for s in major_species_cfg if s in TARGET_SPECIES]
    if not major_species_cfg:
        major_species_cfg = DEFAULT_MAJOR_SPECIES

    # Pre-allocate array to avoid doubling RAM during np.stack
    # Use float16 which is sufficient for absorbance precision and halves memory
    X = np.zeros((n_samples, GRID_NPTS), dtype=np.float16)
    y = np.zeros((n_samples, len(TARGET_SPECIES)), dtype=np.float32)

    sampler: CurriculumSamplerV2 | HybridTraceSampler | None = None
    if sampling_mode == "curriculum_v2":
        sampler = CurriculumSamplerV2(
            lib=lib,
            rng=rng,
            n_samples=n_samples,
            major_species=major_species_cfg,
            stage1_cap_policy=stage1_cap_policy,
            stage1_frac=curriculum_stage1_frac,
            lhs_frac=lhs_frac,
            min_active_species=min_active_species,
            max_active_species=max_active_species,
        )
        log.info(
            "Sampling mode: curriculum_v2 (stage1=%d, stage2=%d, cap=%s, active=%d-%d, aug=%s)",
            sampler.stage1_n,
            sampler.stage2_n,
            stage1_cap_policy,
            min_active_species,
            max_active_species,
            augmentation_profile,
        )
    elif sampling_mode == "hybrid_v4":
        sampler = HybridTraceSampler(
            lib=lib,
            rng=rng,
            n_samples=n_samples,
            trace_species=DEFAULT_TRACE_SPECIES,
            trace_fraction=hybrid_trace_fraction,
            min_active_species=min_active_species,
            max_active_species=max_active_species,
            augment_profile=augmentation_profile,
        )
        log.info(
            "Sampling mode: hybrid_v4 (trace_frac=%.2f, active=%d-%d, aug=%s)",
            hybrid_trace_fraction,
            min_active_species,
            max_active_species,
            augmentation_profile,
        )
    else:
        log.info("Sampling mode: default (active=%d-%d, aug=%s)", min_active_species, max_active_species, augmentation_profile)

    log_interval = max(1, n_samples // 20)
    successful = 0
    failures = 0

    for i in range(n_samples):
        if sampling_mode == "curriculum_v2" and sampler is not None:
            chosen = sampler.sample(i)
            result = _build_sample_from_concentrations(
                lib,
                rng,
                chosen,
                augment_profile=augmentation_profile,
            )
        elif sampling_mode == "hybrid_v4" and sampler is not None:
            result = sampler.sample(i)
        else:
            result = build_one_sample(
                lib,
                rng,
                min_species=min_active_species,
                max_species=max_active_species,
                interference_prob=0.20,
                augment_profile=augmentation_profile,
            )
        if result is None:
            failures += 1
            continue
        
        X[successful] = result[0].astype(np.float16)
        y[successful] = result[1]
        successful += 1
        
        if verbose and (i + 1) % log_interval == 0:
            log.info("  Generated %d / %d  (failures: %d)", i + 1, n_samples, failures)

    X = X[:successful]
    y = y[:successful]
    failure_rate = failures / max(n_samples, 1)
    if failure_rate > 0.10:
        log.warning(
            "High failure rate during generation: %d / %d (%.1f%%). "
            "Check that the SPC library has sufficient reference spectra.",
            failures, n_samples, failure_rate * 100,
        )
    log.info("Generation complete: X=%s (float16)  y=%s  (%d failures)", X.shape, y.shape, failures)

    stats = _species_stats_from_library(lib)
    extra_diag = sampler.diagnostics() if sampler is not None else None
    diag = _log_generation_diagnostics(
        y,
        species=TARGET_SPECIES,
        major_species=[s for s in major_species_cfg if s in TARGET_SPECIES],
        stats=stats,
        extra=extra_diag,
    )

    if out_path is None:
        out_path = Path(SYNTHETIC_DIR) / "spectra.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y, target_species=TARGET_SPECIES)
    log.info("Saved → %s", out_path)

    if diagnostics_json is not None:
        diagnostics_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sampling_mode": sampling_mode,
            "augmentation_profile": augmentation_profile,
            "n_requested": int(n_samples),
            "n_generated": int(successful),
            "seed": int(seed),
            "stage1_cap_policy": stage1_cap_policy,
            "hybrid_trace_fraction": float(hybrid_trace_fraction),
            "major_species": major_species_cfg,
            "diagnostics": diag,
        }
        diagnostics_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Saved diagnostics → %s", diagnostics_json)

    return X, y


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic FTIR training spectra")
    parser.add_argument("--n-samples", type=int, default=10_000, help="Number of synthetic samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="Output .npz path")
    parser.add_argument(
        "--reference-root", type=str, default=str(REFERENCE_ROOT), help="Path to data/reference/"
    )
    parser.add_argument(
        "--sampling-mode",
        type=str,
        default="default",
        choices=["default", "curriculum_v2", "hybrid_v4"],
        help="Sampling strategy for synthetic concentration vectors.",
    )
    parser.add_argument(
        "--augmentation-profile",
        type=str,
        default="mild",
        choices=["mild", "strong"],
        help="Synthetic augmentation strength profile.",
    )
    parser.add_argument(
        "--curriculum-stage1-frac",
        type=float,
        default=0.70,
        help="Relative stage-1 weight in curriculum_v2 (normalized with --lhs-frac).",
    )
    parser.add_argument(
        "--major-species",
        type=str,
        default=",".join(DEFAULT_MAJOR_SPECIES),
        help="Comma-separated major species list for curriculum_v2 biasing.",
    )
    parser.add_argument(
        "--stage1-cap-policy",
        type=str,
        default="p95",
        choices=["p90", "p95", "max"],
        help="Upper bound policy for stage-1 curriculum concentrations.",
    )
    parser.add_argument(
        "--lhs-frac",
        type=float,
        default=0.30,
        help="Relative stage-2 (LHS) weight in curriculum_v2.",
    )
    parser.add_argument(
        "--hybrid-trace-fraction",
        type=float,
        default=0.15,
        help="Fraction of hybrid_v4 samples that force a trace-species inclusion.",
    )
    parser.add_argument("--min-active-species", type=int, default=1)
    parser.add_argument("--max-active-species", type=int, default=4)
    parser.add_argument(
        "--diagnostics-json",
        type=str,
        default=None,
        help="Optional output path for generation diagnostics JSON.",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    out = Path(args.out) if args.out else None
    diag_out = Path(args.diagnostics_json) if args.diagnostics_json else None
    major_species = [s.strip() for s in args.major_species.split(",") if s.strip()]
    generate(
        n_samples=args.n_samples,
        seed=args.seed,
        out_path=out,
        reference_root=Path(args.reference_root),
        sampling_mode=args.sampling_mode,
        augmentation_profile=args.augmentation_profile,
        curriculum_stage1_frac=args.curriculum_stage1_frac,
        major_species=major_species,
        stage1_cap_policy=args.stage1_cap_policy,
        lhs_frac=args.lhs_frac,
        hybrid_trace_fraction=args.hybrid_trace_fraction,
        min_active_species=args.min_active_species,
        max_active_species=args.max_active_species,
        diagnostics_json=diag_out,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    _cli()
