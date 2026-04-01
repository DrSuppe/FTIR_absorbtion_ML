"""Shared FTIR feature engineering for training, evaluation, and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .constants import (
    DEFAULT_TARGET_SPECIES,
    REFERENCE_ROOT,
    SATURATION_AU,
    WAVENUMBER_STEP,
)
from .spectra import interpolate_to_grid, load_spectrum


@dataclass(frozen=True)
class InputTransformConfig:
    """Configuration for building normalized multi-channel model inputs."""

    raw_scale: float = SATURATION_AU
    derivative_scale: float = 1.0
    saturation_epsilon: float = 1e-3

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SpectralWindow:
    """Half-open `[start, end)` spectral window on the fixed grid."""

    start: int
    end: int

    def as_dict(self) -> dict[str, int]:
        return {"start": int(self.start), "end": int(self.end)}


def smooth_saturate(spectrum: np.ndarray, *, ceiling: float = SATURATION_AU) -> np.ndarray:
    """Smooth saturation: S * tanh(y / S).

    Unlike hard clipping, this preserves gradient information near the ceiling
    and avoids discontinuities that the model cannot learn through.
    """
    s = np.asarray(spectrum, dtype=np.float32)
    return (ceiling * np.tanh(s / ceiling)).astype(np.float32)


def saturation_mask(spectrum: np.ndarray, *, epsilon: float = 1e-3) -> np.ndarray:
    """Return a binary mask for saturated points."""
    return (np.asarray(spectrum) >= (SATURATION_AU - epsilon)).astype(np.float32)


def derivative_on_grid(spectrum: np.ndarray) -> np.ndarray:
    """Return first derivative with respect to wavenumber."""
    return np.gradient(np.asarray(spectrum, dtype=np.float32), WAVENUMBER_STEP).astype(np.float32)


def fit_input_transform(
    spectra: np.ndarray | Sequence[np.ndarray],
    *,
    saturation_epsilon: float = 1e-3,
) -> InputTransformConfig:
    """Estimate robust normalization scales from training spectra."""
    if isinstance(spectra, np.ndarray):
        iterable: Iterable[np.ndarray] = spectra
    else:
        iterable = spectra

    per_spectrum_scales: list[float] = []
    for spec in iterable:
        deriv = np.abs(derivative_on_grid(spec))
        per_spectrum_scales.append(float(np.percentile(deriv, 95.0)))

    derivative_scale = float(np.median(per_spectrum_scales)) if per_spectrum_scales else 1.0
    derivative_scale = max(derivative_scale, 1e-6)
    return InputTransformConfig(
        raw_scale=SATURATION_AU,
        derivative_scale=derivative_scale,
        saturation_epsilon=saturation_epsilon,
    )


def build_input_channels(
    spectrum: np.ndarray,
    cfg: InputTransformConfig,
) -> np.ndarray:
    """Build normalized `[raw, derivative, saturation_mask]` channels."""
    raw = np.asarray(spectrum, dtype=np.float32)
    sat = saturation_mask(raw, epsilon=cfg.saturation_epsilon)
    raw = smooth_saturate(raw, ceiling=cfg.raw_scale)
    deriv = derivative_on_grid(raw) / cfg.derivative_scale
    raw = raw / max(cfg.raw_scale, 1e-6)
    return np.stack([raw, deriv, sat], axis=0).astype(np.float32)


def _contiguous_regions(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return `[start, end)` regions where mask is true."""
    if mask.size == 0:
        return []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    regions: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for cur in idx[1:]:
        cur = int(cur)
        if cur != prev + 1:
            regions.append((start, prev + 1))
            start = cur
        prev = cur
    regions.append((start, prev + 1))
    return regions


def _dominant_windows(
    template: np.ndarray,
    *,
    max_windows: int,
    threshold_ratio: float,
    pad_points: int,
) -> list[SpectralWindow]:
    """Extract up to `max_windows` dominant contiguous windows from a template."""
    energy = np.clip(np.asarray(template, dtype=np.float32), 0.0, None)
    peak = float(energy.max())
    if peak <= 0:
        return []

    mask = energy >= (peak * threshold_ratio)
    regions = _contiguous_regions(mask)
    scored: list[tuple[float, SpectralWindow]] = []
    for start, end in regions:
        start = max(0, start - pad_points)
        end = min(int(energy.shape[0]), end + pad_points)
        score = float(energy[start:end].sum())
        scored.append((score, SpectralWindow(start=start, end=end)))

    if not scored:
        center = int(np.argmax(energy))
        return [
            SpectralWindow(
                start=max(0, center - pad_points),
                end=min(int(energy.shape[0]), center + pad_points + 1),
            )
        ]

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [window for _, window in scored[:max_windows]]
    selected.sort(key=lambda window: window.start)
    return selected


def _load_species_spectra(
    manifest: pd.DataFrame,
    *,
    species: str,
    splits: Sequence[str],
) -> list[np.ndarray]:
    """Load reference spectra for one species on the fixed grid."""
    rows = manifest[
        (manifest["species"] == species)
        & (manifest["split"].isin(splits))
        & (manifest["source_format"] == "spc")
    ]
    spectra: list[np.ndarray] = []
    for source_path in rows["source_path"].tolist():
        raw_path = Path(source_path)
        path = raw_path if raw_path.is_absolute() else REFERENCE_ROOT / raw_path
        try:
            x_raw, y_raw = load_spectrum(path)
            spectra.append(interpolate_to_grid(x_raw, y_raw).astype(np.float32))
        except Exception:
            continue
    return spectra


class SpectralPriorExtractor:
    """Derive lightweight spectroscopy priors from reference templates."""

    def __init__(
        self,
        *,
        target_species: Sequence[str],
        templates: dict[str, np.ndarray],
        windows: dict[str, list[SpectralWindow]],
        raw_scale: float = SATURATION_AU,
    ) -> None:
        self.target_species = list(target_species)
        self.templates = {
            sp: np.asarray(templates[sp], dtype=np.float32) for sp in self.target_species
        }
        self.windows = {sp: list(windows.get(sp, [])) for sp in self.target_species}
        self.raw_scale = float(raw_scale)
        self.max_windows = max((len(v) for v in self.windows.values()), default=0)
        self.n_features = len(self.target_species) * (self.max_windows + 1)

    @classmethod
    def fit_from_manifest(
        cls,
        manifest: pd.DataFrame,
        *,
        target_species: Sequence[str] | None = None,
        splits: Sequence[str] = ("train", "val"),
        max_windows: int = 2,
        threshold_ratio: float = 0.35,
        pad_points: int = 16,
    ) -> "SpectralPriorExtractor":
        species_list = list(target_species or DEFAULT_TARGET_SPECIES)
        templates: dict[str, np.ndarray] = {}
        windows: dict[str, list[SpectralWindow]] = {}

        for species in species_list:
            spectra = _load_species_spectra(manifest, species=species, splits=splits)
            if spectra:
                stack = np.stack(spectra).astype(np.float32)
                template = np.median(stack, axis=0).astype(np.float32)
            else:
                template = np.zeros(0, dtype=np.float32)

            templates[species] = template
            if template.size > 0:
                windows[species] = _dominant_windows(
                    template,
                    max_windows=max_windows,
                    threshold_ratio=threshold_ratio,
                    pad_points=pad_points,
                )
            else:
                windows[species] = []

        return cls(target_species=species_list, templates=templates, windows=windows)

    def describe(self) -> dict[str, object]:
        return {
            "target_species": list(self.target_species),
            "max_windows": int(self.max_windows),
            "n_features": int(self.n_features),
            "windows": {
                species: [window.as_dict() for window in self.windows.get(species, [])]
                for species in self.target_species
            },
        }

    def transform(self, spectrum: np.ndarray) -> np.ndarray:
        """Build deterministic prior features for one spectrum."""
        raw = np.asarray(spectrum, dtype=np.float32)
        positive = np.clip(raw, 0.0, None)
        spec_norm = float(np.linalg.norm(positive))

        features: list[float] = []
        for species in self.target_species:
            species_windows = self.windows.get(species, [])
            for idx in range(self.max_windows):
                if idx < len(species_windows):
                    window = species_windows[idx]
                    integral = float(
                        np.trapezoid(positive[window.start : window.end], dx=WAVENUMBER_STEP)
                        / max(self.raw_scale, 1e-6)
                    )
                    features.append(integral)
                else:
                    features.append(0.0)

            template = self.templates.get(species)
            if template is None or template.size == 0:
                features.append(0.0)
                continue
            template_pos = np.clip(template, 0.0, None)
            denom = spec_norm * float(np.linalg.norm(template_pos))
            if denom <= 0:
                features.append(0.0)
            else:
                sim = float(np.dot(positive, template_pos) / denom)
                features.append(sim)

        return np.asarray(features, dtype=np.float32)
