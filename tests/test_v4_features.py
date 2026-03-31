from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ftir_analysis.constants import PROJECT_ROOT, SATURATION_AU
from ftir_analysis.datasets import ReferenceSpectraDataset
from ftir_analysis.features import build_input_channels, fit_input_transform, SpectralPriorExtractor
from synthetic_generator import (
    DEFAULT_TRACE_SPECIES,
    GRID_NPTS,
    HybridTraceSampler,
    SPCLibrary,
    augment,
    generate,
)


def test_mild_augmentation_stays_close_to_source() -> None:
    rng = np.random.default_rng(42)
    spectrum = np.full(GRID_NPTS, 0.5, dtype=np.float32)
    out = augment(spectrum, rng, profile="mild")

    assert out.shape == spectrum.shape
    assert float(np.max(np.abs(out - spectrum))) < 0.08
    assert not np.any(out == 0.0)
    assert float(out.max()) <= SATURATION_AU


def test_build_input_channels_marks_saturation() -> None:
    spectrum = np.linspace(0.0, 1.0, GRID_NPTS, dtype=np.float32)
    spectrum[123] = SATURATION_AU

    cfg = fit_input_transform([spectrum])
    channels = build_input_channels(spectrum, cfg)

    assert channels.shape == (3, GRID_NPTS)
    assert np.isclose(channels[0, 123], SATURATION_AU / cfg.raw_scale)
    assert channels[2, 123] == 1.0
    assert channels[2, 122] == 0.0


def test_prior_extractor_is_deterministic() -> None:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "reference" / "manifest_v1.csv")
    ds = ReferenceSpectraDataset(manifest, splits=("train",), log_transform=False)
    raw = ds._X[0]

    first = SpectralPriorExtractor.fit_from_manifest(manifest)
    second = SpectralPriorExtractor.fit_from_manifest(manifest)

    assert first.describe() == second.describe()
    assert np.allclose(first.transform(raw), second.transform(raw))


def test_hybrid_trace_sampler_respects_fraction() -> None:
    manifest = pd.read_csv(PROJECT_ROOT / "data" / "reference" / "manifest_v1.csv")
    lib = SPCLibrary(manifest, train_only=False)
    rng = np.random.default_rng(7)
    sampler = HybridTraceSampler(
        lib=lib,
        rng=rng,
        n_samples=100,
        trace_species=DEFAULT_TRACE_SPECIES,
        trace_fraction=0.15,
        min_active_species=1,
        max_active_species=4,
        augment_profile="mild",
    )

    produced = 0
    for idx in range(100):
        if sampler.sample(idx) is not None:
            produced += 1

    diag = sampler.diagnostics()
    assert produced > 0
    assert diag["stage_counts"]["trace_boost"] == 15
    assert sum(diag["trace_species_counts"].values()) == 15


def test_hybrid_v4_generation_emits_trace_fraction_diagnostics(tmp_path: Path) -> None:
    reference_root = PROJECT_ROOT / "data" / "reference"
    out_path = tmp_path / "hybrid_v4.npz"
    diag_path = tmp_path / "hybrid_v4_diag.json"

    generate(
        n_samples=120,
        seed=42,
        out_path=out_path,
        reference_root=reference_root,
        sampling_mode="hybrid_v4",
        augmentation_profile="mild",
        hybrid_trace_fraction=0.10,
        min_active_species=1,
        max_active_species=4,
        diagnostics_json=diag_path,
        verbose=False,
    )

    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    extra = payload["diagnostics"]["extra"]
    assert payload["sampling_mode"] == "hybrid_v4"
    assert payload["hybrid_trace_fraction"] == pytest.approx(0.10)
    assert extra["trace_fraction"] == pytest.approx(0.10)
    assert extra["stage_counts"]["trace_boost"] == 12
