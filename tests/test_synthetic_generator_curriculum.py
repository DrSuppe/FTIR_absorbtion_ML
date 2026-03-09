from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ftir_analysis.constants import PROJECT_ROOT
from synthetic_generator import DEFAULT_MAJOR_SPECIES, TARGET_SPECIES, generate


def _species_p_quantile(manifest: pd.DataFrame, q: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for sp in TARGET_SPECIES:
        d = manifest[manifest["species"] == sp]["concentration_ppmv"].to_numpy(dtype=np.float64)
        if d.size > 0:
            out[sp] = float(np.percentile(d, q))
    return out


def test_curriculum_stage1_p90_cap_enforced(tmp_path: Path) -> None:
    reference_root = PROJECT_ROOT / "data" / "reference"
    out_path = tmp_path / "curriculum_stage1_only.npz"
    manifest = pd.read_csv(reference_root / "manifest_v1.csv")
    p90 = _species_p_quantile(manifest, 90.0)

    _X, y = generate(
        n_samples=180,
        seed=42,
        out_path=out_path,
        reference_root=reference_root,
        sampling_mode="curriculum_v2",
        curriculum_stage1_frac=1.0,
        lhs_frac=0.0,
        stage1_cap_policy="p90",
        major_species=DEFAULT_MAJOR_SPECIES,
        min_active_species=2,
        max_active_species=6,
        verbose=False,
    )
    assert y.shape[1] == len(TARGET_SPECIES)

    for i, sp in enumerate(TARGET_SPECIES):
        nz = y[:, i][y[:, i] > 0]
        if nz.size == 0 or sp not in p90:
            continue
        assert float(nz.max()) <= p90[sp] * 1.001


def test_curriculum_emits_major_coverage_diagnostics(tmp_path: Path) -> None:
    reference_root = PROJECT_ROOT / "data" / "reference"
    out_path = tmp_path / "curriculum_v2.npz"
    diag_path = tmp_path / "diag.json"

    _X, y = generate(
        n_samples=240,
        seed=42,
        out_path=out_path,
        reference_root=reference_root,
        sampling_mode="curriculum_v2",
        curriculum_stage1_frac=0.7,
        lhs_frac=0.3,
        stage1_cap_policy="p95",
        major_species=DEFAULT_MAJOR_SPECIES,
        min_active_species=2,
        max_active_species=6,
        diagnostics_json=diag_path,
        verbose=False,
    )
    assert y.shape[1] == len(TARGET_SPECIES)
    assert diag_path.exists()

    payload = json.loads(diag_path.read_text(encoding="utf-8"))
    extra = payload["diagnostics"]["extra"]
    assert extra["stage1_cap_violations"] == 0

    presence = payload["diagnostics"]["presence_fraction"]
    for sp in DEFAULT_MAJOR_SPECIES:
        assert float(presence[sp]) > 0.25

    stage2_bins = extra["stage_bin_counts"]["stage2"]
    high_hits = 0
    for sp in DEFAULT_MAJOR_SPECIES:
        bins = stage2_bins.get(sp, [0, 0, 0])
        if int(bins[2]) > 0:
            high_hits += 1
    assert high_hits >= 4
