from pathlib import Path

import pandas as pd

from ftir_analysis.constants import MANIFEST_COLUMNS, PROJECT_ROOT
from ftir_analysis.manifesting import build_manifest


def test_manifest_build_and_units_conversion(tmp_path: Path) -> None:
    reference_root = PROJECT_ROOT / "reference_spectra"
    out = tmp_path / "manifest_v1.csv"

    manifest = build_manifest(reference_root=reference_root, output_path=out, seed=42, primary_threshold=10)

    assert out.exists()
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) == 556

    idx = pd.read_csv(reference_root / "lab_index.csv")
    first = idx.iloc[0]
    first_ppmv = float(first["concentration_ppm"]) * 1e6

    # Confirm conversion appears in manifest for matching species/path entry.
    species_rows = manifest[manifest["species"] == first["species"]]
    assert (species_rows["concentration_ppmv"] > 1).all()
    assert first_ppmv in species_rows["concentration_ppmv"].values

    # Sparse class staging policy.
    sparse_rows = manifest[manifest["is_sparse_class"]]
    assert not sparse_rows.empty
    assert (sparse_rows["split"] == "staging").all()
