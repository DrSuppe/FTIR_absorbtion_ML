from pathlib import Path

import pandas as pd

from ftir_analysis.constants import MANIFEST_COLUMNS, PROJECT_ROOT
from ftir_analysis.manifesting import build_manifest


def test_manifest_build_and_units_conversion(tmp_path: Path) -> None:
    reference_root = PROJECT_ROOT / "data" / "reference"
    out = tmp_path / "manifest_v1.csv"

    manifest = build_manifest(reference_root=reference_root, output_path=out, seed=42, primary_threshold=10)

    assert out.exists()
    assert list(manifest.columns) == MANIFEST_COLUMNS
    assert len(manifest) > 500

    # Verify at least some basic parsing worked correctly on real files
    co2_rows = manifest[manifest["species"] == "CO2"]
    assert len(co2_rows) > 0
    row = co2_rows.iloc[0]
    assert "species" in row
    assert row["concentration_ppmv"] > 0
