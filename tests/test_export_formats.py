from pathlib import Path

import numpy as np
import pytest

from ftir_analysis.spectra import GRID_NPTS, SpectrumLoadError, load_on_grid


def test_csv_spectrum_ingest_two_columns(tmp_path: Path) -> None:
    x = np.arange(800.0, 5000.0, 0.25)
    y = 0.01 * np.sin(x / 1000.0)

    csv_path = tmp_path / "spec.csv"
    with csv_path.open("w", encoding="utf-8") as fh:
        fh.write("wavenumber,absorbance\n")
        for xi, yi in zip(x, y):
            fh.write(f"{xi:.6f},{yi:.8f}\n")

    out = load_on_grid(csv_path)
    assert out.shape == (GRID_NPTS,)
    assert np.isfinite(out).all()


def test_lab_input_is_explicitly_unsupported(tmp_path: Path) -> None:
    lab_path = tmp_path / "dummy.lab"
    lab_path.write_bytes(b"not-a-real-lab")

    with pytest.raises(SpectrumLoadError):
        load_on_grid(lab_path)
