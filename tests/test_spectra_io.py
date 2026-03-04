from pathlib import Path

import numpy as np

from ftir_analysis.constants import PROJECT_ROOT
from ftir_analysis.spectra import GRID_NPTS, load_on_grid, parse_mks_spc


def test_spc_parse_and_grid_interpolation() -> None:
    spc = sorted((PROJECT_ROOT / "example_data").glob("*.spc"))[0]
    x, y = parse_mks_spc(spc)

    assert len(x) > 1000
    assert len(y) == len(x)
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    y_grid = load_on_grid(spc)
    assert y_grid.shape == (GRID_NPTS,)
    assert np.isfinite(y_grid).all()
