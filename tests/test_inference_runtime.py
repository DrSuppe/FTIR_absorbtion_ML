import shutil
from pathlib import Path

import pandas as pd
import pytest

from ftir_analysis.constants import PROJECT_ROOT
from ftir_analysis.inference_runtime import InferenceConfig, run_inference


def test_inference_outputs_ppmv_csv(tmp_path: Path) -> None:
    src_spc = sorted((PROJECT_ROOT / "example_data").glob("*.spc"))[0]
    work_dir = tmp_path / "spc_only"
    work_dir.mkdir(parents=True)
    shutil.copy2(src_spc, work_dir / src_spc.name)

    out_csv = tmp_path / "pred.csv"
    cfg = InferenceConfig(
        data_dir=work_dir,
        checkpoint_path=PROJECT_ROOT / "checkpoints" / "ftir_solver_best.pth",
        output_csv=out_csv,
    )

    out = run_inference(cfg)
    assert out == out_csv
    assert out.exists()

    df = pd.read_csv(out)
    assert len(df) == 1
    ppm_cols = [c for c in df.columns if c.endswith("_ppmv")]
    assert len(ppm_cols) == 7
    assert (df[ppm_cols] >= 0).all().all()
    assert df[ppm_cols].sum(axis=1).iloc[0] > 0


def test_incompatible_checkpoint_fails_fast(tmp_path: Path) -> None:
    src_spc = sorted((PROJECT_ROOT / "example_data").glob("*.spc"))[0]
    work_dir = tmp_path / "spc_only"
    work_dir.mkdir(parents=True)
    shutil.copy2(src_spc, work_dir / src_spc.name)

    cfg = InferenceConfig(
        data_dir=work_dir,
        checkpoint_path=PROJECT_ROOT / "checkpoints" / "ftir_solver_v1.pth",
        output_csv=tmp_path / "pred.csv",
    )

    with pytest.raises(Exception):
        run_inference(cfg)
