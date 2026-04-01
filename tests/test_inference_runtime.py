import shutil
from pathlib import Path

import pandas as pd
import pytest
import torch

import numpy as np

from ftir_analysis.checkpointing import build_checkpoint_metadata, save_checkpoint
from ftir_analysis.constants import DEFAULT_TARGET_SPECIES, PROJECT_ROOT
from ftir_analysis.inference_runtime import InferenceConfig, run_inference
from ftir_analysis.modeling import FTIRModel
from ftir_analysis.utils import LabelNormalizer


def _write_v4_checkpoint(path: Path) -> None:
    target_species = DEFAULT_TARGET_SPECIES
    n = len(target_species)
    model = FTIRModel(n_species=n, in_channels=3, aux_features=0)

    # Minimal valid LabelNormalizer: identity transform with uniform weights
    label_normalizer = LabelNormalizer(
        means=np.zeros(n, dtype=np.float32),
        stds=np.ones(n, dtype=np.float32),
        species_weights=np.ones(n, dtype=np.float32),
    )

    metadata = build_checkpoint_metadata(target_species)
    metadata.update(
        {
            "input_transform": {
                "raw_scale": 8.0,
                "derivative_scale": 1.0,
                "saturation_epsilon": 1e-3,
            },
            "input_channels": ["absorbance", "derivative", "saturation_mask"],
            "use_prior_features": False,
            "prior_feature_config": None,
            "selection_metric": "ref_val_log_mae_mean",
            "label_normalizer": label_normalizer.as_dict(),
        }
    )
    save_checkpoint(path, model.state_dict(), metadata)


def test_inference_outputs_ppmv_csv(tmp_path: Path) -> None:
    src_spc = sorted((PROJECT_ROOT / "archive" / "example_data").glob("*.spc"))[0]
    work_dir = tmp_path / "spc_only"
    work_dir.mkdir(parents=True)
    shutil.copy2(src_spc, work_dir / src_spc.name)

    out_csv = tmp_path / "pred.csv"
    ckpt_path = tmp_path / "v4_best.pt"
    _write_v4_checkpoint(ckpt_path)
    cfg = InferenceConfig(
        data_dir=work_dir,
        checkpoint_path=ckpt_path,
        output_csv=out_csv,
    )

    out = run_inference(cfg)
    assert out == out_csv
    assert out.exists()

    df = pd.read_csv(out)
    assert len(df) == 1
    ppm_cols = [c for c in df.columns if c.endswith("_ppmv")]
    assert len(ppm_cols) == 11
    assert (df[ppm_cols] >= 0).all().all()
    assert df[ppm_cols].sum(axis=1).iloc[0] > 0


def test_incompatible_checkpoint_fails_fast(tmp_path: Path) -> None:
    src_spc = sorted((PROJECT_ROOT / "archive" / "example_data").glob("*.spc"))[0]
    work_dir = tmp_path / "spc_only"
    work_dir.mkdir(parents=True)
    shutil.copy2(src_spc, work_dir / src_spc.name)

    cfg = InferenceConfig(
        data_dir=work_dir,
        checkpoint_path=PROJECT_ROOT / "outputs" / "checkpoints" / "ftir_solver_v1.pth",
        output_csv=tmp_path / "pred.csv",
    )

    with pytest.raises(Exception):
        run_inference(cfg)
