"""Inference runtime with strict preprocessing and checkpoint validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from .checkpointing import CheckpointMetadataError, load_metadata, load_state_dict_or_raise, validate_metadata
from .constants import CHECKPOINT_DIR, DEFAULT_TARGET_SPECIES
from .modeling import FTIRModel
from .spectra import GRID_NPTS, SpectrumLoadError, load_on_grid
from .utils import labels_from_log, resolve_device


@dataclass
class InferenceConfig:
    data_dir: Path
    checkpoint_path: Path = CHECKPOINT_DIR / "ftir_solver_best.pth"
    output_csv: Path | None = None


def _list_input_files(data_dir: Path) -> list[Path]:
    spc_files: list[Path] = []
    for suffix in ("*.spc", "*.SPC"):
        spc_files.extend(sorted(data_dir.glob(suffix)))
    if spc_files:
        return sorted(set(spc_files))

    # Fallback path for exported spectra if no SPC files are present.
    text_files: list[Path] = []
    for suffix in ("*.csv", "*.CSV", "*.txt", "*.TXT"):
        text_files.extend(sorted(data_dir.glob(suffix)))
    return sorted(set(text_files))


def run_inference(cfg: InferenceConfig) -> Path:
    """Run model inference with strict guards and ppmv output conversion."""
    data_dir = Path(cfg.data_dir)
    checkpoint_path = Path(cfg.checkpoint_path)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")

    metadata = load_metadata(checkpoint_path, strict=True)
    validate_metadata(metadata)

    target_species = list(metadata.get("target_species", DEFAULT_TARGET_SPECIES))

    device = resolve_device()
    model = FTIRModel(out_features=len(target_species)).to(device)

    load_state_dict_or_raise(model, checkpoint_path, map_location=device)
    model.eval()

    files = _list_input_files(data_dir)
    if not files:
        raise RuntimeError(f"No supported input files found in {data_dir}")

    rows: list[dict[str, object]] = []
    hidden_state = None

    with torch.no_grad():
        for path in files:
            try:
                y_grid = load_on_grid(path)
            except SpectrumLoadError as exc:
                raise RuntimeError(f"Failed to preprocess {path}: {exc}") from exc

            if y_grid.shape[0] != GRID_NPTS:
                raise RuntimeError(
                    f"Unexpected input length for {path}: {y_grid.shape[0]} (expected {GRID_NPTS})"
                )

            tensor_input = torch.tensor(y_grid, dtype=torch.float32, device=device).view(1, 1, 1, -1)
            preds, hidden_state = model(tensor_input, hidden_state)
            ppmv = labels_from_log(preds.squeeze(0).squeeze(0)).cpu().numpy()

            row = {"File": path.name}
            for i, species in enumerate(target_species):
                row[f"{species}_ppmv"] = float(max(0.0, ppmv[i]))
            rows.append(row)

    out_csv = cfg.output_csv or (data_dir / "ml_solver_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


def make_inference_config(
    data_dir: str | Path,
    checkpoint_path: str | Path | None = None,
    output_csv: str | Path | None = None,
) -> InferenceConfig:
    """Helper for wrapper scripts and CLI."""
    return InferenceConfig(
        data_dir=Path(data_dir),
        checkpoint_path=Path(checkpoint_path) if checkpoint_path else CHECKPOINT_DIR / "ftir_solver_best.pth",
        output_csv=Path(output_csv) if output_csv else None,
    )
