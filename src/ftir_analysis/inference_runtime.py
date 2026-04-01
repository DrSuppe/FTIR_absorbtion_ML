"""Inference runtime with strict preprocessing and checkpoint validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import logging

from .checkpointing import CheckpointMetadataError, load_metadata, load_state_dict_or_raise, validate_metadata
from .constants import CHECKPOINT_DIR, DEFAULT_TARGET_SPECIES, MANIFEST_FILENAME, REFERENCE_ROOT

log = logging.getLogger(__name__)
from .features import InputTransformConfig, SpectralPriorExtractor, build_input_channels
from .modeling import FTIRModel
from .spectra import GRID_NPTS, SpectrumLoadError, load_on_grid
from .utils import LabelNormalizer, labels_from_log, resolve_device


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

    # The model was trained with the target species from metadata.
    if "target_species" in metadata:
        target_species = metadata["target_species"]
    else:
        target_species = DEFAULT_TARGET_SPECIES

    device = resolve_device()
    if "input_transform" not in metadata:
        log.warning(
            "Checkpoint metadata has no 'input_transform' — using defaults. "
            "This may produce incorrect results if the model was trained with different scales."
        )
    input_transform = InputTransformConfig(**metadata.get("input_transform", {}))

    norm_cfg = metadata.get("label_normalizer")
    label_normalizer: LabelNormalizer | None = (
        LabelNormalizer.from_dict(norm_cfg) if norm_cfg is not None else None
    )
    if label_normalizer is None:
        log.warning(
            "Checkpoint has no label_normalizer metadata — outputs will NOT be denormalised. "
            "This checkpoint was likely generated before v4.1."
        )
    use_prior = bool(metadata.get("use_prior_features", False))
    prior_extractor = None
    if use_prior:
        manifest_path = REFERENCE_ROOT / MANIFEST_FILENAME
        if not manifest_path.exists():
            raise CheckpointMetadataError(
                f"Prior-enabled checkpoint requires manifest for template reconstruction: {manifest_path}"
            )
        manifest = pd.read_csv(manifest_path)
        prior_extractor = SpectralPriorExtractor.fit_from_manifest(
            manifest,
            target_species=target_species,
            splits=("train",),
        )

    model = FTIRModel(
        n_species=len(target_species),
        in_channels=3,
        aux_features=(prior_extractor.n_features if prior_extractor is not None else 0),
    ).to(device)

    load_state_dict_or_raise(model, checkpoint_path, map_location=device)
    model.eval()

    files = _list_input_files(data_dir)
    if not files:
        raise RuntimeError(f"No supported input files found in {data_dir}")

    rows: list[dict[str, object]] = []
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

            xb = build_input_channels(y_grid, input_transform)
            aux = (
                prior_extractor.transform(y_grid)
                if prior_extractor is not None
                else np.zeros((0,), dtype=np.float32)
            )
            tensor_input = torch.tensor(xb, dtype=torch.float32, device=device).unsqueeze(0)
            aux_input = torch.tensor(aux, dtype=torch.float32, device=device).unsqueeze(0)
            preds_norm = model(tensor_input, aux=aux_input if aux_input.shape[-1] > 0 else None)
            preds_log = (
                label_normalizer.denormalize(preds_norm)
                if label_normalizer is not None
                else preds_norm
            )
            ppmv = labels_from_log(preds_log.squeeze(0)).cpu().numpy()

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
