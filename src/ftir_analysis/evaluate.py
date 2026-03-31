"""Model and baseline evaluation for manifest-driven datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .baselines import NNLSReferenceBaseline
from .checkpointing import load_metadata, load_state_dict_or_raise, validate_metadata
from .constants import DEFAULT_TARGET_SPECIES, REPORTS_DIR
from .datasets import ReferenceSpectraDataset, manifest_target_species
from .features import InputTransformConfig, SpectralPriorExtractor, build_input_channels
from .modeling import FTIRModel
from .utils import labels_from_log, resolve_device, set_mpl_config_if_needed, write_json


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, np.ndarray]:
    ae = np.abs(y_pred - y_true)
    return {
        "mae": ae.mean(axis=0),
        "median_ae": np.median(ae, axis=0),
    }


def _reference_arrays(
    manifest: pd.DataFrame,
    *,
    splits: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    ds = ReferenceSpectraDataset(manifest, splits=splits, log_transform=False)
    if len(ds) == 0:
        n_species = len(DEFAULT_TARGET_SPECIES)
        return np.zeros((0, n_species), dtype=np.float32), np.zeros((0, n_species), dtype=np.float32)
    return np.stack(ds._X).astype(np.float32), np.stack(ds._y).astype(np.float32)


def _predict_model(
    model: FTIRModel,
    x_raw: np.ndarray,
    *,
    device: torch.device,
    input_transform: InputTransformConfig,
    prior_extractor: SpectralPriorExtractor | None,
    batch_size: int = 64,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    empty_aux = np.zeros((0,), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, x_raw.shape[0], batch_size):
            batch_raw = x_raw[i : i + batch_size]
            xb = np.stack([build_input_channels(spec, input_transform) for spec in batch_raw]).astype(np.float32)
            aux_np = (
                np.stack([prior_extractor.transform(spec) for spec in batch_raw]).astype(np.float32)
                if prior_extractor is not None
                else np.stack([empty_aux] * len(batch_raw)).astype(np.float32)
            )
            xb_t = torch.tensor(xb, dtype=torch.float32, device=device)
            aux_t = torch.tensor(aux_np, dtype=torch.float32, device=device)
            pred_log = model(xb_t, aux=aux_t if aux_t.shape[-1] > 0 else None)
            pred = labels_from_log(pred_log).cpu().numpy()
            out.append(pred)
    return np.concatenate(out, axis=0) if out else np.zeros((0, len(DEFAULT_TARGET_SPECIES)), dtype=np.float32)


def evaluate_manifest(
    manifest_path: Path,
    *,
    checkpoint_path: Path | None = None,
    report_prefix: str = "evaluation",
) -> dict[str, Any]:
    """Evaluate NNLS baseline and optional ML checkpoint on reference splits."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = pd.read_csv(manifest_path)
    target_species = manifest_target_species(manifest, include_sparse=False)

    x_train, y_train = _reference_arrays(manifest, splits=("train",))
    x_val, y_val = _reference_arrays(manifest, splits=("val",))
    x_test, y_test = _reference_arrays(manifest, splits=("test",))

    if x_train.size == 0:
        raise RuntimeError("No train samples available for evaluation baseline")

    baseline = NNLSReferenceBaseline().fit(x_train, y_train)
    baseline_val = baseline.predict(x_val) if x_val.size else np.zeros_like(y_val)
    baseline_test = baseline.predict(x_test) if x_test.size else np.zeros_like(y_test)

    report: dict[str, Any] = {
        "manifest_path": str(manifest_path),
        "target_species": target_species,
        "n_train": int(x_train.shape[0]),
        "n_val": int(x_val.shape[0]),
        "n_test": int(x_test.shape[0]),
        "baseline_val": _metrics(y_val, baseline_val) if x_val.size else None,
        "baseline_test": _metrics(y_test, baseline_test) if x_test.size else None,
    }

    model_test_pred: np.ndarray | None = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        meta = load_metadata(checkpoint_path, strict=True)
        validate_metadata(meta, expected_target_species=target_species)

        input_transform = InputTransformConfig(**meta.get("input_transform", {}))
        use_prior = bool(meta.get("use_prior_features", False))
        prior_extractor = (
            SpectralPriorExtractor.fit_from_manifest(
                manifest,
                target_species=target_species,
                splits=("train", "val"),
            )
            if use_prior
            else None
        )

        device = resolve_device()
        model = FTIRModel(
            n_species=len(target_species),
            in_channels=3,
            aux_features=(prior_extractor.n_features if prior_extractor is not None else 0),
        ).to(device)
        load_state_dict_or_raise(model, checkpoint_path, map_location=device)

        model_val = (
            _predict_model(
                model,
                x_val,
                device=device,
                input_transform=input_transform,
                prior_extractor=prior_extractor,
            )
            if x_val.size
            else np.zeros_like(y_val)
        )
        model_test_pred = (
            _predict_model(
                model,
                x_test,
                device=device,
                input_transform=input_transform,
                prior_extractor=prior_extractor,
            )
            if x_test.size
            else np.zeros_like(y_test)
        )

        report["checkpoint_path"] = str(checkpoint_path)
        report["model_val"] = _metrics(y_val, model_val) if x_val.size else None
        report["model_test"] = _metrics(y_test, model_test_pred) if x_test.size else None

    def _to_list_metrics(obj: Any) -> Any:
        if obj is None:
            return None
        if isinstance(obj, dict):
            return {k: _to_list_metrics(v) for k, v in obj.items()}
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = _to_list_metrics(report)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_json = REPORTS_DIR / f"{report_prefix}_report.json"
    write_json(report_json, serializable)

    if x_test.size:
        set_mpl_config_if_needed()
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            cols = 3
            rows = int(np.ceil(len(target_species) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.8), squeeze=False)

            for i, species in enumerate(target_species):
                r, c = divmod(i, cols)
                ax = axes[r][c]
                ax.scatter(y_test[:, i], baseline_test[:, i], s=16, alpha=0.7, label="NNLS baseline")
                if model_test_pred is not None:
                    ax.scatter(y_test[:, i], model_test_pred[:, i], s=16, alpha=0.7, label="ML model")
                lim_max = max(float(y_test[:, i].max()), float(baseline_test[:, i].max()), 1.0)
                ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1)
                ax.set_title(species)
                ax.set_xlabel("True ppmv")
                ax.set_ylabel("Pred ppmv")
                ax.grid(True, alpha=0.3)

            for j in range(len(target_species), rows * cols):
                r, c = divmod(j, cols)
                axes[r][c].axis("off")

            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper right")

            fig.tight_layout()
            fig.savefig(REPORTS_DIR / f"{report_prefix}_calibration.png", dpi=150)
            plt.close(fig)
        except Exception:
            pass

    return serializable
