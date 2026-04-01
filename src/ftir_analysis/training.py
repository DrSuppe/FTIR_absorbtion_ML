"""FTIR ML Solver v4 — Training pipeline.

Trains on a combined dataset of:
  1. Synthetic multi-gas spectra (Beer-Lambert combination of real reference spectra)
  2. Real single-gas reference spectra from the SPC library

Labels are in log1p(ppmv) space; outputs are inverted with expm1 for reporting.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .checkpointing import build_checkpoint_metadata, save_checkpoint
from .constants import (
    CHECKPOINT_DIR,
    DEFAULT_TARGET_SPECIES,
    MANIFEST_FILENAME,
    MODEL_VERSION,
    PROJECT_ROOT,
    REFERENCE_ROOT,
    REPORTS_DIR,
    SYNTHETIC_DIR,
)
from .datasets import (
    ArraySpectrumDataset,
    ReferenceSpectraDataset,
    load_synthetic_aux_arrays,
)
from .features import InputTransformConfig, SpectralPriorExtractor, fit_input_transform
from .modeling import FTIRModel, count_parameters
from .utils import LabelNormalizer, labels_from_log, resolve_device, seed_everything, set_mpl_config_if_needed

log = logging.getLogger(__name__)

TARGET_SPECIES = DEFAULT_TARGET_SPECIES
MAJOR_SPECIES = ("H2O", "CO2", "CO", "NO", "NO2", "NH3")
TRACE_SPECIES = ("CH4", "N2O", "C2H4", "HCN", "HNCO")
SPECIES_GROUPS: dict[str, tuple[str, ...]] = {
    "major": MAJOR_SPECIES,
    "trace": TRACE_SPECIES,
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainConfig:
    # Data
    n_synthetic: int = 20_000
    synthetic_npz: Optional[str] = None        # override path for train NPZ
    val_synthetic_npz: Optional[str] = None    # override path for independent val NPZ
    manifest_path: Optional[str] = None        # override manifest
    reference_weight: float = 0.2              # fraction of each batch from real spectra
    synthetic_sampling_mode: str = "hybrid_v4"
    synthetic_augmentation_profile: str = "mild"
    hybrid_trace_fraction: float = 0.15
    min_active_species: int = 1
    max_active_species: int = 4
    use_prior_features: bool = False
    saturation_epsilon: float = 1e-3

    # Model
    n_species: int = len(TARGET_SPECIES)       # 11
    dropout: float = 0.15

    # Optimiser
    epochs: int = 16
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 2.0                 # linear warmup duration
    grad_clip: float = 1.0

    # Misc
    seed: int = 42
    device: Optional[str] = None              # None = auto-detect (mps > cuda > cpu)
    val_split_fraction: float = 0.20           # fraction of synthetic data held for val
    checkpoint_dir: Optional[str] = None
    reports_dir: Optional[str] = None
    log_every_n_epochs: int = 1

    # HuberLoss delta (in log-ppmv space)
    huber_delta: float = 1.0
    active_label_weight: float = 4.0
    inactive_label_weight: float = 0.5


@dataclasses.dataclass
class PreparedDatasets:
    train_ds: ConcatDataset | ArraySpectrumDataset
    val_ds: ConcatDataset | ArraySpectrumDataset
    n_ref_train: int
    synth_val_ds: ArraySpectrumDataset
    ref_val_ds: ReferenceSpectraDataset | None
    n_synth_train: int
    input_transform: InputTransformConfig
    prior_extractor: SpectralPriorExtractor | None
    label_normalizer: LabelNormalizer


@dataclasses.dataclass(frozen=True)
class EpochSummary:
    epoch: int
    val_loss: float
    val_log_mae_mean: float
    synth_val_log_mae_mean: float
    ref_val_log_mae_mean: float | None
    zero_baseline_log_mae_mean: float
    zero_baseline_ref_log_mae_mean: float | None
    species_beating_zero_baseline: int
    selection_metric: str
    selection_metric_value: float
    group_log_mae_mean: dict[str, dict[str, float] | None]


# ---------------------------------------------------------------------------
# LR schedule helpers
# ---------------------------------------------------------------------------

def _cosine_with_warmup_fn(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _hybrid_trace_fraction_tag(trace_fraction: float) -> str:
    pct = int(round(float(trace_fraction) * 100))
    return f"tf{pct:03d}"


def _group_metric_means(values: np.ndarray | None) -> dict[str, float] | None:
    if values is None:
        return None
    grouped: dict[str, float] = {}
    for group_name, species_names in SPECIES_GROUPS.items():
        indices = [TARGET_SPECIES.index(species) for species in species_names if species in TARGET_SPECIES]
        grouped[group_name] = float(np.mean(values[indices])) if indices else float("nan")
    return grouped


def _format_group_metrics(values: dict[str, float] | None) -> str:
    if values is None:
        return "n/a"
    return "  ".join(f"{name}={metric:.3f}" for name, metric in values.items())


def _build_epoch_summary(
    *,
    epoch: int,
    val_loss: float,
    val_log_mae: np.ndarray,
    synth_log_mae: np.ndarray,
    ref_log_mae: np.ndarray | None,
    zero_baseline_mean_all: float,
    zero_baseline_mean_ref: float | None,
    species_beating_zero: int,
) -> EpochSummary:
    val_log_mae_mean = float(val_log_mae.mean())
    synth_log_mae_mean = float(synth_log_mae.mean())
    ref_log_mae_mean = float(ref_log_mae.mean()) if ref_log_mae is not None else None
    selection_metric = "ref_val_log_mae_mean" if ref_log_mae_mean is not None else "val_log_mae_mean"
    selection_metric_value = ref_log_mae_mean if ref_log_mae_mean is not None else val_log_mae_mean
    return EpochSummary(
        epoch=epoch,
        val_loss=float(val_loss),
        val_log_mae_mean=val_log_mae_mean,
        synth_val_log_mae_mean=synth_log_mae_mean,
        ref_val_log_mae_mean=ref_log_mae_mean,
        zero_baseline_log_mae_mean=float(zero_baseline_mean_all),
        zero_baseline_ref_log_mae_mean=(
            float(zero_baseline_mean_ref) if zero_baseline_mean_ref is not None else None
        ),
        species_beating_zero_baseline=int(species_beating_zero),
        selection_metric=selection_metric,
        selection_metric_value=float(selection_metric_value),
        group_log_mae_mean={
            "all": _group_metric_means(val_log_mae),
            "synth": _group_metric_means(synth_log_mae),
            "ref": _group_metric_means(ref_log_mae),
        },
    )


def _epoch_summary_payload(summary: EpochSummary) -> dict[str, object]:
    return {
        "epoch": summary.epoch,
        "val_loss": summary.val_loss,
        "val_log_mae_mean": summary.val_log_mae_mean,
        "synth_val_log_mae_mean": summary.synth_val_log_mae_mean,
        "ref_val_log_mae_mean": summary.ref_val_log_mae_mean,
        "zero_baseline_log_mae_mean": summary.zero_baseline_log_mae_mean,
        "zero_baseline_ref_log_mae_mean": summary.zero_baseline_ref_log_mae_mean,
        "species_beating_zero_baseline": summary.species_beating_zero_baseline,
        "selection_metric": summary.selection_metric,
        "selection_metric_value": summary.selection_metric_value,
        "group_log_mae_mean": summary.group_log_mae_mean,
    }


def _log_best_epoch(label: str, summary: EpochSummary | None) -> None:
    if summary is None:
        return
    ref_txt = (
        f"{summary.ref_val_log_mae_mean:.4f}"
        if summary.ref_val_log_mae_mean is not None
        else "n/a"
    )
    log.info(
        "%s epoch %d: ref=%s mixed=%.4f synth=%.4f groups(all)=%s groups(ref)=%s beat_zero=%d/%d",
        label,
        summary.epoch,
        ref_txt,
        summary.val_log_mae_mean,
        summary.synth_val_log_mae_mean,
        _format_group_metrics(summary.group_log_mae_mean["all"]),
        _format_group_metrics(summary.group_log_mae_mean["ref"]),
        summary.species_beating_zero_baseline,
        len(TARGET_SPECIES),
    )


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _ensure_synthetic(
    cfg: TrainConfig,
    *,
    seed: int | None = None,
    n_samples: int | None = None,
    out_path: Path | None = None,
) -> Path:
    """Return path to synthetic .npz, generating it if necessary.

    Optional overrides allow generating a second independent val set:
      seed     — RNG seed (defaults to cfg.seed)
      n_samples — how many samples (defaults to cfg.n_synthetic)
      out_path  — explicit output path (overrides the auto-derived name)
    """
    effective_seed = seed if seed is not None else cfg.seed
    effective_n = n_samples if n_samples is not None else cfg.n_synthetic

    if out_path is not None:
        npz_path = out_path
    else:
        hybrid_tag = (
            f"_{_hybrid_trace_fraction_tag(cfg.hybrid_trace_fraction)}"
            if cfg.synthetic_sampling_mode == "hybrid_v4"
            else ""
        )
        npz_path = (
            Path(cfg.synthetic_npz)
            if cfg.synthetic_npz and seed is None
            else Path(SYNTHETIC_DIR) / f"spectra_{cfg.synthetic_sampling_mode}{hybrid_tag}_{cfg.synthetic_augmentation_profile}.npz"
        )

    regen = False
    if not npz_path.exists():
        log.info("Synthetic data not found — generating %d samples …", effective_n)
        regen = True
    else:
        try:
            data = np.load(npz_path, allow_pickle=False)
            n_existing = data["X"].shape[0]
            if n_existing < effective_n:
                log.info(
                    "Existing synthetic file has %d samples (need %d) — regenerating.",
                    n_existing, effective_n,
                )
                regen = True
        except Exception:
            regen = True

    if regen:
        generator = PROJECT_ROOT / "synthetic_generator.py"
        cmd = [
            sys.executable, str(generator),
            "--n-samples", str(effective_n),
            "--seed", str(effective_seed),
            "--sampling-mode", cfg.synthetic_sampling_mode,
            "--augmentation-profile", cfg.synthetic_augmentation_profile,
            "--min-active-species", str(cfg.min_active_species),
            "--max-active-species", str(cfg.max_active_species),
            "--hybrid-trace-fraction", str(cfg.hybrid_trace_fraction),
            "--out", str(npz_path),
        ]
        log.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return npz_path


def _build_datasets(cfg: TrainConfig) -> PreparedDatasets:
    """Build datasets plus the preprocessing objects they require."""
    # ---- Synthetic train data ----
    train_npz = _ensure_synthetic(cfg)
    loaded = load_synthetic_aux_arrays(train_npz)
    if loaded is None:
        raise RuntimeError("Could not load synthetic training data — check generation logs.")
    X_train, y_train = loaded

    # ---- Independent synthetic val set (different seed → different distribution sample) ----
    val_n = max(1, int(len(X_train) * cfg.val_split_fraction))
    if cfg.val_synthetic_npz:
        val_npz_path = Path(cfg.val_synthetic_npz)
    else:
        hybrid_tag = (
            f"_{_hybrid_trace_fraction_tag(cfg.hybrid_trace_fraction)}"
            if cfg.synthetic_sampling_mode == "hybrid_v4"
            else ""
        )
        val_npz_path = (
            Path(SYNTHETIC_DIR)
            / f"spectra_{cfg.synthetic_sampling_mode}{hybrid_tag}_{cfg.synthetic_augmentation_profile}_val.npz"
        )
    val_npz = _ensure_synthetic(
        cfg,
        seed=cfg.seed + 1,
        n_samples=val_n,
        out_path=val_npz_path,
    )
    val_loaded = load_synthetic_aux_arrays(val_npz)
    if val_loaded is None:
        raise RuntimeError("Could not load synthetic validation data — check generation logs.")
    X_val, y_val = val_loaded

    # ---- Reference spectra ----
    manifest_path = (
        Path(cfg.manifest_path)
        if cfg.manifest_path
        else Path(REFERENCE_ROOT) / MANIFEST_FILENAME
    )
    n_ref_train = 0
    ref_train_raw: ReferenceSpectraDataset | None = None
    ref_val_raw: ReferenceSpectraDataset | None = None
    ref_val: ReferenceSpectraDataset | None = None
    manifest: pd.DataFrame | None = None
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        # Load reference spectra once (no transforms yet).
        ref_train_raw = ReferenceSpectraDataset(manifest, splits=("train",), log_transform=False)
        ref_val_raw = ReferenceSpectraDataset(manifest, splits=("val",), log_transform=False)
        n_ref_train = len(ref_train_raw)
        log.info("Reference spectra: %d train, %d val", n_ref_train, len(ref_val_raw))
    else:
        log.warning("Manifest not found at %s — using synthetic only.", manifest_path)

    # ---- Fit preprocessing from train spectra only ----
    train_raw_spectra: list[np.ndarray] = [np.asarray(spec, dtype=np.float32) for spec in X_train]
    if ref_train_raw is not None:
        train_raw_spectra.extend(np.asarray(spec, dtype=np.float32) for spec in ref_train_raw._X)
    input_transform = fit_input_transform(
        train_raw_spectra,
        saturation_epsilon=cfg.saturation_epsilon,
    )

    prior_extractor = (
        SpectralPriorExtractor.fit_from_manifest(manifest, target_species=TARGET_SPECIES, splits=("train",))
        if cfg.use_prior_features and manifest is not None
        else None
    )

    # ---- Fit LabelNormalizer from synthetic train labels ----
    # Use log-transformed labels (what the datasets will yield at __getitem__).
    y_train_log = np.log1p(np.clip(y_train.astype(np.float32), 0.0, None))
    label_normalizer = LabelNormalizer.fit(y_train_log)
    log.info(
        "LabelNormalizer fitted: means(log)=[%s]  stds(log)=[%s]  species_weights=[%s]",
        " ".join(f"{v:.3f}" for v in label_normalizer.means),
        " ".join(f"{v:.3f}" for v in label_normalizer.stds),
        " ".join(f"{v:.3f}" for v in label_normalizer.species_weights),
    )

    synth_train = ArraySpectrumDataset(
        X_train,
        y_train,
        input_transform=input_transform,
        prior_extractor=prior_extractor,
    )
    synth_val = ArraySpectrumDataset(
        X_val,
        y_val,
        input_transform=input_transform,
        prior_extractor=prior_extractor,
    )

    if manifest is not None and ref_train_raw is not None:
        # Reuse already-loaded spectra — just apply transforms.
        ref_train = ref_train_raw.with_transforms(
            input_transform=input_transform,
            prior_extractor=prior_extractor,
            log_transform=True,
        )
        ref_val = ref_val_raw.with_transforms(
            input_transform=input_transform,
            prior_extractor=prior_extractor,
            log_transform=True,
        )
        log.info("Reference spectra (with transforms): %d train, %d val", len(ref_train), len(ref_val))

        if n_ref_train > 0:
            train_ds = ConcatDataset([synth_train, ref_train])
        else:
            train_ds = synth_train

        if len(ref_val) > 0:
            val_ds = ConcatDataset([synth_val, ref_val])
        else:
            val_ds = synth_val
    else:
        train_ds = synth_train
        val_ds = synth_val

    return PreparedDatasets(
        train_ds=train_ds,
        val_ds=val_ds,
        n_ref_train=n_ref_train,
        synth_val_ds=synth_val,
        ref_val_ds=ref_val,
        n_synth_train=len(X_train),
        input_transform=input_transform,
        prior_extractor=prior_extractor,
        label_normalizer=label_normalizer,
    )


def _build_dataloader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    n_synthetic: int = 0,
    n_reference: int = 0,
    reference_weight: float = 0.2,
) -> DataLoader:
    """Build DataLoader, optionally with weighted sampling to up-sample reference spectra."""
    if shuffle and n_reference > 0 and n_synthetic > 0:
        # Give reference samples higher weight so they appear reference_weight% of batches
        # even when greatly outnumbered by synthetic samples.
        w_synth = (1.0 - reference_weight) / n_synthetic
        w_ref = reference_weight / n_reference
        total = n_synthetic + n_reference
        weights = [w_synth] * n_synthetic + [w_ref] * n_reference
        sampler = WeightedRandomSampler(weights, total, replacement=True)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

class WeightedHuberLoss(nn.Module):
    """Huber loss with per-species and active/inactive weighting.

    active_mask must be passed at call time — it is computed from the original
    (pre-normalisation) labels so it correctly identifies zero/nonzero entries
    regardless of how normalisation shifts the values.

    species_weights: (n_species,) float tensor, registered as a buffer so it
    moves to the correct device automatically. Derived from LabelNormalizer.
    """

    def __init__(
        self,
        *,
        delta: float = 1.0,
        active_label_weight: float = 2.0,
        inactive_label_weight: float = 1.0,
        species_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction="none")
        self.active_label_weight = float(active_label_weight)
        self.inactive_label_weight = float(inactive_label_weight)
        if species_weights is not None:
            self.register_buffer("species_weights", species_weights.float())
        else:
            self.species_weights: torch.Tensor | None = None  # type: ignore[assignment]

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        element_loss = self.huber(pred, target)
        weights = torch.where(active_mask, self.active_label_weight, self.inactive_label_weight)
        if self.species_weights is not None:
            weights = weights * self.species_weights.to(weights.device)
        return (element_loss * weights).sum() / weights.sum().clamp_min(1e-8)

def _train_epoch(
    model: FTIRModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
    scaler: torch.amp.GradScaler,
    label_normalizer: LabelNormalizer,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    is_cuda = (device.type == "cuda")
    autocast_device = "cuda" if is_cuda else "cpu"

    for X_batch, aux_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        aux_batch = aux_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        # Compute active mask before normalisation (y_batch still in log1p space)
        active_mask = y_batch > 0
        y_batch_norm = label_normalizer.normalize(y_batch)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=autocast_device, enabled=is_cuda):
            pred = model(X_batch, aux=aux_batch if aux_batch.shape[-1] > 0 else None)
            loss = criterion(pred, y_batch_norm, active_mask=active_mask)

        scaler.scale(loss).backward()

        if cfg.grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def _eval_epoch(
    model: FTIRModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    label_normalizer: LabelNormalizer,
) -> tuple[float, np.ndarray, np.ndarray, dict[str, float]]:
    """Returns (mean_huber_loss, per_species_mae_ppmv, per_species_mae_log, raw_stats)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []
    all_pred_log: list[np.ndarray] = []
    all_true_log: list[np.ndarray] = []

    for X_batch, aux_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        aux_batch = aux_batch.to(device)
        y_batch = y_batch.to(device)

        active_mask = y_batch > 0
        y_batch_norm = label_normalizer.normalize(y_batch)

        pred_norm = model(X_batch, aux=aux_batch if aux_batch.shape[-1] > 0 else None)
        loss = criterion(pred_norm, y_batch_norm, active_mask=active_mask)
        total_loss += loss.item()
        n_batches += 1

        # Denormalise back to log1p(ppmv) for interpretable metrics
        pred_log = label_normalizer.denormalize(pred_norm)   # (B, 11) log1p(ppmv)
        true_log = y_batch                                    # (B, 11) log1p(ppmv)

        all_pred_log.append(pred_log.cpu().numpy())
        all_true_log.append(true_log.cpu().numpy())

        # Back-transform to ppmv (clip negatives to match inference)
        all_pred.append(np.clip(labels_from_log(pred_log).cpu().numpy(), 0.0, None))
        all_true.append(labels_from_log(true_log).cpu().numpy())

    pred_arr = np.concatenate(all_pred, axis=0)          # (N, 11) ppmv
    true_arr = np.concatenate(all_true, axis=0)          # (N, 11) ppmv
    pred_log_arr = np.concatenate(all_pred_log, axis=0)  # (N, 11) log1p(ppmv)
    true_log_arr = np.concatenate(all_true_log, axis=0)  # (N, 11) log1p(ppmv)

    per_species_mae = np.abs(pred_arr - true_arr).mean(axis=0)
    per_species_log_mae = np.abs(pred_log_arr - true_log_arr).mean(axis=0)

    raw_stats = {
        "min": float(pred_log_arr.min()),
        "max": float(pred_log_arr.max()),
        "mean": float(pred_log_arr.mean()),
    }

    return total_loss / max(n_batches, 1), per_species_mae, per_species_log_mae, raw_stats


def _save_mae_plot(mae: np.ndarray, epoch: int, reports_dir: Path) -> None:
    set_mpl_config_if_needed()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        reports_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2196F3" if m < 100 else "#FF5722" for m in mae]
        ax.bar(TARGET_SPECIES, mae, color=colors)
        ax.set_ylabel("MAE (ppmv)", fontsize=12)
        ax.set_title(f"Per-species MAE — epoch {epoch}", fontsize=13)
        ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(mae):
            ax.text(i, v * 1.05, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        plt.tight_layout()
        out = reports_dir / f"mae_per_species_epoch{epoch:04d}.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        log.info("Saved MAE plot → %s", out)
    except ImportError:
        log.warning("matplotlib not installed — skipping MAE plot.")


def _zero_baseline_log_mae(loader: DataLoader) -> np.ndarray:
    """Per-species log-MAE for constant zero prediction.

    Labels from the DataLoader are in raw log1p(ppmv) space (not normalised),
    so zero-prediction baseline is just abs(true_log).
    """
    all_true_log: list[np.ndarray] = []
    for _, _, y_batch in loader:
        all_true_log.append(y_batch.numpy())
    true_log_arr = np.concatenate(all_true_log, axis=0)
    return np.abs(true_log_arr).mean(axis=0)


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def train_from_manifest(cfg: TrainConfig | None = None) -> FTIRModel:
    """Full training run. Returns the best model."""
    if cfg is None:
        cfg = TrainConfig()

    seed_everything(cfg.seed)
    device = resolve_device(cfg.device)
    log.info("Training on device: %s", device)

    # ---- Datasets ----
    prepared = _build_datasets(cfg)
    train_ds = prepared.train_ds
    val_ds = prepared.val_ds
    n_ref_train = prepared.n_ref_train
    synth_val_ds = prepared.synth_val_ds
    ref_val_ds = prepared.ref_val_ds
    n_synth_train = prepared.n_synth_train

    log.info(
        "Dataset sizes: train=%d (synth=%d ref=%d), val=%d",
        len(train_ds), n_synth_train, n_ref_train, len(val_ds),
    )
    log.info(
        "Input transform: raw_scale=%.3f derivative_scale=%.6f saturation_eps=%.6f",
        prepared.input_transform.raw_scale,
        prepared.input_transform.derivative_scale,
        prepared.input_transform.saturation_epsilon,
    )
    if prepared.prior_extractor is not None:
        log.info("Light prior features enabled: n_features=%d", prepared.prior_extractor.n_features)
    else:
        log.info("Light prior features disabled")

    train_loader = _build_dataloader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        n_synthetic=n_synth_train,
        n_reference=n_ref_train,
        reference_weight=cfg.reference_weight,
    )
    val_batch_size = cfg.batch_size * 2
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    synth_val_loader = DataLoader(synth_val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
    ref_val_loader = (
        DataLoader(ref_val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)
        if ref_val_ds is not None and len(ref_val_ds) > 0
        else None
    )

    # ---- Model ----
    model = FTIRModel(
        n_species=cfg.n_species,
        in_channels=3,
        aux_features=(prepared.prior_extractor.n_features if prepared.prior_extractor is not None else 0),
        dropout=cfg.dropout,
    ).to(device)
    log.info(
        "Model v4: %d parameters (~%.1fM)",
        count_parameters(model),
        count_parameters(model) / 1e6,
    )

    # ---- Optimiser + LR schedule ----
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: _cosine_with_warmup_fn(
            s, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )

    label_normalizer = prepared.label_normalizer
    species_weights_t = torch.tensor(label_normalizer.species_weights, dtype=torch.float32)
    criterion = WeightedHuberLoss(
        delta=cfg.huber_delta,
        active_label_weight=cfg.active_label_weight,
        inactive_label_weight=cfg.inactive_label_weight,
        species_weights=species_weights_t,
    ).to(device)
    log.info(
        "Loss: WeightedHuber(delta=%.2f, active_w=%.2f, inactive_w=%.2f, species_w=[%s])",
        cfg.huber_delta,
        cfg.active_label_weight,
        cfg.inactive_label_weight,
        " ".join(f"{v:.3f}" for v in label_normalizer.species_weights),
    )

    # ---- Paths ----
    checkpoint_dir = Path(cfg.checkpoint_dir or CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg.reports_dir or REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Baseline diagnostics (zero predictor in log space).
    zero_baseline_log_mae_all = _zero_baseline_log_mae(val_loader)
    zero_baseline_log_mae_synth = _zero_baseline_log_mae(synth_val_loader)
    zero_baseline_log_mae_ref = (
        _zero_baseline_log_mae(ref_val_loader) if ref_val_loader is not None else None
    )

    zero_baseline_mean_all = float(zero_baseline_log_mae_all.mean())
    zero_baseline_mean_synth = float(zero_baseline_log_mae_synth.mean())
    zero_baseline_mean_ref = (
        float(zero_baseline_log_mae_ref.mean()) if zero_baseline_log_mae_ref is not None else None
    )

    ref_baseline_txt = (
        f"{zero_baseline_mean_ref:.3f}" if zero_baseline_mean_ref is not None else "n/a"
    )
    log.info(
        "Zero baseline mean log-MAE: all=%.3f  synth=%.3f  ref=%s",
        zero_baseline_mean_all,
        zero_baseline_mean_synth,
        ref_baseline_txt,
    )

    best_selection_metric = float("inf")
    best_ckpt = checkpoint_dir / "best_model.pt"
    best_ref_metric = float("inf")
    best_mixed_metric = float("inf")
    best_ref_summary: EpochSummary | None = None
    best_mixed_summary: EpochSummary | None = None

    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ---- Training loop ----
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg, scaler,
            label_normalizer,
        )
        val_loss, mae, log_mae, raw_stats = _eval_epoch(
            model, val_loader, criterion, device, label_normalizer,
        )
        _, _, synth_log_mae, _ = _eval_epoch(
            model, synth_val_loader, criterion, device, label_normalizer,
        )
        ref_log_mae = None
        if ref_val_loader is not None:
            _, _, ref_log_mae, _ = _eval_epoch(
                model, ref_val_loader, criterion, device, label_normalizer,
            )

        summary = _build_epoch_summary(
            epoch=epoch,
            val_loss=val_loss,
            val_log_mae=log_mae,
            synth_log_mae=synth_log_mae,
            ref_log_mae=ref_log_mae,
            zero_baseline_mean_all=zero_baseline_mean_all,
            zero_baseline_mean_ref=zero_baseline_mean_ref,
            species_beating_zero=int((log_mae < zero_baseline_log_mae_all).sum()),
        )

        delta_vs_zero_all = zero_baseline_mean_all - summary.val_log_mae_mean
        delta_vs_zero_synth = zero_baseline_mean_synth - summary.synth_val_log_mae_mean
        delta_vs_zero_ref = (
            (zero_baseline_mean_ref - summary.ref_val_log_mae_mean)
            if summary.ref_val_log_mae_mean is not None and zero_baseline_mean_ref is not None
            else None
        )

        ref_split_txt = (
            f"{summary.ref_val_log_mae_mean:.3f} (Δvs0={delta_vs_zero_ref:+.3f})"
            if summary.ref_val_log_mae_mean is not None and delta_vs_zero_ref is not None
            else "n/a"
        )
        current_lr = optimizer.param_groups[0]["lr"]
        log.info(
            "Epoch %3d/%d  train=%.4f  val_weighted=%.4f  val_log_mean=%.4f  selection=%s:%.4f  lr=%.2e",
            epoch,
            cfg.epochs,
            train_loss,
            val_loss,
            summary.val_log_mae_mean,
            summary.selection_metric,
            summary.selection_metric_value,
            current_lr,
        )
        log.info(
            "Val log-MAE split: all=%.3f (Δvs0=%+.3f, beat=%d/%d)  synth=%.3f (Δvs0=%+.3f)  ref=%s",
            summary.val_log_mae_mean,
            delta_vs_zero_all,
            summary.species_beating_zero_baseline,
            len(TARGET_SPECIES),
            summary.synth_val_log_mae_mean,
            delta_vs_zero_synth,
            ref_split_txt,
        )
        log.info(
            "Val group log-MAE: all=%s  synth=%s  ref=%s",
            _format_group_metrics(summary.group_log_mae_mean["all"]),
            _format_group_metrics(summary.group_log_mae_mean["synth"]),
            _format_group_metrics(summary.group_log_mae_mean["ref"]),
        )

        if epoch % cfg.log_every_n_epochs == 0 or epoch == cfg.epochs:
            log.info(
                "MAE (ppmv): %s",
                "  ".join(f"{s}={v:.1f}" for s, v in zip(TARGET_SPECIES, mae)),
            )
            log.info(
                "MAE (log):  %s",
                "  ".join(f"{s}={v:.2f}" for s, v in zip(TARGET_SPECIES, log_mae)),
            )
            if epoch % (cfg.log_every_n_epochs * 2) == 0 or epoch == 1:
                log.info(
                    "Raw Pred Stats -> min: %.3f, max: %.3f, mean: %.3f",
                    raw_stats["min"], raw_stats["max"], raw_stats["mean"],
                )
            _save_mae_plot(mae, epoch, reports_dir)

        if summary.ref_val_log_mae_mean is not None and summary.ref_val_log_mae_mean < best_ref_metric:
            best_ref_metric = summary.ref_val_log_mae_mean
            best_ref_summary = summary

        if summary.val_log_mae_mean < best_mixed_metric:
            best_mixed_metric = summary.val_log_mae_mean
            best_mixed_summary = summary

        if summary.selection_metric_value < best_selection_metric:
            best_selection_metric = summary.selection_metric_value
            metadata = build_checkpoint_metadata(
                TARGET_SPECIES,
                model_version=MODEL_VERSION,
                notes="FTIR v4 reference-first training checkpoint",
            )
            metadata.update(
                {
                    "epoch": epoch,
                    **_epoch_summary_payload(summary),
                    "species_groups": {name: list(species) for name, species in SPECIES_GROUPS.items()},
                    "input_transform": prepared.input_transform.as_dict(),
                    "input_channels": ["absorbance", "derivative", "saturation_mask"],
                    "use_prior_features": prepared.prior_extractor is not None,
                    "prior_feature_config": (
                        prepared.prior_extractor.describe() if prepared.prior_extractor is not None else None
                    ),
                    "label_normalizer": prepared.label_normalizer.as_dict(),
                    "selection_metric": summary.selection_metric,
                }
            )
            save_checkpoint(best_ckpt, model.state_dict(), metadata)

    summary_payload = {
        "selection_metric": (
            "ref_val_log_mae_mean" if ref_val_ds is not None and len(ref_val_ds) > 0 else "val_log_mae_mean"
        ),
        "best_checkpoint": str(best_ckpt),
        "species_groups": {name: list(species) for name, species in SPECIES_GROUPS.items()},
        "best_ref_epoch": _epoch_summary_payload(best_ref_summary) if best_ref_summary is not None else None,
        "best_mixed_epoch": _epoch_summary_payload(best_mixed_summary) if best_mixed_summary is not None else None,
    }
    summary_path = reports_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    _log_best_epoch("Best reference", best_ref_summary)
    _log_best_epoch("Best mixed", best_mixed_summary)
    log.info("Saved training summary → %s", summary_path)
    log.info(
        "Best %s: %.4f — checkpoint: %s",
        "ref_val_log_mae_mean" if ref_val_ds is not None and len(ref_val_ds) > 0 else "val_log_mae_mean",
        best_selection_metric,
        best_ckpt,
    )
    return model
