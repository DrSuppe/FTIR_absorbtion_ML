"""FTIR ML Solver v3 — Training pipeline.

Trains on a combined dataset of:
  1. Synthetic multi-gas spectra (Beer-Lambert combination of real reference spectra)
  2. Real single-gas reference spectra from the SPC library

Labels are in log1p(ppmv) space; outputs are inverted with expm1 for reporting.
"""

from __future__ import annotations

import dataclasses
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

from .constants import (
    CHECKPOINT_DIR,
    DEFAULT_TARGET_SPECIES,
    MANIFEST_FILENAME,
    MODEL_VERSION,
    REFERENCE_ROOT,
    REPORTS_DIR,
    SYNTHETIC_DIR,
)
from .datasets import (
    ArraySpectrumDataset,
    ReferenceSpectraDataset,
    load_synthetic_aux_arrays,
)
from .modeling import FTIRModel, count_parameters
from .utils import labels_from_log, resolve_device, seed_everything, set_mpl_config_if_needed

log = logging.getLogger(__name__)

TARGET_SPECIES = DEFAULT_TARGET_SPECIES


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class TrainConfig:
    # Data
    n_synthetic: int = 10_000
    synthetic_npz: Optional[str] = None        # override path
    manifest_path: Optional[str] = None        # override manifest
    reference_weight: float = 0.2              # fraction of each batch from real spectra

    # Model
    n_species: int = len(TARGET_SPECIES)       # 11
    dropout: float = 0.15

    # Optimiser
    epochs: int = 50
    batch_size: int = 64
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: float = 3.0                 # linear warmup duration
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


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _ensure_synthetic(cfg: TrainConfig) -> Path:
    """Return path to synthetic .npz, generating it if necessary."""
    npz_path = (
        Path(cfg.synthetic_npz)
        if cfg.synthetic_npz
        else Path(SYNTHETIC_DIR) / "spectra.npz"
    )
    regen = False
    if not npz_path.exists():
        log.info("Synthetic data not found — generating %d samples …", cfg.n_synthetic)
        regen = True
    else:
        # Check if the .npz has the expected number of samples
        try:
            data = np.load(npz_path, allow_pickle=False)
            n_existing = data["X"].shape[0]
            if n_existing < cfg.n_synthetic:
                log.info(
                    "Existing synthetic file has %d samples (need %d) — regenerating.",
                    n_existing, cfg.n_synthetic,
                )
                regen = True
        except Exception:
            regen = True

    if regen:
        generator = Path(__file__).resolve().parents[2] / "synthetic_generator.py"
        cmd = [
            sys.executable, str(generator),
            "--n-samples", str(cfg.n_synthetic),
            "--seed", str(cfg.seed),
            "--out", str(npz_path),
        ]
        log.info("Running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)

    return npz_path


def _build_datasets(cfg: TrainConfig):
    """Return (train_dataset, val_dataset, n_reference_train)."""
    # ---- Synthetic data ----
    npz_path = _ensure_synthetic(cfg)
    loaded = load_synthetic_aux_arrays(npz_path)
    if loaded is None:
        raise RuntimeError("Could not load synthetic data — check generation logs.")
    X_all, y_all = loaded

    # 80/20 train/val split on synthetic data
    rng = np.random.default_rng(cfg.seed)
    n = len(X_all)
    idx = rng.permutation(n)
    n_val = max(1, int(n * cfg.val_split_fraction))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    synth_train = ArraySpectrumDataset(X_all[train_idx], y_all[train_idx])
    synth_val = ArraySpectrumDataset(X_all[val_idx], y_all[val_idx])

    # ---- Reference spectra ----
    manifest_path = (
        Path(cfg.manifest_path)
        if cfg.manifest_path
        else Path(REFERENCE_ROOT) / MANIFEST_FILENAME
    )
    n_ref_train = 0
    if manifest_path.exists():
        manifest = pd.read_csv(manifest_path)
        ref_train = ReferenceSpectraDataset(manifest, splits=("train", "val"))
        ref_val = ReferenceSpectraDataset(manifest, splits=("test",))
        n_ref_train = len(ref_train)
        log.info("Reference spectra: %d train, %d test", n_ref_train, len(ref_val))

        if n_ref_train > 0:
            train_ds = ConcatDataset([synth_train, ref_train])
        else:
            train_ds = synth_train

        if len(ref_val) > 0:
            val_ds = ConcatDataset([synth_val, ref_val])
        else:
            val_ds = synth_val
    else:
        log.warning("Manifest not found at %s — using synthetic only.", manifest_path)
        train_ds = synth_train
        val_ds = synth_val
        n_ref_train = 0

    return train_ds, val_ds, n_ref_train


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

def _train_epoch(
    model: FTIRModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    criterion: nn.Module,
    device: torch.device,
    cfg: TrainConfig,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()

        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        optimizer.step()
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
) -> tuple[float, np.ndarray]:
    """Returns (mean_loss, per_species_mae_ppmv)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_pred: list[np.ndarray] = []
    all_true: list[np.ndarray] = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item()
        n_batches += 1

        # Back-transform to ppmv for MAE
        all_pred.append(labels_from_log(pred).cpu().numpy())
        all_true.append(labels_from_log(y_batch).cpu().numpy())

    pred_arr = np.concatenate(all_pred, axis=0)   # (N, 11)
    true_arr = np.concatenate(all_true, axis=0)   # (N, 11)
    per_species_mae = np.abs(pred_arr - true_arr).mean(axis=0)  # (11,)

    return total_loss / max(n_batches, 1), per_species_mae


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
    train_ds, val_ds, n_ref_train = _build_datasets(cfg)
    n_synth_train = len(train_ds) - n_ref_train

    log.info(
        "Dataset sizes: train=%d (synth=%d ref=%d), val=%d",
        len(train_ds), n_synth_train, n_ref_train, len(val_ds),
    )

    train_loader = _build_dataloader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        n_synthetic=n_synth_train,
        n_reference=n_ref_train,
        reference_weight=cfg.reference_weight,
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=0)

    # ---- Model ----
    model = FTIRModel(n_species=cfg.n_species, dropout=cfg.dropout).to(device)
    log.info(
        "Model v3: %d parameters (~%.1fM)",
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

    criterion = nn.HuberLoss(delta=cfg.huber_delta)

    # ---- Paths ----
    checkpoint_dir = Path(cfg.checkpoint_dir or CHECKPOINT_DIR)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg.reports_dir or REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt = checkpoint_dir / "best_model.pt"

    # ---- Training loop ----
    for epoch in range(1, cfg.epochs + 1):
        train_loss = _train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device, cfg
        )
        val_loss, mae = _eval_epoch(model, val_loader, criterion, device)

        if epoch % cfg.log_every_n_epochs == 0 or epoch == cfg.epochs:
            current_lr = optimizer.param_groups[0]["lr"]
            log.info(
                "Epoch %3d/%d  train=%.4f  val=%.4f  lr=%.2e",
                epoch, cfg.epochs, train_loss, val_loss, current_lr,
            )
            log.info(
                "MAE (ppmv): %s",
                "  ".join(f"{s}={v:.1f}" for s, v in zip(TARGET_SPECIES, mae)),
            )
            _save_mae_plot(mae, epoch, reports_dir)

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "model_version": MODEL_VERSION,
                    "target_species": TARGET_SPECIES,
                },
                best_ckpt,
            )

    log.info("Best val loss: %.4f — checkpoint: %s", best_val_loss, best_ckpt)
    return model
