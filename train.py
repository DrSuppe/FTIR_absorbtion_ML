"""CLI entrypoint for FTIR model training."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ["FTIR_PROJECT_ROOT"] = str(Path(__file__).resolve().parent)

from ftir_analysis.training import TrainConfig, train_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FTIR ML Solver v4")

    # Data
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu, mps, cuda (default: auto-detect)")
    parser.add_argument("--n-synthetic", type=int, default=20_000,
                        help="Synthetic spectra to generate (default: 20000).")
    parser.add_argument("--synthetic-npz", type=str, default=None,
                        help="Path to pre-generated .npz (skips generation if supplied)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Override manifest_v1.csv path")
    parser.add_argument("--reference-weight", type=float, default=0.2,
                        help="Fraction of each batch drawn from real reference spectra (0–1)")
    parser.add_argument("--synthetic-sampling-mode", type=str, default="hybrid_v4",
                        choices=["default", "curriculum_v2", "hybrid_v4"])
    parser.add_argument("--synthetic-augmentation-profile", type=str, default="mild",
                        choices=["mild", "strong"])
    parser.add_argument("--hybrid-trace-fraction", type=float, default=0.15,
                        help="Fraction of hybrid_v4 synthetic samples that force a trace species")
    parser.add_argument("--min-active-species", type=int, default=1)
    parser.add_argument("--max-active-species", type=int, default=4)
    parser.add_argument("--use-prior-features", action="store_true",
                        help="Enable light spectroscopy prior features derived from references")

    # Training
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=2.0)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--active-label-weight", type=float, default=4.0,
                        help="Relative loss weight for non-zero target entries")
    parser.add_argument("--inactive-label-weight", type=float, default=0.5,
                        help="Relative loss weight for zero target entries")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.20)

    # Paths
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--reports-dir", type=str, default=None)

    parser.add_argument("--log-every", type=int, default=1,
                        help="Save MAE plot and log per-species stats every N epochs")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = TrainConfig(
        device=args.device,
        n_synthetic=args.n_synthetic,
        synthetic_npz=args.synthetic_npz,
        manifest_path=args.manifest,
        reference_weight=args.reference_weight,
        synthetic_sampling_mode=args.synthetic_sampling_mode,
        synthetic_augmentation_profile=args.synthetic_augmentation_profile,
        hybrid_trace_fraction=args.hybrid_trace_fraction,
        min_active_species=args.min_active_species,
        max_active_species=args.max_active_species,
        use_prior_features=args.use_prior_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        dropout=args.dropout,
        huber_delta=args.huber_delta,
        active_label_weight=args.active_label_weight,
        inactive_label_weight=args.inactive_label_weight,
        seed=args.seed,
        val_split_fraction=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        log_every_n_epochs=args.log_every,
    )

    train_from_manifest(cfg)


if __name__ == "__main__":
    main()
