"""CLI entrypoint for FTIR model training."""

from __future__ import annotations

import argparse
import logging

from ftir_analysis.training import TrainConfig, train_from_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FTIR ML Solver v3")

    # Data
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: cpu, mps, cuda (default: auto-detect)")
    parser.add_argument("--n-synthetic", type=int, default=10_000,
                        help="Synthetic spectra to generate (default: 10000). "
                             "Use 50000+ for a full training run.")
    parser.add_argument("--synthetic-npz", type=str, default=None,
                        help="Path to pre-generated .npz (skips generation if supplied)")
    parser.add_argument("--manifest", type=str, default=None,
                        help="Override manifest_v1.csv path")
    parser.add_argument("--reference-weight", type=float, default=0.2,
                        help="Fraction of each batch drawn from real reference spectra (0–1)")

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--dropout", type=float, default=0.15)
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        dropout=args.dropout,
        seed=args.seed,
        val_split_fraction=args.val_split,
        checkpoint_dir=args.checkpoint_dir,
        reports_dir=args.reports_dir,
        log_every_n_epochs=args.log_every,
    )

    train_from_manifest(cfg)


if __name__ == "__main__":
    main()
