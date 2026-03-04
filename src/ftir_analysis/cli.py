"""Command-line interface for FTIR rework pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from .auditing import audit_manifest
from .checkpointing import build_checkpoint_metadata, save_checkpoint
from .constants import CHECKPOINT_DIR, DEFAULT_TARGET_SPECIES, MANIFEST_FILENAME, REFERENCE_ROOT
from .evaluate import evaluate_manifest
from .inference_runtime import make_inference_config, run_inference
from .manifesting import build_manifest
from .training import TrainConfig, train_from_manifest


def _parse_species_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FTIR ML fitting pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser("build-manifest", help="Build canonical manifest_v1.csv")
    p_manifest.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)
    p_manifest.add_argument("--index", type=Path, default=None)
    p_manifest.add_argument("--output", type=Path, default=None)
    p_manifest.add_argument("--seed", type=int, default=42)
    p_manifest.add_argument("--primary-threshold", type=int, default=10)

    p_audit = sub.add_parser("audit-manifest", help="Generate data audit reports")
    p_audit.add_argument("--manifest", type=Path, default=REFERENCE_ROOT / MANIFEST_FILENAME)
    p_audit.add_argument("--reference-root", type=Path, default=REFERENCE_ROOT)

    p_train = sub.add_parser("train", help="Train FTIR model from manifest")
    p_train.add_argument("--manifest", type=Path, default=REFERENCE_ROOT / MANIFEST_FILENAME)
    p_train.add_argument("--target-species", type=str, default=None)
    p_train.add_argument("--epochs", type=int, default=50)
    p_train.add_argument("--batch-size", type=int, default=64)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--weight-decay", type=float, default=1e-4)
    p_train.add_argument("--seed", type=int, default=42)
    p_train.add_argument("--run-name", type=str, default="ftir_solver_primary")
    p_train.add_argument("--use-synthetic-aux", action="store_true")

    p_infer = sub.add_parser("infer", help="Run inference over folder of spectra")
    p_infer.add_argument("--data-dir", type=Path, default=Path("example_data"))
    p_infer.add_argument("--checkpoint", type=Path, default=CHECKPOINT_DIR / "ftir_solver_best.pth")
    p_infer.add_argument("--output", type=Path, default=None)

    p_eval = sub.add_parser("evaluate", help="Evaluate baseline and optional model")
    p_eval.add_argument("--manifest", type=Path, default=REFERENCE_ROOT / MANIFEST_FILENAME)
    p_eval.add_argument("--checkpoint", type=Path, default=None)
    p_eval.add_argument("--report-prefix", type=str, default="evaluation")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "build-manifest":
        manifest = build_manifest(
            reference_root=args.reference_root,
            index_path=args.index,
            output_path=args.output,
            seed=args.seed,
            primary_threshold=args.primary_threshold,
        )
        out = args.output or (args.reference_root / MANIFEST_FILENAME)
        print(f"Manifest rows: {len(manifest)}")
        print(f"Manifest written: {out}")
        return 0

    if args.command == "audit-manifest":
        outputs = audit_manifest(args.manifest, reference_root=args.reference_root)
        for name, path in outputs.items():
            print(f"{name}: {path}")
        return 0

    if args.command == "train":
        cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            run_name=args.run_name,
            use_synthetic_aux=args.use_synthetic_aux,
        )
        target_species = _parse_species_list(args.target_species)
        report = train_from_manifest(args.manifest, target_species=target_species, cfg=cfg)
        print(f"Run name: {report['run_name']}")
        print(f"Best checkpoint: {report['checkpoint_best']}")
        print(f"Final checkpoint: {report['checkpoint_final']}")
        return 0

    if args.command == "infer":
        cfg = make_inference_config(args.data_dir, checkpoint_path=args.checkpoint, output_csv=args.output)
        out = run_inference(cfg)
        print(f"Inference results written: {out}")
        return 0

    if args.command == "evaluate":
        report = evaluate_manifest(args.manifest, checkpoint_path=args.checkpoint, report_prefix=args.report_prefix)
        print(f"Evaluation report written: reports/{args.report_prefix}_report.json")
        if report.get("model_test"):
            print("Model test MAE:", report["model_test"]["mae"])
        if report.get("baseline_test"):
            print("Baseline test MAE:", report["baseline_test"]["mae"])
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
