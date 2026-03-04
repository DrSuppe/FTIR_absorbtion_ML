"""Legacy entrypoint wrapper for FTIR inference.

This script preserves the historical `python inference.py` workflow but delegates
execution to the hardened runtime in `src/ftir_analysis`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ftir_analysis.inference_runtime import make_inference_config, run_inference as run_inference_runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FTIR inference over spectra folder")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "example_data")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "ftir_solver_best.pth")
    parser.add_argument("--output", type=Path, default=None)
    return parser


def run_inference_wrapper(data_dir: Path, model_path: Path, output: Path | None = None) -> Path:
    """Compatibility wrapper for old function-style usage."""
    cfg = make_inference_config(data_dir, checkpoint_path=model_path, output_csv=output)
    return run_inference_runtime(cfg)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = run_inference_wrapper(args.data_dir, args.checkpoint, args.output)
    print(f"Inference complete. Results saved to {out}")
    return 0


# Backward-compatible function name expected by older notebooks/scripts.
run_inference = run_inference_wrapper


if __name__ == "__main__":
    raise SystemExit(main())
