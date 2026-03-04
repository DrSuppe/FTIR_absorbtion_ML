"""Convenience runner for FTIR pipeline commands.

Examples:
  python3 runner.py build-manifest
  python3 runner.py audit-manifest
  python3 runner.py train --manifest reference_spectra/manifest_v1.csv
  python3 runner.py infer --data-dir example_data
  python3 runner.py evaluate --manifest reference_spectra/manifest_v1.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ftir_analysis.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
