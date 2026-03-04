"""Shared constants for FTIR analysis pipelines."""

from __future__ import annotations

from pathlib import Path

# Spectral grid
WAVENUMBER_MIN = 800.0
WAVENUMBER_MAX = 5000.0
WAVENUMBER_STEP = 0.25
SATURATION_AU = 8.0

GRID_NPTS = int((WAVENUMBER_MAX - WAVENUMBER_MIN) / WAVENUMBER_STEP)

# 11-gas target order (model output head).
DEFAULT_TARGET_SPECIES = ["H2O", "CO2", "CO", "NO", "NO2", "NH3", "CH4", "N2O", "C2H4", "HCN", "HNCO"]

# Species present in the SPC library but NOT in the model output head.
# They appear as interference gases in synthetic training samples.
INTERFERENCE_SPECIES = ["C2H6", "C2H6O", "CH4N2O", "HNO3", "H2SO4", "Biodiesel", "Diesel", "NOxP"]

# Manifest defaults.
MANIFEST_FILENAME = "manifest_v1.csv"
PRIMARY_CLASS_THRESHOLD = 10

MANIFEST_COLUMNS = [
    "sample_id",
    "source_path",
    "source_format",
    "species",
    "concentration_ppmv",
    "temperature_c",
    "path_length_cm",
    "is_sparse_class",
    "split",
    "quality_flags",
]

SUPPORTED_SOURCE_FORMATS = {"spc", "csv", "txt"}

# Checkpoint metadata
MODEL_VERSION = "ftir_solver_v3_0"
LABEL_TRANSFORM = "log1p_ppmv"
METADATA_SUFFIX = ".meta.json"

# Paths
_module_dir = Path(__file__).resolve().parent
if (_module_dir.parents[1] / "synthetic_generator.py").exists():
    PROJECT_ROOT = _module_dir.parents[1]
elif Path.cwd().joinpath("synthetic_generator.py").exists():
    PROJECT_ROOT = Path.cwd()
else:
    PROJECT_ROOT = _module_dir.parents[1]

REFERENCE_ROOT = PROJECT_ROOT / "data" / "reference"
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
