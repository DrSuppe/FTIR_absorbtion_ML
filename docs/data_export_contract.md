# Data Export Contract (v1)

## Scope

Reference labels are indexed by `reference_spectra/lab_index.csv`.

## Required Export Formats

The v1 ingestion path supports:

- `.spc` (MKS/Galactic exported spectra)
- `.csv` / `.txt` (numeric spectral export)

`.lab` parsing is intentionally disabled in v1.

## Expected CSV/TXT Layout

One of:

1. Two numeric columns: `wavenumber, absorbance`
2. One numeric column: `absorbance` already aligned to the canonical grid

## Unit Conventions

- Canonical concentration: **ppmv**.
- Index column `concentration_ppm` is interpreted as mole fraction and converted via `* 1e6`.
- Path length is canonicalized to centimeters (`path_length_cm`).

## Quality Flags

Manifest rows may include:

- `requires_export`
- `source_missing`
- `unsupported_source_format`
- `index_path_not_found`
- `sparse_class`
- `path_length_units_assumed_cm`

See `reports/label_anomalies.csv` and `reports/data_audit.md`.
