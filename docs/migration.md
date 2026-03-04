# Migration Notes

## What Changed

- Core implementation moved to `src/ftir_analysis`.
- `inference.py` and `train.py` are wrappers over the package API.
- Checkpoints now require sidecar metadata (`*.meta.json`) for strict compatibility checks.
- Inference enforces fixed-grid preprocessing and log-space inverse (`expm1(clamp(x, 0))`).

## Checkpoint Contract

Each checkpoint must have JSON sidecar fields:

- `model_version`
- `target_species`
- `grid_min`, `grid_max`, `grid_step`
- `label_transform`
- `train_data_manifest_hash`

Inference rejects checkpoints missing sidecar metadata or incompatible target/grid settings.

## Manifest Contract

`reference_spectra/manifest_v1.csv` columns:

- `sample_id`
- `source_path`
- `source_format`
- `species`
- `concentration_ppmv`
- `temperature_c`
- `path_length_cm`
- `is_sparse_class`
- `split`
- `quality_flags`

## Legacy Compatibility

Legacy scripts still exist:

- `python3 inference.py`
- `python3 train.py`

For full workflows, use:

- `python3 runner.py <command>`
