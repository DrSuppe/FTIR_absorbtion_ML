# Changelog

All notable changes to this project are documented here.

---

## [v4.1] — 2026-04-01

### Added
- **`LabelNormalizer`** (`utils.py`): per-species zero-mean/unit-std normalizer fitted on active (nonzero) training labels. Equalizes regression difficulty across species spanning very different concentration ranges (H₂O ~10,000 ppmv vs HCN ~5 ppmv). Stored in checkpoint sidecar metadata and required for denormalization at inference.
- **Per-species loss weights**: `1/log1p(median_active_ppmv)`, normalized to mean=1. Trace species receive proportionally higher loss weight. Computed as part of `LabelNormalizer.fit()`.
- **`smooth_saturate(y, S)`** (`features.py`): `S·tanh(y/S)` replaces the hard clip at `SATURATION_AU=8.0`. Applied in `build_input_channels` and in synthetic generation. Preserves gradient information near the detector saturation ceiling.
- **Independent synthetic val set**: val NPZ is generated with `seed+1` (separate RNG from train). Eliminates the statistical identity between train and val that existed with the old 80/20 in-memory split.
- **`FTIRModel.predict_with_uncertainty()`** (`modeling.py`): MC Dropout inference — N forward passes with dropout active, returns `(mean, std)` per species. Activated via `InferenceConfig.mc_dropout_samples > 0`.
- **Uncertainty columns in inference output**: when MC Dropout is enabled, output CSV gains `{species}_uncertainty_log` columns.
- **`_temperature_perturbation()` and `_sample_temperature()`** (`synthetic_generator.py`): functions for temperature-dependent spectral perturbation (intensity scaling, line broadening, band-specific gain shifts) — implemented but dormant by default (not called during generation). Can be re-enabled for future experiments.
- **Authoritative `GRID`** (`constants.py`): `np.linspace(WAVENUMBER_MIN, WAVENUMBER_MAX, GRID_NPTS, endpoint=False)` as the single source of truth. Eliminates float-rounding drift between `np.arange` and `int(...)` arithmetic.

### Changed
- **`GroupNorm(8)` replaces `BatchNorm1d`** throughout the CNN backbone (`ResBlock1D`, `SpectralCNN.stem`). Batch-size independent; stable when synthetic and reference spectra are mixed in a batch.
- **Model head: softplus → linear**. The model now outputs in normalized label space. Non-negativity is enforced after denormalization (`expm1 + clip(≥0)`) at inference and evaluation time, not by an activation.
- **`WeightedHuberLoss`**: accepts `species_weights` as a registered buffer (moves to device automatically) and requires an external `active_mask` parameter computed from the original (pre-normalization) labels.
- **`_build_datasets`**: rewritten to generate two independent NPZ files (train with `seed`, val with `seed+1`) instead of in-memory 80/20 split.
- **`_ensure_synthetic`**: accepts `seed`, `n_samples`, and `out_path` overrides to support independent val generation.
- **`MODEL_VERSION`**: `ftir_solver_v4_0` → `ftir_solver_v4_1`
- **`LABEL_TRANSFORM`**: `log1p_ppmv` → `log1p_ppmv_normalized`
- **`SCHEMA_VERSION`**: `1` → `2`
- **`GRID_NPTS`**: `int(...)` → `round(...)` for rounding safety.
- **`validate_metadata`**: now requires `label_normalizer` field. Checkpoints without it raise `CheckpointMetadataError`.
- **`build_checkpoint_metadata`**: uses `SCHEMA_VERSION` constant instead of hardcoded `1`.
- **Synthetic generator SPC loader**: hard clip `np.clip(arr, 0, SATURATION_AU)` changed to floor-only `np.clip(arr, 0, None)` — smooth saturation applied later in `augment()`.
- **Colab notebook** (`colab_train.ipynb`): switched from `pip install . --no-deps` to `pip install -e . --no-deps` (editable install). This ensures subprocess calls (`!python3 train.py`) use the same live source as the notebook kernel, preventing stale dist-packages copies from shadowing updates.
- **Colab notebook**: title updated from v3 → v4.1.
- **`make_inference_config`**: accepts new `mc_dropout_samples: int = 0` parameter.

### Fixed
- **`NameError: name 'F' is not defined`** in `modeling.py`: `ResBlock1D.forward` uses `F.relu` but the `import torch.nn.functional as F` was missing after a cleanup pass. Re-added.
- **Test fixture** (`tests/test_inference_runtime.py`): `_write_v4_checkpoint` now includes a minimal valid `label_normalizer` dict in checkpoint metadata, since `validate_metadata` now requires it.
- **`SPCLibrary` train-only flag**: `train_only=True` prevents test-split spectra from leaking into synthetic generation.
- **Spectra grid**: `spectra.py` imports `GRID` from constants instead of recomputing with `np.arange`.

---

## [v4.0] — prior

- Initial CNN → SelfAttention → BiGRU → MLP architecture
- Per-species Huber loss with active/inactive weighting
- Synthetic data via Beer-Lambert combination with concentration interpolation
- Reference spectra manifest with train/val/test splits
- Checkpoint metadata sidecar (schema v1)
- Colab training notebook
