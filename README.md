# FTIR Absorption ML Solver

A physics-informed machine learning pipeline for quantifying gas concentrations from FTIR absorbance spectra.

## Overview

This project trains a neural network to predict the concentrations of **11 gas species** simultaneously from a single FTIR absorbance spectrum.

**Target species:** H₂O, CO₂, CO, NO, NO₂, NH₃, CH₄, N₂O, C₂H₄, HCN, HNCO

Trained on:
1. **Real reference spectra** — single-gas `.spc` files from an MKS MultiGas 2030 analyzer.
2. **Synthetic multi-gas spectra** — Beer-Lambert combinations of reference spectra with concentration interpolation and physical augmentations.

## Architecture — FTIRModel v4.1 (~16M parameters)

```
Input (B, 3, 16800 pts)   [raw, derivative, saturation_mask] channels
                            raw channel compressed via S·tanh(y/S) before normalization
 └── SpectralCNN           5 ResBlocks, channels 32→64→128→256→256, GroupNorm(8)
 └── 8-head SelfAttention  over temporal axis
 └── 2-layer Bidirectional GRU (hidden 512/direction → 1024 concat)
 └── Global Average Pool + MLP refinement (1024 → 2048 → 1024)
 └── Linear head           (B, 11) in normalized label space
     └── Denormalize        → log1p(ppmv)
     └── expm1 + clip(≥0)  → ppmv at inference
```

### Key v4.1 design decisions

| Feature | Detail |
|---------|--------|
| **Label normalization** | Per-species zero-mean/unit-std fitted on active (nonzero) training samples. Equalizes regression difficulty between H₂O (~10,000 ppmv) and HCN (~5 ppmv). |
| **GroupNorm** | Replaces BatchNorm1d throughout the CNN. Stable across mixed synthetic+reference batches regardless of batch size. |
| **Weighted Huber loss** | `active_weight=4, inactive_weight=0.5` plus per-species trace weights `1/log1p(median_ppmv)` (normalized to mean=1). |
| **Linear head** | Non-negativity enforced after denormalization at inference — not by the activation. |
| **Smooth saturation** | `S·tanh(y/S)` replaces hard clip at SATURATION_AU=8 AU. Preserves gradient information near detector ceiling. |
| **Independent val set** | Val NPZ generated with `seed+1` (distinct from train). No 80/20 in-memory split. |
| **MC Dropout uncertainty** | `FTIRModel.predict_with_uncertainty(x, n_samples=20)` — N stochastic forward passes → mean + std per species. Opt-in via `InferenceConfig.mc_dropout_samples > 0`. |
| **Checkpoint sidecar** | `.meta.json` alongside each `.pth`. Contains `label_normalizer`, `input_transform`, `target_species`, grid params. Schema v2. |

## Quick Start

### Install
```bash
git clone https://github.com/DrSuppe/FTIR_absorbtion_ML.git
cd FTIR_absorbtion_ML
pip install -e .
```

### Generate synthetic data
```bash
python synthetic_generator.py --n-samples 20000 --seed 42 --sampling-mode hybrid_v4
```

### Train
```bash
python train.py --n-synthetic 20000 --epochs 16 --batch-size 64
```

### Inference
```bash
python inference.py --data-dir path/to/spc_files/ --checkpoint outputs/checkpoints/run_a/best_model.pt
```

MC Dropout uncertainty (adds `{species}_uncertainty_log` columns to output CSV):
```bash
python inference.py --data-dir path/to/spc_files/ --mc-dropout-samples 20
```

### Train on Google Colab
Open `colab_train.ipynb` in Colab with a T4 GPU and run cells top to bottom.

## Data Pipeline

### Reference Spectra
- Located in `data/reference/spc_files/*.spc`
- Manifest auto-built from filenames: species, concentration, split
- Single-gas only; multi-gas mixtures are synthetic

### Synthetic Data Generation (`synthetic_generator.py`)

Each sample:
1. Draw 1–4 target species + optional interference gases
2. For each species: sample a concentration log-uniformly over the reference range
3. **Beer-Lambert interpolation**: blend the two bracketing reference spectra in absorbance space (`A = (1-t)·A_lo + t·A_hi`)
4. Sum all species
5. Apply augmentations: baseline drift, detector noise, spectral block dropout
6. Apply smooth saturation: `S·tanh(y/S)` with `S = SATURATION_AU = 8.0`

### Sampling modes

| Mode | Description |
|------|-------------|
| `hybrid_v4` | Default. 15% of samples force a trace species. Balances major/trace coverage. |
| `curriculum_v2` | Two-stage: stage-1 caps concentrations at p95, stage-2 uses Latin Hypercube Sampling. |
| `default` | Simple log-uniform sampling. |

### Training Data Mix
- `ConcatDataset(synthetic + reference)` with `WeightedRandomSampler` (~20% real spectra per batch)
- Labels: `log1p(ppmv)` in normalized space during training; inverted to ppmv for reporting

## Training Config

Key defaults in `TrainConfig`:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `n_synthetic` | 20,000 | |
| `epochs` | 16 | |
| `batch_size` | 64 | |
| `lr` | 3e-4 | Cosine with 2-epoch warmup |
| `huber_delta` | 1.0 | In normalized label space |
| `active_label_weight` | 4.0 | |
| `inactive_label_weight` | 0.5 | |
| `reference_weight` | 0.2 | Fraction of each batch from real spectra |

## Checkpoints

Checkpoints are saved as two files:
- `best_model.pt` — PyTorch state dict
- `best_model.meta.json` — full metadata sidecar (schema v2)

The sidecar contains everything needed to reconstruct the preprocessing: `input_transform`, `label_normalizer`, `target_species`, grid bounds.

## Running on Google Colab

### Step-by-step

1. **Open Colab**: [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub tab** → paste repo URL → select `colab_train.ipynb`
3. **Switch to GPU**: Runtime → Change runtime type → GPU → T4 → Save
4. **Run Cell 1** — downloads and installs the package as an editable install (~30 sec)
5. **Run Cell 2** — builds the manifest
6. **Run one or more training cells** (Cells 3-5) — each takes ~1-2 hours
7. **Run Cell 6** — compares runs, selects winner
8. **Run Cell 8** — downloads the best checkpoint

### Tips
- Free Colab gives ~4-6 hrs per session on T4. Save your checkpoint before it expires.
- Connect Google Drive to persist checkpoints:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !python train.py --checkpoint-dir /content/drive/MyDrive/ftir_checkpoints
  ```
- Cell 1 uses `pip install -e .` (editable install) so subprocess calls (`!python3 train.py`) use the same source as the notebook kernel.

## Project Structure

```
FTIR_absorbtion_ML/
├── data/
│   ├── reference/           # .spc files and manifest_v1.csv
│   └── synthetic/           # Generated .npz training data
├── outputs/
│   ├── checkpoints/         # .pt + .meta.json checkpoint pairs
│   └── reports/             # Per-epoch MAE plots, training_summary.json
├── src/
│   └── ftir_analysis/
│       ├── checkpointing.py  # Checkpoint save/load/validate
│       ├── constants.py      # Grid, species list, paths
│       ├── datasets.py       # Dataset classes
│       ├── evaluate.py       # Post-training evaluation
│       ├── features.py       # Input channel engineering, smooth_saturate
│       ├── inference_runtime.py  # Production inference with MC Dropout
│       ├── modeling.py       # FTIRModel architecture
│       ├── training.py       # Training loop, LabelNormalizer, WeightedHuberLoss
│       └── utils.py          # LabelNormalizer, seed_everything, etc.
├── colab_train.ipynb
├── synthetic_generator.py
├── train.py
├── inference.py
└── pyproject.toml
```

## License

MIT
