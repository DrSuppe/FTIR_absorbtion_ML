# FTIR Absorption ML Solver

A physics-informed machine learning pipeline for quantifying gas concentrations from FTIR absorbance spectra.

## Overview

This project trains a neural network (CNN → Transformer → GRU → output head) to predict the concentrations of **11 gas species** simultaneously from a single FTIR absorbance spectrum.

**Target species:** H₂O, CO₂, CO, NO, NO₂, NH₃, CH₄, N₂O, C₂H₄, HCN, HNCO

Trained on:
1. **Real reference spectra** — 557 single-gas `.spc` files measured on an MKS MultiGas 2030 analyzer.
2. **Synthetic multi-gas spectra** — built by linearly combining reference spectra (Beer-Lambert additivity) with concentration interpolation and physical augmentations.

## Quick Start

### Install
```bash
git clone https://github.com/YOUR_USERNAME/FTIR_absorbtion_ML.git
cd FTIR_absorbtion_ML
pip install -e .
```

### Generate synthetic data
```bash
python3 synthetic_generator.py --n-samples 50000 --out data/synthetic/spectra.npz (local, Apple Silicon)
```

### Train (local, Apple Silicon)
```bash
python train.py --n-synthetic 10000 --epochs 50 --batch-size 64
```

### Train (Google Colab)
Open `colab_train.ipynb` in Colab with a T4 GPU. See **Colab Instructions** below.

## Architecture — FTIRModel v3 (~29M parameters)

```
Input (B, 16800 pts, 800–5000 cm⁻¹ @ 0.25 cm⁻¹)
 └── SpectralCNN (5 ResBlocks, channels 32→64→128→256→256)
 └── 8-head Self-Attention
 └── 2-layer Bidirectional GRU (hidden 512/direction)
 └── Global Average Pool → 2-layer Transformer Encoder
 └── Output head → 11-species concentrations (log1p ppmv)
```

## Data Pipeline

### Reference Spectra
- Located in `reference_spectra/spc_files/*.spc`
- Manifest auto-built from filenames: species, concentration, temperature
- No `.lab` files used

### Synthetic Data Generation (`synthetic_generator.py`)
Each sample:
1. Draw 1–5 target species + optional interference gases
2. For each species: sample a target concentration (log-uniform over reference range)
3. **Concentration interpolation**: linearly blend the two bracketing reference spectra in absorbance space — `A = (1-t)·A_lo + t·A_hi`. This fills gaps between measured concentrations and prevents overfitting to specific values.
4. Sum all species (Beer-Lambert additivity)
5. Apply augmentations: axis shift, regional gain jitter, 3rd-order baseline, detector noise, spectral block dropout, saturation clipping

### Training Data Mix
- `ConcatDataset(synthetic + reference)` with `WeightedRandomSampler` (~20% real spectra per batch)
- Labels: `log1p(ppmv)` — inverted with `expm1` for reporting

## Training Config

| Parameter | Quick test | Full run (Colab) |
|-----------|-----------|------------------|
| `--n-synthetic` | 5 000 | 50 000 |
| `--epochs` | 20 | 100 |
| `--batch-size` | 32 | 128 |
| `--lr` | 3e-4 | 3e-4 |

## Colab Instructions

See [**Colab Guide**](#running-on-google-colab) below.

## Project Structure

```
FTIR_absorbtion_ML/
├── data/
│   ├── reference/         # Your raw .spc files and manifest_v1.csv
│   └── synthetic/         # Generated training data (.npz)
├── outputs/
│   ├── checkpoints/       # Saved PyTorch models
│   └── reports/           # Training plots, MAE per species, etc.
├── src/
│   └── ftir_analysis/     # Core package
├── train.py
├── inference.py
├── synthetic_generator.py
└── runner.py              # CLI entrypoint
```

## Running on Google Colab

### Step-by-step

1. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
2. **File → Open notebook → GitHub tab** → paste your repo URL → select `colab_train.ipynb`
3. **Switch to GPU**: Runtime → Change runtime type → GPU → T4 → Save
4. **Run the first cell** — it clones the repo and installs dependencies (~30 sec)
5. **Upload reference spectra** (if not in the repo):
   - Zip `data/reference/spc_files/` and `data/reference/manifest_v1.csv` locally
   - In Colab: uncomment the `files.upload()` cell and upload the zip
   - Then run: `!unzip -q /content/uploaded.zip -d /content/ftir/data/reference/`
6. **Run the training cell** — choose Quick (~10 min) or Full (~2 hrs)
7. **Download your checkpoint**: run `python3 inference.py --data-dir archive --checkpoint outputs/checkpoints/ftir_solver_best.pth`

### Tips
- Free Colab gives ~4–6 hrs per session on T4. Save your checkpoint before it expires.
- Use `--log-every 5` to reduce plot generation overhead during long runs.
- Connect Google Drive to persist checkpoints across sessions:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  !python train.py --checkpoint-dir /content/drive/MyDrive/ftir_checkpoints
  ```

## License

MIT
