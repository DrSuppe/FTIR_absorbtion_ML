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

## Scientific Basis and Justification

### 1. Why a linear combination of reference spectra equals the spectrum of the real mixture

This is a direct consequence of the **Beer-Lambert law of additivity**, which holds exactly under the fixed measurement conditions of this system (T = 191 °C, P = 1 atm, no chemical reactions in the cell).

For a single species $i$ at concentration $c_i$ in a cell of path length $\ell$:

$$A_i(\tilde{\nu}) = \varepsilon_i(\tilde{\nu}) \cdot c_i \cdot \ell$$

where $\varepsilon_i(\tilde{\nu})$ [L mol⁻¹ cm⁻¹] is the **molar absorptivity** — a molecular property that depends only on species identity, temperature, and pressure. Because T and P are fixed and identical for all measurements, $\varepsilon_i(\tilde{\nu})$ is a **fixed constant array** for each species.

For a mixture of $N$ non-reacting species, the transmitted intensity is:

$$I(\tilde{\nu}) = I_0(\tilde{\nu}) \cdot \exp\!\left(-\ell \sum_{i=1}^{N} \varepsilon_i(\tilde{\nu}) \cdot c_i\right)$$

Taking $-\log_{10}$ gives the **total absorbance**:

$$A_{\text{mix}}(\tilde{\nu}) = \sum_{i=1}^{N} \varepsilon_i(\tilde{\nu}) \cdot c_i \cdot \ell = \sum_{i=1}^{N} A_i(\tilde{\nu})$$

Since $A_i \propto c_i$ (linear), a reference spectrum $A_i^{\text{ref}}(\tilde{\nu})$ measured at $c_i^{\text{ref}}$ scales exactly:

$$A_i(\tilde{\nu})\big|_{c_i} = \frac{c_i}{c_i^{\text{ref}}} \cdot A_i^{\text{ref}}(\tilde{\nu})$$

Combining:

$$\boxed{A_{\text{mix}}(\tilde{\nu}) = \sum_{i=1}^{N} \frac{c_i}{c_i^{\text{ref}}} \cdot A_i^{\text{ref}}(\tilde{\nu})}$$

The synthetic data generation in `synthetic_generator.py` implements this exactly. The **Beer-Lambert interpolation** between two bracketing references is also exact: for $c \in [c_{\text{lo}}, c_{\text{hi}}]$, the weight $t = (c - c_{\text{lo}})/(c_{\text{hi}} - c_{\text{lo}})$ gives $A = (1-t)A_{\text{lo}} + tA_{\text{hi}}$, which is algebraically identical to $A = \varepsilon \cdot c \cdot \ell$.

**Conditions under which this holds exactly in this system:**
- T = 191 °C and P = 1 atm are fixed for all measurements (reference and sample) — no temperature or pressure shift in line positions or intensities
- No chemical reactions between species in the measurement cell
- Linear detector response (violations are handled by the smooth-saturation channel)
- The 11 target species do not react with each other under these conditions

**Important implication:** because T and P are fixed, the molar absorptivities $\varepsilon_i(\tilde{\nu})$ are fixed constants. Temperature augmentation in the synthetic generator is **not physically motivated** for this specific setup and should be disabled or removed to avoid training the model to expect spectral shifts that will never occur in deployment.

---

### 2. Advantages of ML over a classical least-squares optimizer

The classical approach is **Non-Negative Least Squares (NNLS)**: given the reference matrix $\mathbf{R} \in \mathbb{R}^{W \times N}$ (W = 16800 wavenumber points, N = 11 species), solve $\min_{\mathbf{c} \geq 0} \|\mathbf{A}_{\text{mix}} - \mathbf{R}\mathbf{c}\|_2^2$.

| Aspect | Classical NNLS | This ML model |
|---|---|---|
| **Spectral overlap** | Sensitive — columns of **R** are correlated (CO/CO₂/N₂O share regions); small overlap errors inflate concentration estimates directly | Learns nonlinear representations that disentangle overlapping species |
| **Baseline and drift** | Any additive baseline component corrupts $\hat{c}$ directly | Trained with baseline augmentation; derivative channel is baseline-insensitive by construction |
| **Detector saturation** | Saturated pixels violate Beer-Lambert; must be masked manually, and masking discards information | Smooth-saturation channel signals the detector ceiling; model is trained on saturated examples |
| **Uncertainty** | Propagates only photon noise analytically; no account for model error or spectral drift | MC Dropout gives empirical uncertainty per species per measurement |
| **Trace species at low SNR** | Conditioning number of **R** worsens when one species dominates; trace species estimates become numerically unstable | `hybrid_v4` sampling explicitly trains on trace-species scenarios; per-species weighted loss equalizes gradient signal |
| **Inference speed** | $O(WN)$ linear algebra, very fast | Single forward pass, also fast (~ms per spectrum) |

The core argument: NNLS is the **optimal solution when Beer-Lambert holds perfectly** and the noise is Gaussian. In practice, baseline drift, partially saturated pixels, and instrument line-function variations all break this optimality. The ML model is trained to be robust to these deviations through data augmentation, at the cost of requiring synthetic training data.

---

### 3. Drawbacks and mitigations

| Drawback | Root cause | Mitigation in this codebase |
|---|---|---|
| **Requires retraining per instrument** | Reference spectra encode the instrument line function, path length, and cell geometry | Architecture is generic; retrain with new `.spc` reference files |
| **Fails outside training concentration range** | Neural networks extrapolate poorly beyond training support | Log-uniform concentration sampling covers the full reference range; MC Dropout uncertainty increases out-of-distribution |
| **No guarantee of physical constraints during training** | Loss is unconstrained | Non-negativity enforced post-hoc; `inactive_label_weight=0.5` suppresses false detections |
| **Unknown interferents** | A species not in the training set will have its signal distributed across the 11 outputs | No mitigation; monitor residuals $(A_{\text{meas}} - A_{\text{reconstructed}})$ in deployment |
| **Data-hungry** | NNLS needs only the reference matrix; this model needs 20k+ spectra | Synthetic generator makes this cost-free; `--n-synthetic` is the main quality lever |
| **Black-box decisions** | Nonlinear model is not interpretable in terms of spectral assignments | Training data is physically grounded in Beer-Lambert; residual analysis on real measurements remains valid |

The most important operational risk: **the model will be confidently wrong when presented with conditions not covered by training** (new interferents, out-of-range concentrations, or instrument drift). The MC Dropout uncertainty output (`predict_with_uncertainty`) is the primary tool for detecting this at runtime.

---

## License

MIT
