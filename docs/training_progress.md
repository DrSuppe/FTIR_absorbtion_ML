# Training Progress Log

## 2026-03-09 — Colab T4 Mixed-Validation Recovery

### Context
- Goal: improve **mixed validation** performance (synthetic + reference) using mean val log-MAE.
- Model: FTIR v3 (`~29.0M` params), unchanged architecture.
- Data mix: `train=4431 (4000 synth + 431 ref)`, `val=1076 (1000 synth + 76 ref)`.

### What We Changed
- Added weighted Huber loss (`active_label_weight`, `inactive_label_weight`).
- Added diagnostics:
  - zero-predictor baseline mean log-MAE (`all`, `synth`, `ref`)
  - per-epoch split log-MAE (`all`, `synth`, `ref`)
  - delta vs zero baseline and species-count beating baseline
- Changed checkpoint selection metric:
  - from `val_loss`
  - to `val_log_mae_mean`
- Updated notebook for A/B runs with separate output dirs:
  - `outputs/checkpoints/run_a`, `outputs/reports/run_a`
  - `outputs/checkpoints/run_b`, `outputs/reports/run_b`

### A/B Results

#### Baseline (zero predictor)
- mean val log-MAE:
  - `all=1.676`
  - `synth=1.755`
  - `ref=0.637`

#### Run A
- Config:
  - `active=1.5`, `inactive=1.0`
  - `reference_weight=0.05`
  - `epochs=12`, `batch_size=64`, `warmup=1.5`
- Best metric:
  - `val_log_mae_mean=1.8388`
- Baseline comparison:
  - `Δvs0 = -0.163` (worse than zero baseline)
  - species beating zero baseline: `0/11`

#### Run B
- Config:
  - `active=2.0`, `inactive=1.0`
  - `reference_weight=0.10`
  - `epochs=12`, `batch_size=64`, `warmup=1.5`
- Best metric:
  - `val_log_mae_mean=1.9581`
- Baseline comparison:
  - `Δvs0 = -0.283` (worse than zero baseline)
  - species beating zero baseline: `0/11`

#### Outcome
- Winner by plan rule: **Run A** (lower mean val log-MAE than Run B).
- However, both A and B are still worse than the zero baseline.

### Interpretation
- Increasing active-label pressure moved predictions, but did not improve mixed-val generalization.
- Current issue appears dominated by **objective/data-distribution mismatch** rather than architecture.
- Heaviest error contributors remain high-range species (notably H2O/CO2/CH4 tails).

### Notebook Fixes Applied
- Cell 6 now searches `run_a` / `run_b` report directories for latest MAE plot.
- Cell 7 now downloads the winner checkpoint from `run_a`/`run_b` automatically.

### Next Suggested Direction
- Keep architecture fixed.
- Next iteration should target data/target distribution handling:
  - concentration range capping for extreme tails, or
  - stratified synthetic sampling by concentration bins.

## 2026-03-09 — Synthetic Data V2 Implementation

### Goal
- Implement curriculum synthetic sampling to improve concentration-space coverage while avoiding early over-emphasis on extreme tails.

### Implemented
- Added `curriculum_v2` sampler to `synthetic_generator.py`:
  - stage-1 realistic prior sampling
  - stage-2 LHS coverage in log-concentration space
  - strong major-species bias (`H2O, CO2, CO, NO, NO2, NH3`)
  - active-species constraints (`min_active_species`/`max_active_species`)
  - stage-1 concentration cap policy (`p90`/`p95`/`max`)
  - generator diagnostics (presence rates, quantiles, concentration-bin occupancy)
- Added new CLI options:
  - `--sampling-mode {default,curriculum_v2}`
  - `--curriculum-stage1-frac`
  - `--major-species`
  - `--stage1-cap-policy {p90,p95,max}`
  - `--lhs-frac`
  - `--min-active-species` / `--max-active-species`
  - `--diagnostics-json`
- Updated notebook A/B setup:
  - Run A: `default` generator + Run-A training recipe
  - Run B: `curriculum_v2` generator + same Run-A training recipe
  - both runs now write per-run generator diagnostics JSON.

### Added Tests
- `tests/test_synthetic_generator_curriculum.py`
  - stage-1 `p90` cap enforcement
  - curriculum diagnostics health checks:
    - major species presence floor
    - stage-2 high-bin coverage
    - zero stage-1 cap violations

### Status
- Implementation complete and committed.
- Next action: run Colab A/B with the new generator and compare to zero-baseline gate (`val_log_mae_mean < 1.676`).
