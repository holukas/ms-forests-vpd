"""
XGBoost SHAP Analysis & Hyperparameter Tuning Pipeline

## Overview
Trains XGBoost models per site and calculates out-of-sample SHAP values for feature importance.
Alternatively, can run hyperparameter tuning mode to optimize model parameters.

## Dual-Mode Design

### Mode 1: SHAP Analysis (default, TUNE_HYPERPARAMETERS=False)
**Purpose**: Calculate unbiased feature importance via SHAP values
- Each site gets 5 models (5-fold CV)
- Every data point gets SHAP values from a model that never trained on it
- Per-fold performance metrics (R², RMSE) show generalization
- Global out-of-sample metrics aggregate across all folds
- Output: Parquet/CSV with features, SHAP values, predictions, and metrics

**Workflow:**
1. Load site configuration and data
2. For each site:
   - Run 5-fold cross-validation
   - Per fold: train on 68%, use 12% for early stopping, explain 20%
     (80% fold further split 85/15 for training vs validation)
   - Collect per-fold metrics and SHAP values
   - Reassemble in original order
3. Aggregate CV metrics across all sites to CSV

**Output Files:**
- `1_models_xgboost_shap-{TYPE}_{FLUX}.txt` — Per-site fold metrics and logs
- `{SITE}_shap-{TYPE}_{FLUX}.parquet/csv` — Full results per site
- `2_cv_results_all_sites-{TYPE}_{FLUX}.csv` — Aggregated metrics (all sites, one row per site)

### Mode 2: Hyperparameter Tuning (TUNE_HYPERPARAMETERS=True)
**Purpose**: Find optimal n_estimators, max_depth, learning_rate per site
- Tests {n_iter} parameter combinations per site (default: 25)
- Uses 5-fold CV to score each combination
- Fixed regularization (reg_lambda, reg_alpha, gamma, etc.) for consistency
- Output: Best parameters and CV R² score per site

**Workflow:**
1. Load site configuration and data
2. For each site:
   - Run RandomizedSearchCV with 5-fold CV
   - Test 25 random parameter combinations
   - Track best parameters and best CV R²
3. Aggregate best parameters across all sites to CSV

**Output Files:**
- `1_models_xgboost_shap-{TYPE}_{FLUX}.txt` — Tuning mode indicator
- `{SITE}_hyperparameter_tuning_{FLUX}.txt` — Best params per site
- `0_hyperparameter_tuning_results_{FLUX}.csv` — Aggregated best params (all sites)

## Configuration Variables

FLUX : str
    Target variable to analyze (options: NEP_ZSCORE, ET_ZSCORE, GPP_ZSCORE, RECO_ZSCORE)
    Default: NEP_ZSCORE

FEATURES : list
    Feature variables to use as predictors
    Default: ['TA_ZSCORE', 'SWIN_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']

CONDITIONAL : bool
    If True: Calculate conditional SHAP (respects feature correlations) — RECOMMENDED
    If False: Calculate standard/marginal SHAP (assumes independence)
    Default: True

TUNE_HYPERPARAMETERS : bool
    If False (default): Run SHAP analysis
    If True: Run hyperparameter tuning instead
    Default: False

TUNE_N_ITER : int
    Number of parameter combinations to test in tuning mode (default: 25)
    Higher = more thorough but slower (~25 fits × 5 folds × 100+ sites = hours)

## How to Use

**For SHAP Analysis (default):**
```bash
python scripts/30_shap/31_shap.py
```
Generates feature importance explanations. Check output for:
- Per-site fold metrics: data/outputs/50_shap_analysis/{FLUX}/{TYPE}/
- Aggregated CSV with global R² per site

**For Hyperparameter Tuning:**
Edit the script:
```python
TUNE_HYPERPARAMETERS = True
TUNE_N_ITER = 25  # Adjust if needed (25 = ~2-3 mins per site)
```
Then run:
```bash
python scripts/30_shap/31_shap.py
```
Check output for best parameters per site and aggregated results.

**For Single-Site Testing:**
Uncomment in the loop:
```python
if ix > 1:  # Only test first 2 sites
    break
if siteconfig['SITE'] != "CH-Dav":  # Only test this site
    continue
```

## Key Design Decisions

1. **Fixed Regularization Parameters**: Keep reg_lambda=1, reg_alpha=0.1, gamma=0.2, etc.
   - Prevents overfitting consistently across sites
   - Tuning only core capacity (n_estimators, max_depth, learning_rate)
   - Faster, more stable results

2. **5-Fold CV for SHAP**: Each data point gets out-of-sample explanations
   - In-sample SHAP can be misleading (model may have memorized)
   - Out-of-sample SHAP is unbiased: high value = feature truly predicts target

3. **Per-Fold + Global Metrics**: Show both generalization and overall performance
   - Per-fold: How model performs on unseen data (5 estimates)
   - Global: Single R²/RMSE across all 5 folds combined

4. **Aggregation to CSV**: Easy cross-site analysis
   - One row per site (SHAP mode) or per site (tuning mode)
   - Supports weighting by N_RECORDS in meta-analysis

## Performance Tips

- **Faster runs**: Set TUNE_HYPERPARAMETERS=True and TUNE_N_ITER=10 for quick tests
- **Single site**: Comment out loop, test CH-Dav first
- **Different flux**: Change FLUX to ET_ZSCORE, GPP_ZSCORE, or RECO_ZSCORE
- **Memory**: Results are modest (~50-100 MB per site for SHAP)
"""

from pathlib import Path

import pandas as pd

import src.files as files
from src.models import train_xgboost_models_and_shap, tune_xgboost_hyperparameters

# ------------------------------
# Variables
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'
FEATURES = ['TA_ZSCORE', 'SWIN_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']
CONDITIONAL = True  # Use conditional SHAP instead of standard SHAP

# Tuning mode: Set to True to tune hyperparameters instead of computing SHAP
TUNE_HYPERPARAMETERS = False  # Set to True to run hyperparameter tuning
TUNE_N_ITER = 25  # Number of parameter combinations to test (default: 25)

# ------------------------------
# Calculate SHAP values for:
# [x] NEP_ZSCORE
# [ ] ET_ZSCORE
# [ ] GPP_ZSCORE
# [ ] RECO_ZSCORE
# ------------------------------


# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'

# Load subsets info
infile = Path('../../data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv')
subsets_df = pd.read_csv(infile)

# Create output directory
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type
# parents=True: Creates any necessary parent directories that don't exist.
# exist_ok=True: Prevents an error if the directory already exists.
results_outdir.mkdir(parents=True, exist_ok=True)

# Write to file (overwrites if file exists, creates if not)
substr = "conditional" if CONDITIONAL else "standard"
modelstxt = Path(results_outdir) / f"1_models_xgboost_shap-{substr}_{FLUX}.txt"
with open(modelstxt, 'w') as file:
    file.write("XGBOOST MODELS and SHAP CALCULATIONS\n")
    file.write("------------------------------------\n")
    file.write(f"Target: {FLUX}\n")
    file.write(f"Features: {FEATURES}\n")
    file.write(f"Conditional SHAP: {CONDITIONAL}\n")

_subsets_df = subsets_df.copy()
cv_results_all = []

# Add mode indicator to header
if TUNE_HYPERPARAMETERS:
    with open(modelstxt, 'a') as file:
        file.write(f"\nMODE: HYPERPARAMETER TUNING (n_iter={TUNE_N_ITER})\n")
    print(f"\n{'=' * 80}")
    print(f"HYPERPARAMETER TUNING MODE - Testing {TUNE_N_ITER} combinations per site")
    print(f"{'=' * 80}\n")
else:
    print(f"\n{'=' * 80}")
    print(f"SHAP ANALYSIS MODE - 5-Fold Cross-Validation")
    print(f"{'=' * 80}\n")

for ix, siteconfig in _subsets_df.iterrows():
    # if ix > 1:
    #     break
    # if siteconfig['SITE'] != "CH-Dav":
    #     continue

    if TUNE_HYPERPARAMETERS:
        # Run hyperparameter tuning
        tuning_result = tune_xgboost_hyperparameters(
            features=FEATURES,
            target=FLUX,
            siteconfig=siteconfig,
            ix=ix,
            results_outdir=results_outdir,
            n_iter=TUNE_N_ITER
        )
        if tuning_result:
            cv_results_all.append(tuning_result)
    else:
        # Run SHAP analysis
        cv_result = train_xgboost_models_and_shap(
            features=FEATURES,
            target=FLUX,
            siteconfig=siteconfig,
            ix=ix, modelstxt=modelstxt,
            conditional=CONDITIONAL,
            results_outdir=results_outdir
        )
        if cv_result:  # Only add if result is not empty
            cv_results_all.append(cv_result)

# Save aggregated results to CSV
if cv_results_all:
    cv_results_df = pd.DataFrame(cv_results_all)
    if TUNE_HYPERPARAMETERS:
        cv_csv_path = Path(
            results_outdir) / f"0_hyperparameter_tuning_results_{FLUX}.csv"
        print(f"\n{'=' * 80}")
        print(f"Hyperparameter tuning results saved to:")
    else:
        cv_csv_path = Path(
            results_outdir) / f"2_cv_results_all_sites-{('conditional' if CONDITIONAL else 'standard')}_{FLUX}.csv"
        print(f"\n{'=' * 80}")
        print(f"Saved aggregated CV results for all sites to:")
    print(f"{cv_csv_path}")
    cv_results_df.to_csv(cv_csv_path, index=False)
    print(f"{'=' * 80}\n")
    # train_rf_models_and_shap(
    #     features=FEATURES,
    #     target=FLUX,
    #     siteconfig=siteconfig,
    #     ix=ix, modelstxt=modelstxt,
    #     conditional=CONDITIONAL,
    #     results_outdir=results_outdir
    # )
