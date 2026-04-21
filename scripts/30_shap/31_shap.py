"""
Train XGBoost model for each site and save SHAP values to file.
Optionally tune hyperparameters instead of computing SHAP values.
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
