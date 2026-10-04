"""
Train XGBoost per site and compute out-of-sample SHAP values, or tune hyperparameters.

Runs `train_xgboost_models_and_shap` (src/models.py) for every site in
21_SUBSETS_parquet_vars_stats_subsets.csv. With TUNE_HYPERPARAMETERS = True it
runs `tune_xgboost_hyperparameters` instead, testing TUNE_N_ITER combinations
per site (about 2 to 3 min per site at 25).

Settings at the top of the file: FLUX, FEATURE_SET, CONDITIONAL, CV_STRATEGY,
TUNE_HYPERPARAMETERS, TUNE_N_ITER, VARIANT, SITES, MAX_SITES. Set VARIANT for
any run other than the main analysis, or the main results are overwritten.
Script 21 must have run with the same VARIANT.

Writes to 30_shap/<FLUX>/<TYPE>/<VARIANT>/ (TYPE: conditional or interventional):
    1_models_xgboost_shap-<TYPE>_<FLUX>.txt    run log and fold metrics
    <SITE>_shap-<TYPE>_<FLUX>.parquet, .csv    SHAP values and predictions
    2_cv_results_all_sites-<TYPE>_<FLUX>.csv   CV metrics, one row per site
In tuning mode: <SITE>_hyperparameter_tuning_<FLUX>.txt and
0_hyperparameter_tuning_results_<FLUX>.csv.
"""

import time
from pathlib import Path

import pandas as pd

import src.files as files
from src.models import train_xgboost_models_and_shap, tune_xgboost_hyperparameters
from src.paths import load_settings

# ------------------------------
# Variables
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'
# Predictor set. The reduced sets test whether TA and VPD can be told apart,
# given how strongly they covary, by dropping one of them.
#
#   full     the main-analysis set, all four drivers
#   no_vpd   TA kept, VPD dropped
#   no_ta    VPD kept, TA dropped
#
# A third variant, TA plus an alternative humidity variable, is not reachable. No
# site in the analysis carries RH, and any humidity variable that could be derived
# here is an exact function of TA and VPD, so it would add nothing that separates
# them.
#
# Set VARIANT as well, or the results overwrite the main-analysis ones.
FEATURE_SETS = {
    'full': ['TA_ZSCORE', 'SWIN_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE'],
    'no_vpd': ['TA_ZSCORE', 'SWIN_ZSCORE', 'SWC_ZSCORE'],
    'no_ta': ['VPD_ZSCORE', 'SWIN_ZSCORE', 'SWC_ZSCORE'],
}
FEATURE_SET = 'full'
FEATURES = FEATURE_SETS[FEATURE_SET]
CONDITIONAL = True  # False runs interventional SHAP, which breaks feature correlations

# Tuning mode: Set to True to tune hyperparameters instead of computing SHAP
TUNE_HYPERPARAMETERS = False  # Set to True to run hyperparameter tuning
TUNE_N_ITER = 25  # Number of parameter combinations to test (default: 25)

# Run variant. An empty string reads the baseline subsets and overwrites the
# main-analysis results. Any other value adds a folder level on both sides, so the
# subsets come from 20_subsets/<VARIANT>/ and the results go to
# 30_shap/<FLUX>/<shap_type>/<VARIANT>/. Stage 21 must have run with the same
# value, otherwise there are no subsets to read.
VARIANT = ""

# Which sites to run. SITES wins if it is not empty, otherwise MAX_SITES takes
# the first n rows of the subsets file and 0 means all of them. Both are meant
# for timing a short run before starting the full campaign. For a timing run,
# name sites of different sizes in SITES: run time follows the number of
# records, and the first n rows are simply the first n site names.
SITES = []
MAX_SITES = 0

# Cross-validation strategy. "random" is the main-analysis setting, a shuffled 5-fold
# split of the complete rows. "blocked" leaves one calendar year out at a time:
# neighboring half-hours are correlated, so a
# shuffled split can put a record and its neighbor on opposite sides and flatter
# the score. Expect lower scores under "blocked". The drop is the size of the
# leakage, not a fault. Sites with a single year are skipped, since a year-wise
# split needs at least two. Set VARIANT as well, or the results overwrite the
# main-analysis ones.
CV_STRATEGY = "random"

# ------------------------------
# Calculate SHAP values for:
# [x] NEP_ZSCORE
# [x] ET_ZSCORE
# [x] GPP_ZSCORE
# [x] RECO_ZSCORE
# ------------------------------


# Load settings
settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'

# Load subsets info, from the same variant the results are written to
infile = (Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
          / "21_SUBSETS_parquet_vars_stats_subsets.csv")
subsets_df = pd.read_csv(infile)

# Create output directory
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT
# parents=True: Creates any necessary parent directories that don't exist.
# exist_ok=True: Prevents an error if the directory already exists.
results_outdir.mkdir(parents=True, exist_ok=True)

# Write to file (overwrites if file exists, creates if not)
substr = "conditional" if CONDITIONAL else "interventional"
modelstxt = Path(results_outdir) / f"1_models_xgboost_shap-{substr}_{FLUX}.txt"
with open(modelstxt, 'w') as file:
    file.write("XGBOOST MODELS and SHAP CALCULATIONS\n")
    file.write("------------------------------------\n")
    file.write(f"Target: {FLUX}\n")
    file.write(f"Features: {FEATURES}\n")
    file.write(f"Conditional SHAP: {CONDITIONAL}\n")
    file.write(f"CV strategy: {CV_STRATEGY}\n")
    file.write(f"Feature set: {FEATURE_SET}\n")

_subsets_df = subsets_df.copy()
if SITES:
    _subsets_df = _subsets_df[_subsets_df['SITE'].isin(SITES)]
    missing = sorted(set(SITES) - set(_subsets_df['SITE']))
    if missing:
        raise ValueError(f"Not in {infile.name}: {missing}")
elif MAX_SITES:
    _subsets_df = _subsets_df.head(MAX_SITES)
cv_results_all = []
site_seconds = []

# Add mode indicator to header
if TUNE_HYPERPARAMETERS:
    with open(modelstxt, 'a') as file:
        file.write(f"\nMODE: HYPERPARAMETER TUNING (n_iter={TUNE_N_ITER})\n")
    print(f"\n{'=' * 80}")
    print(f"HYPERPARAMETER TUNING MODE - Testing {TUNE_N_ITER} combinations per site")
    print(f"{'=' * 80}\n")
else:
    print(f"\n{'=' * 80}")
    print(f"SHAP ANALYSIS MODE - {'5-fold' if CV_STRATEGY == 'random' else 'leave-one-year-out'} cross-validation")
    print(f"{'=' * 80}\n")

print(f"Subsets:  {infile}")
print(f"Results:  {results_outdir}")
print(f"Sites:    {len(_subsets_df)} of {len(subsets_df)}")
print(f"CV:       {CV_STRATEGY}")
print(f"Features: {FEATURE_SET} {FEATURES}")

for ix, siteconfig in _subsets_df.iterrows():
    site_started = time.perf_counter()

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
            results_outdir=results_outdir,
            cv_strategy=CV_STRATEGY
        )
        if cv_result:  # Only add if result is not empty
            cv_results_all.append(cv_result)

    site_seconds.append(time.perf_counter() - site_started)
    print(f"[{len(site_seconds)}/{len(_subsets_df)}] {siteconfig['SITE']} "
          f"took {site_seconds[-1] / 60:.1f} min")

# Timing, so a short run says what the full campaign costs
if site_seconds:
    total_min = sum(site_seconds) / 60
    mean_min = total_min / len(site_seconds)
    print("")
    print(f"{len(site_seconds)} site(s) in {total_min:.1f} min, "
          f"{mean_min:.1f} min per site on average")
    print(f"At that rate {len(subsets_df)} sites take "
          f"{mean_min * len(subsets_df) / 60:.1f} h")

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
            results_outdir) / f"2_cv_results_all_sites-{('conditional' if CONDITIONAL else 'interventional')}_{FLUX}.csv"
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
