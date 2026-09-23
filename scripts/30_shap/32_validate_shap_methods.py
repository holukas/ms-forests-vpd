"""
Compute ALE (accumulated local effects) curves per site as a check on the SHAP results.

For each site, `train_xgboost_models_and_ale` (src/models.py) fits one XGBoost
model (85/15 split for early stopping, same hyperparameters as script 31) and
computes an ALE curve per feature (grid_size=100). Partial correlations, path
analysis and a combined summary are available in src/models.py and are
commented out in the main loop.

Settings at the top of the file: FLUX, FEATURES, VARIANT, SITES, MAX_SITES.
Script 21 must have run with the same VARIANT, and 31 should have too.

Writes to 30_shap/<FLUX>/ale/<VARIANT>/:
    <SITE>_ale_curves_<FLUX>.csv, .parquet   ALE curves
    <SITE>_ale_combined_<FLUX>.png           one panel per feature
    <SITE>_ale_<FLUX>.parquet, .csv          data with predictions
    1_ale_analysis_<FLUX>.txt                run log
    0_ale_validation_sites_<FLUX>.csv        fit metrics, one row per site
"""

import time
from pathlib import Path

import pandas as pd

import src.files as files
from src.models import train_xgboost_models_and_ale
from src.paths import load_settings
# Optional validation methods (commented out - uncomment to enable):
# from src.models import calculate_partial_correlations, calculate_path_analysis, create_validation_summary

# ------------------------------
# Variables
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'
FEATURES = ['TA_ZSCORE', 'SWIN_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']

# Run variant. An empty string reads the baseline subsets and overwrites the
# main-analysis results. Any other value adds a folder level on both sides, so the
# subsets come from 20_subsets/<VARIANT>/ and the curves go to
# 30_shap/<FLUX>/ale/<VARIANT>/. Stage 21 must have run with the same value,
# and 31 should have run with it too, since the two are read side by side.
VARIANT = ""

# Which sites to run. SITES wins if it is not empty, otherwise MAX_SITES takes
# the first n rows of the subsets file and 0 means all of them. Both are meant
# for a short test run. ALE refits a model per site, so this is not free.
SITES = []
MAX_SITES = 0

# ALE is the main validation method
# (Other methods like partial correlations and path analysis can be enabled below)

# ------------------------------
# Calculate ALE values for:
# [x] NEP_ZSCORE
# [ ] ET_ZSCORE
# [ ] GPP_ZSCORE
# [ ] RECO_ZSCORE
# ------------------------------


# Load settings
settings = load_settings()

# Load subsets info, from the same variant the curves are written to
infile = (Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
          / "21_SUBSETS_parquet_vars_stats_subsets.csv")
subsets_df = pd.read_csv(infile)

# Create output directory
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'ale' / VARIANT
# parents=True: Creates any necessary parent directories that don't exist.
# exist_ok=True: Prevents an error if the directory already exists.
results_outdir.mkdir(parents=True, exist_ok=True)

# Write to file (overwrites if file exists, creates if not)
modelstxt = Path(results_outdir) / f"1_ale_analysis_{FLUX}.txt"
with open(modelstxt, 'w', encoding='utf-8') as file:
    file.write("ALE (ACCUMULATED LOCAL EFFECTS) VALIDATION\n")
    file.write("--------------------------------------------\n")
    file.write(f"Target: {FLUX}\n")
    file.write(f"Features: {FEATURES}\n")
    file.write("\nPurpose: Independent validation for SHAP feature importance\n")
    file.write("- ALE shows isolated feature effects accounting for correlations\n")
    file.write("- Compare with SHAP direction/strength for confidence in findings\n")

_subsets_df = subsets_df.copy()
if SITES:
    _subsets_df = _subsets_df[_subsets_df['SITE'].isin(SITES)]
    missing = sorted(set(SITES) - set(_subsets_df['SITE']))
    if missing:
        raise ValueError(f"Not in {infile.name}: {missing}")
elif MAX_SITES:
    _subsets_df = _subsets_df.head(MAX_SITES)
ale_results_all = []
site_seconds = []

# Add mode indicator to header
with open(modelstxt, 'a', encoding='utf-8') as file:
    file.write(f"\nMODE: ALE VALIDATION (grid_size=100)\n")

print(f"\n{'=' * 80}")
print(f"ALE (ACCUMULATED LOCAL EFFECTS) VALIDATION")
print(f"{'=' * 80}\n")

print(f"Subsets:  {infile}")
print(f"Results:  {results_outdir}")
print(f"Sites:    {len(_subsets_df)} of {len(subsets_df)}")

for ix, siteconfig in _subsets_df.iterrows():
    site_started = time.perf_counter()

    # Main: ALE Analysis (returns metrics for aggregation)
    ale_result = train_xgboost_models_and_ale(
        features=FEATURES,
        target=FLUX,
        siteconfig=siteconfig,
        ix=ix,
        modelstxt=modelstxt,
        results_outdir=results_outdir
    )

    # Collect results for aggregation (only if not empty)
    if ale_result:
        ale_results_all.append(ale_result)

    site_seconds.append(time.perf_counter() - site_started)
    print(f"[{len(site_seconds)}/{len(_subsets_df)}] {siteconfig['SITE']} "
          f"took {site_seconds[-1] / 60:.1f} min")

    # Optional validation methods (uncomment to enable):
    # ====================================================

    # Partial Correlations: Statistical feature importance
    # calculate_partial_correlations(
    #     features=FEATURES,
    #     target=FLUX,
    #     siteconfig=siteconfig,
    #     ix=ix,
    #     modelstxt=modelstxt,
    #     results_outdir=results_outdir
    # )

    # Path Analysis: Direct vs indirect causal effects
    # calculate_path_analysis(
    #     features=FEATURES,
    #     target=FLUX,
    #     siteconfig=siteconfig,
    #     ix=ix,
    #     modelstxt=modelstxt,
    #     results_outdir=results_outdir
    # )

    # Integrated Summary: Combines all three methods
    # create_validation_summary(
    #     target=FLUX,
    #     features=FEATURES,
    #     siteconfig=siteconfig,
    #     ix=ix,
    #     results_outdir=results_outdir
    # )

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
if ale_results_all:
    ale_results_df = pd.DataFrame(ale_results_all)
    ale_csv_path = Path(results_outdir) / f"0_ale_validation_sites_{FLUX}.csv"
    print(f"\n{'=' * 80}")
    print(f"Saved ALE validation results for all sites to:")
    print(f"{ale_csv_path}")
    ale_results_df.to_csv(ale_csv_path, index=False)
    print(f"{'=' * 80}\n")

print("Output files per site:")
print(f"  - {{SITE}}_ale_curves_{FLUX}.csv — ALE curve data (100 grid points)")
print(f"  - {{SITE}}_ale_combined_{FLUX}.png — 2x2 combined ALE plot")
print(f"  - {{SITE}}_ale_{FLUX}.parquet/csv — Full dataset with predictions")
print("\nUse ALE results to validate SHAP importance:")
print("  Match: SHAP direction agrees with the ALE effect, high confidence")
print("  Disagreement: check feature correlations or interactions")
