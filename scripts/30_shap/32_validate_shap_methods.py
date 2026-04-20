"""
Train XGBoost model for each site and calculate validation metrics.

Implements multiple validation methods for SHAP results:
1. ALE (Accumulated Local Effects) - Shows how features affect predictions
2. Partial Correlations - Isolates feature effects while controlling for confounders
3. Path Analysis - Tests direct and indirect causal pathways
"""

from pathlib import Path

import pandas as pd

import src.files as files
from src.models import (
    train_xgboost_models_and_ale,
    calculate_partial_correlations,
    calculate_path_analysis,
    create_validation_summary
)

# ------------------------------
# Variables
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'
FEATURES = ['TA_ZSCORE', 'SWIN_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']

# ------------------------------
# Calculate ALE values for:
# [x] NEP_ZSCORE
# [ ] ET_ZSCORE
# [ ] GPP_ZSCORE
# [ ] RECO_ZSCORE
# ------------------------------


# Load settings
settings = files.read_settings_file("../../config/settings.yaml")

# Load subsets info
infile = Path('../../data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv')
subsets_df = pd.read_csv(infile)

# Create output directory
# Renamed from 'ale' to 'validation_methods' to reflect all three methods:
# - ALE (Accumulated Local Effects)
# - Partial Correlations
# - Path Analysis
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'validation_methods'
# parents=True: Creates any necessary parent directories that don't exist.
# exist_ok=True: Prevents an error if the directory already exists.
results_outdir.mkdir(parents=True, exist_ok=True)

# Write to file (overwrites if file exists, creates if not)
# Use UTF-8 encoding to support special characters (β, etc.)
modelstxt = Path(results_outdir) / f"1_models_xgboost_validation_results_{FLUX}.txt"
with open(modelstxt, 'w', encoding='utf-8') as file:
    file.write("=" * 80 + "\n")
    file.write("SHAP VALIDATION ANALYSIS\n")
    file.write("=" * 80 + "\n\n")
    file.write("This script validates SHAP feature importance using three complementary methods:\n\n")
    file.write("1. ALE (Accumulated Local Effects)\n")
    file.write("   - Shows the isolated effect of each feature on model predictions\n")
    file.write("   - Accounts for correlations with other features\n")
    file.write("   - Produces plots and curve data for visualization\n\n")
    file.write("2. Partial Correlations\n")
    file.write("   - Calculates the correlation between target and feature\n")
    file.write("   - Controls statistically for all other features\n")
    file.write("   - Provides p-values for statistical significance testing\n\n")
    file.write("3. Path Analysis (Structural Equation Modeling)\n")
    file.write("   - Calculates standardized path coefficients (direct effects)\n")
    file.write("   - Assesses feature intercorrelations (mediation potential)\n")
    file.write("   - Tests for direct vs. indirect causal pathways\n\n")
    file.write("=" * 80 + "\n")
    file.write(f"Target: {FLUX}\n")
    file.write(f"Features: {FEATURES}\n")
    file.write("=" * 80 + "\n\n")

print("\n" + "=" * 80)
print("SHAP VALIDATION: ALE + PARTIAL CORRELATIONS + PATH ANALYSIS")
print("=" * 80 + "\n")

_subsets_df = subsets_df.copy()
for ix, siteconfig in _subsets_df.iterrows():

    print("\n" + "-" * 80)
    print(f"SITE {ix + 1}/{len(_subsets_df)}: {siteconfig['SITE']}")
    print("-" * 80)

    # 1. ALE Analysis
    print("\n[1/3] ALE - Accumulated Local Effects")
    train_xgboost_models_and_ale(
        features=FEATURES,
        target=FLUX,
        siteconfig=siteconfig,
        ix=ix,
        modelstxt=modelstxt,
        results_outdir=results_outdir
    )

    # 2. Partial Correlations
    print("\n[2/3] Partial Correlations")
    calculate_partial_correlations(
        features=FEATURES,
        target=FLUX,
        siteconfig=siteconfig,
        ix=ix,
        modelstxt=modelstxt,
        results_outdir=results_outdir
    )

    # 3. Path Analysis
    print("\n[3/3] Path Analysis")
    calculate_path_analysis(
        features=FEATURES,
        target=FLUX,
        siteconfig=siteconfig,
        ix=ix,
        modelstxt=modelstxt,
        results_outdir=results_outdir
    )

    # 4. Integrated Summary
    print("\n[4/4] Creating Integrated Summary")
    create_validation_summary(
        target=FLUX,
        features=FEATURES,
        siteconfig=siteconfig,
        ix=ix,
        results_outdir=results_outdir
    )

print("\n" + "=" * 80)
print("VALIDATION ANALYSIS COMPLETE")
print("=" * 80 + "\n")
print(f"Results saved to: {results_outdir}\n")
print("Output files per site:")
print("  - {SITE}_ale_curves_{TARGET}.csv: Consolidated ALE curve data (all features)")
print("  - {SITE}_ale_combined_{TARGET}.png: Combined ALE plot (2x2 subpanels)")
print("  - {SITE}_partial_correlations_{TARGET}.csv: Partial correlation coefficients")
print("  - {SITE}_path_analysis_{TARGET}.csv: Standardized path coefficients")
print("  - 1_models_xgboost_validation_results_{TARGET}.txt: Summary report")
