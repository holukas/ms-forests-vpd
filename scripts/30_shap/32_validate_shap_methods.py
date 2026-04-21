"""
ALE (Accumulated Local Effects) Validation for SHAP Results

## Purpose
Calculate ALE curves as an independent validation method for SHAP feature importance.
ALE plots show how features affect model predictions in isolation, accounting for
feature correlations—complementary to SHAP value interpretations.

## Why ALE Validates SHAP?
- SHAP: Feature importance from model explanations (per-sample contributions)
- ALE: Isolated feature effects on predictions (across the feature range)
- Agreement: If SHAP and ALE agree on effect direction/strength → high confidence
- Disagreement: Suggests complex interactions or data-specific patterns

## Output Per Site
- {SITE}_ale_curves_{TARGET}.csv — ALE curve data (100 grid points per feature)
- {SITE}_ale_combined_{TARGET}.png — 2x2 subplot with all 4 features
- {SITE}_ale_{TARGET}.parquet/csv — Full dataset with predictions

## Future Extensions (commented out)
- Partial Correlations: Statistical feature importance (correlation after removing confounders)
- Path Analysis: Direct vs indirect causal effects
These can be enabled by uncommenting imports and function calls in the main loop.
"""

from pathlib import Path

import pandas as pd

import src.files as files
from src.models import train_xgboost_models_and_ale
# Optional validation methods (commented out - uncomment to enable):
# from src.models import calculate_partial_correlations, calculate_path_analysis, create_validation_summary

# ------------------------------
# Variables
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'
FEATURES = ['TA_ZSCORE', 'SWIN_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']

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
settings = files.read_settings_file("../../config/settings.yaml")

# Load subsets info
infile = Path('../../data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv')
subsets_df = pd.read_csv(infile)

# Create output directory
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'ale'
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
ale_results_all = []

# Add mode indicator to header
with open(modelstxt, 'a', encoding='utf-8') as file:
    file.write(f"\nMODE: ALE VALIDATION (grid_size=100)\n")

print(f"\n{'=' * 80}")
print(f"ALE (ACCUMULATED LOCAL EFFECTS) VALIDATION")
print(f"{'=' * 80}\n")

for ix, siteconfig in _subsets_df.iterrows():
    # if ix > 1:
    #     break
    # if siteconfig['SITE'] != "CH-Dav":
    #     continue

    # Main: ALE Analysis
    train_xgboost_models_and_ale(
        features=FEATURES,
        target=FLUX,
        siteconfig=siteconfig,
        ix=ix,
        modelstxt=modelstxt,
        results_outdir=results_outdir
    )

    # Collect results for aggregation
    ale_result = {
        'site': siteconfig['SITE'],
        'target': FLUX,
        'n_records': siteconfig.get('N_RECORDS', 'N/A')
    }
    if ale_result:
        ale_results_all.append(ale_result)

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
print("  ✓ Match SHAP direction ↔ ALE effect → High confidence")
print("  ✗ Disagreement → Investigate feature correlations or interactions\n")
