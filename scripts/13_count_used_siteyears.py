"""
Train XGBoost model for each site and save SHAP values to file.
"""

from pathlib import Path

import pandas as pd

import src.files as files
from src.models import train_xgboost_models_and_shap

# ------------------------------
# Variables
FLUX = 'RECO'  # NEP, ET, GPP, RECO, NEE, LE
FEATURES = ['TA', 'SWIN', 'VPD', 'SWC']
CONDITIONAL = True  # Use conditional SHAP instead of standard SHAP
# ------------------------------

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
subfolder = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / subfolder

# Load datasets info
infile = Path('../data/outputs/12_datasets_parquet_vars_stats_subsets.csv')
datasets_df = pd.read_csv(infile)

# Sites where SWC is available and that are not DNF (only 2 sites)
subset = datasets_df.loc[datasets_df['SWC_AVG'] != '-MISSING-'].copy()
subset = subset.loc[subset['IGBP'] != 'DNF'].copy()
subset = subset.reset_index(drop=True)
print(f"{subset[['SITE', 'ELEVATION', 'N_YEARS']]}")
print(f"\nSite years:\n  "
      f"Total: {int(subset['N_YEARS'].sum())}\n  "
      f"Min: {int(subset['N_YEARS'].min())}\n  "
      f"Max: {int(subset['N_YEARS'].max())}\n  "
      f"Top 10 most years: {subset.sort_values(by='N_YEARS', ascending=False).head(10)['SITE'].values}\n  "
      f"Top 10 least years: XXX")
print(f"\nElevation:\n  "
      f"Max: {int(subset['ELEVATION'].max())}\n  "
      f"Min: {int(subset['ELEVATION'].min())}")
