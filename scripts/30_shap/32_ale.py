"""
Train XGBoost model for each site and calculate ALE (Accumulated Local Effects) values.
"""

from pathlib import Path

import pandas as pd

import src.files as files
from src.models import train_xgboost_models_and_ale

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
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'ale'
# parents=True: Creates any necessary parent directories that don't exist.
# exist_ok=True: Prevents an error if the directory already exists.
results_outdir.mkdir(parents=True, exist_ok=True)

# Write to file (overwrites if file exists, creates if not)
modelstxt = Path(results_outdir) / f"1_models_xgboost_ale_{FLUX}.txt"
with open(modelstxt, 'w') as file:
    file.write("XGBOOST MODELS and ALE (Accumulated Local Effects) CALCULATIONS\n")
    file.write("--------------------------------------------------------------\n")
    file.write(f"Target: {FLUX}\n")
    file.write(f"Features: {FEATURES}\n")
    file.write("ALE shows the isolated effect of each feature on predictions,\n")
    file.write("accounting for correlations with other features.\n")

_subsets_df = subsets_df.copy()
for ix, siteconfig in _subsets_df.iterrows():
    train_xgboost_models_and_ale(
        features=FEATURES,
        target=FLUX,
        siteconfig=siteconfig,
        ix=ix,
        modelstxt=modelstxt,
        results_outdir=results_outdir
    )
