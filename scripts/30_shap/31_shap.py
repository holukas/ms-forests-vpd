"""
Train XGBoost model for each site and save SHAP values to file.
"""

from pathlib import Path

import pandas as pd

import src.files as files
from src.models import train_xgboost_models_and_shap

# ------------------------------
# Variables
FLUX = 'ET'  # NEP, ET, GPP, RECO, NEE, LE
FEATURES = ['TA', 'SWIN', 'VPD', 'SWC']
CONDITIONAL = True  # Use conditional SHAP instead of standard SHAP
# ------------------------------

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
subfolder = 'conditional' if CONDITIONAL else 'standard'

# Load datasets info
infile = Path('../../data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv')
subsets_df = pd.read_csv(infile)

# Create output directory
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / subfolder
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
for ix, siteconfig in _subsets_df.iterrows():
    train_xgboost_models_and_shap(
        features=FEATURES,
        target=FLUX,
        siteconfig=siteconfig,
        ix=ix, modelstxt=modelstxt,
        conditional=CONDITIONAL,
        results_outdir=results_outdir
    )
