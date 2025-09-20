"""
Train XGBoost model for each site and save SHAP values to file.
"""

from pathlib import Path

import pandas as pd

import src.files as files
from src.models import train_xgboost_models_and_shap

# ------------------------------
# Variables
FLUX = 'NEP'  # NEP, NEE, LE, GPP, RECO
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

# Write to file (overwrites if file exists, creates if not)
substr = "conditional" if CONDITIONAL else "standard"
modelstxt = Path(results_outdir) / f"1_models_xgboost_shap-{substr}_{FLUX}.txt"
with open(modelstxt, 'w') as file:
    file.write("XGBOOST MODELS and SHAP CALCULATIONS\n")
    file.write("------------------------------------\n")
    file.write(f"Target: {FLUX}\n")
    file.write(f"Features: {FEATURES}\n")
    file.write(f"Conditional SHAP: {CONDITIONAL}\n")

_datasets_df = datasets_df.copy()
for ix, siteconfig in _datasets_df.iterrows():
    train_xgboost_models_and_shap(
        features=FEATURES,
        target=FLUX,
        siteconfig=siteconfig,
        ix=ix, modelstxt=modelstxt,
        conditional=CONDITIONAL,
        results_outdir=results_outdir
    )
