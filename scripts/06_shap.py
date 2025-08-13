"""
Train XGBoost model for each site and save SHAP values to file.
"""

import src.files as files
from src.models import train_xgboost_models_and_shap
from pathlib import Path

# Variables
FLUX = 'NEP'  # NEP, NEE, LE, GPP, RECO
FEATURES = ['TA', 'SWIN', 'VPD', 'SWC']
CONDITIONAL = True  # Use conditional SHAP instead of standard SHAP

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
subfolder = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / subfolder

# Load site info
siteinfo_df = files.load_siteinfo(settings)

# Writing to a file (overwrites if file exists, creates if not)
modelstxt = Path(results_outdir) / "1_models_xgboost.txt"
with open(modelstxt, 'w') as file:
    file.write("XGBOOST MODELS and SHAP CALCULATIONS\n")
    file.write(f"Target: {FLUX}\n")
    file.write(f"Features: {FEATURES}\n")
    file.write(f"Conditional SHAP: {CONDITIONAL}\n")


_siteinfo_df = siteinfo_df.copy()
for ix, siteconfig in _siteinfo_df.iterrows():
    siteinfo_df = train_xgboost_models_and_shap(
        features=FEATURES,
        target=FLUX,
        settings=settings,
        siteinfo_df=siteinfo_df,
        siteconfig=siteconfig,
        ix=ix, modelstxt=modelstxt,
        conditional=CONDITIONAL
    )

# Save updated site info
files.save_siteinfo(siteinfo_df=siteinfo_df, settings=settings)
