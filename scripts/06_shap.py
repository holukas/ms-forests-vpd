"""
Train XGBoost model for each site and save SHAP values to file.
"""
import src.files as files
from src.models import train_xgboost_models_and_shap

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(settings)

_siteinfo_df = siteinfo_df.copy()
for ix, siteconfig in _siteinfo_df.iterrows():
    siteinfo_df = train_xgboost_models_and_shap(
        settings=settings,
        siteinfo_df=siteinfo_df,
        siteconfig=siteconfig,
        ix=ix
    )

# Save updated site info
files.save_siteinfo(siteinfo_df=siteinfo_df, settings=settings)
