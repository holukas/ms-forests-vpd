"""
Train XGBoost model for each site and save SHAP values to file.
"""
import src.files as files
from src.models import train_xgboost_models_and_shap

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(filename="04_siteinfo.csv")

fluxcol = 'NEE_VUT_50'
fluxqc = "NEE_VUT_50_QC"

siteinfo_df = train_xgboost_models_and_shap(
    settings=settings,
    siteinfo_df=siteinfo_df,
    fluxcol=fluxcol,
    fluxqc=fluxqc,
)

# Save updated site info
files.save_siteinfo(siteinfo_df=siteinfo_df, filename="05_siteinfo.csv")
