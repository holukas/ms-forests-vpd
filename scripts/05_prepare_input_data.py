"""
Train XGBoost model for each site and save SHAP values to file.
"""
import src.files as files

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(filename="04_siteinfo.csv")

_siteinfo_df = siteinfo_df.copy()
for ix, siteconfig in _siteinfo_df.iterrows():
    siteinfo_df = files.prepare_input_data(
        ix=ix,
        siteconfig=siteconfig,
        settings=settings,
        siteinfo_df=siteinfo_df
    )

# Save updated site info
files.save_siteinfo(siteinfo_df=siteinfo_df, filename="05_siteinfo.csv")
