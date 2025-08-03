from pathlib import Path

import diive as dv

import src.files as files
from src.aggregation import aggregate_shap_values_for_site

# VARIABLES (site-level)
# ----------------------
# 'ta_var', 'vpd_var', 'swc_var', 'swin_var'
xvar = 'ta_var'
yvar = 'vpd_var'
zvar = 'vpd_var'
aggfunc = 'median'
conditional = False  # SHAP

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(settings)

# Aggregate SHAP values for each site and collect in dataframe
shapvals_sites_agg_long_df = None
for ix, siteconfig in siteinfo_df.iterrows():
    shapvals_sites_agg_long_df = aggregate_shap_values_for_site(
        siteconfig=siteconfig, ix=ix,
        shapvals_sites_agg_long_df=shapvals_sites_agg_long_df,
        xvar=xvar, yvar=yvar, zvar=zvar, aggfunc=aggfunc,
        conditional=conditional
    )

# Save SHAP values aggregated per site to Parquet and CSV
outfilepath = dv.save_parquet(
    filename=f"2_PerSite_Aggregated_SHAPValues",
    data=shapvals_sites_agg_long_df,
    outpath=Path(settings['DIR_DATA_OUT_SHAPVALS_STANDARD']))
shapvals_sites_agg_long_df.to_csv(outfilepath.replace('.parquet', '.csv'))
