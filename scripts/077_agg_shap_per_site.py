
# TODO TESTING ----------------------------------------------------
# TODO TESTING ----------------------------------------------------
# TODO TESTING ----------------------------------------------------

import diive as dv

import src.files as files

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
ta = []
vpd = []
swc = []
swin = []
for ix, siteconfig in siteinfo_df.iterrows():
    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")
    filepath = siteconfig['_FILEPATH_SHAP_VALUES_STANDARD']
    if filepath == '-MISSING-':
        continue
    shapvals_df = dv.load_parquet(filepath)
    keepcols = [c for c in shapvals_df.columns if "_SHAPVALS" in c]
    df = shapvals_df[keepcols].copy()
    res = df.abs().mean().dropna()
    ta.append(res['TA_F_SHAPVALS'])
    vpd.append(res['VPD_F_SHAPVALS'])
    swc.append(res['SWC_F_MDS_1_SHAPVALS'])
    swin.append(res['SW_IN_F_SHAPVALS'])

import numpy as np

print(np.mean(swin), np.mean(ta), np.mean(vpd), np.mean(swc))
