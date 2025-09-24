from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import pandas as pd

import src.files as files

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'NEP'
# FLUX = 'LE'
# FLUX = 'GPP'
# FLUX = 'RECO'
xvar = 'TA'
yvar = 'VPD'
aggfunc = 'median'
CONDITIONAL = True  # SHAP
# ------------------------------

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
subfolder = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / subfolder
shap_type = 'conditional' if CONDITIONAL else 'standard'
folder = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Load datasets info
infile = Path('../data/outputs/12_datasets_parquet_vars_stats_subsets.csv')
datasets_df = pd.read_csv(infile)


shapvals_sites_agg_long_df = None
sites_df = None
for ix, siteconfig in datasets_df.iterrows():

    # TODO testing ----
    if ix > 5:
        break
    # TODO testing ----

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        # Skip files that do not have a parquet subset, b/c of missing SWC
        continue

    igbp = siteconfig['IGBP']

    if igbp == 'DNF':  # Not used, only 2 sites
        continue

    site = siteconfig['SITE']
    filename = f"{site}_shap-{shap_type}_{FLUX}.parquet"
    filepath = folder / filename

    print(f"\nLoading data for site #{ix + 1} {site} ...")
    shapvals_df = dv.load_parquet(filepath)

    keepcols = [c for c in shapvals_df.columns if "_SHAPVALS" in c]
    subset = shapvals_df[keepcols].copy()
    n_records = len(subset.index)

    cur_site_dict = {}
    for k in keepcols:
        series = subset[k].copy()
        cur_site_dict['SITE'] = site
        cur_site_dict['IGBP'] = igbp
        cur_site_dict[f'{k}_POSSUM'] = series[series > 0].sum() / n_records
        cur_site_dict[f'{k}_NEGSUM'] = series[series < 0].sum() / n_records
        cur_site_dict[f'{k}_OVRSUM'] = cur_site_dict[f'{k}_POSSUM'] + cur_site_dict[f'{k}_NEGSUM']
        cur_site_dict[f'{k}_OVRSUMRANGE'] = abs(cur_site_dict[f'{k}_POSSUM']) + abs(cur_site_dict[f'{k}_NEGSUM'])

    new_row_df = pd.DataFrame([cur_site_dict], index=[site])

    if ix == 0:
        sites_df = new_row_df.copy()
    else:
        sites_df = pd.concat([sites_df, new_row_df], axis=0)

    # plt.hist(subset['SWC_SHAPVALS'].values.flatten(), bins=100)
    # plt.show()

    # site_results = aggregate_shap_values_for_site(
    #     site=site, igbp=igbp, filepath=filepath, ix=ix,
    #     xvar=xvar, yvar=yvar, aggfunc=aggfunc, binsize=0.1
    # )
    # if ix == 0:
    #     shapvals_sites_agg_long_df = site_results.copy()
    # else:
    #     shapvals_sites_agg_long_df = pd.concat([shapvals_sites_agg_long_df, site_results], axis=0)

outfilepath = dv.save_parquet(
    filename=f"3_AllSites_FeatureImportanceSHAPsums-{shap_type}_{FLUX}",
    data=sites_df,
    outpath=folder)
sites_df.to_csv(outfilepath.replace('.parquet', '.csv'))
