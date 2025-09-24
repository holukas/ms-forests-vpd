from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

import src.files as files

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
# FLUX = 'NEP'
# FLUX = 'LE'
FLUX = 'GPP'
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

    # # TODO testing ----
    # if (ix < 55) | (ix > 77):
    #     continue
    # # TODO testing ----

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

    shapvals_df = shapvals_df.loc[shapvals_df['SWC'] < -1].copy()
    shapvals_df = shapvals_df.loc[shapvals_df['TA'] > 1].copy()
    shapvals_df = shapvals_df.loc[shapvals_df['VPD'] > 1].copy()
    # shapvals_df[['SWIN_SHAPVALS', 'VPD_SHAPVALS']].cumsum().plot()
    # plt.show()

    keepcols = [c for c in shapvals_df.columns if "_SHAPVALS" in c]
    subset = shapvals_df[keepcols].copy()
    n_records = len(subset.index)

    cur_site_dict = {}
    for k in keepcols:
        series = subset[k].copy()
        cur_site_dict['SITE'] = site
        cur_site_dict['IGBP'] = igbp
        cur_site_dict[f'{k}_POS_AVG'] = series[series > 0].mean()
        cur_site_dict[f'{k}_NEG_AVG'] = series[series < 0].mean()
        cur_site_dict[f'{k}_OVR_AVG'] = series.mean()
        cur_site_dict[f'{k}_OVR_ABS_AVG'] = series.abs().mean()

    new_row_df = pd.DataFrame([cur_site_dict], index=[site])

    if ix == 0:
        sites_df = new_row_df.copy()
    else:
        sites_df = pd.concat([sites_df, new_row_df], axis=0)

sites_df = sites_df.drop('SITE', axis=1, inplace=False)
sites_df = sites_df.groupby('IGBP').mean()

fig = plt.figure(figsize=(9, 6), dpi=150, facecolor="white")
gs = gridspec.GridSpec(1, 1)  # rows, cols
# gs.update(wspace=.3, hspace=.2, left=0.03, right=0.94, top=0.97, bottom=0.04)
ax = fig.add_subplot(gs[0, 0])
plotdf = sites_df[['TA_SHAPVALS_OVR_AVG', 'VPD_SHAPVALS_OVR_AVG',
                   'SWIN_SHAPVALS_OVR_AVG', 'SWC_SHAPVALS_OVR_AVG']].copy()
plotdf.plot.bar(ax=ax)
# plotdf = plotdf.pivot(index='SITE', columns='Group', values='Value')
# sites_df['TA_SHAPVALS_OVRSUMRANGE'].plot.bar(color='#EF5350', ax=ax, legend=False, width=.7)
# sites_df['TA_SHAPVALS_OVRSUMRANGE'].plot.bar(color='#03A9F4', ax=ax, legend=False, width=.7)
# sites_df['TA_SHAPVALS_NEGSUM'].plot.bar(color='#03A9F4', legend=False, width=.7)
fig.tight_layout()
fig.show()

outfilepath = dv.save_parquet(
    filename=f"3_AllSites_FeatureImportanceSHAPsums-{shap_type}_{FLUX}",
    data=sites_df,
    outpath=folder)
sites_df.to_csv(outfilepath.replace('.parquet', '.csv'))

print(sites_df)
