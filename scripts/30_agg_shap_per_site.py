from pathlib import Path

import diive as dv
import pandas as pd

import src.files as files
from src.aggregation import aggregate_shap_values_for_site

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

# Aggregate SHAP values for each site and collect in dataframe
shapvals_sites_agg_long_df = None
for ix, siteconfig in datasets_df.iterrows():

    # # TODO testing ----
    # if ix > 1:
    #     break
    # # TODO testing ----

    if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        # Skip files that do not have a parquet subset, b/c of missing SWC
        continue

    site = siteconfig['SITE']
    igbp = siteconfig['IGBP']
    filename = f"{site}_shap-{shap_type}_{FLUX}.parquet"
    filepath = folder / filename

    site_results = aggregate_shap_values_for_site(
        site=site, igbp=igbp, filepath=filepath, ix=ix,
        xvar=xvar, yvar=yvar, aggfunc=aggfunc, binsize=0.1
    )
    if ix == 0:
        shapvals_sites_agg_long_df = site_results.copy()
    else:
        shapvals_sites_agg_long_df = pd.concat([shapvals_sites_agg_long_df, site_results], axis=0)

    # # ---todo testing
    # biny = f"BIN_VPD_F"
    # binx = f"BIN_TA_F"
    # z = f"VPD_F_SHAPVALS"
    # if ix == 0:
    #     dummydf = shapvals_sites_agg_long_df.copy()
    # else:
    #     dummydf = pd.concat([dummydf, shapvals_sites_agg_long_df], axis=0)
    # dummydf2 = aggregate_shap_values_across_all_sites(df=dummydf, binx=binx, biny=biny, z=z)
    # hm = dv.heatmapxyz(
    #     title=f"All sites + {siteconfig['SITE']}",
    #     x=dummydf2[binx], y=dummydf2[biny], z=dummydf2[z],
    #     cb_digits_after_comma=1, xlabel=f'{binx} (z-score)', ylabel=f'{biny} (z-score)',
    #     zlabel=f'{aggfunc} {z} (z-score)',
    #     # show_values_n_dec_places=1,
    #     # show_values=True,
    #     # show_values_fontsize=4,
    #     figdpi=300,
    #     # vmin=-3,
    #     # vmax=3
    # )
    # hm.show()
    # # ---todo testing

outfilepath = dv.save_parquet(
    filename=f"2_PerSite_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}",
    data=shapvals_sites_agg_long_df,
    outpath=folder)
# shapvals_sites_agg_long_df.to_csv(outfilepath.replace('.parquet', '.csv'))
