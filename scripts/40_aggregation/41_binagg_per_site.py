"""
Aggregate SHAP values in bins of 2 variables, per site.
"""
from pathlib import Path

import diive as dv
import pandas as pd

import src.files as files
from src.aggregation import aggregate_shap_values_for_site
from src.paths import load_settings

# ------------------------------
# Variables
# NEP, ET, GPP, RECO, TA, VPD, SWIN, SWC

# Settings for searching in the correct (sub)folder
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'
CONDITIONAL = True  # SHAP

# Run variant. An empty string reads the baseline results and overwrites the
# submitted aggregation. Any other value adds a folder level on both sides, so
# the inputs come from <stage>/<FLUX>/<shap_type>/<VARIANT>/ and the outputs go
# to the matching variant folder. The earlier stages must have run with the same
# value, otherwise there is nothing to read.
VARIANT = ""

# Aggregation combos: xvar / yvar. Every pair listed here is processed in one run,
# which keeps a variant complete: the figures need all five.
VAR_PAIRS = [
    ['TA_ZSCORE', 'VPD_ZSCORE'],
    ['SWC_ZSCORE', 'VPD_ZSCORE'],
    ['TA_ZSCORE', 'SWC_ZSCORE'],
    ['ET_ZSCORE', 'VPD_ZSCORE'],
    ['ET_ZSCORE', 'SWC_ZSCORE'],
]

aggfunc = 'mean'

# ------------------------------
# Agg groups, use z-scores:
# NEP:  [x]TA/VPD [x]SWC/VPD [x]TA/SWC [x] ET/VPD [x] ET/SWC [ ]SWIN/TA [ ]SWIN/VPD
# ET:   [ ]TA/VPD [ ]SWC/VPD [ ]TA/SWC [ ] ET/VPD [ ] ET/SWC [ ]SWIN/TA [ ]SWIN/VPD
# GPP:  [ ]TA/VPD [ ]SWC/VPD [ ]TA/SWC [ ] ET/VPD [ ] ET/SWC [ ]SWIN/TA [ ]SWIN/VPD
# RECO: [ ]TA/VPD [ ]SWC/VPD [ ]TA/SWC [ ] ET/VPD [ ] ET/SWC [ ]SWIN/TA [ ]SWIN/VPD
# ------------------------------

for VARS in VAR_PAIRS:
    print('')
    print('=' * 70)
    print(f'{VARS[0]} x {VARS[1]}')
    print('=' * 70)

    xvar = VARS[0]
    yvar = VARS[1]

    # Load settings
    settings = load_settings()
    shap_type = 'conditional' if CONDITIONAL else 'standard'
    dir_prev_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / VARIANT

    # Load subsets info
    infile = (Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
              / "21_SUBSETS_parquet_vars_stats_subsets.csv")
    subsets_df = pd.read_csv(infile)

    # Create output directory
    dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT
    # parents=True: Creates any necessary parent directories that don't exist.
    # exist_ok=True: Prevents an error if the directory already exists.
    dir_out.mkdir(parents=True, exist_ok=True)

    # Aggregate SHAP values for each site and collect in dataframe
    shapvals_sites_agg_long_df = None
    for ix, siteconfig in subsets_df.iterrows():

        # # TODO testing ----
        # if ix > 5:
        #     break
        # # TODO testing ----

        # if siteconfig['_FILEPATH_PARQUET_SUBSET'] == '-MISSING-':
        #     # Skip files that do not have a parquet subset, b/c of missing SWC
        #     continue

        site = siteconfig['SITE']
        igbp = siteconfig['IGBP']
        filename = f"{site}_shap-{shap_type}_{FLUX}.parquet"
        filepath = dir_prev_results / filename

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
        filename=f"41_SHAPVALUES-{shap_type}_{aggfunc}AggregatedPerSite_BIN-{xvar}+BIN-{yvar}+{FLUX}",
        data=shapvals_sites_agg_long_df,
        outpath=dir_out)
    shapvals_sites_agg_long_df.to_csv(outfilepath.replace('.parquet', '.csv'))
