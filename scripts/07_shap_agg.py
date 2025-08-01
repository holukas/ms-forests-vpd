"""
Train XGBoost model for each site and save SHAP values to file.
"""
from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd

import src.files as files
from common import get_variable_names

# Load settings
settings = files.read_settings_file("../config/settings.yaml")

# Load site info
siteinfo_df = files.load_siteinfo(settings)

df_all = None

for ix, siteconfig in siteinfo_df.iterrows():

    # Get variable names for this site
    varnames = get_variable_names(siteconfig)

    x = varnames['ta_var']
    y = varnames['vpd_var']
    z = f"{varnames['vpd_var']}_SHAPVALS"

    binx = f"BIN_{x}"
    biny = f"BIN_{y}"
    aggfunc = 'median'

    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")
    filepath = siteconfig['_FILEPATH_SHAP_VALUES']

    if filepath == '-MISSING-':
        continue

    shapvals_df = dv.load_parquet(filepath)
    # print(shapvals_df)

    # Round custom bins for exactly two digits after the comma
    custom_x_bins = list(np.arange(-8, 10, .2))
    rounded_custom_x_bins_float = [round(num, 1) for num in custom_x_bins]
    custom_y_bins = list(np.arange(-8, 10, .2))
    rounded_custom_y_bins_float = [round(num, 1) for num in custom_y_bins]

    q = dv.ga(
        x=shapvals_df[x],
        y=shapvals_df[y],
        z=shapvals_df[z],
        binning_type='custom',
        custom_x_bins=rounded_custom_x_bins_float,
        custom_y_bins=rounded_custom_y_bins_float,
        # binning_type='quantiles',
        # binning_type='equal_width',
        # n_bins=20,
        min_n_vals_per_bin=3,  # Number of 30MIN values
        aggfunc=aggfunc
    )
    # print(q.df_agg_wide)
    # print(q.df_agg_long['BIN_VPD_F'].unique())

    # hm = dv.heatmapxyz(
    #     x=q.df_agg_long[binx],
    #     y=q.df_agg_long[biny],
    #     z=q.df_agg_long[z],
    #     title=siteconfig['SITE'],
    #     cb_digits_after_comma=1,
    #     xlabel=f'{binx} (z-score)',
    #     ylabel=f'{biny} (z-score)',
    #     zlabel=f'{aggfunc} {z} (z-score)',
    #     # vmin=-3,
    #     # vmax=3
    # )
    # hm.show()

    if ix == 0:
        df_all = q.df_agg_long.copy()
    else:
        df_all = pd.concat([df_all, q.df_agg_long], axis=0)

    # # TODO testing --------------------
    # # _df_all = df_all.copy()
    # # _df_all['BIN_COMBINED_STR'] = (_df_all[binx].astype(str) + "+" + _df_all[biny].astype(str))
    # # _df_all = _df_all.groupby('BIN_COMBINED_STR').median()
    # hm = dv.heatmapxyz(
    #     title="All sites",
    #     x=df_all[binx],
    #     y=df_all[biny],
    #     z=df_all[z],
    #     cb_digits_after_comma=1,
    #     xlabel=f'{binx} (z-score)',
    #     ylabel=f'{biny} (z-score)',
    #     zlabel=f'{aggfunc} {z} (z-score)',
    #     # vmin=-3,
    #     # vmax=3
    # )
    # hm.show()
    # print("X")
    # # TODO testing --------------------

df_all['BIN_COMBINED_STR'] = (df_all[binx].astype(str) + "+" + df_all[biny].astype(str))
df_all_median = df_all.groupby('BIN_COMBINED_STR').median()
df_all_counts = df_all.groupby('BIN_COMBINED_STR').count()

df_all_median[f'{z}_COUNTS'] = df_all_counts[z].copy()

outfilepath = dv.save_parquet(
    filename=f"2_ALLSITES_shap_values_median",
    data=df_all_median,
    outpath=Path(settings['DIR_DATA_OUT_SHAPVALS_SITE']))
# print(f"Saved SHAP values across all files as mean to file {outfilepath}.")
df_all_median.to_csv(outfilepath.replace('.parquet', '.csv'))

# print(df_all)
