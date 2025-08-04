import diive as dv
import numpy as np
import pandas as pd

from common import get_variable_names


def aggregate_shap_values_across_all_sites(df, binx, biny, z) -> pd.DataFrame:
    df['BIN_COMBINED_STR'] = df[binx].astype(str) + "+" + df[biny].astype(str)
    df_grouped_agg = df.groupby('BIN_COMBINED_STR').median()
    df_grouped_agg[f'{z}_COUNTS'] = df.groupby('BIN_COMBINED_STR').count()[z].copy()
    df_grouped_agg[f'{z}_P25'] = df.groupby('BIN_COMBINED_STR').quantile(0.25)[z].copy()
    df_grouped_agg[f'{z}_P75'] = df.groupby('BIN_COMBINED_STR').quantile(0.75)[z].copy()
    return df_grouped_agg


def aggregate_shap_values_for_site(siteconfig, xvar, yvar, zvar, aggfunc, ix,
                                   shapvals_sites_agg_long_df, conditional=False):
    # Get variable names for this site
    varnames = get_variable_names(siteconfig)

    x = varnames[xvar]
    y = varnames[yvar]
    z = f"{varnames[zvar]}_SHAPVALS"

    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")
    filepath = siteconfig['_FILEPATH_SHAP_VALUES_STANDARD']
    if filepath == '-MISSING-':
        return shapvals_sites_agg_long_df
    shapvals_df = dv.load_parquet(filepath)

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
        min_n_vals_per_bin=1,  # Number of 30MIN values
        aggfunc=aggfunc
    )
    # print(q.df_agg_wide)
    # print(q.df_agg_long['BIN_VPD_F'].unique())

    # binx = f"BIN_VPD_F"
    # biny = f"BIN_TA_F"
    # z = f"VPD_F_SHAPVALS"
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

    # if ix == 0:
    #     df_all = q.df_agg_long.copy()
    # else:
    shapvals_sites_agg_long_df = pd.concat([shapvals_sites_agg_long_df, q.df_agg_long], axis=0)
    return shapvals_sites_agg_long_df
