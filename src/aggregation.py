import diive as dv
import numpy as np
import pandas as pd

from common import get_variable_names


def aggregate_shap_values_across_all_sites(df, binx, biny, z) -> pd.DataFrame:

    print("Aggregating across all sites ...")

    # Remove all rows where all records are NaN,
    # binx and biny columns are ignored for this check
    cols_to_ignore = [binx, biny]
    cols_to_check = [col for col in df.columns if col not in cols_to_ignore]

    # Filter out rows where all values in the selected columns are NaN
    df_cleaned = df.loc[~df[cols_to_check].isna().all(axis=1)].copy()

    df_cleaned['BIN_COMBINED_STR'] = df_cleaned[binx].astype(str) + "+" + df_cleaned[biny].astype(str)

    # Aggregations for all columns, excluding binx and biny
    other_aggregations = ['mean', 'median', 'max', 'min', 'count', 'std',
                          lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]

    # Aggregation for binx and biny (only median)
    bin_aggregations = ['median']

    # Create dictionary to hold the custom aggregations
    agg_dict = {}

    # Get list of columns to apply the 'other_aggregations' to
    cols_for_other_agg = [col for col in df_cleaned.columns if col not in [binx, biny, 'BIN_COMBINED_STR']]

    agg_dict[binx] = bin_aggregations
    agg_dict[biny] = bin_aggregations

    # Populate aggregation dictionary
    for col in cols_for_other_agg:
        agg_dict[col] = other_aggregations

    # Group and aggregate using the aggregation dictionary
    df_grouped_agg = df_cleaned.groupby('BIN_COMBINED_STR').agg(agg_dict)
    return df_grouped_agg


def aggregate_shap_values_for_site(siteconfig, xvar, yvar, zvar, aggfunc, ix,
                                   conditional=False, binsize: float = 0.2):
    site_res = pd.DataFrame()

    # Get variable names for this site
    varnames = get_variable_names(siteconfig)

    x = varnames[xvar]
    y = varnames[yvar]
    z = f"{varnames[zvar]}_SHAPVALS"

    print(f"\nLoading data for site #{ix + 1} {siteconfig['SITE']} ...")
    if conditional:
        filepath = siteconfig['_FILEPATH_SHAP_VALUES_CONDITIONAL']
    else:
        filepath = siteconfig['_FILEPATH_SHAP_VALUES_STANDARD']
    if filepath == '-MISSING-' or pd.isna(filepath):
        return site_res
    shapvals_df = dv.load_parquet(filepath)

    # Round custom bins for exactly two digits after the comma
    custom_x_bins = list(np.arange(-8, 10, binsize))
    rounded_custom_x_bins_float = [round(num, 1) for num in custom_x_bins]
    custom_y_bins = list(np.arange(-8, 10, binsize))
    rounded_custom_y_bins_float = [round(num, 1) for num in custom_y_bins]

    # Calculate aggregates in x/y bins
    available_cols = shapvals_df.columns
    ga_settings = dict(
        binning_type='custom',
        custom_x_bins=rounded_custom_x_bins_float,
        custom_y_bins=rounded_custom_y_bins_float,
        # binning_type='quantiles',
        # binning_type='equal_width',
        # n_bins=20,
        min_n_vals_per_bin=1,  # Number of 30MIN values
        aggfunc=aggfunc
    )

    for cix, c in enumerate(available_cols):
        agg = dv.ga(x=shapvals_df[x], y=shapvals_df[y], z=shapvals_df[c], **ga_settings)
        res = agg.df_agg_long.copy()
        if cix == 0:
            site_res = res.copy()
        else:
            site_res = pd.concat([site_res, res[c]], axis=1)

    # Add site name
    site_res['SITE'] = siteconfig['SITE']

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

    return site_res
