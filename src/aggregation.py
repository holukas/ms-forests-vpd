import diive as dv
import numpy as np
import pandas as pd


def aggregate_shap_values_across_sites(df, binx, biny) -> pd.DataFrame:
    print("Aggregating across all sites ...")

    df['BIN_COMBINED_STR'] = df[binx].astype(str) + "+" + df[biny].astype(str)

    # Aggregations for all columns, excluding binx and biny
    # other_aggregations = ['mean', 'median', 'max', 'min', 'count', 'std']
    other_aggregations = ['mean', 'median', 'max', 'min', 'count', 'std',
                          lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]

    # Aggregation for binx and biny (only median)
    bin_aggregations = ['median']

    # Create dictionary to hold the custom aggregations
    agg_dict = {}

    # Get list of columns to apply the 'other_aggregations' to
    cols_for_other_agg = [col for col in df.columns if col not in [binx, biny, 'BIN_COMBINED_STR']]

    agg_dict[binx] = bin_aggregations
    agg_dict[biny] = bin_aggregations

    # Populate aggregation dictionary
    for col in cols_for_other_agg:
        agg_dict[col] = other_aggregations

    # Group and aggregate using the aggregation dictionary
    df_grouped_agg = df.groupby('BIN_COMBINED_STR').agg(agg_dict)
    return df_grouped_agg


def aggregate_shap_values_for_site(site, igbp, filepath, xvar, yvar, aggfunc, ix,
                                   binsize: float = 0.2):
    shapvals_agg_df = pd.DataFrame()

    print(f"\nLoading data for site #{ix + 1} {site} ...")
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
        agg = dv.ga(x=shapvals_df[xvar], y=shapvals_df[yvar], z=shapvals_df[c], **ga_settings)
        res = agg.df_agg_long.copy()
        if cix == 0:
            shapvals_agg_df = res.copy()
        else:
            shapvals_agg_df = pd.concat([shapvals_agg_df, res[c]], axis=1)

    # Add site name
    shapvals_agg_df['SITE'] = site
    shapvals_agg_df['IGBP'] = igbp

    # binx = f"BIN_{xvar}"
    # biny = f"BIN_{yvar}"
    # z = f"TA_ZSCORE_SHAPVALS"
    # hm = dv.heatmapxyz(
    #     x=shapvals_agg_df[binx],
    #     y=shapvals_agg_df[biny],
    #     z=shapvals_agg_df[z],
    #     title=site,
    #     cb_digits_after_comma=1,
    #     xlabel=f'{xvar} (z-score)',
    #     ylabel=f'{yvar} (z-score)',
    #     zlabel=f'{aggfunc} {z} (z-score)',
    #     # vmin=-3,
    #     # vmax=3
    # )
    # hm.show()

    return shapvals_agg_df
