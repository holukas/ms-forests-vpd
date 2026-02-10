import diive as dv
import numpy as np
import pandas as pd


def aggregate_shap_values_across_sites(df, binx, biny) -> pd.DataFrame:
    print("Aggregating across all sites ...")

    # Strict rounding of binx and biny to 1 decimal place
    # This ensures -3.70000001 becomes -3.7 before string conversion
    df[binx] = df[binx].round(1)
    df[biny] = df[biny].round(1)

    df['BIN_COMBINED_STR'] = df[binx].astype(str) + "+" + df[biny].astype(str)

    # Aggregations for all columns, excluding binx and biny
    other_aggregations = [
        'mean', 'median', 'max', 'min', 'count', 'std',
        ('sem', lambda x: x.std() / np.sqrt(x.count())),
        ('q25', lambda x: x.quantile(0.25)),
        ('q75', lambda x: x.quantile(0.75))
    ]

    # Aggregation for binx and biny class/bin identifiers
    # FORCE 'median' (or 'min'/'max') for the bin labels, we need unique bin IDs
    # Note: using 'mean' here will introduce floating point errors every once in a while,
    # resulting in some strange bins (e.g., -3.70000001 instead of -3.7)
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

    # Round custom bins for exactly one digit after the comma
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
        min_n_vals_per_bin=1,  # Number of 30MIN values
        aggfunc=aggfunc
    )

    # Define the bin column names (diive usually names them like BIN_xvar)
    binx_col = f"BIN_{xvar}"
    biny_col = f"BIN_{yvar}"

    for cix, c in enumerate(available_cols):
        agg = dv.ga(x=shapvals_df[xvar], y=shapvals_df[yvar], z=shapvals_df[c], **ga_settings)
        res = agg.df_agg_long.copy()

        # Ensure bin columns are rounded to avoid merge mismatches
        if binx_col in res.columns:
            res[binx_col] = res[binx_col].round(1)
        if biny_col in res.columns:
            res[biny_col] = res[biny_col].round(1)

        if cix == 0:
            shapvals_agg_df = res.copy()
        else:
            # Merge on bin coordinates
            subset = res[[binx_col, biny_col, c]]

            # Outer merge ensures we keep bins that exist in one var but not another
            shapvals_agg_df = pd.merge(
                shapvals_agg_df,
                subset,
                on=[binx_col, biny_col],
                how='outer'
            )
            # shapvals_agg_df = pd.concat([shapvals_agg_df, res[c]], axis=1)

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
