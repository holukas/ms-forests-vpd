import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def peak_season_months(sitedata, gpp_col: str, n_months: int = 4) -> list:
    """Find the calendar months with the highest mean GPP at a site.

    This is the peak-season definition of the analysis. It is applied to the full
    record, before quality-control or daytime filtering, so the season does not
    move when those filters change.

    Args:
        sitedata: Site data with a DatetimeIndex.
        gpp_col: Name of the GPP column.
        n_months: How many months to keep.

    Returns:
        Month numbers (1 to 12), highest mean GPP first.
    """
    monthly_mean = sitedata.groupby(sitedata.index.month)[gpp_col].mean()
    return monthly_mean.nlargest(n_months).index.to_list()


def findpoi(df, k: int = 9, agg: str = 'mean', what: str = 'max'):
    """Find point of interest (POI) in a DataFrame grid.

    Calculates a local aggregation (e.g., mean) for each point based on its
    k-nearest neighbors (including the point itself) and then finds the
    maximum or minimum of these aggregated values.
    """
    # Prepare Coordinates and Data
    data_array = df.values
    rows, cols = data_array.shape

    # Create a list of all (row, col) coordinates in the grid
    all_coords = np.indices((rows, cols)).reshape(2, -1).T

    # Build a KD-Tree for fast neighbor searches
    # The cKDTree method is the most accurate and robust way to derive every point's value
    # from its true 9 nearest neighbors, regardless of its position in the grid.
    tree = cKDTree(all_coords)

    # Query the tree to find the k nearest neighbors for EVERY point.
    # IMPORTANT: Since the query points are the same as the data points,
    # the first neighbor (index 0) for any point is ALWAYS the point itself.
    # So, with k=9, you get the center point + its 8 closest neighbors.
    distances, neighbor_indices = tree.query(all_coords, k=k)

    # Use indices to get neighbor values and calculate aggregations
    # Get the coordinates of the neighbors using the indices from the query
    neighbor_coords = all_coords[neighbor_indices]

    # Use the neighbor coordinates to get the values from the original data array.
    # This uses advanced NumPy indexing to fetch all neighbor values at once.
    neighbor_values = data_array[neighbor_coords[:, :, 0], neighbor_coords[:, :, 1]]

    # Calculate the aggregation for each set of k neighbors, ignoring NaNs
    # The result is a 1D array of aggregations.
    if agg == 'mean':
        # 1. Count the number of non-NaN values for each point's neighborhood
        non_nan_counts = np.sum(~np.isnan(neighbor_values), axis=1)
        # 2. Calculate the mean, ignoring NaNs (as before)
        aggs = np.nanmean(neighbor_values, axis=1)
        # 3. Set the aggregation to NaN if the count of valid neighbors is less than k
        aggs[non_nan_counts < k] = np.nan

    else:
        raise NotImplementedError(f"{agg} not supported.")

    # Reshape Results Back to the Grid
    # Reshape the 1D means array back into the original 2D grid shape.
    result_array = aggs.reshape(rows, cols)

    # Use the original data as a mask. Where it was NaN, make the result NaN.
    result_array[np.isnan(data_array)] = np.nan

    # Convert the final array back to a DataFrame
    true_knn_df = pd.DataFrame(
        result_array,
        index=df.index,
        columns=df.columns
    )

    # Find the maximum value in the entire DataFrame
    if what == 'max':
        value = true_knn_df.stack().max()
        # Find the location (row, column) of maximum value
        location = true_knn_df.stack().idxmax()
    elif what == 'min':
        value = true_knn_df.stack().min()
        location = true_knn_df.stack().idxmin()
    else:
        raise NotImplementedError(f"{what} not implemented.")

    return location, value


DEEPEST_SWC_MIN_PEAK_RATIO = 0.90


def deepest_swc_per_site(min_peak_ratio: float = DEEPEST_SWC_MIN_PEAK_RATIO) -> dict:
    """
    Pick the deepest usable soil water layer for every site (deep SWC sensitivity run).

    The main analysis mostly uses `SWC_F_MDS_1`, the shallowest layer. A deeper layer
    must keep at least `min_peak_ratio` of layer 1's records within the four
    peak months, so that a change in results is not caused by a smaller sample.
    The deepest layer that passes is chosen; sites where none passes stay on
    layer 1. Depths in cm are not known, so this is the deepest available
    layer, not a fixed rooting depth.

    Reads `13b_variables_per_site.csv` (from `13b_check_available_vars_SWC.py`).

    Args:
        min_peak_ratio: Records the deeper layer must keep within the peak
            months, as a fraction of what layer 1 has.

    Returns:
        dict: Site to `{'swc_var': name, 'swc_qc_var': name, 'layer': int}` for
        every site that moves off layer 1. Sites that stay are not included.
    """
    from src.paths import data_path

    filepath = data_path("data/outputs/10_datasets/13b_variables_per_site.csv")
    df = pd.read_csv(filepath)

    is_layer = df['VARIABLE_NAME'].str.match(r'SWC_F_MDS_\d+$', na=False)
    layers = df[df['USED_SITE'] & is_layer].copy()
    layers['LAYER'] = layers['VARIABLE_NAME'].str.extract(r'_(\d+)$').astype(int)

    chosen = {}
    for site, site_layers in layers.groupby('SITE'):
        deeper = site_layers[(site_layers['LAYER'] > 1)
                             & (site_layers['PEAK_VS_SWC1'] >= min_peak_ratio)]
        if deeper.empty:
            continue
        best = deeper.loc[deeper['LAYER'].idxmax()]
        chosen[str(site)] = {'swc_var': str(best['VARIABLE_NAME']),
                             'swc_qc_var': f"{best['VARIABLE_NAME']}_QC",
                             'layer': int(best['LAYER'])}
    return chosen


def deeper_swc_sites() -> set:
    """
    Sites that do not use the shallowest soil water layer in the deep run.

    The deep run keeps every site, and sites without a usable deeper layer stay
    on layer 1. This set is for the unmixed comparison: run it through both the
    shallow and the deep variant, and only the sensor depth differs.

    Includes the sites from `deepest_swc_per_site()` plus sites whose `SWC_VAR`
    in the main analysis is already not layer 1.

    Returns:
        set: Site names.
    """
    from src.paths import data_path

    sites = set(deepest_swc_per_site())

    info = pd.read_csv(
        data_path("data/outputs/10_datasets/15_datasets_info_parquet_vars_stats_usedsites.csv"))
    already_deep = info.loc[~info['SWC_VAR'].astype(str).str.endswith('_1'), 'SITE']
    sites.update(str(site) for site in already_deep)
    return sites


def get_variable_names(siteconfig: pd.Series):
    """
    Get the variable names for a site from its configuration row.

    Returns:
        dict: Role (e.g. `nee_var`) to column name.
    """
    return {
        'nee_var': str(siteconfig['NEE_VAR']),
        'nee_qc_var': str(siteconfig['NEE_QC_VAR']),
        'le_var': str(siteconfig['LE_VAR']),
        'le_qc_var': str(siteconfig['LE_QC_VAR']),
        'gpp_var': str(siteconfig['GPP_VAR']),
        'reco_var': str(siteconfig['RECO_VAR']),
        'swin_var': str(siteconfig['SWIN_VAR']),
        'swin_qc_var': str(siteconfig['SWIN_QC_VAR']),
        'ta_var': str(siteconfig['TA_VAR']),
        'ta_qc_var': str(siteconfig['TA_QC_VAR']),
        'vpd_var': str(siteconfig['VPD_VAR']),
        'vpd_qc_var': str(siteconfig['VPD_QC_VAR']),
        'swc_var': str(siteconfig['SWC_VAR']),
        'swc_qc_var': str(siteconfig['SWC_QC_VAR']),
        'swinpot_var': 'SW_IN_POT',
        'prec_var': 'P_F',
        # 'rh_var': str(siteconfig['RH_VAR']),
    }
