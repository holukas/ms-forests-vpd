import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def peak_season_months(sitedata, gpp_col: str, n_months: int = 4) -> list:
    """Find the calendar months with the highest mean GPP at a site.

    This is the peak-season definition used throughout the analysis. It is applied
    to the full record, before any quality-control or daytime filtering, so the
    season does not move when those filters change.

    An earlier version picked the warmest months by mean air temperature instead.
    Some comments and file names elsewhere still say "warmest" for that reason;
    the selection has been GPP-based since the subsets were built.

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
    Pick the deepest usable soil water layer for every site.

    Reviewers 2 and 3 and the editor all ask whether the results depend on
    soil moisture being measured near the surface. `SWC_F_MDS_1` is the
    shallowest layer and is what the submitted analysis used, so the
    sensitivity run needs the deepest layer instead.

    Deepest alone is not enough. A deeper layer is often gappier, and a gappier
    driver means the subset keeps fewer records, so a difference in the results
    could come from the change in sample rather than from the change in depth.
    The layer therefore has to keep at least `min_peak_ratio` of layer 1's
    records within the four peak months. The deepest layer that clears that bar
    wins, and a site where none of them does stays on layer 1.

    Depths in cm are not recoverable, only 3 sites carry `GRP_SWC` metadata in
    the AmeriFlux BIF, so this is "the deepest available layer", never "the
    root zone at X cm".

    Coverage comes from `13b_variables_per_site.csv`, written by
    `13b_check_available_vars_SWC.py`, so nothing in the 13 to 17 chain needs
    rerunning.

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
    Sites that do not use the shallowest soil water layer in a deep run.

    The deep run keeps every site, because a site with no usable deeper layer
    still contributes on layer 1. Roughly two fifths of the network ends up in
    that group, so aggregated deep results are a mixture and the difference
    against the shallow run is damped by sites that cannot move.

    This is the set for the unmixed comparison: sites whose soil water comes
    from below layer 1. Run the same set through both the shallow and the deep
    variant and the two are directly comparable, same sites, same records, only
    the sensor depth differs.

    Two sources, because two different things can put a site here. Most come
    from `deepest_swc_per_site()`, which is the rule the deep run applies. One
    more already used `SWC_F_MDS_2` in the submitted analysis, since layer 1 was
    missing there, so it never was a shallow site to begin with.

    Returns:
        set: Site names, 128 of the 208 analysed sites.
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
    Get variable names for this site.

    Args:
        siteconfig (dict): A dictionary containing site configuration.

    Returns:
        dict: A dictionary of variable names.
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
