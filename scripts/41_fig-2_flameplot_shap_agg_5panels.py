"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from diive.core.plotting.styles import LightTheme as theme
# from scipy import ndimage
from scipy.spatial import cKDTree

import src.files as files
import src.plot as plot

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, ET, GPP, RECO, TA, VPD, SWIN, SWC
# FLUX = 'NEP'
# FLUX = 'ET'
FLUX = 'GPP'
# FLUX = 'RECO'
xvar = 'TA'
yvar = 'VPD'
zvar = 'VPD'
aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Heatmap settings
xlabel = f'{xvar} (z-score)'
ylabel = f'{yvar} (z-score)'
zlabel = f'{aggfunc} SHAP value of {zvar} (z-score)'
n_sites_min = 30
n_sites_used = 171
cmap = 'RdYlBu'
cb_digits_after_comma = 1
# cmap = 'RdYlBu_r'
# ------------------------------

binx = (f"BIN_{xvar}", aggfunc)
biny = (f"BIN_{yvar}", aggfunc)
z = (f"{zvar}_SHAPVALS", aggfunc)
z_counts = (f"{zvar}_SHAPVALS", "count")

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Start figure
fig = plt.figure(figsize=(21, 9), dpi=150, facecolor="white")
gs = gridspec.GridSpec(2, 4)  # rows, cols
# gs.update(wspace=.2, hspace=.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
ax_all = fig.add_subplot(gs[0:2, 0:2])
ax2 = fig.add_subplot(gs[0, 2], sharex=ax_all, sharey=ax_all)
ax3 = fig.add_subplot(gs[0, 3], sharex=ax_all, sharey=ax_all)
ax4 = fig.add_subplot(gs[1, 2], sharex=ax_all, sharey=ax_all)
ax5 = fig.add_subplot(gs[1, 3], sharex=ax_all, sharey=ax_all)

# Load SHAP values aggregated across all sites
filepath = Path(results_outdir) / f"3_AllSites_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
keeplocs = shapvals_df[z_counts] >= n_sites_min
shapvals_df = shapvals_df[keeplocs].copy()
n_sites_all_min = shapvals_df[z_counts].min()
n_sites_all_max = shapvals_df[z_counts].max()
subset_all = shapvals_df[[binx, biny, z]].copy()
vmin = subset_all[z].min()
vmax = subset_all[z].max()
subset_all.columns = ['_'.join(col).strip() for col in subset_all.columns.values]  # Heatmap needs flat column index
p = plot.flameplot(df=subset_all, fig=fig, ax=ax_all, cmap=cmap,
                   title=None, cb_digits_after_comma=cb_digits_after_comma,
                   xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, cb_extend='both')
ax_all.set_aspect('equal')

ax_all.text(0.1, 0.95, f"(a) All sites (n={n_sites_used}, min. {n_sites_all_min})",
            transform=ax_all.transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
            ha='left', va='bottom', zorder=99, backgroundcolor='white')
ax_all.axhline(0, color='black', linestyle='--', linewidth=1, zorder=100)
ax_all.axvline(0, color='black', linestyle='--', linewidth=1, zorder=100)
# ax_all.text(x=2.1, y=-0.03, s='Negative impact\nreduced uptake/increased release',
#         fontsize=9, color='black', ha='left', va='top', zorder=100)

# Info texts
params = dict(size=theme.AX_LABELS_FONTSIZE, color='k', zorder=100)
ax_all.text(2, 0.1, r"$\uparrow$ dry", horizontalalignment='left', verticalalignment='bottom', **params)
ax_all.text(2, -0.1, r"$\downarrow$ humid", horizontalalignment='left', verticalalignment='top', **params)
ax_all.text(-0.1, 3.5, r"$\leftarrow$ cool", horizontalalignment='right', verticalalignment='center', **params)
ax_all.text(0.1, 3.5, r"warm $\rightarrow$", horizontalalignment='left', verticalalignment='center', **params)


# Highlight points of interest
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

    # Calculate the aggregation for each set of 9 neighbors, ignoring NaNs
    # The result is a 1D array of aggregations.
    if agg == 'mean':
        aggs = np.nanmean(neighbor_values, axis=1)
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


# Find optimum and pessimum
pivot_df = subset_all.pivot(index='BIN_TA_median', columns='BIN_VPD_median', values='VPD_SHAPVALS_median')
max_location, max_value = findpoi(df=pivot_df, k=9, agg='mean', what='max')
min_location, min_value = findpoi(df=pivot_df, k=9, agg='mean', what='min')

# Optimum (smallest SHAP)
x = max_location[0] + 0.05
y = max_location[1] + 0.05
params_max = dict(size=theme.AX_LABELS_FONTSIZE, color='k', zorder=100)
color = "#90A4AE"
ax_all.text(x, y, r'$\oplus$', horizontalalignment='center', verticalalignment='center',
            color="white", size=30, zorder=100, alpha=.7)
# ax_all.scatter(x, y, color='white', marker='+', edgecolors=color, linewidth=3, s=500, zorder=100, alpha=0.9)
ax_all.plot([x - 0.15, -2.5], [y, y], color=color, linestyle='--', linewidth=1, zorder=100)
ax_all.plot([-2.5, -2.5], [y, 1], color=color, linestyle='--', linewidth=1, zorder=100)
params = dict(size=theme.AX_LABELS_FONTSIZE, color='k', zorder=100)
ax_all.text(-3, 1.1, "highest NEP increase", horizontalalignment='left', verticalalignment='bottom', **params)
print(f"Minimum found at {min_location[0], min_location[1]}")

# Pessimum (largest SHAP)
x = min_location[0] + 0.05
y = min_location[1] + 0.05
color = "#90A4AE"
ax_all.text(x, y, r'$\ominus$', horizontalalignment='center',
            verticalalignment='center', color="black", size=30, zorder=100, alpha=.7)
# ax_all.scatter(x, y, color='none', marker='v', edgecolors=color, linewidth=2, s=500, zorder=100)
# , alpha=1, s=120, c="white", zorder=99, edgecolors='#e63946'
# ax_all.scatter(x, y, color='none', edgecolors=color, linewidth=2, s=300, zorder=100)
ax_all.plot([x - 0.15, -2.5], [y, y], color=color, linestyle='--', linewidth=1, zorder=100)
ax_all.plot([-2.5, -2.5], [y, 2], color=color, linestyle='--', linewidth=1, zorder=100)
params = dict(size=theme.AX_LABELS_FONTSIZE, color='k', zorder=100)
ax_all.text(-3, 1.9, "highest NEP decrease", horizontalalignment='left', verticalalignment='top', **params)
print(f"Maximum found at {max_location[0], max_location[1]}")

# Load SHAP values aggregated per IGBP
igbps = ['ENF', 'DBF', 'MF', 'EBF']
igbps_n_sites = [87, 56, 14, 14]  # Counted in #33
axes = [ax2, ax3, ax4, ax5]
xlabels = [" ", " ", xlabel, xlabel]
ylabels = [ylabel, " ", ylabel, " "]
letter = ['b', 'c', 'd', 'e']
data_per_igbp = {}
for ix, i in enumerate(igbps):
    filepath = Path(
        results_outdir) / f"4_All-{i}_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}.parquet"
    igbp_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

    # Next line uses the same keeplocs like defined above, i.e. for each site we
    # get the same locations as for the overall (all sites) plot.
    igbp_df = igbp_df[keeplocs].copy()

    # Now we only want to keep those locations where at least 1 value
    # is available. This way the correct min. value is shown in the plot.
    # It is important to note that there are more locations (bins) available
    # for each IGBP that are not shown in the IGBP plots, only the same
    # locations as in the overall plot.
    availablelocs = igbp_df[z_counts] >= 1
    igbp_df = igbp_df[availablelocs].copy()

    n_sites_igbp_min = igbp_df[z_counts].min()
    n_sites_igbp_max = igbp_df[z_counts].max()
    data_per_igbp[i] = igbp_df[[binx, biny, z]].copy()
    data_per_igbp[i].columns = ['_'.join(col).strip() for col in data_per_igbp[i].columns.values]  # Flat
    xlabel = xlabel
    ylabel = ylabel
    plot.flameplot(df=data_per_igbp[i], fig=fig, ax=axes[ix], cmap=cmap,
                   title=None, show_colormap=False,
                   vmin=vmin, vmax=vmax, xlabel=xlabels[ix], ylabel=ylabels[ix])
    title = f"({letter[ix]}) {i} (n={igbps_n_sites[ix]}, min. {n_sites_igbp_min})"
    axes[ix].text(0.1, 1, title,
                  transform=axes[ix].transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
                  ha='left', va='top', zorder=100, backgroundcolor='white')
    axes[ix].axhline(0, color='black', linestyle='--', linewidth=1, zorder=99)
    axes[ix].axvline(0, color='black', linestyle='--', linewidth=1, zorder=99)
    axes[ix].set_aspect('equal')

    pivot_df = data_per_igbp[i].pivot(index='BIN_TA_median', columns='BIN_VPD_median', values='VPD_SHAPVALS_median')
    max_location, max_value = findpoi(df=pivot_df, k=9, agg='mean', what='max')
    min_location, min_value = findpoi(df=pivot_df, k=9, agg='mean', what='min')

    # Optimum (smallest SHAP)
    x = max_location[0] + 0.05
    y = max_location[1] + 0.05
    axes[ix].text(x, y, r'$\oplus$', horizontalalignment='center',
                verticalalignment='center', color="white", size=30, zorder=100, alpha=.7)
    # axes[ix].scatter(x, y, color='none', marker='o', edgecolors=color, linewidth=2, s=500, zorder=100)

    # Pessimum (largest SHAP)
    x = min_location[0] + 0.05
    y = min_location[1] + 0.05
    axes[ix].text(x, y, r'$\ominus$', horizontalalignment='center',
                verticalalignment='center', color="black", size=30, zorder=100, alpha=.7)
    # axes[ix].scatter(x, y, color='none', edgecolors=color, linewidth=2, s=500, zorder=100)

fig.tight_layout()
fig.show()

# # Find local minimum/maximum
# heatmap_df = subset_all.pivot(index='BIN_TA_median', columns='BIN_VPD_median', values='VPD_SHAPVALS_median')
#
# min_required_values = 9
#
#
# # 2. CREATE A MORE ADVANCED FILTER FUNCTION
# def create_mean_calculator(min_vals):
#     """This function returns another function that will be used by the filter."""
#     def calculate_mean_if_valid(arr):
#         """
#         Calculates the mean only if the number of valid points
#         in the window meets the threshold.
#         """
#         # Count the number of non-NaN values in the current window (arr)
#         valid_count = np.count_nonzero(~np.isnan(arr))
#
#         # If the count is sufficient, return the mean. Otherwise, return NaN.
#         if valid_count >= min_vals:
#             return np.nanmean(arr)
#         else:
#             return np.nan
#     return calculate_mean_if_valid
#
# # 3. APPLY THE FILTER WITH THE NEW FUNCTION
# # Create the specific calculator function with our threshold
# mean_calculator_func = create_mean_calculator(min_required_values)
#
# # Apply the filter to the DataFrame's values
# # The result is a NumPy array
# neighbor_means_array = ndimage.generic_filter(
#     heatmap_df.values,
#     function=mean_calculator_func,
#     size=(3,3),  # 36 values in total
#     mode='constant',  # How to handle edges
#     cval=np.nan  # Fill value for edges
# )
#
# # Convert the result back to a DataFrame for clarity
# neighbor_means_df = pd.DataFrame(
#     neighbor_means_array,
#     index=heatmap_df.index,
#     columns=heatmap_df.columns
# )
#
# print("\n--- DataFrame of Neighbor Means ---")
# print(neighbor_means_df)
#
# # Find the flattened index of the maximum value in the NumPy array
# max_idx_flat = np.nanargmax(neighbor_means_array)
# min_idx_flat = np.nanargmin(neighbor_means_array)
#
# # Convert the flattened index to (row, column) coordinates
# max_coords = np.unravel_index(max_idx_flat, neighbor_means_array.shape)
# min_coords = np.unravel_index(min_idx_flat, neighbor_means_array.shape)
# max_row_idx, max_col_idx = max_coords
# min_row_idx, min_col_idx = min_coords
#
# # Get the labels (index and column name) from the original DataFrame
# max_mean_loc_index = heatmap_df.index[max_row_idx]
# max_mean_loc_column = heatmap_df.columns[max_col_idx]
# min_mean_loc_index = heatmap_df.index[min_row_idx]
# min_mean_loc_column = heatmap_df.columns[min_col_idx]
#
# # Get the original value and the calculated mean
# original_value_max = heatmap_df.iloc[max_row_idx, max_col_idx]
# original_value_min = heatmap_df.iloc[min_row_idx, min_col_idx]
# max_neighbor_mean = neighbor_means_array[max_row_idx, max_col_idx]
# min_neighbor_mean = neighbor_means_array[min_row_idx, min_col_idx]
#
# ax_all.scatter(max_mean_loc_index, max_mean_loc_column, color='none', edgecolors='#18FFFF', s=300, zorder=100)
# ax_all.scatter(min_mean_loc_index, min_mean_loc_column, color='none', edgecolors='#18FFFF', s=300, zorder=100)
# print(f"Maximum found at {max_mean_loc_index, max_mean_loc_column}")
# print(f"Minimum found at {min_mean_loc_index, min_mean_loc_column}")
