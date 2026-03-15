"""
VPD response curve

Colors: https://www.pinterest.com/pin/914862421161100/
Colors: https://www.pinterest.com/pin/11118330331981167/
"""
from pathlib import Path

import matplotlib.colors
import numpy as np
import pandas as pd

import src.files as files
import src.plot as plot
from src.fit import fit_polynomial

# Settings & variables

# Main Fig. 3
# plotvars = [FLUX, xvar, yvar, zvar, x_in_filename, y_in_filename]
# plotvars = ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'TA_ZSCORE_SHAPVALS', 'SWC_ZSCORE', 'BIN-TA_ZSCORE', 'BIN-SWC_ZSCORE']
# plotvars = ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'SWC_ZSCORE_SHAPVALS', 'TA_ZSCORE', 'BIN-TA_ZSCORE', 'BIN-SWC_ZSCORE']
plotvars = ['NEP_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'TA_ZSCORE', 'BIN-TA_ZSCORE', 'BIN-VPD_ZSCORE']

FLUX = plotvars[0]
xvar, yvar, zvar = plotvars[1], plotvars[2], plotvars[3]
x_in_filename, y_in_filename = plotvars[4], plotvars[5]

# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'

# Agg groups, use z-scores:
# xvar, yvar, zvar = 'TA_ZSCORE', 'TA_ZSCORE_SHAPVALS', 'TA_ZSCORE'
# x_in_filename = 'SWC_ZSCORE'
# y_in_filename = 'VPD_ZSCORE'
# xvar, yvar, zvar = 'SWC_ZSCORE', 'SWC_ZSCORE_SHAPVALS', 'TA_ZSCORE'

# xvar, yvar, zvar = 'TA_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'TA_ZSCORE'
# xvar, yvar, zvar = 'SWC_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'SWC_ZSCORE'
# xvar, yvar, zvar = 'SWIN_ZSCORE', 'TA_ZSCORE', 'VPD_ZSCORE'
# xvar, yvar, zvar = 'SWC_ZSCORE', 'VPD_ZSCORE', 'VPD_ZSCORE'
# xvar, yvar, zvar = 'SWIN_ZSCORE', 'VPD_ZSCORE', 'VPD_ZSCORE'
# xvar, yvar, zvar = 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE'
# ------------------------------

aggfunc, CONDITIONAL = 'mean', True
# aggfunc, CONDITIONAL = 'median', True


colors_list = ['#9C27B0', '#0984E3', '#00B894', '#636E72', '#FDCB6E', '#FF8C00', '#B71C1C']
# colors_list = ['#9C27B0', '#0984E3', '#00B894', '#636E72', '#FDCB6E', '#E17055', '#D63031']
# colors_list = ['grey', '#9C27B0', '#0984E3', '#00B894', '#FDCB6E', '#E17055', '#D63031']
# colors_list = ['#9C27B0', '#6C5CE7', '#0984E3', '#00B894', '#FDCB6E', '#E17055', '#D63031']
# colors_list = ['#2D3436', '#6C5CE7', '#0984E3', '#00B894', '#FDCB6E', '#E17055', '#D63031']
# colors_list = ['black', '#9C27B0', '#26C6DA', '#546E7A', '#FB8C00', '#C62828', 'red']
custom_cmap = matplotlib.colors.ListedColormap(colors_list)
igbps = ['ENF', 'DBF', 'MF', 'EBF']

beautify = {
    "NEP_ZSCORE": "NEP",
    "ET_ZSCORE": "ET",
    "GPP_ZSCORE": "GPP",
    "RECO_ZSCORE": "RECO",
    "TA_ZSCORE": "TA",
    "BIN_TA_ZSCORE": "TA",
    "TA_ZSCORE_SHAPVALS": "TA",
    "BIN_VPD_ZSCORE": "Vapor pressure deficit",
    "VPD_ZSCORE": "Vapor pressure deficit",
    "VPD_ZSCORE_SHAPVALS": "Vapor pressure deficit",
    "SWC_ZSCORE": "SWC",
    "BIN_SWC_ZSCORE": "SWC",
    "SWC_ZSCORE_SHAPVALS": "SWC",
    "SWIN_ZSCORE": "SWIN",
}
AX_LABELS_FONTSIZE = 12

# Labels & Columns
xlabel = rf'{beautify[xvar]} ($\sigma$)'
ylabel = rf'{beautify[yvar]} effect on {beautify[FLUX]} ($\sigma$)'

# Bins always use 'median', b/c using 'mean' results in floating point errors
xagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
yagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
xcol, ycol, zcol = (f"{xvar}", xagg), (f"{yvar}", yagg), (f"{zvar}", aggfunc)
count_vals_col = (f"{zvar}", "count")
ycol_sem = (f"{yvar}", "sem")
ycol_iqr25 = (f"{yvar}", "q25")
ycol_iqr75 = (f"{yvar}", "q75")

# Options
show_txt_effect = True
show_shap_thresholds = True
show_z_colors = True
show_fit = True
color_fitline = '#263238'
color_points = '#607D8B'
colors_symbols = ['black', 'black', 'black']
# colors_symbols = ['#F9A825', '#2E7D32', '#8E24AA']
# ------------------------------

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type

# # Load SHAP values aggregated across all sites
# filepath = Path(
#     results_outdir) / f"42_SHAPVALUES-{shap_type}_AggregatedAcrossSites_{filenamex}+{filenamey}+{FLUX}.parquet"
# shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
# keeplocs = shapvals_df[y_counts] >= n_sites_min
# shapvals_df = shapvals_df[keeplocs].copy()


# Paths & Settings
shap_type = 'conditional' if CONDITIONAL else 'standard'
settings = files.read_settings_file("../../config/settings.yaml")
dir_res = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

# FIGURE LAYOUT (5 panels)
fig, gs, ax_all, axes_sub = plot.layout_5panels((13.86, 6.67), add_colorbar_ax=False)

# ---------------------
# MAIN PLOT (all sites)

# Load data
filedf, subsetdf, minmax_counts, n_sites = files.load_data(
    suffix='Sites', shap_type=shap_type, dir_res=dir_res, flux=FLUX,
    count_vals_col=count_vals_col,
    subsetcols=[xcol, ycol, zcol, ycol_sem],
    site_filter=None, x_in_filename=x_in_filename, y_in_filename=y_in_filename, aggfunc=aggfunc)

# Save plot data to csv
_outfilepath = dir_out / f'53_FIG-3_ResponseCurve_ShapMeans_{FLUX}_{xvar}+{yvar}+{zvar}_ALLSITES_DATA.csv'
subsetdf.to_csv(_outfilepath, index=False)

# Extract the data from the DataFrame
X_data = subsetdf.iloc[:, 0].values
Y_data = subsetdf.iloc[:, 1].values
Z_data = subsetdf.iloc[:, 2].values
# todo check SEM
_sem = subsetdf.iloc[:, 3].values
_sem_upper = Y_data + _sem
_sem_lower = Y_data - _sem

# Bin z data into 5 categories
bin_labels = ['coldest', 'cold', 'cool', 'medium',
              'warm', 'hot', 'hottest']
# binned_z = pd.cut(Z_data, bins=5, labels=bin_labels)
# retbins=True returns (categories, bins)
binned_z, bin_edges = pd.cut(Z_data, bins=7, labels=bin_labels, retbins=True)

# Fit polynomial
poly_func, poly_coeffs, x_fit, y_fit, r_squared, pi_upper, pi_lower = fit_polynomial(X_data=X_data, Y_data=Y_data)
print(f"Polynomial fit ALL SITES: "
      f"y={poly_coeffs[0]:.3f}x4+{poly_coeffs[1]:.3f}x3+{poly_coeffs[2]:.3f}x2"
      f"+{poly_coeffs[3]:.3f}x+{poly_coeffs[3]:.3f}; r2={r_squared:.3f}\n")

# Iterate through each temperature bin and plot the corresponding data points
scatterhandles = []
for i, label in list(enumerate(bin_labels)):
    indices = np.where(binned_z.codes == i)[0]
    # Use color and edgecolors with the same color, but adjust alpha

    if show_z_colors:
        fill_color = colors_list[i]
        # edge_color = 'none'
        edge_color = fill_color
    else:
        fill_color = color_points
        edge_color = color_points

    scatterplot = ax_all.scatter(X_data[indices], Y_data[indices],
                                 label=f'{label}',
                                 # label=f'{label} {beautify[zvar]}',
                                 alpha=.5,
                                 s=30,
                                 color=fill_color,
                                 edgecolors=edge_color,
                                 zorder=98)
    scatterhandles.append(scatterplot)

    # plotparams = dict(marker='o', s=5, zorder=1, alpha=1, edgecolors='none', color=fill_color)
    # semplot = ax_all.scatter(X_data[indices], _sem_lower[indices], label="Standard error", **plotparams)
    # ax_all.scatter(X_data[indices], _sem_upper[indices], **plotparams)

    # 3. Plot the error bars
    semplot = ax_all.errorbar(X_data[indices], Y_data[indices],
                              yerr=[_sem[indices], _sem[indices]],
                              fmt='none',  # 'none' ensures it only plots the bars, no markers/lines
                              ecolor=fill_color,  # Color of the error bars
                              elinewidth=3,  # Thickness of the error bar line
                              capsize=0,  # Length of the horizontal caps at the ends
                              alpha=0.3,  # Match your scatter alpha, or set to 1
                              zorder=1)  # Keeps it behind the scatter points

if show_fit:
    fillbetweenplot = plot.add_fit(
        ax=ax_all, x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper,
        poly_func=poly_func, r_squared=r_squared, show_annotate=True,
        fontsize=AX_LABELS_FONTSIZE, color=color_fitline, linewidth=3)
else:
    fillbetweenplot = None

# # SEM
# _sem = subsetdf.iloc[:, 3].values
# _sem_upper = Y_data + _sem
# _sem_lower = Y_data - _sem
# plotparams = dict(marker='o', s=5, zorder=1, alpha=1, edgecolors='none', color=color_points)
# semplot = ax_all.scatter(X_data, _sem_lower, label="Standard error", **plotparams)
# ax_all.scatter(X_data, _sem_upper, **plotparams)

# Legend 2 for IQR and fill_between plot
if show_fit:
    handles = [fillbetweenplot, semplot]
else:
    handles = [semplot]

# Legends for main figure
legend1 = ax_all.legend(handles=scatterhandles,
                        bbox_to_anchor=(0.44, 0.86),
                        frameon=False, ncol=2, fontsize=AX_LABELS_FONTSIZE, labelspacing=.3,
                        title="Aggregated site data",
                        title_fontsize=AX_LABELS_FONTSIZE)
# legend2 = ax_all.legend(handles=handles,
#                         bbox_to_anchor=(0.36, 0.3),
#                         frameon=False, ncol=1, labelspacing=.3, fontsize=AX_LABELS_FONTSIZE)
# Add Legend 1 back to the figure
# This is the crucial step to prevent the first legend from being removed
ax_all.add_artist(legend1)

min_ix = np.argmin(y_fit)
max_ix = np.argmax(y_fit)

# Panel letter
letter = 'a'
igbp = 'Global forests'
ax_all.text(0, 1.05, f"{letter} | {igbp}", transform=ax_all.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')
ax_all.text(0.4, 1.05, f"(n={n_sites}, min. {minmax_counts[0]})", transform=ax_all.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='normal', ha='left', va='top')

# Find value closest to "SHAP zero"
idx = (np.abs(y_fit - 0)).argmin()

# Threshold for 95% PI lines
# x_fit[idx]
threshold_lower = x_fit[(np.abs(pi_lower - 0)).argmin()]
threshold_upper = x_fit[(np.abs(pi_upper - 0)).argmin()]


# ---
# ---------------------------------------------------------
# CALCULATE THRESHOLD AND 95% CI FROM POLYNOMIAL BANDS
# ---------------------------------------------------------

# Ensure y values are sorted ascending for numpy interpolation
# Since SHAP values go from positive to negative as VPD increases,
# we need to reverse the arrays for np.interp to work correctly.
x_fit_rev = x_fit[::-1]
y_fit_rev = y_fit[::-1]
pi_lower_rev = pi_lower[::-1]
pi_upper_rev = pi_upper[::-1]

# Find where the main fit crosses 0
threshold_main = np.interp(0, y_fit_rev, x_fit_rev)

# Find where the confidence bands cross 0
threshold_upper_bound = np.interp(0, pi_lower_rev, x_fit_rev) # Lower PI yields the UPPER VPD threshold
threshold_lower_bound = np.interp(0, pi_upper_rev, x_fit_rev) # Upper PI yields the LOWER VPD threshold

# (Optional) Print the results for your manuscript text
print(f"VPD Threshold: {threshold_main:.2f} sigma (95% CI: [{threshold_lower_bound:.2f}, {threshold_upper_bound:.2f}])")

# You can now use these exact variables to plot your dashed vertical lines!
# ---

# Detect min/max value shown in plot, is also used for subplots
ydim_max = _sem_upper.max() * 1.6
ydim_min = _sem_lower.min() * 1.15
ax_all.set_ylim(ydim_min, ydim_max)
xdim_min = X_data.min()
xdim_max = X_data.max()
ax_all.set_xlim(xdim_min, xdim_max * 1.06)
# ax_all.set_xlim(-1, 1)

# Add text and connecting dashed lines for SHAP max, zero, and min
if show_shap_thresholds:
    plot.show_shap_thresholds(ax=ax_all, x_fit=x_fit, y_fit=y_fit, max_ix=max_ix, min_ix=min_ix,
                              idx=idx, ydim_max=ydim_max, ydim_min=ydim_min, show_annotate=True,
                              fontsize=AX_LABELS_FONTSIZE, show_annotate_short=False,
                              colors_symbols=colors_symbols)

# # todo Add arrow to highlight one of the IQR/SEM data points
# select_x = 1.7
# locations1 = (X_data == select_x)  # Create a boolean mask for locations where X_data is 1.7
# filtered_iqr = iqr_low25[locations1]  # Filter iqr_low25 using the boolean mask
# min_iqr_value = np.min(filtered_iqr)  # Find the minimum value in the filtered array
#
# # Combine both conditions: X_data is *select_x* AND median_minus_sd is the minimum value
# final_location_mask = (X_data == select_x) & (iqr_low25 == min_iqr_value)
# iqr_point_index = np.where(final_location_mask)[0][0]  # Get the index of the element that meets both criteria
# iqr_x = float(X_data[iqr_point_index])
# iqr_y = float(iqr_low25[iqr_point_index])
#
# ax_all.annotate(
#     f'IQR for site data',
#     xy=(iqr_x, iqr_y),
#     xytext=(iqr_x + 0, iqr_y - 0.2),  # Adjust text position as needed
#     arrowprops=dict(arrowstyle="->", color='#6c757d', lw=1.5),
#     fontsize=AX_LABELS_FONTSIZE, color='#6c757d', ha='right', va='center'
# )


ax_all.set_xlabel(xlabel, fontsize=AX_LABELS_FONTSIZE)
ax_all.set_ylabel(ylabel, fontsize=AX_LABELS_FONTSIZE)

# ax_all.legend(bbox_to_anchor=(0.05, 0.95), loc='upper right', frameon=False)
plot.format(ax=ax_all, fontsize=AX_LABELS_FONTSIZE, showyticklabels=True, showxticklabels=True,
            xtickdigits=0, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)
ax_all.axhline(0, color='k', linestyle='--', linewidth=1)

# ---------------
# SUBPLOTS (IGBP)
configs = zip(
    axes_sub, igbps,
    [" ", " ", xlabel, xlabel],
    [" ", " ", " ", " "],  # No y label
    ['b', 'c', 'd', 'e'],
    [True, False, True, False],  # For showing yticklabels
    [False, False, True, True]  # For showing xticklabels
)
for ax, igbp, xl, yl, letter, showyticklabels, showxticklabels in configs:
    # Filter using index from main dataset (keeplocs logic)
    df_igbp, df_subset, minmax_counts_sub, n_sites_sub = files.load_data(
        suffix=f"IGBP-{igbp}", shap_type=shap_type, dir_res=dir_res, flux=FLUX,
        count_vals_col=count_vals_col,
        subsetcols=[xcol, ycol, zcol, ycol_sem],
        site_filter=filedf.index, x_in_filename=x_in_filename, y_in_filename=y_in_filename, aggfunc=aggfunc)
    # Extract the data from the DataFrame
    X_data = df_subset.iloc[:, 0].values
    Y_data = df_subset.iloc[:, 1].values
    Z_data = df_subset.iloc[:, 2].values

    ax.set_xlabel(xl, fontsize=AX_LABELS_FONTSIZE)
    ax.set_ylabel(yl, fontsize=AX_LABELS_FONTSIZE)

    # Fit polynomial (only works on non-NaN data)
    df_subset_nonan = df_subset.copy()
    df_subset_nonan = df_subset_nonan.dropna()
    X_data_nonan = df_subset_nonan.iloc[:, 0].values
    Y_data_nonan = df_subset_nonan.iloc[:, 1].values
    Z_data_nonan = df_subset_nonan.iloc[:, 2].values
    poly_func, poly_coeffs, x_fit, y_fit, r_squared, pi_upper, pi_lower = fit_polynomial(X_data=X_data_nonan,
                                                                                         Y_data=Y_data_nonan)
    # Find value closest to "SHAP zero"
    idx = (np.abs(y_fit - 0)).argmin()

    print(f"{igbp} Polynomial coefficients: "
          f"a={poly_coeffs[0]:.3f}, b={poly_coeffs[1]:.3f}, c={poly_coeffs[2]:.3f}, "
          f"d={poly_coeffs[3]:.3f}, e={poly_coeffs[4]:.3f}; r2={r_squared:.3f}\n")

    if show_fit:
        fillbetweenplot = plot.add_fit(ax=ax, x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper,
                                       poly_func=poly_func, r_squared=r_squared, show_annotate=False,
                                       fontsize=AX_LABELS_FONTSIZE, color=color_fitline,
                                       linewidth=2)
    else:
        fillbetweenplot = None

    # Recalculate bins for this specific subplot using GLOBAL edges
    # This ensures "High" in ENF is the same value range as "High" in All Sites
    binned_z_sub = pd.cut(Z_data, bins=bin_edges, labels=bin_labels)

    for i, label in list(enumerate(bin_labels)):
        # Get indices from the LOCAL binned object
        indices = np.where(binned_z_sub.codes == i)[0]
        # indices = np.where(binned_z.codes == i)[0]

        # Skip if this bin is empty for this IGBP
        if len(indices) == 0:
            continue

        if show_z_colors:
            fill_color = colors_list[i]
            edge_color = fill_color
        else:
            fill_color = '#546E7A'
            edge_color = '#546E7A'

        scatterplot = ax.scatter(
            X_data[indices], Y_data[indices],
            label=f'{label} {beautify[zvar]}',
            alpha=0.25,
            s=15,
            color=fill_color,
            edgecolors=edge_color,
            zorder=98)

    # Panel letters
    ax.text(0, 1.1, f"{letter} | {igbp}", transform=ax.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')
    ax.text(0.3, 1.1, f"(n={n_sites_sub}, min. {minmax_counts_sub[0]})", transform=ax.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='normal', ha='left', va='top')

    # Add text and connecting dashed lines for SHAP max, zero, and min
    if show_shap_thresholds:
        min_ix = np.argmin(y_fit)
        max_ix = np.argmax(y_fit)
        plot.show_shap_thresholds(
            ax=ax, x_fit=x_fit, y_fit=y_fit, max_ix=max_ix, min_ix=min_ix, idx=idx, ydim_max=ydim_max,
            ydim_min=ydim_min, show_annotate=True, fontsize=AX_LABELS_FONTSIZE, show_annotate_short=True,
            colors_symbols=colors_symbols)

    # Format subplot
    plot.format(ax=ax, fontsize=AX_LABELS_FONTSIZE, showyticklabels=showyticklabels, showxticklabels=showxticklabels,
                xtickdigits=0, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)

    # Save plot data to csv
    _outfilepath = dir_out / f'53_FIG-3_ResponseCurve_ShapMeans_{FLUX}_{xvar}+{yvar}+{zvar}_{igbp}_DATA.csv'
    df_subset_nonan.to_csv(_outfilepath, index=False)

fig.tight_layout()
gs.update(wspace=.1)
fig.show()

# Save fig to file
outfilepath = dir_out / f'53_FIG-3_ResponseCurve_ShapMeans_{FLUX}_{xvar}+{yvar}+{zvar}.png'
fig.savefig(outfilepath, dpi=300, bbox_inches='tight')
