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

aggfunc, CONDITIONAL = 'median', True
n_sites_min = 20
colors_list = ['#9C27B0', '#26C6DA', '#546E7A', '#FB8C00', '#C62828']
custom_cmap = matplotlib.colors.ListedColormap(colors_list)
igbps = ['ENF', 'DBF', 'MF', 'EBF']

beautify = {
    "NEP_ZSCORE": "NEP",
    "ET_ZSCORE": "ET",
    "GPP_ZSCORE": "GPP",
    "RECO_ZSCORE": "RECO",
    "TA_ZSCORE": "TA",
    "TA_ZSCORE_SHAPVALS": "TA",
    "BIN_VPD_ZSCORE": "VPD",
    "VPD_ZSCORE_SHAPVALS": "VPD",
    "SWC_ZSCORE": "SWC",
    "SWC_ZSCORE_SHAPVALS": "SWC",
    "SWIN_ZSCORE": "SWIN",
}
AX_LABELS_FONTSIZE = 10

# Labels & Columns
xlabel = f'{beautify[xvar]} (z-score)'
ylabel = f'{beautify[yvar]} effect (z-score)'

# Column names in dataframe
xcol = (f"{xvar}", aggfunc)
ycol = (f"{yvar}", aggfunc)
ycol_iqr25 = (f"{yvar}", "<lambda_0>")
ycol_iqr75 = (f"{yvar}", "<lambda_1>")
zcol = (f"BIN_{zvar}", aggfunc)
count_vals_col = (f"{yvar}", "count")

# Options
show_txt_effect = True
show_shap_thresholds = True
show_z_colors = True
show_fit = True
# ------------------------------

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

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
filedf, subsetdf, n_sites = files.load_data(
    suffix='Sites', shap_type=shap_type, dir_res=dir_res, flux=FLUX,
    count_vals_col=count_vals_col, n_sites_min=n_sites_min, subsetcols=[xcol, ycol, zcol, ycol_iqr25, ycol_iqr75],
    site_filter=None, x_in_filename=x_in_filename, y_in_filename=y_in_filename)

# Extract the data from the DataFrame
X_data = subsetdf.iloc[:, 0].values
Y_data = subsetdf.iloc[:, 1].values
Z_data = subsetdf.iloc[:, 2].values

# Bin z data into 5 categories
bin_labels = ['Lowest', 'Low', 'Medium', 'High', 'Highest']
binned_z = pd.cut(Z_data, bins=5, labels=bin_labels)

# Fit polynomial
poly_func, poly_coeffs, x_fit, y_fit, r_squared, pi_upper, pi_lower = fit_polynomial(X_data=X_data, Y_data=Y_data)
print(f"Polynomial fit ALL SITES: "
      f"y={poly_coeffs[0]:.3f}x4+{poly_coeffs[1]:.3f}x3+{poly_coeffs[2]:.3f}x2"
      f"+{poly_coeffs[3]:.3f}x+{poly_coeffs[4]:.3f}; r2={r_squared:.3f}\n")

# Iterate through each temperature bin and plot the corresponding data points
scatterhandles = []
for i, label in reversed(list(enumerate(bin_labels))):
    indices = np.where(binned_z.codes == i)[0]
    # Use color and edgecolors with the same color, but adjust alpha

    if show_z_colors:
        fill_color = colors_list[i]
        edge_color = fill_color
    else:
        fill_color = '#546E7A'
        edge_color = '#546E7A'

    scatterplot = ax_all.scatter(X_data[indices], Y_data[indices],
                                 label=f'{label} {beautify[zvar]}',
                                 alpha=0.4,
                                 s=30,
                                 color=fill_color,
                                 edgecolors=edge_color,
                                 zorder=98)
    scatterhandles.append(scatterplot)

if show_fit:
    fillbetweenplot = plot.add_fit(ax=ax_all, x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper,
                                   poly_func=poly_func, r_squared=r_squared, show_annotate=True,
                                   fontsize=AX_LABELS_FONTSIZE)
else:
    fillbetweenplot = None

# IQR
iqr_low25 = subsetdf.iloc[:, 3].values
iqr_high75 = subsetdf.iloc[:, 4].values
plotparams = dict(marker='o', s=5, zorder=1, alpha=.2, edgecolors='none', color='#6c757d')
iqrplot = ax_all.scatter(X_data, iqr_low25, label="IQR", **plotparams)
ax_all.scatter(X_data, iqr_high75, **plotparams)

# Legend 2 for IQR and fill_between plot
if show_fit:
    handles = [fillbetweenplot, iqrplot]
else:
    handles = [iqrplot]

# Legends for main figure
legend1 = ax_all.legend(handles=scatterhandles, loc='upper left', bbox_to_anchor=(0.38, 0.98),
                        frameon=False, ncol=1, fontsize=AX_LABELS_FONTSIZE, labelspacing=.3,
                        title="Aggregated site data",
                        title_fontsize=AX_LABELS_FONTSIZE)
legend2 = ax_all.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.68, 0.98),
                        frameon=False, ncol=1, labelspacing=.3, fontsize=AX_LABELS_FONTSIZE)
# Add Legend 1 back to the figure
# This is the crucial step to prevent the first legend from being removed
ax_all.add_artist(legend1)

min_ix = np.argmin(y_fit)
max_ix = np.argmax(y_fit)

plot.style_ax(ax=ax_all, title=f"(a) All sites (n={n_sites[1]}, min. {n_sites[0]})",
              ax_labels_fontsize=AX_LABELS_FONTSIZE)

# Find value closest to "SHAP zero"
idx = (np.abs(y_fit - 0)).argmin()

# Detect min/max value shown in plot, is also used for subplots
ydim_max = iqr_high75.max() * 1.2
ydim_min = iqr_low25.min()
ax_all.set_ylim(ydim_min, ydim_max)

# Add text and connecting dashed lines for SHAP max, zero, and min
if show_shap_thresholds:
    plot.show_shap_thresholds(ax=ax_all, x_fit=x_fit, y_fit=y_fit, max_ix=max_ix, min_ix=min_ix,
                              idx=idx, ydim_max=ydim_max, ydim_min=ydim_min, show_annotate=True,
                              fontsize=AX_LABELS_FONTSIZE, show_annotate_short=False)

# Add arrow to highlight one of the IQR data points
select_x = 1.7
locations1 = (X_data == select_x)  # Create a boolean mask for locations where X_data is 1.7
filtered_iqr = iqr_low25[locations1]  # Filter iqr_low25 using the boolean mask
min_iqr_value = np.min(filtered_iqr)  # Find the minimum value in the filtered array

# Combine both conditions: X_data is *select_x* AND median_minus_sd is the minimum value
final_location_mask = (X_data == select_x) & (iqr_low25 == min_iqr_value)
iqr_point_index = np.where(final_location_mask)[0][0]  # Get the index of the element that meets both criteria
iqr_x = float(X_data[iqr_point_index])
iqr_y = float(iqr_low25[iqr_point_index])

ax_all.annotate(
    f'IQR for site data',
    xy=(iqr_x, iqr_y),
    xytext=(iqr_x + 0, iqr_y - 0.2),  # Adjust text position as needed
    arrowprops=dict(arrowstyle="->", color='#6c757d', lw=1.5),
    fontsize=AX_LABELS_FONTSIZE, color='#6c757d', ha='right', va='center'
)

if show_txt_effect:
    # Add text for negative effect
    ax_all.text(x=2.1, y=-0.03, s='reduced uptake/increased release\n' + r'$\downarrow$Negative effect',
                fontsize=AX_LABELS_FONTSIZE, color='black', ha='left', va='top')

    # Add text for positive effect
    ax_all.text(x=2.1, y=0.03, s=r'$\uparrow$' + 'Positive effect\nincreased uptake/reduced release',
                fontsize=AX_LABELS_FONTSIZE, color='black', ha='left', va='bottom')

ax_all.set_xlabel(xlabel, fontsize=AX_LABELS_FONTSIZE)
ax_all.set_ylabel(ylabel, fontsize=AX_LABELS_FONTSIZE)

# ax_all.legend(bbox_to_anchor=(0.05, 0.95), loc='upper right', frameon=False)
plot.format(ax=ax_all, fontsize=AX_LABELS_FONTSIZE, showyticklabels=True, showxticklabels=True,
            xtickdigits=0, ytickdigits=1)
ax_all.axhline(0, color='k', linestyle='--', linewidth=1)

# ---------------
# SUBPLOTS (IGBP)
configs = zip(
    axes_sub, igbps,
    [" ", " ", xlabel, xlabel],
    [ylabel, " ", ylabel, " "],
    ['b', 'c', 'd', 'e'],
    [True, False, True, False],  # For showing yticklabels
    [False, False, True, True]  # For showing xticklabels
)
for ax, igbp, xl, yl, letter, showyticklabels, showxticklabels in configs:
    # Filter using index from main dataset (keeplocs logic)
    df_igbp, df_subset, n_sites_sub = files.load_data(
        suffix=f"IGBP-{igbp}", shap_type=shap_type, dir_res=dir_res, flux=FLUX,
        count_vals_col=count_vals_col, n_sites_min=n_sites_min, subsetcols=[xcol, ycol, zcol, ycol_iqr25, ycol_iqr75],
        site_filter=filedf.index, x_in_filename=x_in_filename, y_in_filename=y_in_filename)
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
    print(f"Polynomial coefficients {igbp}: "
          f"a={poly_coeffs[0]:.3f}, b={poly_coeffs[1]:.3f}, c={poly_coeffs[2]:.3f}, "
          f"d={poly_coeffs[3]:.3f}, e={poly_coeffs[4]:.3f}; r2={r_squared:.3f}\n")

    if show_fit:
        fillbetweenplot = plot.add_fit(ax=ax, x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper,
                                       poly_func=poly_func, r_squared=r_squared, show_annotate=False,
                                       fontsize=AX_LABELS_FONTSIZE)
    else:
        fillbetweenplot = None

    for i, label in reversed(list(enumerate(bin_labels))):
        indices = np.where(binned_z.codes == i)[0]
        if show_z_colors:
            fill_color = colors_list[i]
            edge_color = fill_color
        else:
            fill_color = '#546E7A'
            edge_color = '#546E7A'
        scatterplot = ax.scatter(
            X_data[indices], Y_data[indices], label=f'{label} {beautify[zvar]}', alpha=0.2,
            s=30, color=fill_color, edgecolors=edge_color, zorder=98)
    plot.style_ax(ax, f"({letter}) {igbp} (n={n_sites_sub[1]}, min. {n_sites_sub[0]})",
                  ax_labels_fontsize=AX_LABELS_FONTSIZE)

    # Add text and connecting dashed lines for SHAP max, zero, and min
    if show_shap_thresholds:
        min_ix = np.argmin(y_fit)
        max_ix = np.argmax(y_fit)
        plot.show_shap_thresholds(
            ax=ax, x_fit=x_fit, y_fit=y_fit, max_ix=max_ix, min_ix=min_ix, idx=idx, ydim_max=ydim_max,
            ydim_min=ydim_min, show_annotate=True, fontsize=AX_LABELS_FONTSIZE, show_annotate_short=True)

    # Format subplot
    plot.format(ax=ax, fontsize=AX_LABELS_FONTSIZE, showyticklabels=showyticklabels, showxticklabels=showxticklabels,
                xtickdigits=0, ytickdigits=1)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)

fig.tight_layout()
gs.update(wspace=.2)
fig.show()

# Save fig to file
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'53_FIG-3_ResponseCurve_ShapMedians_{FLUX}_{xvar}+{yvar}+{zvar}.png'
fig.savefig(outfilepath, dpi=300, bbox_inches='tight')
