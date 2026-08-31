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
import src.fit as fit
import src.plot as plot
from src.paths import load_settings

# Open the figure in a window after saving. False by default so a script can run
# unattended: matplotlib picks the interactive TkAgg backend here, and plt.show()
# then blocks until the window is closed by hand.
SHOW_PLOT = False

# Settings & variables

# Main Fig. 3
# plotvars = [FLUX, xvar, yvar, zvar, x_in_filename, y_in_filename]
# plotvars = ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'TA_ZSCORE_SHAPVALS', 'SWC_ZSCORE', 'BIN-TA_ZSCORE', 'BIN-SWC_ZSCORE']
# plotvars = ['NEP_ZSCORE', 'BIN_SWC_ZSCORE', 'SWC_ZSCORE_SHAPVALS', 'TA_ZSCORE', 'BIN-TA_ZSCORE', 'BIN-SWC_ZSCORE']
plotvars = ['NEP_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'TA_ZSCORE', 'BIN-TA_ZSCORE', 'BIN-VPD_ZSCORE']

FLUX = plotvars[0]

# Wording for the two zones above and below the zero line. A positive SHAP value means
# more of the flux, which is more carbon uptake for NEP and GPP but more respiration for
# RECO and more water loss for ET. Only the first two can be called stimulation without
# saying something wrong, so the other two get neutral wording.
ZONE_LABELS = {
    'NEP_ZSCORE': ("Stimulation", "Suppression"),
    'GPP_ZSCORE': ("Stimulation", "Suppression"),
    'RECO_ZSCORE': ("Increase", "Decrease"),
    'ET_ZSCORE': ("Increase", "Decrease"),
}
LABEL_POS, LABEL_NEG = ZONE_LABELS[FLUX]

# Run variant. An empty string reads the results behind the submitted figures and
# writes to the baseline plot folder. Any other value reads the matching variant
# folder and writes the figures next to it, so a sensitivity run cannot overwrite a
# published figure. The aggregation must have run with the same value.
VARIANT = ""
# Site subset. An empty string reads the aggregation over every site.
# "deeper-only" reads the run restricted to the 128 sites whose soil water comes
# from below layer 1, and writes the figures next to it. The value has to match
# the one the aggregation ran with.
SITE_SUBSET = ""
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


# Colorblind-safe, perceptually uniform cool-to-warm gradient
colors_list = [
    '#313695',  # Dark Blue
    '#4575b4',  # Medium Blue
    '#74add1',  # Light Blue
    '#abd9e9',  # Pale Blue
    '#e0f3f8',  # Ice Blue
    '#fee090',  # Pale Yellow
    '#fdae61',  # Light Orange
    '#f46d43',  # Orange
    '#d73027',  # Red
    '#a50026'   # Dark Red
]
bin_labels = [
    'Extreme cold',
    'Very cold',
    'Cold',
    'Cool',
    'Neutral-cool',
    'Neutral-warm',
    'Warm',
    'Hot',
    'Very hot',
    'Extreme heat'
]
# colors_list = ['#4575b4', '#91bfdb', '#e0f3f8', '#fee090', '#fc8d59', '#d73027', '#a50026']
# colors_list = ['#9C27B0', '#0984E3', '#00B894', '#636E72', '#FDCB6E', '#FF8C00', '#B71C1C']
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
    "BIN_VPD_ZSCORE": "VPD",
    "VPD_ZSCORE": "VPD",
    "VPD_ZSCORE_SHAPVALS": "VPD",
    "SWC_ZSCORE": "SWC",
    "BIN_SWC_ZSCORE": "SWC",
    "SWC_ZSCORE_SHAPVALS": "SWC",
    "SWIN_ZSCORE": "SWIN",
}
AX_LABELS_FONTSIZE = 12

# Labels & Columns
xlabel = rf'{beautify[xvar]} ($\sigma$)'
ylabel = rf'{beautify[yvar]} effect on daytime {beautify[FLUX]} ($\sigma$)'

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
settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET

dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)

# # Load SHAP values aggregated across all sites
# filepath = Path(
#     results_outdir) / f"42_SHAPVALUES-{shap_type}_AggregatedAcrossSites_{filenamex}+{filenamey}+{FLUX}.parquet"
# shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
# keeplocs = shapvals_df[y_counts] >= n_sites_min
# shapvals_df = shapvals_df[keeplocs].copy()


# Paths & Settings
shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
dir_res = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET
df_coeffs = pd.DataFrame(columns=['IGBP', 'a', 'b', 'c', 'd', 'e', 'R2', 'Threshold'])  # Collect coefficients

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
_outfilepath = dir_out / f'54_FIG-4_ResponseCurve_ShapMeans_{FLUX}_{xvar}+{yvar}+{zvar}_ALLSITES_DATA.csv'
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

# binned_z = pd.cut(Z_data, bins=5, labels=bin_labels)
# retbins=True returns (categories, bins)

# Bin z data into 10 categories
binned_z, bin_edges = pd.cut(Z_data, bins=len(bin_labels), labels=bin_labels, retbins=True)

# Fit polynomial
poly_func, poly_coeffs, x_fit, y_fit, r_squared, pi_upper, pi_lower = fit.fit_polynomial(X_data=X_data, Y_data=Y_data)

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
                              elinewidth=2,  # Thickness of the error bar line
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
                        bbox_to_anchor=(0.4, 0.86),
                        frameon=False, ncol=2, fontsize=AX_LABELS_FONTSIZE * 0.9, labelspacing=.3,
                        title="Aggregated site data by temperature regime",
                        title_fontsize=AX_LABELS_FONTSIZE * 0.85)
# legend2 = ax_all.legend(handles=handles,
#                         bbox_to_anchor=(0.36, 0.3),
#                         frameon=False, ncol=1, labelspacing=.3, fontsize=AX_LABELS_FONTSIZE)
# Add Legend 1 back to the figure
# This is the crucial step to prevent the first legend from being removed
ax_all.add_artist(legend1)

min_ix = np.argmin(y_fit)
max_ix = np.argmax(y_fit)

# Panel letter (Bold)
letter = 'a'
igbp = 'Global forests'
ax_all.text(0, 1.05, letter, transform=ax_all.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')

# Panel title (Normal, shifted slightly to the right)
ax_all.text(0.03, 1.05, f'| {igbp}', transform=ax_all.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')

# Site count (Normal)
ax_all.text(0.4, 1.05, f"(n={n_sites}, min. {minmax_counts[0]})", transform=ax_all.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='normal', ha='left', va='top')

# # Panel letter
# ax_all.text(0, 1.05, f"{letter} {igbp}", transform=ax_all.transAxes, zorder=99,
#             size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')
# ax_all.text(0.4, 1.05, f"(n={n_sites}, min. {minmax_counts[0]})", transform=ax_all.transAxes, zorder=99,
#             size=AX_LABELS_FONTSIZE * 1.2, weight='normal', ha='left', va='top')



# Calc threshold incl. upper and lower bounds from 95% fit
threshold_main, threshold_lower, threshold_upper = (
    fit.calc_threshold(x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper))

# Collect coefficients in dataframe
_threshold = f'{threshold_main:.2f} [{threshold_lower:.2f}, {threshold_upper:.2f}]'
df_coeffs.loc[len(df_coeffs)] = ['ALL SITES', poly_coeffs[0], poly_coeffs[1],
                                 poly_coeffs[2], poly_coeffs[3], poly_coeffs[4], r_squared, _threshold]
print(f"Polynomial fit ALL SITES: "
      f"y={poly_coeffs[0]:.3f}x4+{poly_coeffs[1]:.3f}x3+{poly_coeffs[2]:.3f}x2"
      f"+{poly_coeffs[3]:.3f}x+{poly_coeffs[4]:.3f}; r2={r_squared:.3f}\n")

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
                              threshold_main=threshold_main, show_annotate=True,
                              fontsize=AX_LABELS_FONTSIZE, show_annotate_short=False,
                              colors_symbols=colors_symbols,
                              label_pos=LABEL_POS, label_neg=LABEL_NEG)

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

ax_all.spines['bottom'].set_color('black')
ax_all.spines['left'].set_color('black')

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
    poly_func, poly_coeffs, x_fit, y_fit, r_squared, pi_upper, pi_lower = fit.fit_polynomial(X_data=X_data_nonan,
                                                                                             Y_data=Y_data_nonan)

    # Calc threshold incl. upper and lower bounds from 95% fit
    threshold_main, threshold_lower, threshold_upper = (
        fit.calc_threshold(x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper))

    # Find value closest to "SHAP zero"
    main_thres_idx = (np.abs(y_fit - 0)).argmin()

    if show_fit:
        fillbetweenplot = plot.add_fit(ax=ax, x_fit=x_fit, y_fit=y_fit, pi_lower=pi_lower, pi_upper=pi_upper,
                                       poly_func=poly_func, r_squared=r_squared, show_annotate=False,
                                       fontsize=AX_LABELS_FONTSIZE, color=color_fitline,
                                       linewidth=2)
    else:
        fillbetweenplot = None

    # Recalculate bins for this specific subplot using GLOBAL edges
    # This ensures "High" in ENF is the same value range as "High" in All Sites
    # binned_z_sub = pd.cut(Z_data, bins=bin_edges, labels=bin_labels)
    # Recalculate bins for this specific subplot using GLOBAL edges
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
    # Panel letter (Bold)
    ax.text(0, 1.1, letter, transform=ax.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')

    # Panel title (Normal, shifted slightly to the right)
    ax.text(0.06, 1.1, f'| {igbp}', transform=ax.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')

    # Site count (Normal)
    ax.text(0.4, 1.1, f"(n={n_sites_sub}, min. {minmax_counts_sub[0]})", transform=ax.transAxes, zorder=99,
            size=AX_LABELS_FONTSIZE * 1.2, weight='normal', ha='left', va='top')
    # ax.text(0, 1.1, f"{letter} | {igbp}", transform=ax.transAxes, zorder=99,
    #         size=AX_LABELS_FONTSIZE * 1.2, weight='bold', ha='left', va='top')
    # ax.text(0.3, 1.1, f"(n={n_sites_sub}, min. {minmax_counts_sub[0]})", transform=ax.transAxes, zorder=99,
    #         size=AX_LABELS_FONTSIZE * 1.2, weight='normal', ha='left', va='top')

    # Add text and connecting dashed lines for SHAP max, zero, and min
    if show_shap_thresholds:
        min_ix = np.argmin(y_fit)
        max_ix = np.argmax(y_fit)
        plot.show_shap_thresholds(
            ax=ax, x_fit=x_fit, y_fit=y_fit, max_ix=max_ix, min_ix=min_ix,
            threshold_main=threshold_main,
            show_annotate=True, fontsize=AX_LABELS_FONTSIZE, show_annotate_short=True,
            colors_symbols=colors_symbols,
            label_pos=LABEL_POS, label_neg=LABEL_NEG)

    # Format subplot
    plot.format(ax=ax, fontsize=AX_LABELS_FONTSIZE, showyticklabels=showyticklabels, showxticklabels=showxticklabels,
                xtickdigits=0, ytickdigits=1, showbottomspine=True, showleftspine=True, showymajorticks=True)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)

    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')

    # Collect coefficients in dataframe
    _threshold = f'{threshold_main:.2f} [{threshold_lower:.2f}, {threshold_upper:.2f}]'
    df_coeffs.loc[len(df_coeffs)] = [igbp, poly_coeffs[0], poly_coeffs[1], poly_coeffs[2], poly_coeffs[3],
                                     poly_coeffs[4], r_squared, _threshold]

    print(f"{igbp} Polynomial coefficients: "
          f"a={poly_coeffs[0]:.3f}, b={poly_coeffs[1]:.3f}, c={poly_coeffs[2]:.3f}, "
          f"d={poly_coeffs[3]:.3f}, e={poly_coeffs[4]:.3f}; R2={r_squared:.3f}\n")

    # Save plot data to csv
    _outfilepath = dir_out / f'54_FIG-4_ResponseCurve_ShapMeans_{FLUX}_{xvar}+{yvar}+{zvar}_{igbp}_DATA.csv'
    df_subset_nonan.to_csv(_outfilepath, index=False)

fig.tight_layout()
gs.update(wspace=.1)
if SHOW_PLOT:
    fig.show()
# Save coefficients to file
_outfilepath = dir_out / f'54_FIG-4_ResponseCurve_ShapMeans_{FLUX}_{xvar}+{yvar}+{zvar}_DATA_COEFFICIENTS.csv'
df_coeffs.to_csv(_outfilepath, index=False)

# ---------------------------------------------------------------------------
# Supplementary table: the same coefficients, with the threshold in kPa
# ---------------------------------------------------------------------------
# This table used to be assembled by hand from two csv files, which is what Reviewer 2
# could not follow in the code. It is written here because the coefficients are fitted
# here, so the table can never disagree with the figure.
#
# Each site standardizes VPD against its own mean and standard deviation, so a threshold
# in sigma means a different absolute VPD at every site:
#
#     VPD_abs = VPD_mean + z * VPD_sd
#
# with both statistics in hPa, over the same records the models saw, stored by stage 21.
# Divide by 10 for kPa, then average sites with equal weight, the aggregation used
# everywhere else. The biome value is the mean over the sites of that biome.
#
# The published range of 1.22 to 1.32 kPa is the span of the four biome means. It is not
# a confidence interval, and the biome intervals overlap heavily.
#
# CD-Ygb records VPD in Pa where every other site uses hPa. The factor is exactly 100:
# dividing gives a site mean of 18.10 hPa and a standard deviation of 6.56 hPa, both
# inside the 4.5 to 31.8 hPa range the other sites span. The per-site z-scores do not
# depend on the unit, so only this mapping changes and the site is converted, not dropped.
PA_UNIT_SITES = ['CD-Ygb']
PA_TO_HPA = 100

_sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                     / "21_SUBSETS_parquet_vars_stats_subsets.csv")
_converted = _sites['SITE'].isin(PA_UNIT_SITES)
_sites.loc[_converted, ['VPD_Z0', 'VPD_SD']] /= PA_TO_HPA


def _to_kpa(sites_df, z):
    """Site-wise absolute threshold in kPa, averaged with equal weight per site."""
    return ((sites_df['VPD_Z0'] + z * sites_df['VPD_SD']) / 10).mean()


def _parse_threshold(text):
    """Split '0.20 [0.05, 0.34]' into its three numbers."""
    point, rest = text.split('[')
    lo, hi = rest.rstrip(']').split(',')
    return float(point), float(lo), float(hi)


_rows = []
for _, _c in df_coeffs.iterrows():
    _group = _c['IGBP']
    _z, _z_lo, _z_hi = _parse_threshold(_c['Threshold'])
    _sub = _sites if _group == 'ALL SITES' else _sites[_sites['IGBP'] == _group]
    _rows.append({
        'Group': _group,
        'Sites': len(_sub),
        'a': _c['a'], 'b': _c['b'], 'c': _c['c'], 'd': _c['d'], 'e': _c['e'],
        'R2': round(_c['R2'], 3),
        'Threshold (sigma)': _c['Threshold'],
        'Threshold (kPa)': f"{_to_kpa(_sub, _z):.2f} [{_to_kpa(_sub, _z_lo):.2f}, "
                           f"{_to_kpa(_sub, _z_hi):.2f}]",
    })
supptable = pd.DataFrame(_rows)

_stem = dir_out / f'54_SUPPTABLE-X_ThresholdPolynomials_{FLUX}'
supptable.to_csv(f'{_stem}.csv', index=False)
with pd.ExcelWriter(f'{_stem}.xlsx', engine='openpyxl') as _writer:
    supptable.to_excel(_writer, sheet_name='Threshold polynomials', index=False)
    _sheet = _writer.sheets['Threshold polynomials']
    for _col, _w in zip('ABCDEFGHIJ', (14, 7, 11, 11, 11, 11, 11, 8, 20, 22)):
        _sheet.column_dimensions[_col].width = _w
print()
print(supptable.to_string(index=False))
print(f"Saved {_stem}.xlsx and {_stem}.csv")

# Save fig to file
outfilepath = dir_out / f'54_FIG-4_ResponseCurve_ShapMeans_{FLUX}_{xvar}+{yvar}+{zvar}.png'
fig.savefig(outfilepath, dpi=300, bbox_inches='tight')
