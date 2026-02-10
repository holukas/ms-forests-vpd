"""
Flame plot.
"""
from pathlib import Path

import matplotlib as mpl

import src.files as files
import src.plot as plot
from src.plot import plot_markers, style_ax

# Settings & variables

# Main Fig. 1
# plotvars = [FLUX, xvar, yvar, zvar, x_in_filename, y_in_filename]
plotvars = ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'BIN-TA_ZSCORE', 'BIN-VPD_ZSCORE']

# # Extended Figures
# plotvars = ['NEP_ZSCORE', 'BIN_TA_ZSCORE', 'BIN_SWC_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'BIN-TA_ZSCORE', 'BIN-SWC_ZSCORE']

FLUX = plotvars[0]
xvar, yvar, zvar = plotvars[1], plotvars[2], plotvars[3]
x_in_filename, y_in_filename = plotvars[4], plotvars[5]

# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP_ZSCORE'
# FLUX = 'RECO_ZSCORE'

# Agg groups, use z-scores:
# xvar, yvar, zvar = 'TA_ZSCORE', 'VPD_ZSCORE', 'VPD_ZSCORE'
# xvar, yvar, zvar = 'TA_ZSCORE', 'VPD_ZSCORE', 'TA_ZSCORE'
# xvar, yvar, zvar = 'SWC_ZSCORE', 'SWC_ZSCORE_SHAPVALS', 'VPD_ZSCORE'
# xvar, yvar, zvar = 'SWIN_ZSCORE', 'TA_ZSCORE', 'VPD_ZSCORE'
# xvar, yvar, zvar = 'SWC_ZSCORE', 'VPD_ZSCORE', 'VPD_ZSCORE'
# xvar, yvar, zvar = 'SWIN_ZSCORE', 'VPD_ZSCORE', 'VPD_ZSCORE'
# xvar, yvar, zvar = 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE'
# ------------------------------

aggfunc, CONDITIONAL = 'mean', True
n_sites_min, cb_digits, area_size = 10, 1, 50
cmap, igbps = 'RdYlBu', ['ENF', 'DBF', 'MF', 'EBF']

beautify = {
    "NEP_ZSCORE": "NEP",
    "ET_ZSCORE": "ET",
    "BIN_TA_ZSCORE": "Air temperature",
    "BIN_VPD_ZSCORE": "VPD",
    "BIN_SWC_ZSCORE": "SWC",
    "SWC_ZSCORE_SHAPVALS": "SWC",
    "VPD_ZSCORE_SHAPVALS": "VPD",
    "SWC_ZSCORE": "SWC",
    "SWIN_ZSCORE": "SWIN",
}
AX_LABELS_FONTSIZE = 16

# Labels & Columns
xlabel = f'{beautify[xvar]} (z-score)'
ylabel = f'{beautify[yvar]} (z-score)'
zlabel = f'{beautify[zvar]} effect (z-score)'

# Column names in dataframe, bins use 'median' to have unique bin IDs
xagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
yagg = 'median' if str(xvar).startswith('BIN_') else aggfunc
xcol = (f"{xvar}", xagg)
ycol = (f"{yvar}", yagg)
zcol = (f"{zvar}", aggfunc)
count_vals_col = (f"{zvar}", "count")

# Paths & Settings
shap_type = 'conditional' if CONDITIONAL else 'standard'
settings = files.read_settings_file("../../config/settings.yaml")
dir_res = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

# FIGURE LAYOUT (5 panels)
fig, gs, ax_all, axes_sub, cax = plot.layout_5panels(figsize=(19.8, 8.1), add_colorbar_ax=True)

# ---------------------
# MAIN PLOT (all sites)

# Load data
filedf, subsetdf, n_sites = files.load_data(
    suffix='Sites', shap_type=shap_type, dir_res=dir_res, flux=FLUX, count_vals_col=count_vals_col,
    n_sites_min=n_sites_min, subsetcols=[xcol, ycol, zcol], site_filter=None, x_in_filename=x_in_filename,
    y_in_filename=y_in_filename, aggfunc=aggfunc)

# Plot
plot.flameplot(df=subsetdf, fig=fig, ax=ax_all, cmap=cmap, title=None, cb_digits_after_comma=cb_digits,
               xlabel=xlabel, ylabel=ylabel, zlabel=None, cb_extend='both', show_colormap=False)
vmin, vmax = subsetdf.iloc[:, 2].min(), subsetdf.iloc[:, 2].max()
ax_all.set_ylim(subsetdf.iloc[:, 1].min() * 1.15, subsetdf.iloc[:, 1].max() * 1.05)
ax_all.set_xlim(subsetdf.iloc[:, 0].min() * 1.15, subsetdf.iloc[:, 0].max() * 1.15)
style_ax(ax=ax_all, title=f"(a) All sites (n={n_sites[1]}, min. {n_sites[0]})", ax_labels_fontsize=AX_LABELS_FONTSIZE)
ax_all.axhline(0, c='k', ls='--', lw=1, zorder=99)
ax_all.axvline(0, c='k', ls='--', lw=1, zorder=99)
ax_all.set_aspect('equal')
t_params = dict(size=AX_LABELS_FONTSIZE, color='0.3', fontstyle='italic', zorder=100)
texts = [(2, 0.1, r"$\uparrow$ dry", 'left', 'bottom'), (2, -0.1, r"$\downarrow$ humid", 'left', 'top'),
         (-0.1, 3.5, r"$\leftarrow$ cool", 'right', 'center'), (0.1, 3.5, r"warm $\rightarrow$", 'left', 'center')]
for x, y, s, h, v in texts:
    ax_all.text(x, y, s, ha=h, va=v, **t_params)
# TODO
# minmaxlocs = plot_markers(ax_all, subsetdf,
#                           xvals=f'{xvar}_{aggfunc}', yvals=f'{yvar}_{aggfunc}', zvals=f'{zvar}_{aggfunc}',
#                           flux_txt=beautify[FLUX], annotate=True, ax_labels_fontsize=AX_LABELS_FONTSIZE,
#                           area_size=area_size)
# print(f"[ALL SITES] Highest increase found at x={minmaxlocs['max'][0]}, y={minmaxlocs['max'][1]}")
# print(f"[ALL SITES] Highest decrease found at x={minmaxlocs['min'][0]}, y={minmaxlocs['min'][1]}")

# ---------------
# SUBPLOTS (IGBP)
configs = zip(
    axes_sub,
    igbps,
    [" ", " ", xlabel, xlabel],
    [ylabel, " ", ylabel, " "],
    ['b', 'c', 'd', 'e'],
    [True, False, True, False],  # For showing yticklabels
    [False, False, True, True]  # For showing xticklabels
)
for ax, igbp, xl, yl, letter, showyticklabels, showxticklabels in configs:
    # Filter using index from main dataset (keeplocs logic)
    df_igbp, df_subset, n_sites_sub = files.load_data(
        suffix=f"IGBP-{igbp}", shap_type=shap_type, dir_res=dir_res, flux=FLUX, count_vals_col=count_vals_col,
        n_sites_min=n_sites_min, subsetcols=[xcol, ycol, zcol], site_filter=filedf.index, x_in_filename=x_in_filename,
        y_in_filename=y_in_filename, aggfunc=aggfunc)
    plot.flameplot(df=df_subset, fig=fig, ax=ax, cmap=cmap, title=None, show_colormap=False,
                   vmin=vmin, vmax=vmax, xlabel=xl, ylabel=yl)
    plot.style_ax(ax, f"({letter}) {igbp} (n={n_sites_sub[1]}, min. {n_sites_sub[0]})",
                  ax_labels_fontsize=AX_LABELS_FONTSIZE)
    ax.axhline(0, c='k', ls='--', lw=1, zorder=99)
    ax.axvline(0, c='k', ls='--', lw=1, zorder=99)
    ax.set_aspect('equal')
    minmaxlocs = plot_markers(ax, df_subset,
                              xvals=f'{xvar}_{xagg}', yvals=f'{yvar}_{yagg}', zvals=f'{zvar}_{aggfunc}',
                              flux_txt=beautify[FLUX], annotate=False, ax_labels_fontsize=AX_LABELS_FONTSIZE,
                              area_size=area_size)
    # Format subplot
    plot.format(ax=ax, fontsize=AX_LABELS_FONTSIZE, showyticklabels=showyticklabels, showxticklabels=showxticklabels,
                xtickdigits=0, ytickdigits=0)
    print(f"[{igbp}] Highest increase found at x={minmaxlocs['max'][0]}, y={minmaxlocs['max'][1]}")
    print(f"[{igbp}] Highest decrease found at x={minmaxlocs['min'][0]}, y={minmaxlocs['min'][1]}")

# ---------------
# SHARED COLORBAR
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cb = fig.colorbar(sm, cax=cax, extend='both')
cb.set_label(zlabel, size=AX_LABELS_FONTSIZE, labelpad=20)
cb.ax.tick_params(labelsize=AX_LABELS_FONTSIZE)

fig.tight_layout()
gs.update(wspace=.2)
fig.show()

# Save fig to file
usedx = subsetdf.columns[0]
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / (f'51_FIG-1_Flameplots_ShapMedians_{FLUX}_'
                         f'{subsetdf.columns[0]}+{subsetdf.columns[1]}+{subsetdf.columns[2]}.png')
fig.savefig(outfilepath, dpi=300, bbox_inches='tight')
