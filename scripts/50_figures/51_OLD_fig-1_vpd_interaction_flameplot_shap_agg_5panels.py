"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from diive.core.plotting.styles import LightTheme as theme

import src.files as files
import src.plot as plot
# from scipy import ndimage
from src.common import findpoi

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, ET, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'NEP_ZSCORE'
# FLUX = 'ET_ZSCORE'
# FLUX = 'GPP'_ZSCORE
# FLUX = 'RECO_ZSCORE'
xvar = 'TA_ZSCORE'
yvar = 'VPD_ZSCORE'
zvar = 'VPD_ZSCORE'
aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Heatmap settings
label_longforms = dict(TA='Air temperature', VPD='Vapor pressure deficit', SWC='Soil water content')
xlabel = f'{xvar} (z-score)'
ylabel = f'{yvar} (z-score)'
# xlabel = f'{label_longforms[xvar]} (z-score)'
# ylabel = f'{label_longforms[yvar]} (z-score)'
zlabel = f'Impact of {zvar} on {FLUX} (SHAP {aggfunc} z-score)'
# zlabel = f'{aggfunc} SHAP value of {zvar} (z-score)'
n_sites_min = 30  # 30
n_sites_used = 171
cb_digits_after_comma = 1
cmap = 'RdYlBu'
# cmap = 'RdYlBu_r'
area_size_minmax = 25  # Number of bins used to calculate min/max areas
# ------------------------------


binx = (f"BIN_{xvar}", aggfunc)
biny = (f"BIN_{yvar}", aggfunc)
z = (f"{zvar}_SHAPVALS", aggfunc)
z_counts = (f"{zvar}_SHAPVALS", "count")

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
dir_prev_results = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
dir_out = Path(settings['DIR_PLOTS_OUT'])

# Start figure
fig = plt.figure(figsize=(20.7, 9), dpi=150, facecolor="white")
gs = gridspec.GridSpec(2, 4)  # rows, cols
# gs.update(wspace=.2, hspace=.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
ax_all = fig.add_subplot(gs[0:2, 0:2])
ax2 = fig.add_subplot(gs[0, 2], sharex=ax_all, sharey=ax_all)
ax3 = fig.add_subplot(gs[0, 3], sharex=ax_all, sharey=ax_all)
ax4 = fig.add_subplot(gs[1, 2], sharex=ax_all, sharey=ax_all)
ax5 = fig.add_subplot(gs[1, 3], sharex=ax_all, sharey=ax_all)

# Load SHAP values aggregated across all sites
filepath = Path(dir_prev_results) / f"42_SHAPVALUES-{shap_type}_AggregatedAcrossSites_BIN-{xvar}+BIN-{yvar}+{FLUX}.parquet"
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

ymin = subset_all.iloc[:, 1].min() * 1.15
ymax = subset_all.iloc[:, 1].max() * 1.05
ax_all.set_ylim(ymin, ymax)
xmin = subset_all.iloc[:, 0].min() * 1.15
xmax = subset_all.iloc[:, 0].max() * 1.15
ax_all.set_xlim(xmin, xmax)

ax_all.text(0.03, 0.98, f"(a) All sites (n={n_sites_used}, min. {n_sites_all_min})",
            transform=ax_all.transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
            ha='left', va='bottom', zorder=99, backgroundcolor='white')
ax_all.axhline(0, color='black', linestyle='--', linewidth=1, zorder=100)
ax_all.axvline(0, color='black', linestyle='--', linewidth=1, zorder=100)

# Info texts
params = dict(size=theme.AX_LABELS_FONTSIZE, color='k', zorder=100)
ax_all.text(2, 0.1, r"$\uparrow$ dry", horizontalalignment='left', verticalalignment='bottom', **params)
ax_all.text(2, -0.1, r"$\downarrow$ humid", horizontalalignment='left', verticalalignment='top', **params)
ax_all.text(-0.1, 3.5, r"$\leftarrow$ cool", horizontalalignment='right', verticalalignment='center', **params)
ax_all.text(0.1, 3.5, r"warm $\rightarrow$", horizontalalignment='left', verticalalignment='center', **params)

# Find optimum and pessimum
pivot_df = subset_all.pivot(index=f'BIN_{xvar}_median', columns=f'BIN_{yvar}_median', values=f'{zvar}_SHAPVALS_median')
max_location, max_value = findpoi(df=pivot_df, k=area_size_minmax, agg='mean', what='max')
min_location, min_value = findpoi(df=pivot_df, k=area_size_minmax, agg='mean', what='min')

# Optimum (largest SHAP)
maxx = max_location[0] + 0.05
maxy = max_location[1] + 0.05
params_max = dict(size=theme.AX_LABELS_FONTSIZE, color='k', zorder=100)
color = "#90A4AE"
ax_all.scatter(maxx, maxy, color='black', marker='+', edgecolors='none', linewidth=3, s=650, zorder=100, alpha=0.5)
ax_all.scatter(maxx, maxy, color='none', marker='o', edgecolor='black', linewidth=3, s=650, zorder=100, alpha=0.5)
ax_all.annotate(f'highest {FLUX} increase',
                xy=(maxx, maxy),
                xytext=(maxx - 3, maxy + 1.5),  # Adjust text position as needed
                arrowprops=dict(arrowstyle="->", color='black', lw=3, shrinkB=15),
                fontsize=16, color='black', ha='left', va='center', zorder=100, )
print(f"Maximum found at x={max_location[0]}, y={max_location[1]}")

# Pessimum (smallest SHAP)
minx = min_location[0] + 0.05
miny = min_location[1] + 0.05
ax_all.scatter(minx, miny, color='black', marker='_', edgecolors='none', linewidth=3, s=650, zorder=100, alpha=0.5)
ax_all.scatter(minx, miny, color='none', marker='o', edgecolor='black', linewidth=3, s=650, zorder=100, alpha=0.5)
ax_all.annotate(f'highest {FLUX} decrease',
                xy=(minx, miny),
                xytext=(minx - 1.1, miny + 0.6),  # Adjust text position as needed
                arrowprops=dict(arrowstyle="->", color='black', lw=3, shrinkB=15),
                fontsize=16, color='black', ha='center', va='center', zorder=100, )
print(f"Minimum found at x={min_location[0]}, y={min_location[1]}")
# ax_all.text(0.03, 0.9, f"Min(x={min_location[0]}, y={min_location[1]})",
#             transform=ax_all.transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
#             ha='left', va='bottom', zorder=99, backgroundcolor='white')

# Load SHAP values aggregated per IGBP
igbps = ['DBF', 'MF', 'EBF']
# igbps = ['ENF', 'DBF', 'MF', 'EBF'] TODO ACT
igbps_n_sites = [87, 56, 14, 14]  # Counted in #33
axes = [ax2, ax3, ax4, ax5]
xlabels = [" ", " ", xlabel, xlabel]
ylabels = [ylabel, " ", ylabel, " "]
letter = ['b', 'c', 'd', 'e']
data_per_igbp = {}
for ix, i in enumerate(igbps):
    filepath = Path(dir_prev_results) / f"43_SHAPVALUES-{shap_type}_AggregatedAcrossIGBP-{i}_BIN-{xvar}+BIN-{yvar}+{FLUX}.parquet"
    igbp_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

    # Next line uses the same keeplocs like defined above, i.e. for each site we
    # get the same locations as for the overall (all sites) plot.
    # Filter igbp_df to keep only rows whose index exists in keeplocs index
    igbp_df = igbp_df[igbp_df.index.isin(keeplocs.index)].copy()

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

    pivot_df = data_per_igbp[i].pivot(index=f'BIN_{xvar}_median', columns=f'BIN_{yvar}_median', values=f'{zvar}_SHAPVALS_median')
    max_location, max_value = findpoi(df=pivot_df, k=area_size_minmax, agg='mean', what='max')
    min_location, min_value = findpoi(df=pivot_df, k=area_size_minmax, agg='mean', what='min')

    # Optimum (smallest SHAP)
    x = max_location[0] + 0.05
    y = max_location[1] + 0.05
    axes[ix].scatter(x, y, color='black', marker='+', edgecolors='none', linewidth=3, s=650, zorder=100, alpha=0.5)
    axes[ix].scatter(x, y, color='none', marker='o', edgecolor='black', linewidth=3, s=650, zorder=100, alpha=0.5)

    # Pessimum (largest SHAP)
    x = min_location[0] + 0.05
    y = min_location[1] + 0.05
    axes[ix].scatter(x, y, color='black', marker='_', edgecolors='none', linewidth=3, s=650, zorder=100, alpha=0.5)
    axes[ix].scatter(x, y, color='none', marker='o', edgecolor='black', linewidth=3, s=650, zorder=100, alpha=0.5)

fig.tight_layout()
gs.update(wspace=.2)
fig.show()
