"""
Flame plot.
"""
from pathlib import Path
import numpy as np
import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from diive.core.plotting.styles import LightTheme as theme

import src.files as files
import src.plot as plot
from src.common import findpoi

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, ET, GPP, RECO, TA, VPD, SWIN, SWC
FLUXES = ['NEP', 'GPP', 'RECO', 'ET',
          'NEP', 'GPP', 'RECO', 'ET',
          'NEP', 'GPP', 'RECO', 'ET']
xvars = ['TA', 'TA', 'TA', 'TA',
         'TA', 'TA', 'TA', 'TA',
         'SWC', 'SWC', 'SWC', 'SWC']
yvars = ['VPD', 'VPD', 'VPD', 'VPD',
         'VPD', 'VPD', 'VPD', 'VPD',
         'VPD', 'VPD', 'VPD', 'VPD', ]
zvars = ['VPD', 'VPD', 'VPD', 'VPD',
         'TA', 'TA', 'TA', 'TA',
         'SWC', 'SWC', 'SWC', 'SWC']
zvar_is_shap = [True, True, True, True,
                True, True, True, True,
                True, True, True, True]
aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Heatmap settings
n_sites_min = 30
cmap = 'RdYlBu'
# cmap = 'RdYlBu_r'
# cbar_zlabel = "median flux (z-score)"
xlabels = [f'{xvar} (z-score)' for xvar in xvars]
ylabels = [f'{yvar} (z-score)' for yvar in yvars]
# zlabels = [f'{aggfunc} SHAP value of {zvar} (z-score)' for zvar in zvars]
binsx = [(f"BIN_{xvar}", aggfunc) for xvar in xvars]
binsy = [(f"BIN_{yvar}", aggfunc) for yvar in yvars]
FONTSIZE = theme.AX_LABELS_FONTSIZE

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'

# Start figure
fig = plt.figure(figsize=(26 * 0.9, 16.8 * 0.9), dpi=150, facecolor="white")
gs = gridspec.GridSpec(3, 21)  # rows, cols
# gs.update(wspace=.3, hspace=.2, left=0.03, right=0.94, top=0.97, bottom=0.04)

# Row 1
ax1 = fig.add_subplot(gs[0, 0:5])
ax2 = fig.add_subplot(gs[0, 5:10], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[0, 10:15], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(gs[0, 15:20], sharex=ax1, sharey=ax1)

# Row 2
ax5 = fig.add_subplot(gs[1, 0:5], sharex=ax1, sharey=ax1)
ax6 = fig.add_subplot(gs[1, 5:10], sharex=ax1, sharey=ax1)
ax7 = fig.add_subplot(gs[1, 10:15], sharex=ax1, sharey=ax1)
ax8 = fig.add_subplot(gs[1, 15:20], sharex=ax1, sharey=ax1)

# Row 3
ax9 = fig.add_subplot(gs[2, 0:5], sharey=ax1)
ax10 = fig.add_subplot(gs[2, 5:10], sharex=ax9, sharey=ax1)
ax11 = fig.add_subplot(gs[2, 10:15], sharex=ax9, sharey=ax1)
ax12 = fig.add_subplot(gs[2, 15:20], sharex=ax9, sharey=ax1)

# Colorbars
ax_cbar_shap_vpd = fig.add_subplot(gs[0, 20])
ax_cbar_shap_ta = fig.add_subplot(gs[1, 20])
ax_cbar_shap_swc = fig.add_subplot(gs[2, 20])

axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']

vmin = 9999  # Will be detected from NEP below
vmax = -9999
p = None
mesh_obj = {}  # Store pcolormesh objects for scaling later
vmins = []
vmaxs = []

for ix, flux in enumerate(FLUXES):

    # TODO testing ---
    # if ix > 7:
    #     continue
    # TODO testing ---

    # Load SHAP values aggregated across all sites
    results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / flux / shap_type
    filepath = Path(
        results_outdir) / f"3_AllSites_Aggregated_SHAPValues-{shap_type}_BIN-{xvars[ix]}_BIN-{yvars[ix]}_{flux}.parquet"
    shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

    if zvar_is_shap[ix]:
        zs = [(f"{zvar}_SHAPVALS", aggfunc) for zvar in zvars]
        zs_counts = [(f"{zvar}_SHAPVALS", "count") for zvar in zvars]
    else:
        zs = [(f"{zvar}", aggfunc) for zvar in zvars]
        zs_counts = [(f"{zvar}", "count") for zvar in zvars]

    keeplocs = shapvals_df[zs_counts[ix]] >= n_sites_min
    shapvals_df = shapvals_df[keeplocs].copy()
    n_sites_all_min = shapvals_df[zs_counts[ix]].min()
    n_sites_all_max = shapvals_df[zs_counts[ix]].max()
    subset_all = shapvals_df[[binsx[ix], binsy[ix], zs[ix]]].copy()

    # Panels 0-7 need same scaling as in #41
    if ix == 0:
        ymin = subset_all.iloc[:, 1].min() * 1.05
        ymax = subset_all.iloc[:, 1].max() * 1.05
        axes[ix].set_ylim(ymin, ymax)
        xmin = subset_all.iloc[:, 0].min() * 1.15
        xmax = subset_all.iloc[:, 0].max() * 1.15
        axes[ix].set_xlim(xmin, xmax)
        axes[ix].set_xticks(np.arange(-3, 3, 1))

    if ix == 8:
        xmin = subset_all.iloc[:, 0].min() * 1.1
        xmax = subset_all.iloc[:, 0].max() * 1.1
        axes[ix].set_xlim(xmin, xmax)
        axes[ix].set_xticks(np.arange(-2, 4, 1))

    # Collect min and max values for scaling later
    vmins.append(subset_all[zs[ix]].min())
    vmaxs.append(subset_all[zs[ix]].max())

    subset_all.columns = ['_'.join(col).strip() for col in subset_all.columns.values]  # Heatmap needs flat column index
    ylabel = ylabels[ix] if any([ix == 0, ix == 4, ix == 8]) else " "

    # Store pcolormesh objects in dict, used later for scaling and colorbars
    mesh_obj[ix] = plot.flameplot(df=subset_all, fig=fig, ax=axes[ix], cmap=cmap,
                                  title=None, vmin=vmin, vmax=vmax, show_colormap=False,
                                  xlabel=xlabels[ix], ylabel=ylabel, zlabel=None, show_grid=False)

    text = f"({letter[ix]}) {flux}"
    # text = f"({letter[ix]}) All sites, {flux} (n=171, min. {n_sites_all_min})"
    axes[ix].axhline(0, color='black', linestyle='--', linewidth=1, zorder=100)
    axes[ix].axvline(0, color='black', linestyle='--', linewidth=1, zorder=100)
    axes[ix].text(0.1, 0.95, text,
                  transform=axes[ix].transAxes, color='black', size=FONTSIZE,
                  ha='left', va='bottom', zorder=99)
    axes[ix].set_aspect('equal')

    # Find optimum and pessimum
    _index = f"{binsx[ix][0]}_{binsx[ix][1]}"
    _cols = f"{binsy[ix][0]}_{binsy[ix][1]}"
    _vals = f"{zs[ix][0]}_{zs[ix][1]}"
    pivot_df = subset_all.pivot(index=_index, columns=_cols, values=_vals)
    max_location, max_value = findpoi(df=pivot_df, k=25, agg='mean', what='max')
    min_location, min_value = findpoi(df=pivot_df, k=25, agg='mean', what='min')

    # Show symbols for min and max flux locations
    params = dict(horizontalalignment='center', verticalalignment='center', color="black",
                  size=35, zorder=100, alpha=.6)

    # Maximum flux
    maxx = max_location[0] + 0.05
    maxy = max_location[1] + 0.05
    color = "#607D8B"  # blue grey 500
    axes[ix].scatter(maxx, maxy, color='black', marker='+', edgecolors='none',
                     linewidth=3, s=650, zorder=100, alpha=0.5)
    axes[ix].scatter(maxx, maxy, color='none', marker='o', edgecolor='black',
                     linewidth=3, s=650, zorder=100, alpha=0.5)

    # Minimum flux
    minx = min_location[0] - 0.05
    miny = min_location[1] + 0.05
    axes[ix].scatter(minx, miny, color='black', marker='_', edgecolors='none',
                     linewidth=3, s=650, zorder=100, alpha=0.5)
    axes[ix].scatter(minx, miny, color='none', marker='o', edgecolor='black',
                     linewidth=3, s=650, zorder=100, alpha=0.5)

    if ix == 0:
        params = dict(color='black', size=FONTSIZE, zorder=99)
        # axes[ix].text(-3.4, 1, 'highest\nflux increase', ha='left', va='center', **params)
        axes[ix].annotate(f'highest\nflux increase',
                          xy=(maxx, maxy),
                          xytext=(maxx - 2.5, maxy + 2),  # Adjust text position as needed
                          arrowprops=dict(arrowstyle="->", color='black', lw=2, shrinkB=15),
                          fontsize=16, color='black', ha='left', va='center', zorder=100, )
        # axes[ix].text(minx - 0.3, miny, 'highest flux decrease', ha='right', va='center', **params)
        axes[ix].annotate(f'highest flux decrease',
                          xy=(minx, miny),
                          xytext=(minx - 5, miny - 0.5),
                          arrowprops=dict(arrowstyle="->", color='black', lw=2, shrinkB=15),
                          fontsize=16, color='black', ha='left', va='center', zorder=100)

vmin_firstrow = -0.9069341723818797  # Use same scaling as in #41
vmax_firstrow = 0.3076493751085945  # Use same scaling as in #41
vmin_secondrow = min(vmins[4:8])
vmax_secondrow = max(vmaxs[4:8])
vmin_thirdrow = min(vmins[8:12])
vmax_thirdrow = max(vmaxs[8:12])
for ix, m in mesh_obj.items():
    if ix < 4:
        m.set_clim(vmin=vmin_firstrow, vmax=vmax_firstrow)
    elif 4 <= ix < 8:
        m.set_clim(vmin=vmin_secondrow, vmax=vmax_secondrow)
    else:
        m.set_clim(vmin=vmin_thirdrow, vmax=vmax_thirdrow)
    print(f"ax{ix}: {m.get_clim()=}")

# Colorbars
cbar = fig.colorbar(mesh_obj[0], cax=ax_cbar_shap_vpd, label='XXX', extend='both')
cbar.ax.tick_params(labelsize=FONTSIZE)
label = "Impact of VPD on flux (SHAP median z-score)"
cbar.set_label(label, fontsize=FONTSIZE, labelpad=20)
# tick_locations = np.linspace(-0.9, 0.3, 13)
# cbar.set_ticks(tick_locations)

cbar = fig.colorbar(mesh_obj[4], cax=ax_cbar_shap_ta, label='XXX', extend='both')
cbar.ax.tick_params(labelsize=FONTSIZE)
label = "Impact of TA on flux (SHAP median z-score)"
cbar.set_label(label, fontsize=FONTSIZE, labelpad=20)
# tick_locations = np.linspace(-0.9, 0.3, 13)
# cbar.set_ticks(tick_locations)

cbar = fig.colorbar(mesh_obj[8], cax=ax_cbar_shap_swc, label='XXX', extend='both')
cbar.ax.tick_params(labelsize=FONTSIZE)
label = "Impact of SWC on flux (SHAP median z-score)"
cbar.set_label(label, fontsize=FONTSIZE, labelpad=20)
# tick_locations = np.linspace(-0.9, 0.3, 13)
# cbar.set_ticks(tick_locations)

fig.tight_layout()
gs.update(wspace=0.5, hspace=.2)
fig.show()
