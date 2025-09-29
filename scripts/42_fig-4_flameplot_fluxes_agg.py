"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from diive.core.plotting.styles import LightTheme as theme

import src.files as files
import src.plot as plot

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, ET, GPP, RECO, TA, VPD, SWIN, SWC
FLUXES = ['NEP', 'GPP', 'RECO', 'ET',
          'NEP', 'GPP', 'RECO', 'ET',
          'NEP', 'GPP', 'RECO', 'ET']
xvars = ['TA', 'TA', 'TA', 'TA',
         'SWC', 'SWC', 'SWC', 'SWC',
         'SWIN', 'SWIN', 'SWIN', 'SWIN']
yvars = ['VPD', 'VPD', 'VPD', 'VPD',
         'VPD', 'VPD', 'VPD', 'VPD',
         'VPD', 'VPD', 'VPD', 'VPD', ]
zvars = ['NEP', 'GPP', 'RECO', 'ET',
         'NEP', 'GPP', 'RECO', 'ET',
         'NEP', 'GPP', 'RECO', 'ET']
zvar_is_shap = [False, False, False, False,
                False, False, False, False,
                False, False, False, False]
aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Heatmap settings
n_sites_min = 30
cmap = 'RdYlBu'
# cmap = 'RdYlBu_r'
cbar_zlabel = "median flux (z-score)"
xlabels = [f'{xvar} (z-score)' for xvar in xvars]
ylabels = [f'{yvar} (z-score)' for yvar in yvars]
# zlabels = [f'{aggfunc} SHAP value of {zvar} (z-score)' for zvar in zvars]
binsx = [(f"BIN_{xvar}", aggfunc) for xvar in xvars]
binsy = [(f"BIN_{yvar}", aggfunc) for yvar in yvars]

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'

# Start figure
fig = plt.figure(figsize=(26.1, 19.8), dpi=150, facecolor="white")
gs = gridspec.GridSpec(3, 21)  # rows, cols
# gs.update(wspace=.3, hspace=.2, left=0.03, right=0.94, top=0.97, bottom=0.04)
ax1 = fig.add_subplot(gs[0, 0:5])
ax2 = fig.add_subplot(gs[0, 5:10], sharex=ax1, sharey=ax1)
ax3 = fig.add_subplot(gs[0, 10:15], sharex=ax1, sharey=ax1)
ax4 = fig.add_subplot(gs[0, 15:20], sharex=ax1, sharey=ax1)
ax5 = fig.add_subplot(gs[1, 0:5], sharey=ax1)
ax6 = fig.add_subplot(gs[1, 5:10], sharex=ax5, sharey=ax1)
ax7 = fig.add_subplot(gs[1, 10:15], sharex=ax5, sharey=ax1)
ax8 = fig.add_subplot(gs[1, 15:20], sharex=ax5, sharey=ax1)
ax9 = fig.add_subplot(gs[2, 0:5], sharey=ax1)
ax10 = fig.add_subplot(gs[2, 5:10], sharex=ax9, sharey=ax1)
ax11 = fig.add_subplot(gs[2, 10:15], sharex=ax9, sharey=ax1)
ax12 = fig.add_subplot(gs[2, 15:20], sharex=ax9, sharey=ax1)
ax_cbar = fig.add_subplot(gs[0:3, 20])
axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]
letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
vmin = None  # Will be detected from NEP below
vmax = None
p = None

for ix, flux in enumerate(FLUXES):
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
    if ix == 0:  # NEP used to get the scaling numbers
        ymin = subset_all.iloc[:, 1].min() * 1.05
        ymax = subset_all.iloc[:, 1].max() * 1.05
        axes[ix].set_ylim(ymin, ymax)
    # vmin = subset_all[zs[ix]].min()
    # vmax = subset_all[zs[ix]].max()
    vmin = -1.2
    vmax = 1.2
    subset_all.columns = ['_'.join(col).strip() for col in subset_all.columns.values]  # Heatmap needs flat column index
    ylabel = ylabels[ix] if ix == 4 else " "

    p = plot.flameplot(df=subset_all, fig=fig, ax=axes[ix], cmap=cmap,
                       title=None, vmin=vmin, vmax=vmax, show_colormap=False,
                       xlabel=xlabels[ix], ylabel=ylabel, zlabel=None, show_grid=False)
    text = f"({letter[ix]}) {flux}"
    # text = f"({letter[ix]}) All sites, {flux} (n=171, min. {n_sites_all_min})"
    axes[ix].axhline(0, color='black', linestyle='--', linewidth=1, zorder=100)
    axes[ix].axvline(0, color='black', linestyle='--', linewidth=1, zorder=100)
    axes[ix].text(0.1, 0.95, text,
                  transform=axes[ix].transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
                  ha='left', va='bottom', zorder=99)
    axes[ix].set_aspect('equal')

# Colorbar for all subplots
cbar = fig.colorbar(p, cax=ax_cbar, label='XXX', extend='both')
cbar.ax.tick_params(labelsize=theme.AX_LABELS_FONTSIZE)
cbar.set_label(cbar_zlabel, fontsize=theme.AX_LABELS_FONTSIZE, labelpad=20)
tick_locations = np.linspace(-1.2, 1.2, 13)
cbar.set_ticks(tick_locations)  # This is the key line


fig.tight_layout()
gs.update(wspace=0.5, hspace=.2)
fig.show()
