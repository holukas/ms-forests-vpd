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

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
# FLUX = 'NEP'
# FLUX = 'LE'
# FLUX = 'GPP'
# FLUX = 'RECO'
FLUXES = ['NEP', 'LE', 'GPP', 'RECO']
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
cmap = 'RdYlBu'
# cmap = 'RdYlBu_r'
# ------------------------------

binx = (f"BIN_{xvar}", aggfunc)
biny = (f"BIN_{yvar}", aggfunc)
z = (f"{zvar}_SHAPVALS", aggfunc)
z_counts = (f"{zvar}_SHAPVALS", "count")

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'

# Start figure
fig = plt.figure(figsize=(21, 6), dpi=150, facecolor="white")
gs = gridspec.GridSpec(1, 3)  # rows, cols
# gs.update(wspace=.2, hspace=.3, left=0.1, right=0.9, top=0.9, bottom=0.1)
ax_le = fig.add_subplot(gs[0, 0])
ax_gpp = fig.add_subplot(gs[0, 1], sharex=ax_le, sharey=ax_le)
ax_reco = fig.add_subplot(gs[0, 2], sharex=ax_le, sharey=ax_le)
axes = [None, ax_le, ax_gpp, ax_reco]
letter = [None, 'a', 'b', 'c']
vmin = None  # Will be detected from NEP below
vmax = None

for ix, flux in enumerate(FLUXES):
    # Load SHAP values aggregated across all sites
    results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / flux / shap_type
    filepath = Path(
        results_outdir) / f"3_AllSites_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{flux}.parquet"
    shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
    keeplocs = shapvals_df[z_counts] >= n_sites_min
    shapvals_df = shapvals_df[keeplocs].copy()
    n_sites_all_min = shapvals_df[z_counts].min()
    n_sites_all_max = shapvals_df[z_counts].max()
    subset_all = shapvals_df[[binx, biny, z]].copy()
    if ix == 0:  # NEP is only used to get the scaling numbers
        vmin = subset_all[z].min()
        vmax = subset_all[z].max()
        continue
    subset_all.columns = ['_'.join(col).strip() for col in subset_all.columns.values]  # Heatmap needs flat column index
    plot.flameplot(df=subset_all, fig=fig, ax=axes[ix], cmap=cmap,
                   title=None, vmin=vmin, vmax=vmax,
                   xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
    axes[ix].text(0.1, 0.95, f"({letter[ix]}) All sites, {flux}\n    (n={n_sites_all_max}, min. {n_sites_all_min})",
                  transform=axes[ix].transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
                  ha='left', va='bottom', zorder=99)

# # Load SHAP values aggregated per IGBP
# igbps = ['ENF', 'DBF', 'MF', 'EBF']
# axes = [ax2, ax3, ax4, ax5]
# xlabels = [" ", " ", xlabel, xlabel]
# ylabels = [ylabel, " ", ylabel, " "]
# letter = ['b', 'c', 'd', 'e']
# data_per_igbp = {}
# for ix, i in enumerate(igbps):
#     filepath = Path(
#         results_outdir) / f"4_All-{i}_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}.parquet"
#     igbp_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
#     # keeplocs = igbp_df[z_counts] >= n_sites_min
#     igbp_df = igbp_df[keeplocs].copy()
#     n_sites_igbp_min = igbp_df[z_counts].min()
#     n_sites_igbp_max = igbp_df[z_counts].max()
#     data_per_igbp[i] = igbp_df[[binx, biny, z]].copy()
#     data_per_igbp[i].columns = ['_'.join(col).strip() for col in data_per_igbp[i].columns.values]  # Flat
#     xlabel = xlabel
#     ylabel = ylabel
#     plot.flameplot(df=data_per_igbp[i], fig=fig, ax=axes[ix], cmap=cmap,
#                    title=None, show_colormap=False,
#                    vmin=vmin, vmax=vmax, xlabel=xlabels[ix], ylabel=ylabels[ix])
#     axes[ix].text(0.1, 1, f"({letter[ix]}) {i} (n={n_sites_igbp_max}, min. {n_sites_igbp_min})",
#                   transform=axes[ix].transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
#                   ha='left', va='top', zorder=99)

fig.tight_layout()
fig.show()
