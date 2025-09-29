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
# NEP, ET, GPP, RECO, TA, VPD, SWIN, SWC
FLUXES = ['NEP', 'GPP', 'RECO', 'ET']
xvars = ['SWC', 'TA', 'TA', 'TA']
yvars = ['VPD', 'VPD', 'VPD', 'VPD']
zvars = ['VPD', 'VPD', 'VPD', 'VPD']
aggfunc = 'median'
CONDITIONAL = True  # SHAP
zvar_is_shap = False

# Heatmap settings
n_sites_min = 30
cmap = 'RdYlBu'
# cmap = 'RdYlBu_r'
xlabels = [f'{xvar} (z-score)' for xvar in xvars]
ylabels = [f'{yvar} (z-score)' for yvar in yvars]
zlabels = [f'{aggfunc} SHAP value of {zvar} (z-score)' for zvar in zvars]
binsx = [(f"BIN_{xvar}", aggfunc) for xvar in xvars]
binsy = [(f"BIN_{yvar}", aggfunc) for yvar in yvars]

if zvar_is_shap:
    zs = [(f"{zvar}_SHAPVALS", aggfunc) for zvar in zvars]
    zs_counts = [(f"{zvar}_SHAPVALS", "count") for zvar in zvars]
else:
    zs = [(f"{zvar}", aggfunc) for zvar in zvars]
    zs_counts = [(f"{zvar}", "count") for zvar in zvars]

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'

# Start figure
fig = plt.figure(figsize=(30.4, 6.4), dpi=150, facecolor="white")
gs = gridspec.GridSpec(1, 21)  # rows, cols
# gs.update(wspace=.3, hspace=.2, left=0.03, right=0.94, top=0.97, bottom=0.04)
ax_nep_swc_vpd = fig.add_subplot(gs[0, 0:5])
ax_gpp_ta_vpd = fig.add_subplot(gs[0, 5:10], sharey=ax_nep_swc_vpd)
ax_reco_ta_vpd = fig.add_subplot(gs[0, 10:15], sharex=ax_gpp_ta_vpd, sharey=ax_nep_swc_vpd)
ax_et_ta_vpd = fig.add_subplot(gs[0, 15:20], sharex=ax_gpp_ta_vpd, sharey=ax_nep_swc_vpd)
ax_cbar = fig.add_subplot(gs[0, 20])
axes = [ax_nep_swc_vpd, ax_gpp_ta_vpd, ax_reco_ta_vpd, ax_et_ta_vpd]
letter = ['a', 'b', 'c', 'd']
vmin = None  # Will be detected from NEP below
vmax = None
p = None

for ix, flux in enumerate(FLUXES):
    # Load SHAP values aggregated across all sites
    results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / flux / shap_type
    filepath = Path(
        results_outdir) / f"3_AllSites_Aggregated_SHAPValues-{shap_type}_BIN-{xvars[ix]}_BIN-{yvars[ix]}_{flux}.parquet"
    shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
    keeplocs = shapvals_df[zs_counts[ix]] >= n_sites_min
    shapvals_df = shapvals_df[keeplocs].copy()
    n_sites_all_min = shapvals_df[zs_counts[ix]].min()
    n_sites_all_max = shapvals_df[zs_counts[ix]].max()
    subset_all = shapvals_df[[binsx[ix], binsy[ix], zs[ix]]].copy()
    if ix == 0:  # NEP used to get the scaling numbers
        vmin = subset_all[zs[ix]].min()
        vmax = subset_all[zs[ix]].max()
    subset_all.columns = ['_'.join(col).strip() for col in subset_all.columns.values]  # Heatmap needs flat column index
    p = plot.flameplot(df=subset_all, fig=fig, ax=axes[ix], cmap=cmap,
                       title=None, vmin=vmin, vmax=vmax, show_colormap=False,
                       xlabel=xlabels[ix], ylabel=ylabels[ix], zlabel=zlabels[ix], show_grid=False)
    axes[ix].text(0.1, 0.95, f"({letter[ix]}) All sites, {flux} (n=171, min. {n_sites_all_min})",
                  transform=axes[ix].transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
                  ha='left', va='bottom', zorder=99)
    axes[ix].set_aspect('equal')
    # Create the colorbar on the dedicated axis

# Colorbar for all subplots
cbar = fig.colorbar(p, cax=ax_cbar, label='XXX')
cbar.ax.tick_params(labelsize=theme.AX_LABELS_FONTSIZE)
cbar.set_label(zlabels[0], fontsize=theme.AX_LABELS_FONTSIZE, labelpad=20)

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
