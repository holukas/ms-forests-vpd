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
FLUX = 'NEP'
# FLUX = 'ET'
# FLUX = 'GPP'
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
plot.flameplot(df=subset_all, fig=fig, ax=ax_all, cmap=cmap,
               title=None, cb_digits_after_comma=cb_digits_after_comma,
               xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)
ax_all.text(0.1, 0.95, f"(a) All sites (n={n_sites_used}, min. {n_sites_all_min})",
            transform=ax_all.transAxes, color='black', size=theme.AX_LABELS_FONTSIZE,
            ha='left', va='bottom', zorder=99)

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
    # keeplocs = igbp_df[z_counts] >= 1
    # Next line uses the same keeplocs like above, i.e. for each site we
    # get the same locations as for the overall (all sites) plot.
    igbp_df = igbp_df[keeplocs].copy()

    # XXX
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
                  ha='left', va='top', zorder=99)

fig.tight_layout()
fig.show()
