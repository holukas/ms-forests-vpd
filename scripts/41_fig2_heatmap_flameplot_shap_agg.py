"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

import src.files as files
import src.plot as plot

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'LE'
xvar = 'TA'
yvar = 'VPD'
zvar = 'VPD'
aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Heatmap settings
title = f"The effect of {zvar} on {FLUX}"
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
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Load SHAP values aggregated across all sites
filepath = Path(results_outdir) / f"3_AllSites_Aggregated_SHAPValues-{shap_type}_BIN-{xvar}_BIN-{yvar}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
keeplocs = shapvals_df[z_counts] >= n_sites_min
shapvals_df = shapvals_df[keeplocs].copy()
subset = shapvals_df[[binx, biny, z]].copy()

# Heatmap needs flat column index
# Create new flattened column names by joining the MultiIndex levels
subset.columns = ['_'.join(col).strip() for col in subset.columns.values]

fig = plt.figure(figsize=(36, 18), dpi=72, facecolor="white")
gs = gridspec.GridSpec(2, 4)  # rows, cols
gs.update(wspace=.2, hspace=.1, left=0.1, right=0.9, top=0.9, bottom=0.1)
ax1 = fig.add_subplot(gs[0:2, 0:2])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[0, 3])
ax4 = fig.add_subplot(gs[1, 2])
ax5 = fig.add_subplot(gs[1, 3])

plot.flameplot(
    df=subset, fig=fig, ax=ax1, cmap=cmap,
    title=title, xlabel=xlabel, ylabel=ylabel, zlabel=zlabel)

fig.show()
