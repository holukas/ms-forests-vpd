"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt

import src.files as files

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'RECO'
xvar = 'TA'
yvar = 'VPD'
zvar = 'VPD'
aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Heatmap settings
title = f"The effect of {zvar} on {FLUX}"
n_sites_min = 20
# cmap = 'RdYlBu'
cmap = 'RdYlBu_r'
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
filepath = Path(results_outdir) / f"3_AllSites_Aggregated_SHAPValues-{shap_type}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
keeplocs = shapvals_df[z_counts] >= n_sites_min
shapvals_df = shapvals_df[keeplocs].copy()
subset = shapvals_df[[binx, biny, z]].copy()

# Heatmap needs flat column index
# Create new flattened column names by joining the MultiIndex levels
subset.columns = ['_'.join(col).strip() for col in subset.columns.values]

fig, ax = plt.subplots(figsize=(8, 6), dpi=300, facecolor="white")

# Heatmap
hm = dv.heatmapxyz(
    ax=ax,
    title="All sites",
    x=subset.iloc[:, 0],
    y=subset.iloc[:, 1],
    z=subset.iloc[:, 2],
    cb_digits_after_comma=1,
    xlabel=f'{binx} (z-score)',
    ylabel=f'{biny} (z-score)',
    zlabel=f'{aggfunc} {z} (z-score)',
    # show_values_n_dec_places=1,
    # show_values=True,
    # show_values_fontsize=4,
    figdpi=300,
    color_bad='white',
    cmap=cmap,
    # vmin=-3,
    # vmax=3
)
hm.plot()
hm.export_borderless_heatmap(
    name="TEST",
    outpath=r"F:\Sync\luhk_work\20 - CODING\29 - WORKBENCH\ms_co2_penalty\data\outputs\borderless_heatmaps")
# ax = hm.get_ax()

# ax.set_xlabel('Air temperature (z-score)')
# ax.set_ylabel("Vapor pressure deficit (z-score)")
ax.set_title(title, fontsize=14, pad=10, y=1.02)

# Hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.tick_params(axis='both', which='major', width=1, length=5)
ax.tick_params(axis='both', which='minor', width=1, length=2)
# ax.axvline(x=0, color='black', linestyle='-', lw=99)
fig.show()
