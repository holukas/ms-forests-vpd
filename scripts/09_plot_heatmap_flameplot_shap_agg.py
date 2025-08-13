"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt

import src.files as files

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

settings = files.read_settings_file("../config/settings.yaml")

swincol = 'SW_IN_F'
tacol = 'TA_F'
vpdcol = 'VPD_F'
swccol = 'SWC_F_MDS_1'

# Used variable names, from aggregation across sites
binx = (f"BIN_{tacol}", "median")
# biny = (f"{swccol}", "median")
biny = (f"BIN_{vpdcol}", "median")
z = (f"{vpdcol}_SHAPVALS", "median")
z_counts = (f"{vpdcol}_SHAPVALS", "count")
conditional = True

# Heatmap settings
n_sites_min = 20
aggfunc = 'median'

# --------------------------------

pathstr = 'DIR_DATA_OUT_SHAPVALS_CONDITIONAL' if conditional else 'DIR_DATA_OUT_SHAPVALS_STANDARD'
filepath = Path(settings[pathstr]) / "3_AllSites_Aggregated_SHAPValues.parquet"
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
    # vmin=-3,
    # vmax=3
)
hm.plot()
# ax = hm.get_ax()

# ax.set_xlabel('Air temperature (z-score)')
# ax.set_ylabel("Vapor pressure deficit (z-score)")
ax.set_title('The effect of VPD on CO$_2$ uptake and release', fontsize=14, pad=10, y=1.02)

# Hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
ax.tick_params(axis='both', which='major', width=1, length=5)
ax.tick_params(axis='both', which='minor', width=1, length=2)
# ax.axvline(x=0, color='black', linestyle='-', lw=99)
fig.show()
