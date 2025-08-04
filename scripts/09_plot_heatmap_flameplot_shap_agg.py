"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import numpy as np

import src.files as files

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

settings = files.read_settings_file("../config/settings.yaml")

swincol = 'SW_IN_F'
tacol = 'TA_F'
vpdcol = 'VPD_F'
swccol = 'SWC_F_MDS_1'

x = tacol
y = vpdcol
z = f"{vpdcol}_SHAPVALS"

binx = f"BIN_{x}"
biny = f"BIN_{y}"
aggfunc = 'median'

filepath = Path(settings['DIR_DATA_OUT_SHAPVALS_STANDARD']) / "3_AllSites_Aggregated_SHAPValues.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

n_sites_min = 20
keeplocs = shapvals_df[f'{z}_COUNTS'] >= n_sites_min
shapvals_df = shapvals_df[keeplocs].copy()

fig, ax = plt.subplots(figsize=(8, 6), dpi=300, facecolor="white")

# Heatmap
hm = dv.heatmapxyz(
    ax=ax,
    title="All sites",
    x=shapvals_df[binx],
    y=shapvals_df[biny],
    z=shapvals_df[z],
    # z=shapvals_df['VPD_F_SHAPVALS_COUNTS'],
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

ax.set_xlabel('Air temperature (z-score)')
ax.set_ylabel("Vapor pressure deficit (z-score)")
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
