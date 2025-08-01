"""
Flame plot - Highlight z values > 1.
"""
from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# from scipy.interpolate import griddata # Not directly used in this version for hexbin, but kept for context

import src.files as files
from matplotlib.colors import Normalize, LinearSegmentedColormap

settings = files.read_settings_file("../config/settings.yaml")

swincol = 'SW_IN_F'
tacol = 'TA_F'
vpdcol = 'VPD_F'

x = tacol
y = vpdcol
z = f"{vpdcol}_SHAPVALS"

binx = f"BIN_{x}"
biny = f"BIN_{y}"

filepath = Path(settings['DIR_DATA_OUT_SHAPVALS_STANDARD']) / "2_ALLSITES_shap_values_median.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

keeplocs = shapvals_df['VPD_F_SHAPVALS_COUNTS'] >= 30
shapvals_df = shapvals_df[keeplocs].copy()

# Convert relevant columns to numeric, coercing errors to NaN
shapvals_df[binx] = pd.to_numeric(shapvals_df[binx], errors='coerce')
shapvals_df[biny] = pd.to_numeric(shapvals_df[biny], errors='coerce')
shapvals_df[z] = pd.to_numeric(shapvals_df[z], errors='coerce')

# Drop rows with NaN values in the columns
shapvals_df.dropna(subset=[binx, biny, z], inplace=True)

# hexbin returns a PolyCollection and a ColorbarBase.
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_facecolor("gray")

hb = ax.hexbin(shapvals_df[binx], shapvals_df[biny], C=shapvals_df[z], gridsize=20, cmap='RdYlBu_r',
               # reduce_C_function=np.count_nonzero,
               reduce_C_function=np.median,
               mincnt=1)
cbar = fig.colorbar(hb, ax=ax, label='z')

# # Get the paths (vertices) and values of each hexagon
# paths = hb.get_paths()
# values = hb.get_array()
#
# # Define a threshold to highlight bins
# threshold = np.percentile(values, 80)  # Highlight top 20% by count
#
# # Create a new PolyCollection for the highlighted outlines
# highlighted_verts = []
# for i, count in enumerate(values):
#     if count >= threshold:
#         highlighted_verts.append(paths[i].vertices)

# Customize plot
ax.set_title('Interactive Thresholds of VPD Influence on NEE')
ax.set_xlabel(f'{binx} (z-score)')
ax.set_ylabel(f'{biny} (z-score)')
ax.set_aspect('equal')  # Or 'equal' if physical units require it
ax.axhline(y=0, color='black', linestyle='-')
ax.axvline(x=0, color='black', linestyle='-')

# Add grid lines
# plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Save plot
# plt.savefig('vpd_shap_isolines_with_thresholds.png', dpi=300)
# print("Plot saved as 'vpd_shap_isolines_with_thresholds.png'")
