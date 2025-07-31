"""
Flame plot - Attempt to align contours with hexbin data.
"""
from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata

import src.files as files

settings = files.read_settings_file("../config/settings.yaml")

swincol = 'SW_IN_F'
tacol = 'TA_F'
vpdcol = 'VPD_F'

x = tacol
y = vpdcol
z = f"{vpdcol}_SHAPVALS"

binx = f"BIN_{x}"
biny = f"BIN_{y}"

filepath = Path(settings['DIR_DATA_OUT_SHAPVALS_SITE']) / "2_ALLSITES_shap_values_mean.parquet"
df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

# Convert relevant columns to numeric, coercing errors to NaN
df[binx] = pd.to_numeric(df[binx], errors='coerce')
df[biny] = pd.to_numeric(df[biny], errors='coerce')
df[z] = pd.to_numeric(df[z], errors='coerce')

# Drop rows with NaN values in the columns
df.dropna(subset=[binx, biny, z], inplace=True)

# Extract scattered data points
x_scatter = df[binx].values
y_scatter = df[biny].values
z_scatter = df[z].values

# Create a regular grid for interpolation
# The number of points (e.g., 100j) determines the resolution of the grid.
# Adjust min/max based on the actual range of your data.
grid_x, grid_y = np.mgrid[
                 x_scatter.min():x_scatter.max():100j,
                 y_scatter.min():y_scatter.max():100j
                 ]

# Interpolate the SHAP values onto the new grid
grid_z = griddata((x_scatter, y_scatter), z_scatter, (grid_x, grid_y), method='linear')

# Handle NaNs that might result from interpolation (areas outside the original data's convex hull)
grid_z[np.isnan(grid_z)] = np.nan  # Example: fill with 0, adjust as needed

# # Flatten grid coordinates and interpolated values
# # This creates a long format DataFrame
# df_long = pd.DataFrame({
#     'x': grid_x.flatten(),
#     'y': grid_y.flatten(),
#     'interpolated_value': grid_z.flatten()
# })

# df_long = df_long.round(1)

# # hexbin returns a PolyCollection and a ColorbarBase.
fig, ax = plt.subplots(figsize=(10, 8))
# hb = ax.hexbin(df[binx], df[biny], C=df[z], gridsize=40, cmap='RdYlBu_r',
#                reduce_C_function=np.median)
# cbar = fig.colorbar(hb, ax=ax, label='Mean Interpolated Value')

# Isolines (Contour Lines)
# fig, ax = plt.subplots(figsize=(10, 8))

# Plot the filled contours (the colored background)
levels_contourf = np.linspace(np.nanmin(grid_z), np.nanmax(grid_z), 100)  # Many levels for smooth colors
contourf_plot = ax.contourf(grid_x, grid_y, grid_z, levels=levels_contourf, cmap='RdYlBu_r', extend='both', alpha=1)
# cbar = fig.colorbar(contourf_plot, ax=ax, label='mean VPD_F_SHAPVALS (z-score)')

# Define Critical SHAP Value Levels for Isolines
# This is the core of identifying "tipping points."
# You need to determine what SHAP values are "critical" for NEE.
# For example:
#   0.0: No influence of VPD on NEE (or baseline)
#   0.2: Beginning of noticeable positive influence (VPD starts to matter)
#   0.5: Significant positive influence (VPD becomes a major driver, likely stress)
#   1.0: Very strong positive influence (extreme VPD stress/control)
critical_shap_levels = [-0.5, 0, 0.5, 1]

# Filter out levels that are outside the range of the interpolated data to avoid errors
valid_critical_levels = [level for level in critical_shap_levels if np.nanmin(grid_z) <= level <= np.nanmax(grid_z)]

if valid_critical_levels:
    # Plot isolines at the critical thresholds
    # `levels` for contour defines the specific values at which lines will be drawn.
    contours = ax.contour(
        grid_x, grid_y, grid_z,
        levels=valid_critical_levels,
        colors='black',  # Color of the lines
        linestyles='--',  # Style of the lines (dashed)
        linewidths=2  # Thickness of the lines
    )

    # Label the isolines to show their value
    ax.clabel(contours, inline=True, fontsize=10, fmt='%1.1f')
else:
    print("No valid critical SHAP levels found within the data range to plot contours.")
    print(f"Data SHAP range: [{np.nanmin(grid_z):.2f}, {np.nanmax(grid_z):.2f}]")

# Customize plot
ax.set_title('Interactive Thresholds of VPD Influence on NEE')
ax.set_xlabel(f'{binx} (z-score)')
ax.set_ylabel(f'{biny} (z-score)')
ax.set_aspect('equal')  # Or 'equal' if physical units require it

# Add grid lines
# plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.show()

# Save plot
# plt.savefig('vpd_shap_isolines_with_thresholds.png', dpi=300)
# print("Plot saved as 'vpd_shap_isolines_with_thresholds.png'")
