"""
Flame plot.
"""
from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import numpy as np

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
aggfunc = 'median'

filepath = Path(settings['DIR_DATA_OUT_SHAPVALS_STANDARD']) / "2_ALLSITES_shap_values_median.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

n_sites_min = 20
keeplocs = shapvals_df['VPD_F_SHAPVALS_COUNTS'] >= n_sites_min
shapvals_df = shapvals_df[keeplocs].copy()

# # Heatmap
# hm = dv.heatmapxyz(
#     title="All sites",
#     x=shapvals_df[binx],
#     y=shapvals_df[biny],
#     z=shapvals_df[z],
#     # z=shapvals_df['VPD_F_SHAPVALS_COUNTS'],
#     cb_digits_after_comma=1,
#     xlabel=f'{binx} (z-score)',
#     ylabel=f'{biny} (z-score)',
#     zlabel=f'{aggfunc} {z} (z-score)',
#     # show_values_n_dec_places=1,
#     # show_values=True,
#     # show_values_fontsize=4,
#     figdpi=300,
#     # vmin=-3,
#     # vmax=3
# )
# hm.show()

# Scatter with z colors
# plt.scatter(shapvals_df['BIN_VPD_F'], shapvals_df['VPD_F_SHAPVALS'], c=shapvals_df['BIN_TA_F'])
# plt.show()

# Assuming your DataFrame 'df' is already loaded with 'vpd_zscore' and 'shap_value' columns

# Extract the data from the DataFrame
X_data = shapvals_df['BIN_VPD_F'].values
Y_data = shapvals_df['VPD_F_SHAPVALS'].values

# Fit a 4th degree polynomial (you can change the degree)
degree = 4
poly_coeffs = np.polyfit(X_data, Y_data, degree)
poly_func = np.poly1d(poly_coeffs)

# Create a smooth x-range for plotting the fitted curve
x_fit = np.linspace(X_data.min(), X_data.max(), 500)

# Colors: https://www.pinterest.com/pin/914862421161100/
# Colors: https://www.pinterest.com/pin/11118330331981167/

fig, ax = plt.subplots(figsize=(8, 6))

# Plot the original data and the fitted polynomial
ax.scatter(X_data, Y_data, label=f'Aggregated site data (min. {n_sites_min} sites)',
           alpha=0.3, s=40, color="#6c757d", edgecolors='none')
           # alpha=0.6, s=20, color="none", edgecolors='#1982C4')
           # alpha=0.33, color="none", edgecolors='#369CBB')

yerrlow = shapvals_df['VPD_F_SHAPVALS'].sub(shapvals_df['VPD_F_SHAPVALS_P25'])
yerrhigh = shapvals_df['VPD_F_SHAPVALS_P75'].sub(shapvals_df['VPD_F_SHAPVALS'])
yerr_iqr = [yerrlow.to_numpy(), yerrhigh.to_numpy()]
plt.errorbar(X_data, Y_data, yerr=yerr_iqr, fmt='none', capsize=1, elinewidth=0,
             ecolor='#6c757d', markerfacecolor='none', markersize=0, alpha=0.5,
             label="IQR")

ax.plot(x_fit, poly_func(x_fit),
        label=f'Fitted {degree}rd Degree Polynomial',
        color='#004e98', linewidth=3)

y_fit = poly_func(x_fit)
min_ix = np.argmin(y_fit)
max_ix = np.argmax(y_fit)
# Find value closest to "SHAP zero"
idx = (np.abs(y_fit - 0)).argmin()

ax.scatter(x_fit[max_ix], y_fit[max_ix], label=f'Max (x={x_fit[max_ix]:.2f})', marker='^',
           alpha=1, s=120, c="white", zorder=99, edgecolors='#e63946', linewidths=2)
           # alpha=1, s=120, c="white", zorder=99, edgecolors='#FF595E', linewidths=2)
# alpha=1, s=100, c="white", zorder=99, edgecolors='#D05B61', linewidths=2)
ax.scatter(x_fit[idx], y_fit[idx], label=f'SHAP zero (x={x_fit[idx]:.2f})',
           alpha=1, s=100, c="white", zorder=99, edgecolors='#f77f00', linewidths=3)
           # alpha=1, s=100, c="white", zorder=99, edgecolors='#FFCA3A', linewidths=3)
# alpha=1, s=90, c="white", zorder=99, edgecolors='#FFA72C', linewidths=3)
# alpha=1, s=70, c="white", zorder=99, edgecolors='#fdaf62', linewidths=3)
ax.scatter(x_fit[min_ix], y_fit[min_ix], label=f'Min (x={x_fit[min_ix]:.2f})', marker='v',
           # alpha=1, s=100, c="#8AC926", zorder=99, edgecolors='k', linewidths=2)
           alpha=1, s=120, c="white", zorder=99, edgecolors='#66bb6a', linewidths=2)
           # alpha=1, s=120, c="white", zorder=99, edgecolors='#8AC926', linewidths=2)
# alpha=1, s=100, c="white", zorder=99, edgecolors='#369CBB', linewidths=2)

# 426cb0
ax.set_xlabel('VPD Z-score')
ax.set_ylabel('SHAP value of VPD')
# ax.set_title('Fitted Polynomial with DataFrame')
ax.axhline(y=0, color='black', linestyle='-', lw=1)
ax.grid(False)
ax.legend(frameon=False, loc='upper left')
# Hide the top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1)
ax.spines['left'].set_linewidth(1)
# Set the tick width for both x and y axes
ax.tick_params(axis='both', which='major', width=1, length=5)
ax.tick_params(axis='both', which='minor', width=1, length=2)
fig.show()

print(f"Polynomial coefficients: {poly_coeffs}")
