"""
Flame plot.

Colors: https://www.pinterest.com/pin/914862421161100/
Colors: https://www.pinterest.com/pin/11118330331981167/
"""
from pathlib import Path

import diive as dv
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.files as files
from src.fit import fit_polynomial

# Recommended for scientific publications
# plt.rcParams['text.usetex'] = True
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = 'Computer Modern Roman'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'NEP'
xvar = 'VPD'
yvar = 'VPD'  # SHAP values
aggfunc = 'median'
CONDITIONAL = True  # SHAP

filename_x = 'TA'
zvar = 'TA'  # Colors

descriptive_flux = {
    'NEP': 'net CO$_{2}$ exchange',
}

# Plot settings
title = f"The impact of {yvar} on forest {descriptive_flux[FLUX]}"
xlabel = f"{xvar} (z-score)"
ylabel = f"Median impact of {yvar} (z-score)"
n_sites_min = 30

show_txt_effect = True
show_shap_thresholds = True
show_z_colors = True
show_fit = True
# ------------------------------

x = (f"BIN_{xvar}", aggfunc)
y = (f"{yvar}_SHAPVALS", aggfunc)
y_counts = (f"{yvar}_SHAPVALS", "count")
z = (f"BIN_{zvar}", aggfunc)

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Load SHAP values aggregated across all sites
filepath = Path(
    results_outdir) / f"3_AllSites_Aggregated_SHAPValues-{shap_type}_BIN-{filename_x}_BIN-{yvar}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
keeplocs = shapvals_df[y_counts] >= n_sites_min
shapvals_df = shapvals_df[keeplocs].copy()

# Extract the data from the DataFrame
X_data = shapvals_df[x].values
Y_data = shapvals_df[y].values
Z_data = shapvals_df[z].values

# Bin z data into 5 categories
bin_labels = ['Lowest', 'Low', 'Medium', 'High', 'Highest']
binned_z = pd.cut(Z_data, bins=5, labels=bin_labels)
# z_series = shapvals_df[z].copy()
# z_binned = pd.cut(z_series, bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Fit polynomial
poly_func, poly_coeffs, x_fit, y_fit, r_squared, pi_upper, pi_lower = fit_polynomial(X_data=X_data, Y_data=Y_data)

# Custom colormap for z-values

colors_list = ['#9C27B0', '#26C6DA', '#546E7A', '#FB8C00', '#C62828']
# colors_list = ['#2b83ba', '#26C6DA', '#546E7A', '#FF9800', '#C62828']
# colors_list = ['#1565C0', '#26C6DA', '#546E7A', '#FF9800', '#C62828']
# colors_list = ['#2b83ba', '#abdda4', '#ffffbf', '#fdae61', '#d7191c']

# colors_list = ['#08519c', '#7bccc4', '#f7f7f7', '#fec44f', '#ec7014']
# colors_list = ['#440154', '#31688e', '#35b779', '#fde725', '#bada55']
# colors_list = ['#00796B', '#80CBC4', '#ffe0b2', '#F4A460', '#D84315']
custom_cmap = matplotlib.colors.ListedColormap(colors_list)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Iterate through each temperature bin and plot the corresponding data points
scatterhandles = []
for i, label in reversed(list(enumerate(bin_labels))):
    indices = np.where(binned_z.codes == i)[0]
    # Use color and edgecolors with the same color, but adjust alpha

    if show_z_colors:
        fill_color = colors_list[i]
        edge_color = fill_color
    else:
        fill_color = '#546E7A'
        edge_color = '#546E7A'

    scatterplot = ax.scatter(X_data[indices], Y_data[indices],
                             label=f'{label} {zvar}',
                             alpha=0.4,
                             s=30,
                             color=fill_color,
                             edgecolors=edge_color,
                             zorder=98)
    scatterhandles.append(scatterplot)

# Legend for scatter points
legend1 = ax.legend(handles=scatterhandles, loc='upper left', bbox_to_anchor=(0.38, 1.02),
                        frameon=False, ncol=1, fontsize=9, labelspacing=.3, title="Aggregated site data",
                        title_fontsize=9)

if show_fit:
    # Plot fitted polynomial curve
    ax.plot(x_fit, y_fit, color='#004e98', linewidth=3, zorder=99)
    # label=rf'$y = {poly_coeffs[0]:.4f}x^4 - {poly_coeffs[1]:.4f}x^3 + {poly_coeffs[2]:.4f}x^2 + {poly_coeffs[3]:.4f}x - {poly_coeffs[4]:.4f}$'

    # Plot prediction interval
    fillbetweenplot = ax.fill_between(x_fit, pi_lower, pi_upper, color='#004e98', alpha=0.2,
                                      label='95% prediction interval', zorder=1)

    # Add an arrow to the fitted line
    # Find a point on the line to place the arrow.
    # Let's place it a little past the middle of the x-range.
    arrow_x = 3
    arrow_y = poly_func(arrow_x)
    # Find a point slightly to the left to define the arrow direction
    tail_x = arrow_x - 0.1
    tail_y = poly_func(tail_x)
    # Calculate the angle of the line at this point to get the correct arrow orientation
    angle = np.arctan2(arrow_y - tail_y, arrow_x - tail_x) * 180 / np.pi
    ax.annotate(f'Fitted 4th degree\npolynomial (r$^2$={r_squared:.2f})',
                xy=(arrow_x, arrow_y),
                xytext=(arrow_x - 0.2, arrow_y + 0.4),  # Adjust text position as needed
                arrowprops=dict(arrowstyle="->", color='#004e98', lw=1.5),
                fontsize=9, color='#004e98', ha='left', va='center', zorder=100)
else:
    fillbetweenplot = None

# IQR
iqr_low25 = shapvals_df[(f"{yvar}_SHAPVALS", "<lambda_0>")]
iqr_high75 = shapvals_df[(f"{yvar}_SHAPVALS", "<lambda_1>")]
plotparams = dict(marker='o', s=5, zorder=1, alpha=.2, edgecolors='none', color='#6c757d')
iqrplot = ax.scatter(X_data, iqr_low25, label="IQR", **plotparams)
ax.scatter(X_data, iqr_high75, **plotparams)

# Legend 2 for IQR and fill_between plot
if show_fit:
    handles = [fillbetweenplot, iqrplot]
else:
    handles = [iqrplot]

legend2 = ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.65, 1.02),
                    frameon=False, ncol=1, labelspacing=.3, fontsize=9)

# Add Legend 1 back to the figure.
# This is the crucial step to prevent the first legend from being removed.
ax.add_artist(legend1)

min_ix = np.argmin(y_fit)
max_ix = np.argmax(y_fit)

# Find value closest to "SHAP zero"
idx = (np.abs(y_fit - 0)).argmin()

# Detect min/max value shown in plot
_temp = iqr_high75.max() * 1.2
_temp2 = iqr_low25.min()
ax.set_ylim(_temp2, _temp)

# Add text and connecting dashed lines for SHAP max, zero, and min
if show_shap_thresholds:
    y_top = ax.get_ylim()[-1]
    y_bottom = ax.get_ylim()[0]

    _params = dict(color='black', linestyle='--', linewidth=1, zorder=100)
    _params2 = dict(linewidth=2, zorder=100, s=100, alpha=1)
    _params3 = dict(color='black', fontsize=9, linespacing=1.2)

    # Maximum positive impact
    # ax.scatter(x_fit[max_ix], y_fit[max_ix], color='black', marker='^', edgecolors='none', **_params2)
    ax.scatter(x_fit[max_ix], y_fit[max_ix], color='none', marker='^', edgecolor='black', **_params2)
    text_y_pos_max = y_fit[max_ix] - 0.45 * (_temp - _temp2)  # Get the y-position for the text below the plotted points
    ax.text(x_fit[max_ix], text_y_pos_max, f'Maximum positive impact\n(x={x_fit[max_ix]:.2f})',
            va='bottom', ha='center', **_params3)
    ax.plot([x_fit[max_ix], x_fit[max_ix]], [text_y_pos_max + 0.1, y_fit[max_ix]], **_params)
    ax.plot([x_fit[max_ix], x_fit[max_ix]], [y_bottom, text_y_pos_max], **_params)

    # Impact tipping point
    ax.scatter(x_fit[idx], y_fit[idx], c="none", edgecolors='black', **_params2)
    text_y_pos_zero = y_fit[idx] - 0.45 * (_temp - _temp2)
    ax.text(x_fit[idx], text_y_pos_zero, f'Impact tipping point\n(x={x_fit[idx]:.2f})',
            va='bottom', ha='center', **_params3)
    ax.plot([x_fit[idx], x_fit[idx]], [text_y_pos_zero + 0.1, y_fit[idx] - 0.03], **_params)
    ax.plot([x_fit[idx], x_fit[idx]], [y_bottom, text_y_pos_zero], **_params)

    # Maximum negative impact
    # ax.scatter(x_fit[min_ix], y_fit[min_ix], color='black', marker='_', edgecolors='none', **_params2)
    ax.scatter(x_fit[min_ix], y_fit[min_ix], color='none', marker='v', edgecolor='black', **_params2)
    text_y_pos_min = y_fit[min_ix] - 0.15 * (_temp - _temp2)
    ax.text(x_fit[min_ix] + 0.2, text_y_pos_min, f'Maximum negative impact\n(x={x_fit[min_ix]:.2f})',
            va='bottom', ha='right', **_params3)
    ax.plot([x_fit[min_ix], x_fit[min_ix]], [text_y_pos_min + 0.1, y_fit[min_ix] - 0.03], **_params)
    ax.plot([x_fit[min_ix], x_fit[min_ix]], [y_bottom, text_y_pos_min], **_params)

# Add arrow to highlight one of the IQR data points
select_x = 1.8
locations1 = (X_data == select_x)  # Create a boolean mask for locations where X_data is 1.7
filtered_iqr = iqr_low25[locations1]  # Filter iqr_low25 using the boolean mask
min_iqr_value = np.min(filtered_iqr)  # Find the minimum value in the filtered array

# Combine both conditions: X_data is *select_x* AND iqr_low25 is the minimum value
final_location_mask = (X_data == select_x) & (iqr_low25 == min_iqr_value)
iqr_point_index = np.where(final_location_mask)[0][0]  # Get the index of the element that meets both criteria
# iqr_point_index = 1  # Choose a representative index
# iqr_point_index = iqr_point_index if iqr_point_index < len(X_data) else 0
iqr_x = float(X_data[iqr_point_index])
iqr_y = float(iqr_low25[iqr_point_index])
# iqr_y = float(iqr_low25[iqr_point_index])

ax.annotate(
    f'IQR for site data',
    xy=(iqr_x, iqr_y),
    xytext=(iqr_x + 0, iqr_y - 0.2),  # Adjust text position as needed
    arrowprops=dict(arrowstyle="->", color='#6c757d', lw=1.5),
    fontsize=9, color='#6c757d', ha='right', va='center'
)

if show_txt_effect:
    # Add text for negative effect
    ax.text(x=2.1, y=-0.03, s='reduced uptake/increased release\n' + r'$\downarrow$Negative impact',
            fontsize=9, color='black', ha='left', va='top')

    # Add text for positive effect
    # _text =
    ax.text(x=2.1, y=0.03, s=r'$\uparrow$' + 'Positive impact\nincreased uptake/reduced release',
            fontsize=9, color='black', ha='left', va='bottom')

ax.set_xlabel(xlabel)
ax.set_ylabel(ylabel)
ax.set_title(title, fontsize=14, pad=10, y=1.02)
ax.axhline(y=0, color='black', linestyle='-', lw=1, zorder=98)
ax.grid(False)

# ax.legend(bbox_to_anchor=(0.05, 0.95), loc='upper right', frameon=False)
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
