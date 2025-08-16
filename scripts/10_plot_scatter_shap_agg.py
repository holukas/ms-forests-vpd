"""
Flame plot.

Colors: https://www.pinterest.com/pin/914862421161100/
Colors: https://www.pinterest.com/pin/11118330331981167/
"""
from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

import src.files as files

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
yvar = 'VPD'
aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Heatmap settings
n_sites_min = 20
# ------------------------------

x = (f"BIN_{xvar}", aggfunc)
y = (f"{yvar}_SHAPVALS", aggfunc)
y_counts = (f"{yvar}_SHAPVALS", "count")

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Load SHAP values aggregated across all sites
filepath = Path(results_outdir) / f"3_AllSites_Aggregated_SHAPValues-{shap_type}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
keeplocs = shapvals_df[y_counts] >= n_sites_min
shapvals_df = shapvals_df[keeplocs].copy()

# Extract the data from the DataFrame
X_data = shapvals_df[x].values
Y_data = shapvals_df[y].values

# Fit polynomial
degree = 4
# poly_coeffs = np.polyfit(X_data, Y_data, degree)
poly_coeffs, residuals, _, _, _ = np.polyfit(X_data, Y_data, degree, full=True)
poly_func = np.poly1d(poly_coeffs)

# Calculate r2
ss_res = residuals[0]  # The sum of squared residuals (SS_res) is the first element of the residuals array
y_mean = np.mean(Y_data)
ss_tot = np.sum((Y_data - y_mean) ** 2)  # Calculate the total sum of squares (SS_tot)
r_squared = 1 - (ss_res / ss_tot)
# print(f"The R-squared value is: {r_squared:.4f}")


# Create x-range for plotting fitted curve and create fit values
x_fit = np.linspace(X_data.min(), X_data.max(), 500)
y_fit = poly_func(x_fit)

# Calculate the prediction interval
# ----------------------------------------------------
n = len(X_data)  # Number of data points
p = degree + 1  # Number of parameters (coefficients)
alpha = 0.05  # 95% prediction interval

# Calculate Mean Squared Error (MSE)
mse = residuals[0] / (n - p)

# Create Vandermonde matrix for the original data
X_vander = np.vander(X_data, p)

# Calculate the covariance matrix of the coefficients
covariance_matrix = mse * np.linalg.inv(X_vander.T @ X_vander)

# Create Vandermonde matrix for the fitted line
x_fit_vander = np.vander(x_fit, p)

# Calculate the standard error of the fitted line at each point
se_fit = np.sqrt(np.diag(x_fit_vander @ covariance_matrix @ x_fit_vander.T))

# Calculate the standard error for prediction intervals (adds the MSE)
se_pred = np.sqrt(se_fit ** 2 + mse)

# Get the t-statistic for a 95% confidence level
t_value = stats.t.ppf(1 - alpha / 2, n - p)

# Calculate the prediction interval bounds
pi_upper = y_fit + t_value * se_pred
pi_lower = y_fit - t_value * se_pred
# ----------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# Plot the original data and the fitted polynomial
ax.scatter(X_data, Y_data, label=f'Aggregated site data (min. {n_sites_min} sites)',
           alpha=0.4, s=40,
           # color="#004e98",
           color="#6c757d",
           # c=shapvals_df['BIN_TA_F'],
           # cmap='RdYlBu_r',
           edgecolors='none')

# Plot fitted polynomial curve
ax.plot(x_fit, y_fit,
        # label=f'Fitted {degree}th degree polynomial',
        color='#004e98', linewidth=3)
# label=rf'$y = {poly_coeffs[0]:.4f}x^4 - {poly_coeffs[1]:.4f}x^3 + {poly_coeffs[2]:.4f}x^2 + {poly_coeffs[3]:.4f}x - {poly_coeffs[4]:.4f}$'

# Plot prediction interval
ax.fill_between(x_fit, pi_lower, pi_upper, color='#004e98', alpha=0.2,
                label='95% prediction interval', zorder=1)

# IQR
# yerrlow = shapvals_df[y].sub(shapvals_df[(yvar, "<lambda_0>")])  # <lambda_0> is P25
# yerrhigh = shapvals_df[(yvar, "<lambda_1>")].sub(shapvals_df[y])  # <lambda_0> is P75
iqr_low25 = shapvals_df[(f"{yvar}_SHAPVALS", "<lambda_0>")]
iqr_high75 = shapvals_df[(f"{yvar}_SHAPVALS", "<lambda_1>")]
plotparams = dict(marker='o', s=5, zorder=1, alpha=.2, edgecolors='none', color='#6c757d')
ax.scatter(X_data, iqr_low25, label="IQR", **plotparams)
ax.scatter(X_data, iqr_high75, **plotparams)
# yerr_iqr = [yerrlow.to_numpy(), yerrhigh.to_numpy()]
# ax.errorbar(X_data, Y_data, yerr=yerr_iqr, fmt='none', capsize=1, elinewidth=0,
#             ecolor='#6c757d', markerfacecolor='none', markersize=0, alpha=0.5,
#             label="IQR",
# capthick=1)

min_ix = np.argmin(y_fit)
max_ix = np.argmax(y_fit)
# Find value closest to "SHAP zero"
idx = (np.abs(y_fit - 0)).argmin()

ax.scatter(x_fit[max_ix], y_fit[max_ix],
           # label=f'SHAP max (x={x_fit[max_ix]:.2f})',
           marker='^', alpha=1, s=120, c="white", zorder=99, edgecolors='#e63946', linewidths=2)

ax.scatter(x_fit[idx], y_fit[idx],
           # label=f'SHAP zero (x={x_fit[idx]:.2f})',
           alpha=1, s=100, c="white", zorder=99, edgecolors='#f77f00', linewidths=3)

ax.scatter(x_fit[min_ix], y_fit[min_ix],
           # label=f'SHAP min (x={x_fit[min_ix]:.2f})',
           marker='v', alpha=1, s=120, c="white", zorder=99, edgecolors='#66bb6a', linewidths=2)

# Detect min/max value shown in plot
_temp = iqr_high75.max() * 1.2
_temp2 = iqr_low25.min()
ax.set_ylim(_temp2, _temp)

# Get the y-position for the text, slightly above the plotted points
text_y_pos_max = y_fit[max_ix] - 0.3 * (_temp - _temp2)
text_y_pos_zero = y_fit[idx] - 0.3 * (_temp - _temp2)
text_y_pos_min = y_fit[min_ix] - 0.2 * (_temp - _temp2)

# Add text and connecting dashed lines for SHAP max, zero, and min
y_top = ax.get_ylim()[-1]
y_bottom = ax.get_ylim()[0]

ax.text(x_fit[max_ix], text_y_pos_max, f'SHAP max\n(x={x_fit[max_ix]:.2f})',
        va='bottom', ha='center', color='#e63946', fontsize=9, linespacing=1.2)
ax.plot([x_fit[max_ix], x_fit[max_ix]], [text_y_pos_max + 0.15, y_fit[max_ix]], color='#e63946', linestyle='--',
        linewidth=1)
ax.plot([x_fit[max_ix], x_fit[max_ix]], [y_bottom, text_y_pos_max], color='#e63946', linestyle='--', linewidth=1)

ax.text(x_fit[idx], text_y_pos_zero, f'SHAP zero\n(x={x_fit[idx]:.2f})',
        va='bottom', ha='center', color='#f77f00', fontsize=9, linespacing=1.2)
ax.plot([x_fit[idx], x_fit[idx]], [text_y_pos_zero + 0.15, y_fit[idx]], color='#f77f00', linestyle='--', linewidth=1)
ax.plot([x_fit[idx], x_fit[idx]], [y_bottom, text_y_pos_zero], color='#f77f00', linestyle='--', linewidth=1)

ax.text(x_fit[min_ix], text_y_pos_min, f'SHAP min\n(x={x_fit[min_ix]:.2f})',
        va='bottom', ha='center', color='#66bb6a', fontsize=9, linespacing=1.2)
ax.plot([x_fit[min_ix], x_fit[min_ix]], [text_y_pos_min + 0.15, y_fit[min_ix]], color='#66bb6a', linestyle='--',
        linewidth=1)
ax.plot([x_fit[min_ix], x_fit[min_ix]], [y_bottom, text_y_pos_min], color='#66bb6a', linestyle='--', linewidth=1)

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
ax.annotate(f'Fitted 4th degree polynomial (r$^2$={r_squared:.2f})',
            xy=(arrow_x, arrow_y),
            xytext=(arrow_x - 2.5, arrow_y - 0.6),  # Adjust text position as needed
            arrowprops=dict(arrowstyle="->", color='#004e98', lw=1.5),
            fontsize=10, color='#004e98', ha='left', va='center')

# # TODO Add arrow to one of the IQR data points
# for i in range(800, 805):
iqr_point_index = 804  # Choose a representative index
iqr_point_index = iqr_point_index if iqr_point_index < len(X_data) else 0
iqr_x = X_data[iqr_point_index]
iqr_y = iqr_low25[iqr_point_index]
# iqr_yerr_low = yerr_iqr[0][iqr_point_index]
# # iqr_yerr_high = yerr_iqr[1][iqr_point_index]
ax.annotate(
    # f'{i}',
    f'IQR for site data',
    xy=(iqr_x, iqr_y),
    # xy=(iqr_x, iqr_y - iqr_yerr_low),
    xytext=(iqr_x + 0.1, iqr_y - 0.3),  # Adjust text position as needed
    arrowprops=dict(arrowstyle="->", color='#6c757d', lw=1.5),
    fontsize=10, color='#6c757d', ha='right', va='center'
)

# Add text for negative effect
ax.text(x=2.1, y=-0.03, s='Negative impact\nreduced uptake/increased release',
        fontsize=9, color='black', ha='left', va='top')

# Add text for positive effect
ax.text(x=2.1, y=0.03, s='Positive impact\nincreased uptake/reduced release',
        fontsize=9, color='black', ha='left', va='bottom')

# 426cb0
ax.set_xlabel("Vapor pressure deficit (z-score)")
ax.set_ylabel('SHAP value of VPD effect on NEE (z-score)')
ax.set_title('The impact of VPD on NEP', fontsize=14, pad=10, y=1.02)
ax.axhline(y=0, color='black', linestyle='-', lw=1)
ax.grid(False)
ax.legend(loc='upper right', frameon=False)
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
