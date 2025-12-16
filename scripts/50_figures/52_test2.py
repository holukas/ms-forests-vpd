"""
Transition Plot: Tracks changes in SHAP impact per site across scenarios.
Refactored: Removed boxplots, replaced with Median/IQR bars, retained spaghetti lines.
"""
from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

import src.files as files

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Configuration
# ------------------------------
FLUX = 'NEP'
CONDITIONAL = True  # SHAP

# Scenarios to track (in order of the transition)
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal\nConditions', 'Dry & Hot', 'Compound\nExtremes']
N_SCENARIOS = len(SCENARIO_ORDER)

# Variables to plot
VARIABLES_BASE = ['SWIN', 'TA', 'VPD', 'SWC']
SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'
SHAP_COLS = [v + SHAP_SUFFIX for v in VARIABLES_BASE]

# Colors for the 4 variables
# SWIN (Greenish), TA (Orange), VPD (Red), SWC (Blue)
FEATURE_COLORS = ['#b2df8a', '#fdbf6f', '#fb9a99', '#a6cee3']
LINE_COLORS = ['#33a02c', '#ff7f00', '#e31a1c', '#1f77b4']

# ------------------------------
# Data Loading & Preprocessing
# ------------------------------
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

filepath = Path(results_outdir) / f"3_AllSites_SHAP-ScenarioSums-{shap_type}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

df = shapvals_df.copy()
# Keep 'SITE' for connecting lines
df = df[['SITE', 'IGBP', 'SCENARIO'] + SHAP_COLS].copy()
# Filter for specific IGBP and Scenarios
df = df.loc[df['IGBP'] == 'ENF'].copy()
df = df.loc[df['SCENARIO'].isin(SCENARIO_ORDER)].copy()

# --- Calculate Global Y-Limits for Shared Scaling ---
all_shap_values = df[SHAP_COLS].values.flatten()
global_ymin = np.nanmin(all_shap_values)
global_ymax = np.nanmax(all_shap_values)

# Add a small buffer
y_range_buffer = (global_ymax - global_ymin) * 0.05
global_ymin -= y_range_buffer
global_ymax += y_range_buffer

# ------------------------------
# Plotting
# ------------------------------
fig = plt.figure(figsize=(7, 14))
gs = gridspec.GridSpec(4, 1, figure=fig)

axes = []

for i, (feature_col, var_name) in enumerate(zip(SHAP_COLS, VARIABLES_BASE)):
    ax = fig.add_subplot(gs[i, 0])
    axes.append(ax)

    color_line = LINE_COLORS[i]

    # Pivot data: Index=SITE, Columns=SCENARIO
    pivot_df = df.pivot(index='SITE', columns='SCENARIO', values=feature_col)
    pivot_df = pivot_df[SCENARIO_ORDER]

    # Data list
    data_per_scenario = [pivot_df[scen].dropna() for scen in SCENARIO_ORDER]

    # --- A. Draw Connecting Lines (Spaghetti) ---
    x_coords_base = np.arange(N_SCENARIOS)
    x_coords_lines = x_coords_base - 0.15

    # Draw faint lines connecting sites
    ax.plot(x_coords_lines, pivot_df.T.values,
            color=color_line, alpha=0.15, linewidth=1, zorder=1)

    # --- B. Draw Distributions (Sina Points + Median Bar) ---
    for x_idx, scenario_data in enumerate(data_per_scenario):
        if len(scenario_data) < 2:
            continue

        pos_points_center = x_idx - 0.15
        pos_stats_center = x_idx + 0.15

        # 1. Sina Plot (Cloud of points)
        kde = gaussian_kde(scenario_data)
        density = kde(scenario_data)
        sina_width = (density / density.max()) * 0.12
        jitter = np.random.uniform(-1, 1, size=len(scenario_data)) * sina_width

        ax.scatter(pos_points_center + jitter, scenario_data,
                   color=color_line, s=15, alpha=0.5, linewidth=0, zorder=2)

        # 2. Stats (Median + IQR) instead of Boxplot
        q1 = np.percentile(scenario_data, 25)
        median = np.percentile(scenario_data, 50)
        q3 = np.percentile(scenario_data, 75)

        # Draw Vertical Line for IQR (25th to 75th percentile)
        ax.vlines(pos_stats_center, q1, q3, color=color_line, linewidth=2, alpha=0.6, zorder=3)

        # Draw Horizontal Bar for Median
        # "Show median differently" -> Bold horizontal bar
        ax.hlines(median, pos_stats_center - 0.08, pos_stats_center + 0.08,
                  color=color_line, linewidth=3, zorder=4)

    # --- Formatting ---
    ax.set_ylim(global_ymin, global_ymax)
    ax.set_ylabel(f"{var_name}\nImpact (z-score)", fontsize=12, weight='bold', rotation=90, labelpad=15)
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6, zorder=0)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i < len(SHAP_COLS) - 1:
         ax.spines['bottom'].set_visible(False)

    # X-Ticks
    ax.set_xlim(-0.6, N_SCENARIOS - 1 + 0.6)
    ax.set_xticks(np.arange(N_SCENARIOS))

    if i == len(SHAP_COLS) - 1:
        ax.set_xticklabels(SCENARIO_LABELS, fontsize=11)
        ax.tick_params(axis='x', length=5)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    letter = chr(97 + i)
    ax.text(-0.1, 1.02, f"({letter})", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='right')

    ax.tick_params(axis='y', labelsize=10)

# --- Global Cleanup ---
fig.suptitle(f"Transition of {FLUX} Impact per Site (IGBP: EBF)", fontsize=14, y=0.96, weight='bold')
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.1, left=0.15, right=0.95, bottom=0.08)

plt.show()