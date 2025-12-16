"""
Transition Plot: Tracks changes in SHAP impact per site across scenarios.
Refactored:
- Removed boxplots.
- Added continuous Median Line and Filled IQR Ribbon ("River").
- Aligned all elements (ghost lines, points, stats) to the central axis.
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
FEATURE_COLORS = ['#b2df8a', '#fdbf6f', '#fb9a99', '#a6cee3']
LINE_COLORS = ['#33a02c', '#ff7f00', '#e31a1c', '#1f77b4']

# ------------------------------
# Data Loading & Preprocessing
# ------------------------------
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

filepath = Path(results_outdir) / f"3_AllSites_SHAP-ScenarioSums-{shap_type}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

df = shapvals_df.copy()
df = df[['SITE', 'IGBP', 'SCENARIO'] + SHAP_COLS].copy()
df = df.loc[df['IGBP'] == 'EBF'].copy()
df = df.loc[df['SCENARIO'].isin(SCENARIO_ORDER)].copy()

# --- Calculate Global Y-Limits for Shared Scaling ---
all_shap_values = df[SHAP_COLS].values.flatten()
global_ymin = np.nanmin(all_shap_values)
global_ymax = np.nanmax(all_shap_values)
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
    color_fill = FEATURE_COLORS[i]

    # Pivot data: Index=SITE, Columns=SCENARIO
    pivot_df = df.pivot(index='SITE', columns='SCENARIO', values=feature_col)
    pivot_df = pivot_df[SCENARIO_ORDER]

    # --- Statistics for Ribbon/Line ---
    # Calculate stats across the sites (axis=0) for each scenario
    medians = pivot_df.median(axis=0)
    q1 = pivot_df.quantile(0.25, axis=0)
    q3 = pivot_df.quantile(0.75, axis=0)

    # X-coordinates for the categories
    x_coords = np.arange(N_SCENARIOS)

    # --- A. Draw Connecting Lines (Spaghetti) ---
    # Plot faint lines for individual sites
    ax.plot(x_coords, pivot_df.T.values,
            color=color_line, alpha=0.1, linewidth=0.8, zorder=1)

    # --- B. Draw IQR Ribbon ---
    # Filled area between Q1 and Q3
    ax.fill_between(x_coords, q1, q3,
                    color=color_line, alpha=0.2, linewidth=0, zorder=2)

    # Optional: Add thin border lines to the ribbon
    ax.plot(x_coords, q1, color=color_line, alpha=0.4, linewidth=0.5, linestyle='--', zorder=2)
    ax.plot(x_coords, q3, color=color_line, alpha=0.4, linewidth=0.5, linestyle='--', zorder=2)

    # --- C. Draw Overall Median Line ---
    # Thick line connecting the medians
    ax.plot(x_coords, medians,
            color=color_line, linewidth=3, alpha=0.9, zorder=4)
    # Add points at the median nodes
    ax.scatter(x_coords, medians,
               facecolor=color_line, edgecolor='white', linewidth=1.5, s=60, zorder=5)

    # --- D. Draw Sina Points (Distributions) ---
    data_per_scenario = [pivot_df[scen].dropna() for scen in SCENARIO_ORDER]
    for x_idx, scenario_data in enumerate(data_per_scenario):
        if len(scenario_data) < 2:
            continue

        # 1. Sina Plot (Cloud of points)
        kde = gaussian_kde(scenario_data)
        density = kde(scenario_data)
        sina_width = (density / density.max()) * 0.15 # Adjust width as needed

        # Jitter centered around the main axis (x_idx)
        jitter = np.random.uniform(-1, 1, size=len(scenario_data)) * sina_width

        ax.scatter(x_idx + jitter, scenario_data,
                   color=color_line, s=15, alpha=0.5, linewidth=0, zorder=3)

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
    ax.set_xlim(-0.5, N_SCENARIOS - 1 + 0.5)
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