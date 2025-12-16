"""
Transition Plot: Tracks changes in SHAP impact per site across scenarios.
Refactored for side-by-side boxplots/points, colored lines, and shared y-scaling.
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

# Colors for the 4 variables (used for boxplots/points/lines)
# SWIN (Greenish), TA (Orange), VPD (Red), SWC (Blue)
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
# Keep 'SITE' for connecting lines
df = df[['SITE', 'IGBP', 'SCENARIO'] + SHAP_COLS].copy()
# Filter for specific IGBP and Scenarios
df = df.loc[df['IGBP'] == 'ENF'].copy()
df = df.loc[df['SCENARIO'].isin(SCENARIO_ORDER)].copy()

# --- Calculate Global Y-Limits for Shared Scaling ---
# Get all SHAP values across the 4 variables to determine common min/max
all_shap_values = df[SHAP_COLS].values.flatten()
global_ymin = np.nanmin(all_shap_values)
global_ymax = np.nanmax(all_shap_values)

# Add a small buffer (e.g., 5% of range) for visual comfort
y_range_buffer = (global_ymax - global_ymin) * 0.05
global_ymin -= y_range_buffer
global_ymax += y_range_buffer

# ------------------------------
# Plotting
# ------------------------------
# 4 Rows (Variables), 1 Column
fig = plt.figure(figsize=(7, 14))
gs = gridspec.GridSpec(4, 1, figure=fig)

axes = []

for i, (feature_col, var_name) in enumerate(zip(SHAP_COLS, VARIABLES_BASE)):
    ax = fig.add_subplot(gs[i, 0])
    axes.append(ax)

    # Get colors for this variable layer
    color_fill = FEATURE_COLORS[i]
    color_line = LINE_COLORS[i]

    # Pivot data: Index=SITE, Columns=SCENARIO, Values=Feature
    # This aligns rows by site for drawing connecting lines
    pivot_df = df.pivot(index='SITE', columns='SCENARIO', values=feature_col)
    # Ensure correct column order based on transition path
    pivot_df = pivot_df[SCENARIO_ORDER]

    # Data list for distribution plots (sina/box)
    data_per_scenario = [pivot_df[scen].dropna() for scen in SCENARIO_ORDER]

    # --- A. Draw Connecting Lines (The Transition) ---
    # Base X-coordinates for the scenarios (0, 1, 2)
    x_coords_base = np.arange(N_SCENARIOS)
    # Shift lines slightly left to align better with the point clouds
    x_coords_lines = x_coords_base - 0.15

    # Plot lines representing sites.
    # Use corresponding variable color, low alpha for "spaghetti" effect.
    ax.plot(x_coords_lines, pivot_df.T.values,
            color=color_line, alpha=0.15, linewidth=1, zorder=1)

    # --- B. Draw Distributions (Side-by-Side Sina & Boxplot) ---
    for x_idx, scenario_data in enumerate(data_per_scenario):
        if len(scenario_data) < 2: # Need at least 2 points for KDE
            continue

        # Offsets to separate points and boxes
        pos_points_center = x_idx - 0.15
        pos_box_center = x_idx + 0.15

        # 1. Sina Plot (Cloud of points)
        kde = gaussian_kde(scenario_data)
        density = kde(scenario_data)
        # Define width of the cloud based on density
        sina_width = (density / density.max()) * 0.12
        # Add random jitter within that width
        jitter = np.random.uniform(-1, 1, size=len(scenario_data)) * sina_width

        ax.scatter(pos_points_center + jitter, scenario_data,
                   color=color_line, s=15, alpha=0.5, linewidth=0, zorder=2)

        # 2. Boxplot (Summary)
        ax.boxplot(
            scenario_data,
            positions=[pos_box_center],
            showfliers=False,
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor=color_fill, edgecolor=color_line, alpha=0.9),
            medianprops=dict(color='black', linewidth=1.5),
            whiskerprops=dict(color=color_line, linewidth=1),
            capprops=dict(color=color_line, linewidth=1),
            zorder=3
        )

    # --- Formatting ---
    # Apply shared y-scaling
    ax.set_ylim(global_ymin, global_ymax)

    # Y-Label (Variable Name)
    ax.set_ylabel(f"{var_name}\nImpact (z-score)", fontsize=12, weight='bold', rotation=90, labelpad=15)

    # Zero line indicating neutral impact
    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6, zorder=0)

    # Spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # Keep bottom spine only for the last plot
    if i < len(SHAP_COLS) - 1:
         ax.spines['bottom'].set_visible(False)

    # X-Ticks and Labels
    ax.set_xlim(-0.6, N_SCENARIOS - 1 + 0.6)
    # Set ticks at the center of the scenario groups
    ax.set_xticks(np.arange(N_SCENARIOS))

    # Only show x-labels on the bottom-most plot
    if i == len(SHAP_COLS) - 1:
        ax.set_xticklabels(SCENARIO_LABELS, fontsize=11)
        ax.tick_params(axis='x', length=5)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    # Add subplot label letters (a, b, c, d)
    letter = chr(97 + i)
    ax.text(-0.1, 1.02, f"({letter})", transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='right')

    ax.tick_params(axis='y', labelsize=10)

# --- Global Cleanup ---
fig.suptitle(f"Transition of {FLUX} Impact per Site (IGBP: EBF)", fontsize=14, y=0.96, weight='bold')
plt.tight_layout()
# Adjust spacing to prevent label overlap and accommodate suptitle
plt.subplots_adjust(top=0.93, hspace=0.1, left=0.15, right=0.95, bottom=0.08)

plt.show()