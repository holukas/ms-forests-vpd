"""
Transition Plot
- Layout: 5 Columns (All Sites | ENF | DBF | MF | EBF) x 4 Rows (Variables).
- "All Sites" column is slightly wider and highlighted.
- Style: Publication-ready (Arial font, Okabe-Ito palette, minimal chart junk).
- Labeling: Combined Letter and Title inside each panel.
"""
from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import src.files as files

# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# Configuration
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
# Define the ordered columns for the plot
COLUMN_ORDER = ['All Sites'] + IGBP_CLASSES
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal', 'Dry+Hot', 'Compound']
N_SCENARIOS = len(SCENARIO_ORDER)
VARIABLES_BASE = ['SWIN_ZSCORE', 'TA_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']
VAR_TITLES = ['Solar Radiation', 'Air Temperature', 'Vapor Pressure Deficit', 'Soil Water Content']
SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'
SHAP_COLS = [v + SHAP_SUFFIX for v in VARIABLES_BASE]
COLORS = ['#E69F00', '#D55E00', '#CC79A7', '#009E73']

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.grid': False,
    'lines.linewidth': 1.0,
    'figure.titlesize': 12,
    'figure.titleweight': 'bold',
})

# Load data
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
filepath = Path(results_outdir) / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df[['SITE', 'IGBP', 'SCENARIO'] + SHAP_COLS].copy()
df_main = df_main.loc[df_main['SCENARIO'].isin(SCENARIO_ORDER)].copy()
df_main = df_main.loc[df_main['IGBP'].isin(IGBP_CLASSES)].copy()

# Global scaling
all_values_flat = df_main[SHAP_COLS].values.flatten()
Y_MIN_GLOBAL = np.nanmin(all_values_flat)
Y_MAX_GLOBAL = np.nanmax(all_values_flat)
pad = (Y_MAX_GLOBAL - Y_MIN_GLOBAL) * 0.1
Y_LIMITS = (Y_MIN_GLOBAL - pad, Y_MAX_GLOBAL + pad)


# Plotting engine
def plot_panel(ax, df, feature_col, color, show_x=False, show_y=False, is_main=False):
    pivot = df.pivot(index='SITE', columns='SCENARIO', values=feature_col).reindex(columns=SCENARIO_ORDER)

    if pivot.dropna(how='all').empty:
        ax.set_ylim(Y_LIMITS)
        ax.axis('off')
        return

    medians = pivot.median(axis=0)
    q1 = pivot.quantile(0.25, axis=0)
    q3 = pivot.quantile(0.75, axis=0)
    x_coords = np.arange(N_SCENARIOS)

    # Style
    lw_trend = 2 if is_main else 2
    s_node = 20 if is_main else 20
    alpha_ghost = 0.05 if is_main else 0.08
    linewidth = 1 if is_main else 1

    # Ghost lines
    ax.plot(x_coords, pivot.T.values, color="gray", alpha=alpha_ghost, linewidth=linewidth, zorder=1)
    if not is_main:
        # IQR ribbon
        ax.fill_between(x_coords, q1, q3, color=color, alpha=0.25, linewidth=0, zorder=2)
        # ax.plot(x_coords, q1, color=color, alpha=0.3, linewidth=0.5, linestyle=':', zorder=2)
        # ax.plot(x_coords, q3, color=color, alpha=0.3, linewidth=0.5, linestyle=':', zorder=2)
    # Median trend
    ax.plot(x_coords, medians, color=color, linewidth=lw_trend, alpha=0.9, zorder=4)
    ax.scatter(x_coords, medians, facecolor=color, edgecolor='white', linewidth=1.2, s=s_node, zorder=5)

    # Sina Points
    for x_i, scen in enumerate(SCENARIO_ORDER):
        if scen not in pivot:
            continue
        data = pivot[scen].dropna()
        if len(data) < 2:
            ax.scatter([x_i] * len(data), data, color=color, s=5, alpha=0.5, zorder=3)
            continue

        kde = gaussian_kde(data)
        density = kde(data)
        width = (density / density.max()) * 0.22
        rng = np.random.RandomState(42 + x_i)
        jitter = rng.uniform(-1, 1, size=len(data)) * width
        s_sina = 6 if is_main else 5
        ax.scatter(x_i + jitter, data, color=color, s=s_sina, alpha=0.6, linewidth=0, zorder=3)

    # Formatting
    ax.set_ylim(Y_LIMITS)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(x_coords)

    # X-Labels
    if show_x:
        ax.set_xticklabels(SCENARIO_LABELS, color='#333333')
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    # Y-Labels
    if show_y:
        ax.tick_params(axis='y', colors='#333333', length=3)
    else:
        ax.set_yticklabels([])
        ax.set_ylabel("")
        ax.tick_params(axis='y', length=0)

    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')

    if is_main:
        ax.set_facecolor('#F9F9F9')


# Construct layout
fig = plt.figure(figsize=(7.68, 9.12), dpi=200)

# 4 rows (variables) x 5 columns (groups)
# First column ('All Sites') larger width ratio for emphasis
gs = gridspec.GridSpec(4, 5, figure=fig, wspace=0.1, hspace=0.15,
                       width_ratios=[2, 1, 1, 1, 1])

panel_counter = 0

# Iterate rows (variables)
for row, (feature_col, var_title) in enumerate(zip(SHAP_COLS, VAR_TITLES)):
    color = COLORS[row]
    
    # Add row label (variable name) on the far left
    # Position calculated roughly based on row index
    y_pos = 0.82 - (row * 0.215)
    fig.text(0.02, y_pos, var_title, rotation=90,
             va='center', ha='center', fontsize=10, fontweight='bold', color=color)

    # Iterate columns (groups)
    for c, group_name in enumerate(COLUMN_ORDER):
        ax = fig.add_subplot(gs[row, c])

        # Filter data based on column
        if group_name == 'All Sites':
            df_sub = df_main
            is_main = True
        else:
            df_sub = df_main[df_main['IGBP'] == group_name]
            is_main = False

        # Determine axis visibility
        is_left_col = (c == 0)
        is_bottom_row = (row == 3)

        plot_panel(ax, df_sub, feature_col, color,
                   show_x=is_bottom_row, show_y=is_left_col, is_main=is_main)

        # Add column headers on the top row
        if row == 0:
            header_color = '#222222'
            # header_color = '#222222' if is_main else COLORS[0]
            ax.set_title(group_name, fontsize=11, fontweight='bold', pad=10, color=header_color)

        # Combined letter and title label inside panel
        letter = chr(97 + panel_counter)
        # Use black for 'All Sites' label, variable color for IGBPs for visual grouping
        label_color = '#222222' if is_main else color

        ax.text(0.04, 0.94, f"({letter})", transform=ax.transAxes,
                fontsize=9, fontweight='bold', va='top', ha='left', color=label_color)

        if is_left_col:
            ax.set_ylabel("SHAP value (z-score)", labelpad=5, fontsize=9, color='#555555')

        panel_counter += 1

plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08)
plt.show()
