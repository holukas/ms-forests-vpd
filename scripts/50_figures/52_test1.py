from pathlib import Path
from matplotlib import ticker
import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import src.files as files

# --- CONFIGURATION ---
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
COLUMN_ORDER = ['All Sites'] + IGBP_CLASSES
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal', 'Dry+Hot', 'Compound']
N_SCENARIOS = len(SCENARIO_ORDER)

# Variables
VARIABLES_BASE = ['SWIN_ZSCORE', 'TA_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']
VAR_TITLES = ['Solar Rad.', 'Air Temp.', 'VPD', 'Soil Water']  # Shortened for cleaner look
SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'
SHAP_COLS = [v + SHAP_SUFFIX for v in VARIABLES_BASE]

# Okabe-Ito Palette (Colorblind Friendly - Nature Standard)
COLORS = ['#E69F00', '#D55E00', '#CC79A7', '#009E73']

# Nature Style Dimensions (Double Column ~183mm width)
FIG_WIDTH_INCHES = 7.2
FIG_HEIGHT_INCHES = 8.5

# Style Settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,  # Base font size
    'axes.labelsize': 8,  # Axis labels
    'axes.titlesize': 8,  # Panel titles
    'axes.titleweight': 'bold',
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.linewidth': 0.4,  # Spine thickness
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
    'lines.linewidth': 1.0,
    'figure.dpi': 300,  # High res for export
    'savefig.dpi': 300,
})

# --- DATA LOADING ---


settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
filepath = Path(results_outdir) / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df[['SITE', 'IGBP', 'SCENARIO'] + SHAP_COLS].copy()
df_main = df_main.loc[df_main['SCENARIO'].isin(SCENARIO_ORDER)].copy()
df_main = df_main.loc[df_main['IGBP'].isin(IGBP_CLASSES)].copy()

# Calculate Global Limits per Variable (Row) to ensure comparison
Y_LIMITS_PER_ROW = {}
for col in SHAP_COLS:
    vals = df_main[col].dropna()
    vmin, vmax = vals.min(), vals.max()
    pad = (vmax - vmin) * 0.1
    Y_LIMITS_PER_ROW[col] = (vmin - pad, vmax + pad)


# --- PLOTTING ENGINE ---
def plot_panel(ax, df, feature_col, color, show_x=False, show_y=False, is_main=False):
    pivot = df.pivot(index='SITE', columns='SCENARIO', values=feature_col).reindex(columns=SCENARIO_ORDER)

    if pivot.dropna(how='all').empty:
        ax.set_visible(False)
        return

    # Stats
    medians = pivot.median(axis=0)
    q1 = pivot.quantile(0.25, axis=0)
    q3 = pivot.quantile(0.75, axis=0)
    x_coords = np.arange(N_SCENARIOS)
    n_sites = len(pivot)

    # 1. Ghost lines (The "Hairball" - keep very faint)
    # Thinner and more transparent for the background noise
    alpha_ghost = 0.07 if is_main else 0.08
    lw_ghost = 0.5
    ax.plot(x_coords, pivot.T.values, color='gray', alpha=alpha_ghost, linewidth=lw_ghost, zorder=1)

    # 2. IQR Ribbon (Crucial for scientific spread)
    ax.fill_between(x_coords, q1, q3, color=color, alpha=0.15, linewidth=0, zorder=2)

    # 3. Sina / Jitter Points
    # Controlled jitter that respects density but stays tight
    for x_i, scen in enumerate(SCENARIO_ORDER):
        if scen not in pivot:
            continue
        data = pivot[scen].dropna()
        if len(data) < 2:
            ax.scatter([x_i] * len(data), data, color=color, s=2, alpha=0.5, zorder=3)
            continue

        kde = gaussian_kde(data)
        density = kde(data)
        # Normalize width for jitter
        width_factor = 0.15
        width = (density / density.max()) * width_factor
        rng = np.random.RandomState(42 + x_i)
        jitter = rng.uniform(-1, 1, size=len(data)) * width

        # Plot points
        s_sina = 4 if is_main else 4
        alpha_sina = 0.4 if is_main else 0.5
        ax.scatter(x_i + jitter, data, color=color, s=s_sina, alpha=alpha_sina, linewidth=0, zorder=3)

    # 4. Median Trend Line & Nodes
    lw_trend = 2.0
    s_node = 25
    ax.plot(x_coords, medians, color=color, linewidth=lw_trend, alpha=1.0, zorder=5)
    ax.scatter(x_coords, medians, facecolor=color, edgecolor='white', linewidth=1.0, s=s_node, zorder=6)

    # Formatting
    ax.set_ylim(Y_LIMITS_PER_ROW[feature_col])

    # Zero line (subtle)
    ax.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.5, zorder=0)

    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(x_coords)

    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Tick Styling
    if show_x:
        ax.set_xticklabels(SCENARIO_LABELS, rotation=0, color='black')
        ax.tick_params(axis='x', length=4, width=0.8)
    else:
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

    if show_y:
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.tick_params(axis='y', colors='black', length=3, direction='out')
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', length=0)

    # Sample size annotation (n=...)
    # Essential for Nature figures
    ax.text(0.08, 0.08, f'n={n_sites}', transform=ax.transAxes,
            fontsize=6, color='#555555', ha='left', va='bottom')

    # Highlight "All Sites" background slightly
    if is_main:
        ax.patch.set_facecolor('#f7f7f7')
        ax.patch.set_alpha(0.5)
    else:
        ax.patch.set_alpha(0.0)


# --- LAYOUT CONSTRUCTION ---
fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

# GridSpec: 4 Rows x 6 Cols (5 Data + 1 Spacer)
# We insert a narrow spacer column (index 1) between "All Sites" and "IGBP"
gs = gridspec.GridSpec(4, 6, figure=fig,
                       width_ratios=[2.2, 0.05, 1, 1, 1, 1],  # Spacer is 0.15 relative width
                       height_ratios=[1, 1, 1, 1],
                       wspace=0.1, hspace=0.15)

panel_counter = 0

for row, (feature_col, var_title) in enumerate(zip(SHAP_COLS, VAR_TITLES)):
    color = COLORS[row]

    # Row Label (Variable) - Rotated on left
    # Using figure text allows exact placement independent of axis coordinates
    y_pos = 0.82 - (row * 0.22)  # Approximate calculation based on height

    # Add Variable Title on the far left (Y-Axis Label equivalent)
    # Create a dummy axis for the label if needed, or use fig.text
    # Here we put it on the first axes ylabel for alignment

    # Iterate Columns
    # Map visual columns (0, 2, 3, 4, 5) to data groups. Skip col 1 (Spacer)
    current_col_idx = 0

    for grid_col in range(6):
        if grid_col == 1: continue  # Spacer

        group_name = COLUMN_ORDER[current_col_idx]
        current_col_idx += 1

        ax = fig.add_subplot(gs[row, grid_col])

        # Filter Data
        if group_name == 'All Sites':
            df_sub = df_main
            is_main = True
        else:
            df_sub = df_main[df_main['IGBP'] == group_name]
            is_main = False

        is_left_col = (grid_col == 0)
        is_bottom_row = (row == 3)

        plot_panel(ax, df_sub, feature_col, color,
                   show_x=is_bottom_row, show_y=is_left_col, is_main=is_main)

        # Column Headers
        if row == 0:
            ax.set_title(group_name, fontsize=8, fontweight='bold', pad=8, color='black')

        # Y-Axis Label (only for first column)
        if is_left_col:
            ax.set_ylabel(var_title + "\n(z-score)", fontsize=8, fontweight='bold', color=color, labelpad=4)

        # Panel Lettering: (a), (b), ...
        # Nature style: Bold lowercase letter, top left
        letter = chr(97 + panel_counter)
        ax.text(0.05, 0.92, f"({letter})", transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='left',
                color='black', zorder=10)  # Always black for readability

        panel_counter += 1

# Final Layout Adjustment
# We rely on GridSpec, but a final tight_layout with padding helps
plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.06)

# Save function helper
# plt.savefig('nature_transition_plot.pdf', dpi=300, bbox_inches='tight')

plt.show()
