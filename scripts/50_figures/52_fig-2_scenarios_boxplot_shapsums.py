from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.files as files
from src.plot import plot_scenario_panel

# Settings
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
COLUMN_ORDER = ['All Sites'] + IGBP_CLASSES
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal', 'Dry and hot', 'Extremes']
N_SCENARIOS = len(SCENARIO_ORDER)

# Variables
VARIABLES_BASE = ['VPD_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'SWIN_ZSCORE']
VAR_TITLES = ['Vapor pressure deficit', 'Air Temperature',
              'Soil water content', 'Incoming shortwave radiation']
SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'
SHAP_COLS = [v + SHAP_SUFFIX for v in VARIABLES_BASE]

# Okabe-Ito palette (colorblind friendly)
COLORS = ['#D55E00', '#CC79A7', '#009E73', '#E69F00']

# Figure dimensions (double column ~183mm width)
FIG_WIDTH_INCHES = 7.2
FIG_HEIGHT_INCHES = 8.5

# Style settings
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
pad = (Y_MAX_GLOBAL - Y_MIN_GLOBAL) * 0.15
Y_LIMITS = (Y_MIN_GLOBAL - pad, Y_MAX_GLOBAL)

# Plotting engine

# Figure
fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

# GridSpec: 4 rows x 6 cols (5 data + 1 spacer)
gs = gridspec.GridSpec(4, 6, figure=fig,
                       width_ratios=[2.2, 0.025, 1, 1, 1, 1],  # Spacer is 0.15 relative width
                       height_ratios=[1, 1, 1, 1],
                       wspace=0.1, hspace=0.15)

panel_counter = 0
featurestats_df = None  # Collects stats for each feature, scenario and IGBP

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
        if grid_col == 1:
            continue  # Spacer

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
        is_top_row = (row == 0)
        is_bottom_row = (row == 3)
        is_first = (grid_col == 0) and (row == 0)

        # Plot feature effects and collect stats
        cur_featurestats_df = plot_scenario_panel(
            ax, df_sub, feature_col, color, group_name=group_name, columns=SCENARIO_ORDER, n_scenarios=N_SCENARIOS,
            scenario_labels=SCENARIO_LABELS, y_limits=Y_LIMITS, show_x=is_bottom_row, show_y=is_left_col,
            is_main=is_main, is_top_row=is_top_row, is_first=is_first)

        # Collect feature stats in table
        if panel_counter == 0:
            featurestats_df = cur_featurestats_df.copy()
        else:
            featurestats_df = pd.concat([featurestats_df, cur_featurestats_df], axis=0)

        # Column headers
        if row == 0:
            ax.set_title(group_name, fontsize=8, fontweight='bold', pad=8, color='black')

        # y-axis label (only for first column)
        if is_left_col:
            ax.set_ylabel(var_title + "\n(z-score)", fontsize=8, fontweight='bold', color="black", labelpad=4)

        # Panel letters: (a), (b), ...
        letter = chr(97 + panel_counter)
        ax.text(0.05, 0.92, f"({letter})", transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top', ha='left',
                color='black', zorder=10)  # Always black for readability

        panel_counter += 1

# Final layout adjustment
plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.07)

# Save function helper
# plt.savefig('transition_plot.pdf', dpi=300, bbox_inches='tight')

# --------------
# CREATE TABLE 1
# --------------
var_map = {
    'VPD_ZSCORE_SHAPVALS_OVR_MEDIAN': 'Vapor Pressure Deficit',
    'TA_ZSCORE_SHAPVALS_OVR_MEDIAN': 'Air Temperature',
    'SWC_ZSCORE_SHAPVALS_OVR_MEDIAN': 'Soil Water Content',
    'SWIN_ZSCORE_SHAPVALS_OVR_MEDIAN': 'Incoming Shortwave Radiation'
}
df = featurestats_df.copy()
df['Feature'] = df['Feature'].map(var_map)
scen_map = {1: 'Normal', 4: 'Dry and hot', 5: 'Compound extremes'}
df['Scenario'] = df['Scenario'].map(scen_map)
# Construct formatted strings for table cells
df['Stats'] = df.apply(lambda r: f"{r['Median']:.2f} ({r['Min']:.2f}, {r['Max']:.2f})", axis=1)
df['NegImpact'] = df.apply(lambda r: f"{r['Sites < 0']} ({r['% < 0']:.0f})", axis=1)

# Pivot to wide format for table in MS Word
SCENARIO_ORDER = ['Normal', 'Dry and hot', 'Compound extremes']
IGBP_ORDER = ['All Sites', 'ENF', 'DBF', 'MF', 'EBF']
VAR_ORDER = ['Vapor Pressure Deficit', 'Air Temperature',
             'Soil Water Content', 'Incoming Shortwave Radiation']
table_1 = df.pivot_table(index=['Feature', 'IGBP'],
                         columns='Scenario',
                         values='Stats',
                         aggfunc='first')

# Reorder columns (Scenarios) and the second level of the index (IGBP)
table_1 = table_1.reindex(columns=SCENARIO_ORDER)
table_1 = table_1.reindex(level='IGBP', index=IGBP_ORDER)
table_1 = table_1.reindex(level='Feature', index=VAR_ORDER)

# Save table to file
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'52_TABLE-1_Scenarios_ShapMedians_{FLUX}.csv'
table_1.to_csv(outfilepath, index=True)

# Show table
pd.set_option('display.max_rows', 3000)
print(table_1.to_string(index=True))

# Save fig to file
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'52_FIG-2_Scenarios_SinaPlots_ShapMedians_{FLUX}.png'
fig.savefig(outfilepath, dpi=300, bbox_inches='tight')

# Show figure
plt.show()
