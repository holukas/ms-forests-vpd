from pathlib import Path
import numpy as np
import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd

import src.files as files
import src.plot as plot
import src.scenarios as scenarios

# ==========================================
# SETTINGS
# ==========================================
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
COLUMN_ORDER = ['All sites'] + IGBP_CLASSES
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal', 'Hot & dry', 'Compound\nextremes']
N_SCENARIOS = len(SCENARIO_ORDER)

# Variables
VARS = ['VPD_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'SWIN_ZSCORE']

VAR_TITLES = ['Vapor pressure deficit effect', 'Air temperature effect',
              'Soil moisture effect', 'Radiation effect']

SHAP_SUFFIX_AVG = '_SHAPVALS_OVR_AVG'
SHAP_COLS_AVG = [v + SHAP_SUFFIX_AVG for v in VARS]
SHAP_SUFFIX_SD = '_SHAPVALS_OVR_SD'
SHAP_COLS_SD = [v + SHAP_SUFFIX_SD for v in VARS]

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

# Paths
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
filepath = Path(results_outdir) / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet"

# Load data
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()

# Select required IGBPs and scenarios
df_global = df_main[df_main['IGBP'].isin(IGBP_CLASSES)]
df_global = df_global[df_global['SCENARIO'].isin(SCENARIO_ORDER)]



# Global scaling
# Global limits for NEP are: (np.float64(), np.float64(1.536760039509245))
# In case a different flux than NEP is plotted, the global limits for NEP are used
if FLUX == 'NEP_ZSCORE':
    all_values_flat = df_global[SHAP_COLS_AVG].values.flatten()
    GRAND_Y_MIN = np.nanmin(all_values_flat)
    GRAND_Y_MAX = np.nanmax(all_values_flat)
    # Global
    pad = (GRAND_Y_MAX - GRAND_Y_MIN) * 0.15
    FIXED_YLIM = (GRAND_Y_MIN - pad, GRAND_Y_MAX)
else:
    FIXED_YLIM = (-1.5785025622990305, 1.536760039509245)

# # Get limits for y-axis scaling, same for all plots
# GRAND_Y_MIN, GRAND_Y_MAX = plot.get_panel_limits(df=scenario_stats)
# FIXED_YLIM = (GRAND_Y_MIN * 1.05, GRAND_Y_MAX * 1.2)

# ------
# FIGURE
# ------
fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

# GridSpec: 4 rows x 6 cols (5 data + 1 spacer)
gs = gridspec.GridSpec(4, 6, figure=fig,
                       width_ratios=[2.2, 0.025, 1, 1, 1, 1],  # Spacer is 0.15 relative width
                       height_ratios=[1, 1, 1, 1],
                       wspace=0.1, hspace=0.15)

panel_counter = 0
featurestats_df = None  # Collects stats for each feature, scenario and IGBP

for row, (feature_col, var_title) in enumerate(zip(SHAP_COLS_AVG, VAR_TITLES)):
    color = COLORS[row]

    # Row Label (Variable) - Rotated on left
    # Using figure text allows exact placement independent of axis coordinates
    y_pos = 0.82 - (row * 0.22)  # Approximate calculation based on height

    # Add variable title on the far left (y-axis label equivalent)
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
        if group_name == 'All sites':
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
        cur_featurestats_df = plot.plot_scenario_panel(
            ax, df_sub, feature_col, color, group_name=group_name, columns=SCENARIO_ORDER, n_scenarios=N_SCENARIOS,
            scenario_labels=SCENARIO_LABELS, y_limits=FIXED_YLIM, show_x=is_bottom_row, show_y=is_left_col,
            is_main=is_main, is_top_row=is_top_row, is_first=is_first)

        # Collect feature stats in table
        if panel_counter == 0:
            featurestats_df = cur_featurestats_df.copy()
        else:
            featurestats_df = pd.concat([featurestats_df, cur_featurestats_df], axis=0)

        # Column headers
        if row == 0:
            ax.set_title(group_name, fontsize=7, fontweight='bold', pad=8, color='black')

        # y-axis label (only for first column)
        if is_left_col:
            var_title = var_title.replace(" ", r"\ ")
            label_text = r"$\mathbf{" + var_title + "}$" + "\n(z-score)"
            ax.set_ylabel(label_text, fontsize=7, color="black", labelpad=4)

        # Panel letters: (a), (b), ...
        letter = chr(97 + panel_counter)
        ax.text(0.05, 0.92, f"({letter})", transform=ax.transAxes,
                fontsize=7, fontweight='bold', va='top', ha='left',
                color='black', zorder=10)  # Always black for readability

        panel_counter += 1

# Final layout adjustment
plt.subplots_adjust(left=0.1, right=0.98, top=0.95, bottom=0.07)

# Save function helper
# plt.savefig('transition_plot.pdf', dpi=300, bbox_inches='tight')

# --------------
# CREATE TABLE 1
# --------------
# ------------------------
# CALCULATE SCENARIO STATS
# ------------------------
print("Calculating scenario stats...")

# Global
# Calculate scenario stats across all sites
scenario_stats = scenarios.calculate_scenario_stats(
    df_input=df_global, igbp='global', scenario_order=SCENARIO_ORDER, vars=VARS,
    shap_suffix_avg=SHAP_SUFFIX_AVG, shap_suffix_sd=SHAP_SUFFIX_SD)

# IGBPs
# Calculate scenario stats for each IGBP
for i, igbp in enumerate(IGBP_CLASSES):
    df_igbp = df_global[df_global['IGBP'] == igbp]
    igbp_data = scenarios.calculate_scenario_stats(
        df_input=df_igbp, igbp=igbp, scenario_order=SCENARIO_ORDER, vars=VARS,
        shap_suffix_avg=SHAP_SUFFIX_AVG, shap_suffix_sd=SHAP_SUFFIX_SD)
    # s_data, s_net, s_err, s_sd, s_counts = calculate_budget_stats(df_sub)
    scenario_stats = pd.concat([scenario_stats, igbp_data], axis=0)

# ---------------------------------------------------------
# ADJUSTED TABLE 1 CREATION USING scenario_stats
# ---------------------------------------------------------
df = scenario_stats.copy()

# 1. Map Columns for Display
# --------------------------
# Map Variable Names
var_map = {
    'VPD_ZSCORE_SHAPVALS_OVR_AVG': 'Vapor pressure deficit',
    'TA_ZSCORE_SHAPVALS_OVR_AVG': 'Air temperature',
    'SWC_ZSCORE_SHAPVALS_OVR_AVG': 'Soil moisture',
    'SWIN_ZSCORE_SHAPVALS_OVR_AVG': 'Radiation',
    'NET_SHAPVALS': 'Net sum'  # Map the Net row directly
}

# Ensure we map whatever column holds the variable name (e.g., 'Variable' or 'Feature')
# Assuming the column is named 'Variable' based on your previous dataframe printout
if 'Variable' in df.columns:
    df['Driver'] = df['Variable'].map(var_map)
elif 'Feature' in df.columns:
    df['Driver'] = df['Feature'].map(var_map)

# Map Scenarios
scen_map = {1: 'Normal', 4: 'Dry and hot', 5: 'Compound extremes'}
df['Scenario_Name'] = df['scenario'].map(scen_map)

# Map IGBP (Normalize 'global' to 'All sites' to match IGBP_ORDER)
igbp_map = {'global': 'All sites'}
# Fill remaining IGBPs with themselves if not in map
df['IGBP_Display'] = df['igbp'].replace(igbp_map)

# 2. Format Statistics String
# ---------------------------
# Check if Min/Max exist, otherwise formatting will fail
if 'min' in df.columns and 'max' in df.columns:
    df['Stats'] = df.apply(lambda r: f"{r['mean']:.2f}±{r['total_sd']:.2f} ({r['min']:.2f}, {r['max']:.2f})", axis=1)
else:
    # Fallback if Min/Max are missing from the aggregate stats
    print("Warning: Min/Max columns missing, showing Mean±SD only.")
    df['Stats'] = df.apply(lambda r: f"{r['mean']:.2f}±{r['total_sd']:.2f}", axis=1)

# 3. Pivot
# --------
SCENARIO_ORDER_NAMES = ['Normal', 'Dry and hot', 'Compound extremes']
IGBP_ORDER = ['All sites', 'ENF', 'DBF', 'MF', 'EBF']
VAR_ORDER = ['Vapor pressure deficit', 'Air temperature', 'Soil moisture', 'Radiation']

# We only need one pivot now because we don't need to sum numeric values manually
table_str = df.pivot_table(index=['IGBP_Display', 'Driver'],
                           columns='Scenario_Name',
                           values='Stats',
                           aggfunc='first')

# Reorder columns to ensure correct scenario sequence
table_str = table_str.reindex(columns=SCENARIO_ORDER_NAMES)

# 4. Construct Final Ordered List
# -------------------------------
grouped_rows = []

for igbp in IGBP_ORDER:
    # A. Header Row (IGBP Name)
    grouped_rows.append({
        'IGBP / Environmental driver': f"{igbp}",
        'Normal': '', 'Dry and hot': '', 'Compound extremes': ''
    })

    # Check if data exists for this IGBP
    if igbp in table_str.index.get_level_values(0):
        igbp_data = table_str.loc[igbp]

        # B. Component Rows (Iterate specific order)
        for driver in VAR_ORDER:
            if driver in igbp_data.index:
                row = igbp_data.loc[driver]
                grouped_rows.append({
                    'IGBP / Environmental driver': f"  {driver}",
                    'Normal': row.get('Normal', ''),
                    'Dry and hot': row.get('Dry and hot', ''),
                    'Compound extremes': row.get('Compound extremes', '')
                })

        # C. Net Sum Row (Look it up directly)
        if 'Net sum' in igbp_data.index:
            row = igbp_data.loc['Net sum']
            grouped_rows.append({
                'IGBP / Environmental driver': "  Net sum",
                'Normal': row.get('Normal', ''),
                'Dry and hot': row.get('Dry and hot', ''),
                'Compound extremes': row.get('Compound extremes', '')
            })
        else:
            # Fallback if Net sum is missing for some reason
            grouped_rows.append({'IGBP / Environmental driver': "  Net sum", 'Normal': 'n/a', 'Dry and hot': 'n/a', 'Compound extremes': 'n/a'})

# 5. Finalize
table_1_final = pd.DataFrame(grouped_rows)
table_1_final.set_index('IGBP / Environmental driver', inplace=True)

# Output
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 1000)
print(table_1_final)

# # Means
# var_map = {
#     'VPD_ZSCORE_SHAPVALS_OVR_AVG': 'Vapor pressure deficit',
#     'TA_ZSCORE_SHAPVALS_OVR_AVG': 'Air temperature',
#     'SWC_ZSCORE_SHAPVALS_OVR_AVG': 'Soil moisture',
#     'SWIN_ZSCORE_SHAPVALS_OVR_AVG': 'Radiation'
# }
# df = featurestats_df.copy()
# df['Driver'] = df['Feature'].map(var_map)
# scen_map = {1: 'Normal', 4: 'Dry and hot', 5: 'Compound extremes'}
# df['Scenario'] = df['Scenario'].map(scen_map)
#
# # 1. Create the formatted string column (for display)
# df['Stats'] = df.apply(lambda r: f"{r['Mean']:.2f}±{r['total_SD']:.2f} ({r['Min']:.2f}, {r['Max']:.2f})", axis=1)
#
# # 2. Create pivots
# SCENARIO_ORDER = ['Normal', 'Dry and hot', 'Compound extremes']
# IGBP_ORDER = ['All sites', 'ENF', 'DBF', 'MF', 'EBF']
# VAR_ORDER = ['Vapor pressure deficit', 'Air temperature',
#              'Soil moisture', 'Radiation']
#
# # Pivot A: The formatted strings for the table
# table_str = df.pivot_table(index=['IGBP', 'Driver'],
#                            columns='Scenario',
#                            values='Stats',
#                            aggfunc='first')
#
# # Pivot B: The numeric Means for calculating sums
# table_num = df.pivot_table(index=['IGBP', 'Driver'],
#                            columns='Scenario',
#                            values='Mean',
#                            aggfunc='first')
#
# # Reorder
# table_str = table_str.reindex(columns=SCENARIO_ORDER)
# table_num = table_num.reindex(columns=SCENARIO_ORDER)
#
# # Format table for output
# grouped_rows = []
#
# # Iterate through each IGBP class (Outer Layer)
# for igbp in IGBP_ORDER:
#     # 1. Add the Header Row (The Site/Class Name)
#     grouped_rows.append({
#         'IGBP / Environmental driver': f"{igbp}",
#         'Normal': '',
#         'Dry and hot': '',
#         'Compound extremes': ''
#     })
#
#     # Check if data exists for this IGBP
#     if igbp in table_str.index.get_level_values(0):
#         # Slice data for this IGBP
#         igbp_data_str = table_str.loc[igbp]
#         igbp_data_num = table_num.loc[igbp]
#
#         # Initialize sums for this IGBP block
#         sums = {scen: 0.0 for scen in SCENARIO_ORDER}
#
#         # 2. Add rows for each Driver
#         for driver in VAR_ORDER:
#             if driver in igbp_data_str.index:
#                 row_str = igbp_data_str.loc[driver]
#                 row_num = igbp_data_num.loc[driver]
#
#                 # Add to the list
#                 grouped_rows.append({
#                     'IGBP / Environmental driver': f"  {driver}",
#                     'Normal': row_str['Normal'],
#                     'Dry and hot': row_str['Dry and hot'],
#                     'Compound extremes': row_str['Compound extremes']
#                 })
#
#                 # Add to the sums (handling NaNs if any)
#                 for scen in SCENARIO_ORDER:
#                     val = row_num[scen]
#                     if pd.notna(val):
#                         sums[scen] += val
#
#         # 3. Add the Net Sum Row
#         # This matches the "Sum of Means" used in your stacked plots
#         grouped_rows.append({
#             'IGBP / Environmental driver': "  Net sum",
#             'Normal': f"{sums['Normal']:.2f}",
#             'Dry and hot': f"{sums['Dry and hot']:.2f}",
#             'Compound extremes': f"{sums['Compound extremes']:.2f}"
#         })
#
# # Create final DataFrame
# table_1_final = pd.DataFrame(grouped_rows)
# table_1_final.set_index('IGBP / Environmental driver', inplace=True)

# Save
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'52_TABLE-1_Scenarios_ShapMeans_WithNetSum_{FLUX}.csv'
table_1_final.to_csv(outfilepath, index=True)

# Show table
pd.set_option('display.max_rows', 3000)
pd.set_option('display.width', 1000)
print(table_1_final.to_string(index=True))

# # Save fig to file
# dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
# outfilepath = dir_out / f'52_FIG-2_Scenarios_SinaPlots_ShapMeans_{FLUX}.png'
# fig.savefig(outfilepath, dpi=300, bbox_inches='tight')

# Show figure
plt.show()
