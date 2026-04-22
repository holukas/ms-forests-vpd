from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import src.files as files
import src.plot as plot
import src.stages as stages

# ==========================================
# SETTINGS
# ==========================================
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
COLUMN_ORDER = ['All sites'] + IGBP_CLASSES
STAGE_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]
STAGE_LABELS = [1, 2, 3, 4, 5, 6, 7, 8]
N_STAGES = len(STAGE_ORDER)

# Variables
VARS = ['VPD_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'SWIN_ZSCORE']

VAR_TITLES = ['VPD effect', 'TA effect',
              'SM effect', 'SW effect']

SHAP_SUFFIX_AVG = '_SHAPVALS_OVR_AVG'
SHAP_COLS_AVG = [v + SHAP_SUFFIX_AVG for v in VARS]
SHAP_SUFFIX_SD = '_SHAPVALS_OVR_SD'
SHAP_COLS_SD = [v + SHAP_SUFFIX_SD for v in VARS]

# Okabe-Ito palette (colorblind friendly)
COLORS = ['#D55E00', '#CC79A7', '#009E73', '#E69F00']

# Figure dimensions (double column ~183mm width)
FIG_WIDTH_INCHES = 12
# FIG_WIDTH_INCHES = 7.2
FIG_HEIGHT_INCHES = 8.5

# Style settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,  # Base font size
    'axes.labelsize': 8,  # Axis labels
    'axes.titlesize': 8,  # Panel titles
    'axes.titleweight': 'bold',
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
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
df_global = df_global[df_global['SCENARIO'].isin(STAGE_ORDER)]

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
# ------
# ------
# FIGURE
# ------
fig = plt.figure(figsize=(FIG_WIDTH_INCHES, FIG_HEIGHT_INCHES))

# GridSpec: 4 rows x 6 cols (5 data + 1 spacer)
gs = gridspec.GridSpec(4, 6, figure=fig,
                       width_ratios=[2.2, 0.025, 1, 1, 1, 1],  # Spacer is 0.15 relative width
                       height_ratios=[1, 1, 1, 1],
                       wspace=0.05, hspace=0.15)

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
        cur_featurestats_df = plot.plot_stage_panel(
            ax, df_sub, feature_col, color, group_name=group_name, columns=STAGE_ORDER, n_scenarios=N_STAGES,
            stage_labels=STAGE_LABELS, y_limits=FIXED_YLIM, show_x=is_bottom_row, show_y=is_left_col,
            is_main=is_main, is_top_row=is_top_row, is_first=is_first)

        # Collect feature stats in table
        if panel_counter == 0:
            featurestats_df = cur_featurestats_df.copy()
        else:
            featurestats_df = pd.concat([featurestats_df, cur_featurestats_df], axis=0)

        # Column headers
        if row == 0:
            ax.set_title(group_name, fontsize=10, fontweight='bold', pad=14, color='black')

        # y-axis label (only for first column)
        if is_left_col:
            var_title = var_title.replace(" ", r"\ ")
            label_text = r"$\mathbf{" + var_title + "}$" + " ($\sigma$)"
            ax.set_ylabel(label_text, fontsize=9, color="black", labelpad=4)

        # Panel letters: (a), (b), ...
        letter = chr(97 + panel_counter)
        ax.text(0.05, 0.92, f"{letter}", transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left',
                color='black', zorder=10)  # Always black for readability

        panel_counter += 1

# Add shared x-axis label
fig.supxlabel('Stage number', fontsize=9, fontweight='bold')

# Final layout adjustment (you may need to slightly increase the bottom margin from 0.07 to 0.08 so the label doesn't get cut off)
plt.subplots_adjust(left=0.05, right=0.98, top=0.95, bottom=0.08)

# Save function helper
# plt.savefig('transition_plot.pdf', dpi=300, bbox_inches='tight')

# ------------------------
# CONFIGURATION FOR MANUSCRIPT TABLE
# ------------------------
STAGE_ORDER_NAMES = ['1', '2', '3', '4', '5', '6', '7', '8']
scen_map = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8'}

# CHANGED: 'Global forests' instead of 'All sites'
IGBP_ORDER = ['Global forests', 'ENF', 'DBF', 'MF', 'EBF']
VAR_ORDER = ['VPD', 'TA', 'SM', 'SW']

# Map raw variable names to display names
var_map = {
    'VPD_ZSCORE_SHAPVALS_OVR_AVG': 'VPD',
    'TA_ZSCORE_SHAPVALS_OVR_AVG': 'TA',
    'SWC_ZSCORE_SHAPVALS_OVR_AVG': 'SM',
    'SWIN_ZSCORE_SHAPVALS_OVR_AVG': 'SW',
    'NET_SHAPVALS': 'Net sum'
}

# ---------------------------------------------------------
# 1. DEFINE SCENARIO CONDITIONS (THE HEADER LOGIC)
# ---------------------------------------------------------
STAGE_DEFINITIONS = {
    'VPD':
        ['normal', 'unrestricted', 'unrestricted', 'unrestricted', 'unrestricted', 'very dry'],
    'TA':
        ['normal', 'warm', 'warm', 'hot', 'hot', 'hot'],
    'SM':
        ['normal', 'normal', 'dry', 'dry', 'very dry', 'very dry'],
    'SW':
        ['normal', 'unrestricted', 'unrestricted', 'unrestricted', 'unrestricted', 'unrestricted']
}

# ---------------------------------------------------------
# 2. CALCULATE SCENARIO STATS (Existing Code)
# ---------------------------------------------------------
print("Calculating scenario stats...")

# Global
# Calculate scenario stats across all sites
stage_stats = stages.calculate_stage_stats(
    df_input=df_global, igbp='global', stage_order=STAGE_ORDER, vars=VARS,
    shap_suffix_avg=SHAP_SUFFIX_AVG, shap_suffix_sd=SHAP_SUFFIX_SD)

# IGBPs
# Calculate scenario stats for each IGBP
for i, igbp in enumerate(IGBP_CLASSES):
    df_igbp = df_global[df_global['IGBP'] == igbp]
    igbp_data = stages.calculate_stage_stats(
        df_input=df_igbp, igbp=igbp, stage_order=STAGE_ORDER, vars=VARS,
        shap_suffix_avg=SHAP_SUFFIX_AVG, shap_suffix_sd=SHAP_SUFFIX_SD)
    stage_stats = pd.concat([stage_stats, igbp_data], axis=0)

# ---------------------------------------------------------
# 3. PREPARE DATA FOR TABLE
# ---------------------------------------------------------
df = stage_stats.copy()

# Determine column name for variable
if 'Variable' in df.columns:
    var_col = 'Variable'
elif 'Feature' in df.columns:
    var_col = 'Feature'
else:
    var_col = 'Variable'  # Fallback

df['Driver'] = df[var_col].map(var_map).fillna(df[var_col])
df['Stage_Name'] = df['stage'].map(scen_map)

# CHANGED: Map 'global' to 'Global forests'
igbp_map = {'global': 'Global forests'}
df['IGBP_Display'] = df['igbp'].replace(igbp_map)

# Format Statistics
def format_stats(r):
    if pd.isna(r['mean']): return "n/a"
    m = r['mean']
    sd = r['total_sd']
    if 'min' in r and 'max' in r and not pd.isna(r['min']):
        return f"{m:.2f}±{sd:.2f} ({r['min']:.2f}, {r['max']:.2f})"
    else:
        return f"{m:.2f}±{sd:.2f}"

df['Stats'] = df.apply(format_stats, axis=1)

# Pivot
table_str = df.pivot_table(index=['IGBP_Display', 'Driver'],
                           columns='Stage_Name',
                           values='Stats',
                           aggfunc='first')
table_str = table_str.reindex(columns=STAGE_ORDER_NAMES)

# ---------------------------------------------------------
# 4. CONSTRUCT ROWS (HEADER + DATA)
# ---------------------------------------------------------
final_rows = []

# # --- A. ADD SCENARIO NUMBER ROW ---
# # This creates the row: "Scenario" | 1 | 2 | 3 | 4 | 5 | 6
# row_scen_num = {'index': 'Scenario'}
# for col in SCENARIO_ORDER_NAMES:
#     row_scen_num[col] = col
# final_rows.append(row_scen_num)

# --- B. ADD SCENARIO CONDITIONS HEADER ---
# 1. Main Header Title
final_rows.append({'index': 'Scenario conditions',
                   '1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''})

# 2. Condition Rows (VPD, Temp, etc.)
for driver in VAR_ORDER:
    # Get the list of conditions for this driver
    conditions = STAGE_DEFINITIONS.get(driver, ['?'] * 6)

    row_cond = {'index': f"  {driver}"}
    for idx, col_name in enumerate(STAGE_ORDER_NAMES):
        if idx < len(conditions):
            row_cond[col_name] = conditions[idx]
        else:
            row_cond[col_name] = '-'
    final_rows.append(row_cond)

final_rows.append({'index': '',
                   '1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''})

final_rows.append({'index': 'IGBP / Environmental driver',
                   '1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''})

# --- C. ADD DATA ROWS (IGBP GROUPS) ---
# Iterate through 'Global forests' then IGBPs
for igbp in IGBP_ORDER:
    # Header Row (IGBP Name)
    final_rows.append({
        'index': f"{igbp}",
        '1': '', '2': '', '3': '', '4': '', '5': '', '6': '', '7': '', '8': ''
    })

    if igbp in table_str.index.get_level_values(0):
        igbp_data = table_str.loc[igbp]

        # Component Rows
        for driver in VAR_ORDER:
            row_dict = {'index': f"  {driver}"}
            if driver in igbp_data.index:
                for col in STAGE_ORDER_NAMES:
                    val = igbp_data.loc[driver, col]
                    row_dict[col] = val if pd.notna(val) else '-'
            else:
                for col in STAGE_ORDER_NAMES: row_dict[col] = '-'
            final_rows.append(row_dict)

        # Net Sum Row
        net_row_dict = {'index': "  Net sum"}
        if 'Net sum' in igbp_data.index:
            for col in STAGE_ORDER_NAMES:
                val = igbp_data.loc['Net sum', col]
                net_row_dict[col] = val if pd.notna(val) else '-'
        else:
            for col in STAGE_ORDER_NAMES: net_row_dict[col] = '-'
        final_rows.append(net_row_dict)

# ---------------------------------------------------------
# 5. FINALIZE & SAVE
# ---------------------------------------------------------
table_1_final = pd.DataFrame(final_rows)
table_1_final.set_index('index', inplace=True)

# Save
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath_excel = dir_out / f'56_ExtendedData_TABLE-4_Stages_SinaPlots_ShapMeans_{FLUX}.xlsx'
table_1_final.to_excel(outfilepath_excel, index=True)

# Save all collected stats in separate csv
outfilepath = dir_out / f'56_ExtendedData_TABLE-4_Stages_SinaPlots_ShapMeans_{FLUX}_DATA-FeatureStatsFull.csv'
featurestats_df.to_csv(outfilepath, index=False, encoding='utf-8-sig')

# Show table
pd.set_option('display.max_rows', 3000)
pd.set_option('display.width', 1000)
print(f"Table saved to: {outfilepath_excel}")
print(table_1_final.to_string(index=True))

# Save fig to file
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'56_ExtendedData_FIG-1_Stages_SinaPlots_ShapMeans_{FLUX}.png'
fig.savefig(outfilepath, dpi=300, bbox_inches='tight')

# Show figure
plt.show()
