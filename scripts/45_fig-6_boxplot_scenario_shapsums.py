from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt

import src.files as files

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'NEP'
# xvar = 'VPD'
# yvar = 'VPD'  # SHAP values
# aggfunc = 'median'
CONDITIONAL = True  # SHAP

# filename_x = 'TA'
# zvar = 'TA'  # Colors

# # Plot settings
# title = f"The effect of {yvar} on {FLUX}"
# xlabel = f"{xvar} (z-score)"
# ylabel = f"SHAP value of {yvar} (z-score)"
# n_sites_min = 30
#
# show_txt_effect = True
# show_shap_thresholds = True
# show_z_colors = False
# show_fit = True
# # ------------------------------

# x = (f"BIN_{xvar}", aggfunc)
# y = (f"{yvar}_SHAPVALS", aggfunc)
# y_counts = (f"{yvar}_SHAPVALS", "count")
# z = (f"BIN_{zvar}", aggfunc)

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Load SHAP values aggregated across all sites
filepath = Path(results_outdir) / f"3_AllSites_SHAP-ScenarioSums-{shap_type}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

df = shapvals_df.copy()
df = df.drop('SITE', axis=1, inplace=False)
df = df.drop('CONDITION', axis=1, inplace=False)
# df = df.drop('IGBP', axis=1, inplace=False)
df = df[['IGBP', 'SCENARIO', 'VPD_SHAPVALS_OVR_AVG', 'TA_SHAPVALS_OVR_AVG', 'SWC_SHAPVALS_OVR_AVG',
         'SWIN_SHAPVALS_OVR_AVG']].copy()

# Melt the DataFrame to a long format for plotting
shap_value_columns = ['VPD_SHAPVALS_OVR_AVG', 'TA_SHAPVALS_OVR_AVG', 'SWC_SHAPVALS_OVR_AVG', 'SWIN_SHAPVALS_OVR_AVG']
df_melted = df.melt(
    id_vars=['IGBP', 'SCENARIO'],
    value_vars=shap_value_columns,
    var_name='SHAP_FEATURE',
    value_name='SHAP_VALUE'
)

# Drop rows with NaN values
df_melted.dropna(subset=['SHAP_VALUE'], inplace=True)

# Define the order of features to be plotted and their colors
ordered_features = ['VPD_SHAPVALS_OVR_AVG', 'TA_SHAPVALS_OVR_AVG', 'SWC_SHAPVALS_OVR_AVG', 'SWIN_SHAPVALS_OVR_AVG']
# facecolors = ['#9A425A', '#4CC9B3', '#FF9966', '#61A4E7']  # Example colors from a Matplotlib palette
# linecolors = ['#7B3548', '#3D9E8F', '#CC7A52', '#4D83B9']  # Example colors from a Matplotlib palette
facecolors = ['#EF5350', '#FFCA28', '#26C6DA', '#FFEE58']  # Example colors from a Matplotlib palette
linecolors = ['#B71C1C', '#FF6F00', '#01579B', '#F57F17']  # Example colors from a Matplotlib palette
feature_labels = ['VPD', 'TA', 'SWC', 'SWIN']

# Get unique scenarios and sort them for consistent plotting
scenarios = sorted(df_melted['SCENARIO'].unique())

# Prepare data for plotting
data_to_plot = []
labels = []
box_positions = []
gap = .5
group_width = len(ordered_features)
current_pos = 1

for scenario in scenarios:
    for i, feature in enumerate(ordered_features):
        subset = df_melted[(df_melted['SCENARIO'] == scenario) & (df_melted['SHAP_FEATURE'] == feature)]['SHAP_VALUE']
        data_to_plot.append(subset)
        labels.append(f"{feature_labels[i]}")
        box_positions.append(current_pos + i)
    current_pos += group_width + gap

# Create the boxplot
fig, ax = plt.subplots(figsize=(15, 3))
bp = ax.boxplot(data_to_plot, positions=box_positions, patch_artist=True, showfliers=False)

# Assign colors to the box plots
for i in range(len(bp['boxes'])):
    feature_index = i % len(ordered_features)
    bp['boxes'][i].set_facecolor(facecolors[feature_index])
    bp['boxes'][i].set_edgecolor(linecolors[feature_index])
    bp['medians'][i].set_color('black')
    # Set whisker colors in pairs
    # Note: Whiskers are in pairs (2 per box), so we access them with 2*i and 2*i + 1
    bp['whiskers'][2 * i].set_color(linecolors[feature_index])
    bp['whiskers'][2 * i + 1].set_color(linecolors[feature_index])
    # Set cap colors
    # Caps are also in pairs
    bp['caps'][2 * i].set_color(linecolors[feature_index])
    bp['caps'][2 * i + 1].set_color(linecolors[feature_index])

# Add visual grouping boxes for scenarios
scenario_centers = []
start_pos = 1
for s in scenarios:
    end_pos = start_pos + group_width - 1
    rect = plt.Rectangle((start_pos - 0.5, ax.get_ylim()[0]), group_width, ax.get_ylim()[1] - ax.get_ylim()[0],
                         color='lightgray', zorder=0, alpha=0.5)
    ax.add_patch(rect)
    scenario_centers.append((start_pos + end_pos) / 2)
    start_pos += group_width + gap

# Set main x-axis labels for scenarios
ax.set_xticks(scenario_centers)
ax.set_xticklabels([f"SCENARIO {s}" for s in scenarios])
ax.tick_params(axis='x', which='major', pad=15, length=0)

# Add feature labels below the main scenario labels
for i, pos in enumerate(box_positions):
    ax.text(pos, -0.1, labels[i], ha='center', va='top', fontsize=10, rotation=45, transform=ax.get_xaxis_transform())

# Set plot titles and labels
ax.set_title('Distribution of SHAP Values by SCENARIO and Feature', fontsize=16)
ax.set_ylabel('SHAP Value', fontsize=12)
ax.set_xlabel('Scenario and Feature', fontsize=12, labelpad=30)

# Hide spines and grid for a cleaner look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(False)
ax.axhline(0, color='black', linestyle='--', linewidth=1)

plt.tight_layout()
plt.show()
