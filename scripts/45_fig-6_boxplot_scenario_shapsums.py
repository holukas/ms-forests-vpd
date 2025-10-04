from pathlib import Path

import diive as dv
import matplotlib.pyplot as plt

import src.files as files
import numpy as np
import matplotlib.gridspec as gridspec

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'NEP'
# aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Load settings
settings = files.read_settings_file("../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Load SHAP values aggregated across all sites
filepath = Path(results_outdir) / f"3_AllSites_SHAP-ScenarioSums-{shap_type}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

df = shapvals_df.copy()
# df = df.loc[df['IGBP'] == 'ENF'].copy()
df = df.drop('SITE', axis=1, inplace=False)
df = df.drop('CONDITION', axis=1, inplace=False)
# df = df.drop('IGBP', axis=1, inplace=False)
# shap_value_columns = ['VPD_SHAPVALS_NEG_AVG', 'TA_SHAPVALS_NEG_AVG', 'SWC_SHAPVALS_NEG_AVG', 'SWIN_SHAPVALS_NEG_AVG']
# shap_value_columns = ['VPD_SHAPVALS_POS_AVG', 'TA_SHAPVALS_POS_AVG', 'SWC_SHAPVALS_POS_AVG', 'SWIN_SHAPVALS_POS_AVG']
# shap_value_columns = ['VPD_SHAPVALS_OVR_ABS_AVG', 'TA_SHAPVALS_OVR_ABS_AVG', 'SWC_SHAPVALS_OVR_ABS_AVG', 'SWIN_SHAPVALS_OVR_ABS_AVG']
# shap_value_columns = ['VPD_SHAPVALS_OVR_ABS_MEDIAN', 'TA_SHAPVALS_OVR_ABS_MEDIAN', 'SWC_SHAPVALS_OVR_ABS_MEDIAN', 'SWIN_SHAPVALS_OVR_ABS_MEDIAN']
shap_value_columns = ['VPD_SHAPVALS_OVR_MEDIAN', 'TA_SHAPVALS_OVR_MEDIAN', 'SWC_SHAPVALS_OVR_MEDIAN', 'SWIN_SHAPVALS_OVR_MEDIAN']
# shap_value_columns = ['VPD_SHAPVALS_OVR_AVG', 'TA_SHAPVALS_OVR_AVG', 'SWC_SHAPVALS_OVR_AVG', 'SWIN_SHAPVALS_OVR_AVG']
dfcols = ['IGBP', 'SCENARIO'] + shap_value_columns
df = df[dfcols].copy()

# Melt the DataFrame to a long format for plotting

df_melted = df.melt(
    id_vars=['IGBP', 'SCENARIO'],
    value_vars=shap_value_columns,
    var_name='SHAP_FEATURE',
    value_name='SHAP_VALUE'
)

# Drop rows with NaN values
df_melted.dropna(subset=['SHAP_VALUE'], inplace=True)

# Define colors for each feature
feature_colors = ['#EF9A9A', '#FFE082', '#90CAF9', '#C5E1A5']
line_colors = ['#880E4F', '#E65100', '#01579B', '#33691E']
feature_labels = ['VPD', 'TA', 'SWC', 'SWIN']

# Get unique scenarios
scenarios = sorted(df_melted['SCENARIO'].unique())
n_scenarios = len(scenarios)
scenario_description = {
    0: "(a) All data",
    1: "(b) Normal conditions",
    2: "(c) Dry soil",
    3: "(d) Hot temperatures",
    4: "(e) Dry soil and hot temperatures",
    5: "(f) Compound extremes"
}

# Determine subplot grid layout
ncols = 3
nrows = int(np.ceil(n_scenarios / ncols))

# Create the figure and GridSpec
fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
gs = gridspec.GridSpec(nrows, ncols, figure=fig)
ax_objects = []

# Loop through scenarios to create subplots and plot
for i, scenario in enumerate(scenarios):
    ax = fig.add_subplot(gs[i // ncols, i % ncols])
    ax_objects.append(ax)

    # Filter data for the current scenario
    scenario_data = df_melted[df_melted['SCENARIO'] == scenario]


    # Prepare data for boxplot
    data_to_plot = [scenario_data[scenario_data['SHAP_FEATURE'] == f]['SHAP_VALUE'] for f in shap_value_columns]
    n_vals = len(data_to_plot[0])

    # Create the boxplot
    bp = ax.boxplot(data_to_plot, patch_artist=True, showfliers=True)

    # Apply colors
    for j in range(len(bp['boxes'])):
        # Box face and edge color
        bp['boxes'][j].set_facecolor(feature_colors[j])
        bp['boxes'][j].set_edgecolor(line_colors[j])

        # Median color
        bp['medians'][j].set_color('black')

        # Whiskers color
        bp['whiskers'][2 * j].set_color(line_colors[j])
        bp['whiskers'][2 * j + 1].set_color(line_colors[j])

        # Caps color
        bp['caps'][2 * j].set_color(line_colors[j])
        bp['caps'][2 * j + 1].set_color(line_colors[j])

    # The fliers are not ordered by box, so a single loop is needed
    for fix, flier in enumerate(bp['fliers']):
        flier.set(marker='o', markerfacecolor=line_colors[fix], markeredgecolor='none', alpha=0.5)


    # Set subplot labels and title
    if i == 0:
        ax.set_title(f'{scenario_description[scenario]}\nn={n_vals} sites', fontsize=14, x=0.05, y=1, horizontalalignment='left')
    else:
        ax.set_title(f'{scenario_description[scenario]}\nn={n_vals}', fontsize=14, x=0.05, y=1, horizontalalignment='left')
    # ax.set_title(f'SCENARIO {scenario}', fontsize=14)
    ax.set_xticks(range(1, len(shap_value_columns) + 1))
    ax.set_xticklabels(feature_labels, rotation=0, ha='center', fontsize=14)
    ax.tick_params(axis='y', labelsize=14)

    # Add a horizontal line at y=0 for reference
    ax.axhline(0, color='black', linestyle='--', linewidth=1)

    # Hide spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(False)

    # Legend
    if i == 0:
        ax.legend(bp['boxes'], feature_labels, loc='center', bbox_to_anchor=(0.5, 0.8),
                  fontsize=12, frameon=False, ncol=2)

# Share the y-axis across all subplots
y_min = min(ax.get_ylim()[0] for ax in ax_objects)
y_max = max(ax.get_ylim()[1] for ax in ax_objects)
for ax in ax_objects:
    ax.set_ylim(y_min, y_max)

# Add a single ylabel for the entire figure
fig.text(0.04, 0.5, 'SHAP Value', va='center', rotation='vertical', fontsize=14)

# Hide any unused subplots
for i in range(len(scenarios), len(ax_objects)):
    ax_objects[i].axis('off')

fig.tight_layout(rect=[0.05, 0, 1, 1])
gs.update(hspace=.3)
fig.show()