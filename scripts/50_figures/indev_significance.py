from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import scikit_posthocs as sp
from scipy import stats
import src.files as files

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Latin Modern Roman'] + plt.rcParams['font.serif']

# ------------------------------
# Variables
# NEP, NEE, LE, GPP, RECO, TA, VPD, SWIN, SWC
FLUX = 'NEP'
# aggfunc = 'median'
CONDITIONAL = True  # SHAP

# Load settings
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional' if CONDITIONAL else 'standard'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type

# Load SHAP values aggregated across all sites
filepath = Path(results_outdir) / f"3_AllSites_SHAP-ScenarioSums-{shap_type}_{FLUX}.parquet"
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

# Your data processing logic (same as before)
df = shapvals_df.copy()
df = df.drop('SITE', axis=1, inplace=False)
df = df.drop('CONDITION', axis=1, inplace=False)

# df = df.loc[df['N_VALUES'] > 6].copy()

# Stats
subset = df[['SCENARIO', 'N_VALUES']].copy()
subset_stats = subset.groupby('SCENARIO').describe()
print(subset_stats.loc[1])
print(subset_stats.loc[4])
print(subset_stats.loc[5])
# print(f"Used scenarios:\n"
#       f"#1: {subset_stats.loc[1].values[0]} values on average per site\n"
#       f"#4: {subset_stats.loc[4].values[0]} values\n"
#       f"#5: {subset_stats.loc[5].values[0]} values")

shaps = '_SHAPVALS_OVR_MEDIAN'
# shaps = '_SHAPVALS_OVR_ABS_MEDIAN'
# shaps = '_SHAPVALS_POS_AVG'
# shaps = '_SHAPVALS_NEG_AVG'
variables = ['SWIN', 'TA', 'VPD', 'SWC']
shap_value_columns = [variables[0] + shaps, variables[1] + shaps, variables[2] + shaps, variables[3] + shaps]

dfcols = ['IGBP', 'SCENARIO'] + shap_value_columns
df = df[dfcols].copy()
df_melted = df.melt(
    id_vars=['IGBP', 'SCENARIO'],
    value_vars=shap_value_columns,
    var_name='SHAP_FEATURE',
    value_name='SHAP_VALUE'
)
df_melted.dropna(subset=['SHAP_VALUE'], inplace=True)

# Define colors for each feature
feature_colors = ['#b2df8a', '#fdbf6f', '#fb9a99', '#a6cee3']
line_colors = ['#33a02c', '#ff7f00', '#e31a1c', '#1f77b4']
# feature_colors = ['#88c0d0', '#bf616a', '#a3be8c', '#ebcb8b']
# line_colors = ['#5e81ac', '#a4424e', '#7a996b', '#d0af6f']
# feature_colors = ['#48d0b7', '#f8719e', '#a286fc', '#f7a85c']
# line_colors = ['#009688', '#d94575', '#7c59eb', '#f58728']
# feature_colors = ['#9fa6ff', '#ff8673', '#69f2c2', '#d09fff']
# line_colors = ['#636efa', '#ef553b', '#00cc96', '#ab63fa']
# feature_colors = ['#F0E442', '#E69F00', '#56B4E9', '#CC79A7']
# line_colors = ['#cba90d', '#b87f00', '#0072B2', '#a35f85']
# feature_colors = ['#66c2a5', '#fdb462', '#8da0cb', '#e78ac3']
# line_colors = ['#1b7837', '#b15928', '#386cb0', '#c51b7d']
# feature_colors = ['#FFEB3B', '#FF9800', '#EF5350', '#64B5F6', ]
# line_colors = ['#F9A825', '#BF360C', '#B71C1C', '#0D47A1', ]
# feature_colors = ['#EF9A9A', '#FFE082', '#90CAF9', '#C5E1A5']
# line_colors = ['#880E4F', '#E65100', '#01579B', '#33691E']


# Get unique scenarios
# scenarios = sorted(df_melted['SCENARIO'].unique())
scenarios = [1, 4, 5]
n_scenarios = len(scenarios)
scenario_description = {
    0: "(a) All data",
    1: "(a) Normal conditions",
    2: "(c) Dry soil",
    3: "(d) Hot temperatures",
    4: "(b) Dry soil and hot temperatures",
    5: "(c) Compound extremes"
}

# Determine subplot grid layout
ncols = 3
nrows = int(np.ceil(n_scenarios / ncols))

n_vals = -9999

def generate_significance_letters(p_values_df, alpha=0.05):
    """
    Generates significance letters (Compact Letter Display) from a DataFrame of p-values.

    Args:
        p_values_df (pd.DataFrame): A square DataFrame where cell (i, j) contains the
                                    p-value for the comparison between group i and group j.
        alpha (float): The significance level.

    Returns:
        dict: A dictionary mapping group names to their significance letters (e.g., 'a', 'b', 'ab').
    """
    from collections import defaultdict

    # Extract groups and create a boolean DataFrame of significant differences
    groups = p_values_df.columns
    significant = p_values_df < alpha

    # Group variables that are NOT significantly different from each other
    # This is equivalent to finding cliques in the graph of non-significant connections
    non_significant_groups = []
    for i in range(len(groups)):
        for j in range(i, len(groups)):
            if not significant.iloc[i, j]:
                # Find a group to add these to
                found_group = False
                for group in non_significant_groups:
                    if groups[i] in group or groups[j] in group:
                        group.add(groups[i])
                        group.add(groups[j])
                        found_group = True
                        break
                if not found_group:
                    non_significant_groups.append({groups[i], groups[j]})

    # Assign letters to each non-significant group
    letters = 'abcdefghijklmnopqrstuvwxyz'
    group_letters = {letter: group for letter, group in zip(letters, non_significant_groups)}

    # Create the final labels for each variable
    final_labels = defaultdict(str)
    for group_name in groups:
        for letter, group in group_letters.items():
            if group_name in group:
                final_labels[group_name] += letter

    return dict(final_labels)

# Create the figure and GridSpec
fig = plt.figure(figsize=(5 * ncols, 5 * nrows))
gs = gridspec.GridSpec(nrows, ncols, figure=fig)
ax_objects = []

# Loop through scenarios to create subplots and plot
for i, scenario in enumerate(scenarios):
    ax = fig.add_subplot(gs[i // ncols, i % ncols])
    ax_objects.append(ax)

    scenario_data = df_melted[df_melted['SCENARIO'] == scenario]

    # --- START: STATISTICAL ANALYSIS FOR THIS SCENARIO ---
    significance_letters = {}
    if not scenario_data.empty:
        # Perform Dunn's test to get pairwise p-values
        dunn_results = sp.posthoc_dunn(scenario_data, val_col='SHAP_VALUE', group_col='SHAP_FEATURE', p_adjust='holm')

        # Clean up the column and index names for the function
        dunn_results.columns = variables
        dunn_results.index = variables

        # Call the helper function to get the letters
        significance_letters = generate_significance_letters(dunn_results)
    # --- END: STATISTICAL ANALYSIS ---

    # Iterate through each feature to plot it
    for pos, feature_name in enumerate(shap_value_columns):
        # Filter data for the specific feature
        feature_data = scenario_data[scenario_data['SHAP_FEATURE'] == feature_name]['SHAP_VALUE']
        color = feature_colors[pos]

        if pos == 0:
            n_vals = len(feature_data)

        # # The "Rain" (Strip Plot), with random jitter to the x-axis
        # jitter = np.random.normal(loc=pos - 0.25, scale=0.04, size=len(feature_data))
        # ax.scatter(jitter, feature_data, color=line_colors[pos], alpha=0.2, s=5, zorder=2)

        # 1. Calculate KDE first
        kde = gaussian_kde(feature_data)

        # 2. Evaluate density at each data point's y-value
        density_at_points = kde(feature_data)

        # 3. Scale the density to determine the max spread (width) at each point
        # The '0.2' controls the maximum width of the sina plot. Adjust as needed.
        sina_width = (density_at_points / density_at_points.max()) * 0.2

        # 4. Generate random offsets scaled by the calculated width
        random_offsets = np.random.uniform(-1, 1, size=len(feature_data)) * sina_width

        # 5. Define the final x-positions for the scatter plot
        # The 'pos - 0.25' centers the sina plot to the left, matching your old jitter.
        sina_x = pos - 0.15 + random_offsets

        # The "Rain" (Sina Plot)
        ax.scatter(sina_x, feature_data, color=line_colors[pos],
                   edgecolor='none',
                   alpha=0.3, s=16, zorder=2)

        # Boxplot (Summary Plot)
        box = ax.boxplot(
            feature_data,
            positions=[pos + 0.15],
            showfliers=False,
            widths=0.15,
            patch_artist=True,
            boxprops=dict(facecolor=feature_colors[pos], edgecolor=line_colors[pos], alpha=0.8),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color=line_colors[pos]),
            capprops=dict(color=line_colors[pos]),
            zorder=3
        )

        # # Half-violin plot, calculate kernel density estimate
        # if not feature_data.empty:
        #     kde = gaussian_kde(feature_data)
        #     # Create a range of y-values to evaluate the KDE
        #     y_range = np.linspace(feature_data.min(), feature_data.max(), 100)
        #     # Evaluate, scale and shift the KDE
        #     density = kde(y_range)
        #     scaled_density = density / density.max() * 0.4  # Scale width of the violin
        #     # Plot filled density curve
        #     ax.fill_betweenx(
        #         y_range,
        #         pos + 0.05,  # Shift to the right of center
        #         pos + 0.05 + scaled_density,  # Create violin shape
        #         color=color,
        #         alpha=0.5,
        #         zorder=1,
        #         edgecolor=line_colors[pos]
        #     )

        # --- START: ADD SIGNIFICANCE LETTERS ---
        if significance_letters:
            # Get the letter for the current variable
            letter = significance_letters.get(variables[pos], '')

            # Determine y-position for the letter (slightly above the data)
            # We use the top of the upper whisker from the boxplot as a reference
            top_whisker = box['whiskers'][1].get_ydata()[1]
            y_max_plot = ax.get_ylim()[1]
            # Add a small offset (e.g., 2% of the y-axis range)
            offset = (y_max_plot - ax.get_ylim()[0]) * 0.02
            y_pos = top_whisker + offset

            ax.text(pos + 0.15, y_pos, letter,
                    ha='center', va='bottom', fontsize=12, color='black', fontweight='bold')
        # --- END: ADD SIGNIFICANCE LETTERS ---

    # Add title
    ax.set_title(f'{scenario_description[scenario]}', fontsize=14, x=0.05, y=.95, horizontalalignment='left')

    # Add number of values
    infotxt = f"n = {n_vals}"
    if i == 0:
        infotxt += f" sites"
    ax.text(0.7, 0.85, infotxt, transform=ax.transAxes, color='black', size=10,
            ha='left', va='center', zorder=99, backgroundcolor='white')

    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylabel('')
    ax.set_xlabel('')

    # Manually set x-ticks and labels since we control the positions
    ax.set_xticks(range(len(shap_value_columns)))
    if i >= n_scenarios - ncols:  # Check if bottom-row plot
        ax.set_xticklabels(variables, rotation=0, ha='center', fontsize=14)
    else:
        ax.set_xticklabels([])
        ax.set_xticks([])

    if i % ncols != 0:  # Check if it's not a first-column plot
        ax.set_yticks([])

    # Zero line
    ax.axhline(0, color='#B0BEC5', linestyle='-', linewidth=1, zorder=0)

    # Hide spines and grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if i % ncols != 0:
        ax.spines['left'].set_visible(False)
    if i < n_scenarios - ncols:
        ax.spines['bottom'].set_visible(False)
    ax.grid(False)

    # Legend
    # if i == 0:
    #     legend_elements = [Patch(facecolor=color, label=label, edgecolor='gray')
    #                        for color, label in zip(feature_colors, feature_labels)]
    #     ax.legend(handles=legend_elements, loc='lower left', bbox_to_anchor=(0.05, 0.05),
    #               fontsize=12, frameon=False, ncol=2)

# Share y-axis across all subplots
y_min_list = [ax.get_ylim()[0] for ax in ax_objects]
y_max_list = [ax.get_ylim()[1] for ax in ax_objects]
y_min = min(y_min_list)
y_max = max(y_max_list)

for ax in ax_objects:
    ax.set_ylim(y_min, y_max * 1.08)
    ax.set_xlim(-0.5, len(shap_value_columns) - 0.5)  # Adjust x-limits

# Add single ylabel for the entire figure
fig.text(0.04, 0.5, 'Median impact on NEP (z-scores)', va='center', rotation='vertical', fontsize=16)

# Hide any unused subplots
for i in range(n_scenarios, len(ax_objects)):
    ax_objects[i].axis('off')

fig.tight_layout(rect=[0.05, 0, 1, 1])
gs.update(hspace=.1, wspace=.2)
plt.show()
