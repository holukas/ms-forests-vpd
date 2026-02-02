from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import src.files as files
import src.plot as plot
import src.scenarios as scenarios

# ==========================================
# SETTINGS
# ==========================================
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal', 'Hot & dry', 'Compound\nextremes']

# Variables
VARS = ['SWIN_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE']

VAR_LABELS = {
    'VPD_ZSCORE': 'Vapor pressure deficit effect',
    'TA_ZSCORE': 'Air temperature effect',
    'SWC_ZSCORE': 'Soil moisture effect',
    'SWIN_ZSCORE': 'Radiation effect'
}

IGBP_NAMES = {
    'ENF': 'Evergreen needleleaf forests',
    'DBF': 'Deciduous broadleaf forests',
    'MF': 'Mixed forests',
    'EBF': 'Evergreen broadleaf forests'
}

# Column suffixes
SHAP_SUFFIX_AVG = '_SHAPVALS_OVR_AVG'
SHAP_SUFFIX_SD = '_SHAPVALS_OVR_SD'  # Using SD column for uncertainty

PALETTE = {
    'VPD_ZSCORE': '#D55E00',
    'TA_ZSCORE': '#CC79A7',
    'SWC_ZSCORE': '#009E73',
    'SWIN_ZSCORE': '#E69F00'
}
BLUE = '#0072B2'

# Paths
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
filepath = Path(results_outdir) / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet"

# ==========================================
# EXECUTION
# ==========================================

# Data
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()

# Select required IGBPs and scenarios
df_global = df_main[df_main['IGBP'].isin(IGBP_CLASSES)]
df_global = df_global[df_global['SCENARIO'].isin(SCENARIO_ORDER)]

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

# Get limits for y-axis scaling, same for all plots
GRAND_Y_MIN, GRAND_Y_MAX = plot.get_panel_limits(df=scenario_stats)
FIXED_YLIM = (GRAND_Y_MIN * 1.05, GRAND_Y_MAX * 1.2)

# ------
# FIGURE
# ------
fig = plt.figure(figsize=(18, 9), dpi=300)
outer_gs = gridspec.GridSpec(1, 2, width_ratios=[0.55, 0.45], wspace=0.1)
gs_left = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[0])
gs_right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1], wspace=0.15, hspace=0.1)

# Panel settings
panels_data = []

# Global
_data = scenario_stats.loc[scenario_stats['igbp'] == 'global'].copy()
panels_data.append({'data': _data, 'title': "a | Global forest response (all sites)",
                    'is_small': False, 'gs': gs_left[0], 'show_scenario_labels': True})

# IGBP
for i, igbp in enumerate(IGBP_CLASSES):
    row, col = i // 2, i % 2
    letter = chr(98 + i)
    show_scenario_lables = True if row == 1 else False
    _data = scenario_stats.loc[scenario_stats['igbp'] == igbp].copy()
    panels_data.append({'data': _data, 'title': f"{letter} | {IGBP_NAMES[igbp]}",
                        'is_small': True, 'gs': gs_right[row, col], 'show_scenario_labels': show_scenario_lables})



# Draw
print("Drawing panels...")
for pix, p in enumerate(panels_data):
    ax = fig.add_subplot(p['gs'])
    plot.draw_panel(ax=ax, df=p['data'], title=p['title'], fixed_ylim=FIXED_YLIM, is_small=p['is_small'],
                    show_scenario_labels=p['show_scenario_labels'], vars=VARS, palette=PALETTE,
                    scenario_labels=SCENARIO_LABELS)
    if not p['is_small']:
        ax.text(x=-0.97, y=0.04, s=r'$\uparrow$' + 'Positive effect ($\sigma$)\nincreased uptake\nreduced release',
                fontsize=12, color='black', ha='left', va='bottom')
        ax.text(x=-0.97, y=-0.04, s='increased release\nreduced uptake\n' + r'$\downarrow$Negative effect ($\sigma$)',
                fontsize=12, color='black', ha='left', va='top')

        # Draw up and down area arrows
        # Define vertices
        vertices_up = [(-0.3, 0), (-0.3, 0.15), (-0.65, 0.2), (-1, 0.15), (-1, 0), (-0.3, 0)]
        vertices_down = [(-0.3, 0), (-0.3, -0.15), (-0.65, -0.2), (-1, -0.15), (-1, 0), (-0.3, 0)]


        def add_gradient_arrow(vertices, color_main, direction='up'):
            # Create Path
            path = mpath.Path(vertices)
            patch = mpatches.PathPatch(path, facecolor='none', edgecolor='none')
            ax.add_patch(patch)

            # Define gradient (top to bottom)
            # Custom colormap from chosen color to a lighter/faded version
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            if direction == 'down':
                gradient = np.flipud(gradient)  # Flip for the down arrow

            # Display and clip
            # Extent should cover the bounding box of the arrow
            ymin, ymax = (0, 0.2) if direction == 'up' else (-0.2, 0)
            im = ax.imshow(gradient, interpolation='bicubic',
                           extent=[-1, -0.3, ymin, ymax],
                           cmap=plt.cm.colors.LinearSegmentedColormap.from_list('custom', [color_main, '#ffffff']),
                           aspect='auto', alpha=0.6, zorder=3)
            im.set_clip_path(patch)


        # Add arrows
        add_gradient_arrow(vertices_up, '#829460', direction='up')  # Sage Green
        add_gradient_arrow(vertices_down, '#4E6E81', direction='down')  # Slate Blue

# Legend
legend_elements = [Patch(facecolor=c, label=l) for l, c in zip(VAR_LABELS.values(), PALETTE.values())]
# noinspection PyTypeChecker
legend_elements.append(Line2D([0], [0], color='none', marker='D', markerfacecolor='white',
                              markeredgecolor='black', markeredgewidth=2, markersize=10, label='Net effect'))
# noinspection PyTypeChecker
legend_elements.append(Line2D([0], [0], color='black', marker='|', markeredgewidth=2, markersize=10, lw=0,
                              label='Standard error of the mean'))
fig.legend(handles=legend_elements, loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)

# Adjust
plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.1)

dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'51_FIG-1_SankeyPlotSHAPValuesScenarios_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()
