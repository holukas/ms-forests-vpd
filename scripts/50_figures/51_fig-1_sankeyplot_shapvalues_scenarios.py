from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
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
SCENARIO_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]
SCENARIO_LABELS = [f"Scenario {sl}" for sl in SCENARIO_ORDER]

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

AX_LABELS_FONTSIZE = 12

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
    scenario_stats = pd.concat([scenario_stats, igbp_data], axis=0)

# Get limits for y-axis scaling, same for all plots
GRAND_Y_MIN, GRAND_Y_MAX = plot.get_panel_limits(df=scenario_stats)
FIXED_YLIM_MAIN = (GRAND_Y_MIN * 1.05, GRAND_Y_MAX * 2.2)
FIXED_YLIM_SUB = (GRAND_Y_MIN * 1.05, GRAND_Y_MAX * 1)

# ------
# FIGURE
# ------

# Figure settings
gain = 1.3
fig = plt.figure(figsize=(11 * gain, 9 * gain), dpi=150)
outer_gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], width_ratios=[1], wspace=0.1)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[0])
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[1], wspace=0.15, hspace=0.1)

# Panel settings
panels_data = []

# Global
_data = scenario_stats.loc[scenario_stats['igbp'] == 'global'].copy()
panels_data.append({'data': _data, 'title': "a | Global forests",
                    'is_small': False, 'gs': gs_top[0], 'show_scenario_labels': True})

# IGBP
for i, igbp in enumerate(IGBP_CLASSES):
    row, col = 0, i
    letter = chr(98 + i)
    show_scenario_lables = True if row == 1 else False
    _data = scenario_stats.loc[scenario_stats['igbp'] == igbp].copy()
    panels_data.append({'data': _data, 'title': f"{letter} | {IGBP_NAMES[igbp]}",
                        'is_small': True, 'gs': gs_bottom[row, col], 'show_scenario_labels': show_scenario_lables})

# Draw
print("Drawing panels...")
for pix, p in enumerate(panels_data):
    ax = fig.add_subplot(p['gs'])
    fixedy = FIXED_YLIM_MAIN if not p['is_small'] else FIXED_YLIM_SUB

    # Draw panel
    plot.draw_panel(ax=ax, df=p['data'], title=p['title'], fixed_ylim=fixedy, is_small=p['is_small'],
                    show_scenario_labels=p['show_scenario_labels'], vars=VARS, palette=PALETTE,
                    scenario_ids=SCENARIO_ORDER, scenario_labels=SCENARIO_LABELS, shap_suffix_avg=SHAP_SUFFIX_AVG)

    if not p['is_small']:
        color_limzone = '#d6604d'
        color_facilzone = '#4393c3'
        ax.text(x=-1.2, y=0.04, s='Carbon gain', fontweight='bold', zorder=99,
                fontsize=AX_LABELS_FONTSIZE * 1.3, color=color_facilzone, ha='center', va='bottom')
        ax.text(x=-1.2, y=-0.04, s='Carbon penalty', fontweight='bold', zorder=99,
                fontsize=AX_LABELS_FONTSIZE * 1.3, color=color_limzone, ha='center', va='top')

        # Draw up and down area arrows
        # Define vertices
        vertices_up = [(-0.5, 0), (-0.5, 0.15), (-1.2, 0.25), (-1.9, 0.15), (-1.9, 0), (-0.5, 0)]
        vertices_down = [(-0.5, 0), (-0.5, -0.15), (-1.2, -0.25), (-1.9, -0.15), (-1.9, 0), (-0.5, 0)]
        # Add arrows
        plot.add_gradient_arrow(ax=ax, vertices=vertices_up, color_main=color_facilzone, direction='up')  # Sage Green
        plot.add_gradient_arrow(ax=ax, vertices=vertices_down, color_main=color_limzone, direction='down')  # Slate Blue

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

# dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
# outfilepath = dir_out / f'51_FIG-1_SankeyPlotSHAPValuesScenarios_{FLUX}.png'
# print(f"Saved to {outfilepath}")
# plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()
