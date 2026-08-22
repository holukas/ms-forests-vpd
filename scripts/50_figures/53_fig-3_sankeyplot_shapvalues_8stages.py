from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import src.files as files
import src.plot as plot
import src.stages as stages
from src.paths import load_settings

# ==========================================
# SETTINGS
# ==========================================
FLUX = 'NEP_ZSCORE'

# Run variant. An empty string reads the results behind the submitted figures and
# writes to the baseline plot folder. Any other value reads the matching variant
# folder and writes the figures next to it, so a sensitivity run cannot overwrite a
# published figure. The aggregation must have run with the same value.
VARIANT = ""
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
STAGE_ORDER = [1, 2, 3, 4, 5, 6, 7, 8]
STAGE_LABELS = [f"Stage {sl}" for sl in STAGE_ORDER]

# Variables
VARS = ['SWIN_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE']

VAR_LABELS = {
    'VPD_ZSCORE': 'VPD effect',
    'TA_ZSCORE': 'TA effect',
    'SWC_ZSCORE': 'SM effect',
    'SWIN_ZSCORE': 'SW effect'
    # 'VPD_ZSCORE': 'Vapor pressure deficit effect',
    # 'TA_ZSCORE': 'Air temperature effect',
    # 'SWC_ZSCORE': 'Soil moisture effect',
    # 'SWIN_ZSCORE': 'Radiation effect'
}

IGBP_NAMES = {
    'ENF': 'ENF',  # 'Evergreen needleleaf forests',
    'DBF': 'DBF',  # 'Deciduous broadleaf forests',
    'MF': 'MF',  # 'Mixed forests',
    'EBF': 'EBF'  # 'Evergreen broadleaf forests'
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
settings = load_settings()
shap_type = 'conditional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT
filepath = Path(results_outdir) / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet"

# ==========================================
# EXECUTION
# ==========================================

# Data
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()

# Select required IGBPs and stages (scenarios)
df_global = df_main[df_main['IGBP'].isin(IGBP_CLASSES)]
df_global = df_global[df_global['SCENARIO'].isin(STAGE_ORDER)]

# ------------------------
# CALCULATE STAGE (SCENARIO) STATS
# ------------------------
print("Calculating stage stats...")

# Global
# Calculate stage (scenario) stats across all sites
stage_stats = stages.calculate_stage_stats(
    df_input=df_global, igbp='global', stage_order=STAGE_ORDER, vars=VARS,
    shap_suffix_avg=SHAP_SUFFIX_AVG, shap_suffix_sd=SHAP_SUFFIX_SD)

# IGBPs
# Calculate stage (scenario) stats for each IGBP
for i, igbp in enumerate(IGBP_CLASSES):
    df_igbp = df_global[df_global['IGBP'] == igbp]
    igbp_data = stages.calculate_stage_stats(
        df_input=df_igbp, igbp=igbp, stage_order=STAGE_ORDER, vars=VARS,
        shap_suffix_avg=SHAP_SUFFIX_AVG, shap_suffix_sd=SHAP_SUFFIX_SD)
    stage_stats = pd.concat([stage_stats, igbp_data], axis=0)

# Get limits for y-axis scaling, same for all plots
GRAND_Y_MIN, GRAND_Y_MAX = plot.get_panel_limits(df=stage_stats)
FIXED_YLIM_MAIN = (GRAND_Y_MIN * 1.03, GRAND_Y_MAX * 1.55)
FIXED_YLIM_SUB = (GRAND_Y_MIN * 1.05, GRAND_Y_MAX * 1.4)

# ------
# FIGURE
# ------

# Figure settings
gain = 1.3
fig = plt.figure(figsize=(12 * gain, 9 * gain), dpi=150)
outer_gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], width_ratios=[1], hspace=0.05)
gs_top = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[0])
gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer_gs[1], wspace=0.15)

# Panel settings
panels_data = []

# Global
_data = stage_stats.loc[stage_stats['igbp'] == 'global'].copy()
panels_data.append({'data': _data, 'title': "a | Global forests",
                    'is_small': False, 'gs': gs_top[0], 'show_stage_labels': True})

# IGBP
for i, igbp in enumerate(IGBP_CLASSES):
    row, col = 0, i
    letter = chr(98 + i)
    show_stage_lables = True if row == 1 else False
    _data = stage_stats.loc[stage_stats['igbp'] == igbp].copy()
    panels_data.append({'data': _data, 'title': f"{letter} | {IGBP_NAMES[igbp]}",
                        'is_small': True, 'gs': gs_bottom[row, col], 'show_stage_labels': show_stage_lables})

# -----------
# DRAW PANELS
# -----------
first_ax = None
print("Drawing panels...")
for pix, p in enumerate(panels_data):
    ax = fig.add_subplot(p['gs'])
    if pix == 0:
        first_ax = ax  # Store first ax b/c I want the legend here
    fixedy = FIXED_YLIM_MAIN if not p['is_small'] else FIXED_YLIM_SUB

    # Draw panel
    plot.draw_panel(ax=ax, df=p['data'], title=p['title'], fixed_ylim=fixedy, is_small=p['is_small'],
                    show_stage_labels=p['show_stage_labels'], vars=VARS, palette=PALETTE,
                    stage_ids=STAGE_ORDER, stage_labels=STAGE_LABELS, shap_suffix_avg=SHAP_SUFFIX_AVG,
                    fontsize=AX_LABELS_FONTSIZE)

    if not p['is_small']:
        color_limzone = '#d6604d'
        color_facilzone = '#4393c3'
        ax.text(x=-1.2, y=0.04, s='Stimulation', fontweight='bold', zorder=99, style='italic',
                fontsize=AX_LABELS_FONTSIZE * 1.2, color=color_facilzone, ha='center', va='bottom')
        ax.text(x=-1.2, y=-0.04, s='Suppression', fontweight='bold', zorder=99, style='italic',
                fontsize=AX_LABELS_FONTSIZE * 1.2, color=color_limzone, ha='center', va='top')

        # Draw up and down area arrows
        # Define vertices
        vertices_up = [(-0.6, 0), (-0.6, 0.15), (-1.2, 0.25), (-1.8, 0.15), (-1.8, 0), (-0.5, 0)]
        vertices_down = [(-0.6, 0), (-0.6, -0.15), (-1.2, -0.25), (-1.8, -0.15), (-1.8, 0), (-0.5, 0)]
        # Add arrows
        plot.add_gradient_arrow(ax=ax, vertices=vertices_up, color_main=color_facilzone, direction='up')  # Sage Green
        plot.add_gradient_arrow(ax=ax, vertices=vertices_down, color_main=color_limzone, direction='down')  # Slate Blue

        # Y-axis label for the entire figure
        ax.text(-2.2, 0, r'Effect on daytime NEP ($\sigma$)', va='center', ha='center',
                rotation='vertical', fontsize=AX_LABELS_FONTSIZE + 2, fontweight='bold')

# Legend
legend_elements = [Patch(facecolor=c, label=l) for l, c in zip(VAR_LABELS.values(), PALETTE.values())]
# noinspection PyTypeChecker
legend_elements.append(Line2D([0], [0], color='none', marker='D', markerfacecolor='white',
                              markeredgecolor='black', markeredgewidth=2, markersize=10,
                              label=r'Net effect ($\sigma$)'))
# noinspection PyTypeChecker
legend_elements.append(Line2D([0], [0], color='black', marker='|', markeredgewidth=2, markersize=10, lw=0,
                              label=r'Standard error of the mean ($\sigma$)'))

first_ax.legend(handles=legend_elements, loc='lower left', ncol=3,
                bbox_to_anchor=(0.15, 0.14), frameon=False, fontsize=AX_LABELS_FONTSIZE)

# Adjust
plt.subplots_adjust(left=0.035, right=0.975, top=0.95, bottom=0.02)

# Save fig
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT
dir_out.mkdir(parents=True, exist_ok=True)
outfilepath = dir_out / f'53_FIG-3_SankeyPlotStages_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

# Save panels data
stage_stats.to_csv(dir_out / f"53_FIG-3_SankeyPlotStages_{FLUX}_DATA.csv", index=False)

plt.show()
