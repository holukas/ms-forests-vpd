from pathlib import Path

import diive as dv
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import src.files as files

# ==========================================
# 1. SETTINGS
# ==========================================
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal', 'Hot & dry', 'Compound\nextremes']

# Variable Order
VARS = ['SWIN_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE']

VAR_LABELS = {
    'VPD_ZSCORE': 'VPD effect',
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

# Column Suffixes
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
# 2. HELPER FUNCTIONS
# ==========================================
def sigmoid(x, x_start, x_end, y_start, y_end):
    x_norm = (x - x_start) / (x_end - x_start)
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


def calculate_global_sd(means, stds):
    """
    Calculates the pooled Global Standard Deviation from site-level statistics.
    Formula: sqrt(Mean(Within-Site Variance) + Variance(Between-Site Means))
    """
    means = np.array(means)
    stds = np.array(stds)
    # 1. Average Within-Site Variance
    mean_variance = np.mean(stds ** 2) if len(stds) > 0 else 0
    # 2. Variance Between Sites
    between_site_variance = np.var(means) if len(means) > 0 else 0
    # 3. Total Global SD
    return np.sqrt(mean_variance + between_site_variance)


def calculate_budget_stats(df_input):
    scen_data = []
    net_vals = []
    net_errs = []
    site_net_counts = []

    for scen_id in SCENARIO_ORDER:
        df_scen = df_input[df_input['SCENARIO'] == scen_id].copy()

        if len(df_scen) == 0:
            # scen_dict = {v: {'mean': 0} for v in VARS}
            # scen_data.append(scen_dict)
            # net_vals.append(0)
            # net_errs.append(0)
            # continue
            raise ValueError(f"Scenario {scen_id} has no data.")

        scen_dict = {}

        # Individual drivers (calculate mean for bars)
        for var in VARS:
            col_avg = var + SHAP_SUFFIX_AVG
            if col_avg not in df_scen.columns:
                raise ValueError(f"Column '{col_avg}' not found in dataframe.")

            # Get the data vector for the specific driver
            data_vec = df_scen[col_avg]

            # Calculate the Mean (for the bar height)
            mean_val = data_vec.mean()

            # Calculate the SEM (Standard Error of the Mean)
            # Note: sem = std / sqrt(n)
            sem_val = data_vec.sem()
            count_val = data_vec.count()

            # Store in dict
            scen_dict[var] = {'mean': mean_val, 'sem': sem_val}

        # Net effect (calculate mean and pooled SD for error bars)
        # Site-level means (sum of drivers per site)
        all_avg_cols = [v + SHAP_SUFFIX_AVG for v in VARS if (v + SHAP_SUFFIX_AVG) in df_scen.columns]
        site_net_means = df_scen[all_avg_cols].sum(axis=1)
        site_net_counts.append(len(df_scen[all_avg_cols].dropna()))

        # # Site-level SDs (approximate via propagation if direct column missing)
        # all_sd_cols = [v + SHAP_SUFFIX_SD for v in VARS if (v + SHAP_SUFFIX_SD) in df_scen.columns]
        # if len(all_sd_cols) > 0:
        #     # Sqrt(Sum of Variances) assuming independence (Lower bound approx)
        #     site_net_variances = (df_scen[all_sd_cols] ** 2).sum(axis=1)
        #     site_net_sds = np.sqrt(site_net_variances)
        # else:
        #     site_net_sds = np.zeros(len(df_scen))

        # # 3. Calculate Global Pooled SD
        global_net_mean = site_net_means.mean()
        global_net_sem = site_net_means.sem()
        # global_net_sd = calculate_global_sd(site_net_means, site_net_sds)

        net_vals.append(global_net_mean)
        net_errs.append(global_net_sem)

        scen_data.append(scen_dict)

    return scen_data, net_vals, net_errs, site_net_counts


def get_panel_limits(data, net_vals, net_errs):
    max_vals = []
    min_vals = []

    # Check Stack Heights
    for d in data:
        pos_sum = sum([d[v]['mean'] for v in VARS if d[v]['mean'] > 0])
        neg_sum = sum([d[v]['mean'] for v in VARS if d[v]['mean'] < 0])
        max_vals.append(pos_sum)
        min_vals.append(neg_sum)

    # Check Net Lines with SD Errors
    for val, err in zip(net_vals, net_errs):
        max_vals.append(val + err)
        min_vals.append(val - err)

    return min(min_vals), max(max_vals)


def draw_panel(ax, data, net_vals, net_errs, net_counts, title, fixed_ylim, show_scenario_labels, is_small=False):
    x_centers = [0, 1, 2]
    bar_width = 0.28 if not is_small else 0.28
    x_centers_shifted_left = np.array(x_centers) - bar_width / 3
    x_centers_shifted_right = np.array(x_centers) + bar_width / 2.5
    alpha_ribbon = 0.25

    fs_val = 12 if not is_small else 12
    fs_label = 12 if not is_small else 12
    fs_tick = 12 if not is_small else 12

    node_pos = [{} for _ in range(len(data))]

    # Draw bars
    for i, d in enumerate(data):
        cx = x_centers[i]

        # Positive Stack
        current_y = 0.0
        for var in VARS:
            val = d[var]['mean']
            if val < 0:
                continue
            top = current_y + val
            bottom = current_y
            ax.bar(cx, val, width=bar_width, bottom=bottom, color=PALETTE[var], edgecolor='white', linewidth=0.5,
                   zorder=10)
            if (abs(val) > 0.01) and not is_small:
                ax.text(x_centers_shifted_right[i], bottom + val / 2, f"+{val:.2f}", ha='right', va='center',
                        fontsize=fs_val - 1, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.2, foreground=PALETTE[var])], zorder=20)
            node_pos[i][var] = (bottom, top)
            current_y += val

        # Negative Stack
        current_y = 0.0
        for var in VARS:
            val = d[var]['mean']
            if val >= 0:
                continue
            top = current_y
            bottom = current_y + val
            ax.bar(cx, abs(val), width=bar_width, bottom=bottom, color=PALETTE[var], edgecolor='white', linewidth=0.5,
                   zorder=10)
            if (abs(val) > 0.01) and not is_small:
                ax.text(x_centers_shifted_right[i], bottom + abs(val) / 2, f"{val:.2f}", ha='right', va='center',
                        fontsize=fs_val - 1, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.2, foreground=PALETTE[var])], zorder=20)
            node_pos[i][var] = (bottom, top)
            current_y += val

        # Scenario Labels
        if show_scenario_labels:
            ax.text(cx, fixed_ylim[0] + (abs(fixed_ylim[0]) * 0.07), SCENARIO_LABELS[i],
                    ha='center', va='top', fontsize=fs_tick, fontweight='bold')

    # Ribbons
    for i in range(len(data) - 1):
        x_start, x_end = x_centers[i] + bar_width / 2, x_centers[i + 1] - bar_width / 2
        x_curve = np.linspace(x_start, x_end, 100)
        for var in VARS:
            if var not in node_pos[i] or var not in node_pos[i + 1]: continue
            s_bot, s_top = node_pos[i][var]
            e_bot, e_top = node_pos[i + 1][var]
            if abs(s_top - s_bot) < 0.005 and abs(e_top - e_bot) < 0.005:
                continue
            y_top = sigmoid(x_curve, x_start, x_end, s_top, e_top)
            y_bot = sigmoid(x_curve, x_start, x_end, s_bot, e_bot)
            ax.fill_between(x_curve, y_bot, y_top, color=PALETTE[var], alpha=alpha_ribbon, edgecolor='none', zorder=1)

    # Net values (marker + pooled sd)
    ax.plot(x_centers_shifted_left, net_vals, '-', color='#808080', linewidth=2, markersize=10, zorder=19)

    mew = 1.5 if is_small else 2
    elinewidth = 1.5 if is_small else 2
    ax.errorbar(x_centers_shifted_left, net_vals, yerr=net_errs, fmt='D', color='white',
                ecolor='black', elinewidth=elinewidth, capsize=4, zorder=23, ms=10, mec='black', mew=mew)

    # Net Labels
    for x, y in zip(x_centers_shifted_left, net_vals):
        offset = 0
        bbox = dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6)
        ax.text(x - 0.1, y + offset, f"{y:+.2f}", fontsize=fs_val, fontweight='bold',
                ha='right', va='center', bbox=bbox, zorder=25)

    # Counts
    for x, y in zip(x_centers, net_counts):
        ypos = 0.42 if is_small else 0.37
        smaller = 1 if is_small else 0
        ax.text(x, ypos, f"n={y}", fontsize=fs_val - smaller, fontweight='normal',
                ha='center', va='center', zorder=25)

    # Styling
    ax.axhline(0, color='black', linewidth=1, linestyle='--', zorder=5)
    ax.set_title(title, fontsize=fs_label, fontweight='bold', loc='left', pad=10)
    ax.set_ylim(fixed_ylim)
    ax.set_xlim(-0.5, 2.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ==========================================
# 3. EXECUTION
# ==========================================
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()

# Figure setup
fig = plt.figure(figsize=(18, 9), dpi=300)
outer_gs = gridspec.GridSpec(1, 2, width_ratios=[0.55, 0.45], wspace=0.1)
gs_left = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=outer_gs[0])
gs_right = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer_gs[1], wspace=0.15, hspace=0.1)

# --- PASS 1: CALC & LIMITS ---
print("Calculating stats...")
panels_data = []

# Global
df_global = df_main[df_main['IGBP'].isin(IGBP_CLASSES)]
df_global = df_global[df_global['SCENARIO'].isin(SCENARIO_ORDER)]
g_data, g_net, g_err, g_counts = calculate_budget_stats(df_global)
panels_data.append({
    'data': g_data, 'net': g_net, 'err': g_err, 'counts': g_counts,
    'title': "a | Global forest response (all sites)",
    'is_small': False, 'gs': gs_left[0], 'show_scenario_labels': True
})

# Subpanels
for i, igbp in enumerate(IGBP_CLASSES):
    df_sub = df_main[df_main['IGBP'] == igbp]
    s_data, s_net, s_err, s_counts = calculate_budget_stats(df_sub)

    row, col = i // 2, i % 2
    letter = chr(98 + i)
    show_scenario_lables = True if row == 1 else False
    panels_data.append({
        'data': s_data, 'net': s_net, 'err': s_err, 'counts': s_counts,
        'title': f"{letter} | {IGBP_NAMES[igbp]}",
        'is_small': True, 'gs': gs_right[row, col],
        'show_scenario_labels': show_scenario_lables
    })

# Get limits for y-axis scaling, same for all plots
all_mins, all_maxs = [], []
for p in panels_data:
    p_min, p_max = get_panel_limits(p['data'], p['net'], p['err'])
    all_mins.append(p_min)
    all_maxs.append(p_max)
GRAND_Y_MIN = min(all_mins)
GRAND_Y_MAX = max(all_maxs)
FIXED_YLIM = (GRAND_Y_MIN * 1.05, GRAND_Y_MAX * 1.2)

# Draw
print("Drawing panels...")
for p in panels_data:
    ax = fig.add_subplot(p['gs'])
    draw_panel(ax, p['data'], p['net'], p['err'], p['counts'], p['title'], FIXED_YLIM, is_small=p['is_small'],
               show_scenario_labels=p['show_scenario_labels'])
    if not p['is_small']:
        ax.text(-0.5, 0.05, "Mean Expected Value (0$\sigma$)", ha='left', va='bottom',
                fontsize=10, style='italic', color='#333333')

# Legend
legend_elements = [Patch(facecolor=c, label=l) for l, c in zip(VAR_LABELS.values(), PALETTE.values())]
# noinspection PyTypeChecker
legend_elements.append(Line2D([0], [0], color='none', marker='D', markerfacecolor='white',
                              markeredgecolor='black', markeredgewidth=2, markersize=10, label='Net effect'))
# noinspection PyTypeChecker
legend_elements.append(Line2D([0], [0], color='black', marker='|', markeredgewidth=2, markersize=10, lw=0,
                              label='Standard error'))
fig.legend(handles=legend_elements, loc='lower center', ncol=6,
           bbox_to_anchor=(0.5, 0.02), frameon=False, fontsize=12)

# Adjust
plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.1)

dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'55_FIG-SHAP_Budget_Unified_Connected_SD_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()
