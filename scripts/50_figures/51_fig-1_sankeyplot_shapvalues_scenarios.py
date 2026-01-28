from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import diive as dv
import src.files as files
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ==========================================
# 1. SETTINGS
# ==========================================
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal\nConditions', 'Hot & Dry\nTransition', 'Compound\nExtremes']

# Variable Order (Bottom to Top for Positive Stack)
VARS = ['SWIN_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE']

VAR_LABELS = {
    'VPD_ZSCORE': 'VPD',
    'SWC_ZSCORE': 'Soil Water',
    'TA_ZSCORE': 'Temp',
    'SWIN_ZSCORE': 'Radiation'
}

# Column Suffixes (Adjust to match your parquet file exactly)
SHAP_SUFFIX_AVG = '_SHAPVALS_OVR_AVG'  # Site-level Mean
SHAP_SUFFIX_SD = '_SHAPVALS_SD'  # Site-level SD

# Palette (High Contrast)
PALETTE = {
    'VPD_ZSCORE': '#D55E00',  # Vermillion
    'SWC_ZSCORE': '#009E73',  # Bluish Green
    'TA_ZSCORE': '#CC79A7',  # Reddish Purple
    'SWIN_ZSCORE': '#E69F00'  # Orange/Yellow
}

# Paths
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
filepath = Path(results_outdir) / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet"


# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def calculate_global_sd(means, stds):
    """
    Calculates the pooled Global Standard Deviation from site-level statistics.
    Formula: sqrt(Mean(Within-Site Variance) + Variance(Between-Site Means))
    """
    means = np.array(means)
    stds = np.array(stds)

    # 1. Average Within-Site Variance
    mean_variance = np.mean(stds ** 2)

    # 2. Variance Between Sites
    between_site_variance = np.var(means)

    # 3. Total Global SD
    return np.sqrt(mean_variance + between_site_variance)


def sigmoid(x, x_start, x_end, y_start, y_end):
    """Sigmoid curve for smooth ribbons."""
    x_norm = (x - x_start) / (x_end - x_start)
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


# ==========================================
# 3. DATA LOAD & CALCULATION
# ==========================================
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()
df_main = df_main.loc[df_main['IGBP'].isin(IGBP_CLASSES)].copy()

scenario_data = []
net_nep_values = []  # Means for black line
global_sd_values = []  # Error bars for black line

print("Calculating Net SHAP Budgets and Pooled SD...")
for scen_id in SCENARIO_ORDER:
    df_scen = df_main[df_main['SCENARIO'] == scen_id].copy()

    scen_dict = {}
    total_shap_sum = 0

    # --- A. Calculate Individual Drivers (Bars) ---
    for var in VARS:
        col_avg = var + SHAP_SUFFIX_AVG

        # Check if SD column exists, otherwise assume 0 (robustness)
        col_sd = var + SHAP_SUFFIX_SD
        if col_sd not in df_scen.columns:
            # Try to infer or warn
            # print(f"Warning: {col_sd} not found. Assuming 0 SD.")
            site_stds = np.zeros(len(df_scen))
        else:
            site_stds = df_scen[col_sd].copy()

        site_means = df_scen[col_avg].copy()

        # Calculate Global Stats
        net_impact = site_means.mean()
        global_sd = calculate_global_sd(site_means, site_stds)

        # Store
        scen_dict[var] = {'mean': net_impact, 'std': global_sd}
        total_shap_sum += net_impact

    # --- B. Calculate Net System State (Black Line) ---
    # 1. Site-Level NET Means (Sum of drivers per site)
    all_avg_cols = [v + SHAP_SUFFIX_AVG for v in VARS]
    site_net_means = df_scen[all_avg_cols].sum(axis=1)

    # 2. Site-Level NET SDs
    # Ideally, use the actual Target Flux SD if available
    # target_sd_col = 'NEP_ZSCORE_SD'
    # if target_sd_col in df_scen.columns:
    #     site_net_sds = df_scen[target_sd_col]
    # else:

    # Approx: Sqrt(Sum of Variances) assuming independence (Lower bound approx)
    all_sd_cols = [v + SHAP_SUFFIX_SD for v in VARS if (v + SHAP_SUFFIX_SD) in df_scen.columns]
    if len(all_sd_cols) > 0:
        site_net_variances = (df_scen[all_sd_cols] ** 2).sum(axis=1)
        site_net_sds = np.sqrt(site_net_variances)
    else:
        site_net_sds = np.zeros(len(df_scen))

    # 3. Calculate Global Net SD
    net_global_sd = calculate_global_sd(site_net_means, site_net_sds)

    net_nep_values.append(total_shap_sum)
    global_sd_values.append(net_global_sd)

    scenario_data.append(scen_dict)


# ==========================================
# 4. PLOTTING ENGINE
# ==========================================
def draw_budget_with_pooled_sd(ax, data, var_list, palette, x_labels, net_means, net_sds):
    x_centers = [0, 1, 2]
    bar_width = 0.22
    alpha_ribbon = 0.35

    # Valid Color Tuple for Error Bars (Dark Grey, 60% Opacity)
    err_color = (0.2, 0.2, 0.2, 0.6)

    # Track positions for ribbons
    node_pos = [{} for _ in range(len(data))]

    # -----------------------------
    # A. DRAW BARS, VALUES & SDs
    # -----------------------------
    for i, d in enumerate(data):
        cx = x_centers[i]

        # --- 1. Positive Stack ---
        current_y = 0.0
        for var in var_list:
            val = d[var]['mean']
            sd = d[var]['std']

            if val < 0: continue

            top = current_y + val
            bottom = current_y

            # Draw Bar
            ax.bar(cx, val, width=bar_width, bottom=bottom,
                   color=palette[var], edgecolor='white', linewidth=0.5, zorder=10)

            # Draw SD Bar
            if val > 0.01:
                y_center = bottom + val / 2
                ax.errorbar(cx, y_center, yerr=sd, fmt='none',
                            ecolor=err_color, elinewidth=0.8, capsize=2, zorder=15)

            # Label Value
            if val > 0.02:
                fs = 8 if val > 0.1 else 6
                ax.text(cx + 0.02, bottom + val / 2, f"{val:.2f}",
                        ha='left', va='center', fontsize=fs, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.2, foreground=palette[var])], zorder=20)

            node_pos[i][var] = (bottom, top)
            current_y += val

        # Positive Total
        if current_y > 0.1:
            ax.text(cx, current_y + 0.15, f"+{current_y:.2f}", ha='center', va='bottom',
                    fontsize=8, color='#555555', fontweight='bold')

        # --- 2. Negative Stack ---
        current_y = 0.0
        for var in var_list:
            val = d[var]['mean']
            sd = d[var]['std']

            if val >= 0: continue

            top = current_y
            bottom = current_y + val

            # Draw Bar
            ax.bar(cx, abs(val), width=bar_width, bottom=bottom,
                   color=palette[var], edgecolor='white', linewidth=0.5, zorder=10)

            # Draw SD Bar
            if abs(val) > 0.01:
                y_center = current_y + val / 2
                ax.errorbar(cx, y_center, yerr=sd, fmt='none',
                            ecolor=err_color, elinewidth=0.8, capsize=2, zorder=15)

            # Label Value
            if abs(val) > 0.02:
                fs = 8 if abs(val) > 0.1 else 6
                ax.text(cx + 0.02, bottom + abs(val) / 2, f"{val:.2f}",
                        ha='left', va='center', fontsize=fs, color='white', fontweight='bold',
                        path_effects=[pe.withStroke(linewidth=1.2, foreground=palette[var])], zorder=20)

            node_pos[i][var] = (bottom, top)
            current_y += val

        # Negative Total
        if abs(current_y) > 0.1:
            ax.text(cx, current_y - 0.15, f"{current_y:.2f}", ha='center', va='top',
                    fontsize=8, color='#555555', fontweight='bold')

        # Scenario Label
        ax.text(cx, -2.1, x_labels[i], ha='center', va='top', fontsize=10, fontweight='bold')

    # -----------------------------
    # B. DRAW RIBBONS
    # -----------------------------
    for i in range(len(data) - 1):
        x_start = x_centers[i] + bar_width / 2
        x_end = x_centers[i + 1] - bar_width / 2
        x_curve = np.linspace(x_start, x_end, 300)

        for var in var_list:
            if var not in node_pos[i] or var not in node_pos[i + 1]: continue
            start_bot, start_top = node_pos[i][var]
            end_bot, end_top = node_pos[i + 1][var]

            if abs(start_top - start_bot) < 0.005 and abs(end_top - end_bot) < 0.005: continue

            y_top_curve = sigmoid(x_curve, x_start, x_end, start_top, end_top)
            y_bot_curve = sigmoid(x_curve, x_start, x_end, start_bot, end_bot)

            ax.fill_between(x_curve, y_bot_curve, y_top_curve,
                            color=palette[var], alpha=alpha_ribbon, edgecolor='none', zorder=1)

    # -----------------------------
    # C. NET LINE WITH SD
    # -----------------------------
    # White halo
    # ax.plot(x_centers, net_means, color='white', linewidth=5, linestyle='-', zorder=19)
    # Black line
    # ax.plot(x_centers, net_means, color='black', markers=9, linewidth=2.5, linestyle='none', zorder=20)

    # Error Bars on Net Line
    ax.errorbar(x_centers, net_means, yerr=net_sds, fmt='o', color='black', ms=9,
                ecolor='black', elinewidth=1.5, capsize=4, zorder=21, label='Net Anomaly (±SD)')

    # Labels
    for x, y in zip(x_centers, net_means):
        bbox_props = dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8, lw=0.5)
        ax.text(x - 0.05, y, f"{y:+.2f}",
                fontsize=9, fontweight='bold', ha='right', va='center', bbox=bbox_props, zorder=25)

    # -----------------------------
    # D. STYLING
    # -----------------------------
    ax.axhline(0, color='black', linewidth=1, linestyle='--', zorder=99)
    ax.text(-0.5, -0.05, "Expected Value (0$\sigma$)", ha='left', va='bottom',
            fontsize=8, style='italic', color='#333333')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    # Calculate limits dynamically
    all_y = []
    for i, d in enumerate(data):
        pos_sum = sum([v['mean'] for v in d.values() if v['mean'] > 0])
        neg_sum = sum([v['mean'] for v in d.values() if v['mean'] < 0])
        all_y.extend([pos_sum + 0.3, neg_sum - 0.3])  # Add buffer for text

    # Add net line extent
    all_y.extend([m + s for m, s in zip(net_means, net_sds)])
    all_y.extend([m - s for m, s in zip(net_means, net_sds)])

    y_max = max(all_y)
    y_min = min(all_y)

    ax.set_ylim(y_min * 1.1, y_max * 1.1)
    ax.set_xlim(-0.5, 2.5)

    # Zone Text
    ax.text(-0.45, y_max * 0.8, "Enhancement\n(+Contribution)",
            ha='left', va='center', fontsize=9, color='gray', alpha=0.6)
    ax.text(-0.45, y_min * 0.8, "Suppression\n(-Contribution)",
            ha='left', va='center', fontsize=9, color='gray', alpha=0.6)


# ==========================================
# 5. EXECUTE
# ==========================================
fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

draw_budget_with_pooled_sd(ax, scenario_data, VARS, PALETTE, SCENARIO_LABELS, net_nep_values, global_sd_values)

# plt.suptitle("Biophysical Attribution of Forest Carbon Anomalies", fontsize=13, fontweight='bold', y=0.98)
# plt.title("Net SHAP contribution ± Pooled Standard Deviation across sites",
#           fontsize=9, color='#404040', pad=10)

legend_elements = [Patch(facecolor=c, label=l) for l, c in zip(VAR_LABELS.values(), PALETTE.values())]
legend_elements.append(
    Line2D([0], [0], color='black', marker='|', markeredgewidth=1.5, markersize=10, lw=0, label='Std. Dev.'))
legend_elements.append(Line2D([0], [0], color='black', lw=2, marker='o', label='Net Anomaly'))

ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.08),
          ncol=6, frameon=False, fontsize=9)

plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)


# # Save
# dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
# outfilepath = dir_out / f'55_FIG-SHAP_Budget_Net_PooledSD_{FLUX}.png'
# print(f"Saved to {outfilepath}")
# plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()