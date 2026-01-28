from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import diive as dv
import src.files as files

# ==========================================
# 1. SETTINGS
# ==========================================
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_LABELS = ['Normal', 'Dry & Hot', 'Compound\nExtremes']

# Variables to decompose
# We stack them in order of importance for visual clarity
VARS = ['SWIN_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE']

VAR_LABELS = {
    'VPD_ZSCORE': 'VPD',
    'SWC_ZSCORE': 'Soil Water',
    'TA_ZSCORE': 'Temperature',
    'SWIN_ZSCORE': 'Radiation'
}
SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'

# Palette (Okabe-Ito)
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
# 2. DATA LOAD & SPLIT LOGIC
# ==========================================
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()
df_main = df_main.loc[df_main['IGBP'].isin(IGBP_CLASSES)].copy()

# Structure: data[scenario_idx][variable] = {'pos': value, 'neg': value}
scenario_data = []
net_nep_values = []

print("Calculating split budgets...")
for scen_id in SCENARIO_ORDER:
    df_scen = df_main[df_main['SCENARIO'] == scen_id].copy()

    # 1. Calculate Net NEP Anomaly (Sum of all SHAP)
    # This is the "Truth" line
    all_cols = [v + SHAP_SUFFIX for v in VARS]
    df_scen['NET_SHAP'] = df_scen[all_cols].sum(axis=1)
    net_nep = df_scen['NET_SHAP'].mean()
    net_nep_values.append(net_nep)

    scen_dict = {}

    # 2. Split each variable into Positive and Negative contributions
    for var in VARS:
        col = var + SHAP_SUFFIX
        vals = df_scen[col].values

        # Calculate the mean POSITIVE impact (Enhancement)
        # We sum all positive values and divide by TOTAL count (to get population impact)
        pos_impact = np.sum(vals[vals > 0]) / len(vals)

        # Calculate the mean NEGATIVE impact (Suppression)
        neg_impact = np.sum(vals[vals < 0]) / len(vals)

        scen_dict[var] = {'pos': pos_impact, 'neg': neg_impact}

    scenario_data.append(scen_dict)


# ==========================================
# 3. PLOTTING ENGINE
# ==========================================
def sigmoid(x, x_start, x_end, y_start, y_end):
    x_norm = (x - x_start) / (x_end - x_start)
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


def draw_anomaly_budget(ax, data, var_list, palette, x_labels):
    x_centers = [0, 1, 2]
    bar_width = 0.15
    alpha_ribbon = 0.6

    # Tracking positions for connections
    # node_pos[scen][var] = {'pos_top', 'pos_bot', 'neg_top', 'neg_bot'}
    node_pos = [{} for _ in range(len(data))]

    # -----------------------------
    # A. DRAW BARS (NODES)
    # -----------------------------
    for i, d in enumerate(data):
        cx = x_centers[i]

        # --- POSITIVE STACK (Going Up from 0) ---
        current_y = 0.0
        # We reverse sort positive stack so Radiation (usually largest) is at bottom?
        # Let's keep consistent order.
        for var in var_list:
            val = d[var]['pos']
            if val < 0.01:
                # Register empty pos for connectivity
                node_pos[i][var] = {**node_pos[i].get(var, {}), 'pos_top': current_y, 'pos_bot': current_y}
                continue

            top = current_y + val
            bottom = current_y

            # Draw Bar
            ax.bar(cx, val, width=bar_width, bottom=bottom,
                   color=palette[var], edgecolor='white', linewidth=0.3, zorder=10)

            # Save coords
            node_pos[i][var] = {**node_pos[i].get(var, {}), 'pos_top': top, 'pos_bot': bottom}

            # Label (Enhancers) - Only on first scenario
            if i == 0 and val > 0.1:
                ax.text(cx - 0.1, bottom + val / 2, VAR_LABELS[var],
                        ha='right', va='center', fontsize=8, color=palette[var], fontweight='bold')

            current_y += val

        # --- NEGATIVE STACK (Going Down from 0) ---
        current_y = 0.0
        for var in var_list:
            val = d[var]['neg']  # Negative value
            if abs(val) < 0.01:
                node_pos[i][var] = {**node_pos[i].get(var, {}), 'neg_top': current_y, 'neg_bot': current_y}
                continue

            # Stack downwards
            top = current_y
            bottom = current_y + val

            # Draw Bar (Height is abs(val))
            ax.bar(cx, abs(val), width=bar_width, bottom=bottom,
                   color=palette[var], edgecolor='white', linewidth=0.3, zorder=10)

            # Save coords
            node_pos[i][var] = {**node_pos[i].get(var, {}), 'neg_top': top, 'neg_bot': bottom}

            # Label (Suppressors) - Only on first scenario
            if i == 0 and abs(val) > 0.1:
                ax.text(cx - 0.1, bottom + val / 2, VAR_LABELS[var],
                        ha='right', va='center', fontsize=8, color=palette[var], fontweight='bold')

            current_y += val

        # Scenario Label
        ax.text(cx, -1.9, x_labels[i], ha='center', va='top', fontweight='bold', fontsize=11)

    # -----------------------------
    # B. DRAW RIBBONS
    # -----------------------------
    for i in range(len(data) - 1):
        x_start = x_centers[i] + bar_width / 2
        x_end = x_centers[i + 1] - bar_width / 2
        x_curve = np.linspace(x_start, x_end, 300)

        # Connect Positives
        for var in var_list:
            start = node_pos[i].get(var, {})
            end = node_pos[i + 1].get(var, {})

            # Draw Positive Ribbon
            if 'pos_top' in start and 'pos_top' in end:
                h_start = start['pos_top'] - start['pos_bot']
                h_end = end['pos_top'] - end['pos_bot']
                if h_start > 0.005 or h_end > 0.005:
                    y_top = sigmoid(x_curve, x_start, x_end, start['pos_top'], end['pos_top'])
                    y_bot = sigmoid(x_curve, x_start, x_end, start['pos_bot'], end['pos_bot'])
                    ax.fill_between(x_curve, y_bot, y_top, color=palette[var], alpha=alpha_ribbon, edgecolor='none')

            # Draw Negative Ribbon
            if 'neg_top' in start and 'neg_top' in end:
                h_start = abs(start['neg_top'] - start['neg_bot'])
                h_end = abs(end['neg_top'] - end['neg_bot'])
                if h_start > 0.005 or h_end > 0.005:
                    y_top = sigmoid(x_curve, x_start, x_end, start['neg_top'], end['neg_top'])
                    y_bot = sigmoid(x_curve, x_start, x_end, start['neg_bot'], end['neg_bot'])
                    ax.fill_between(x_curve, y_bot, y_top, color=palette[var], alpha=alpha_ribbon, edgecolor='none')

    # -----------------------------
    # C. NET ANOMALY LINE
    # -----------------------------
    ax.plot(x_centers, net_nep_values, color='black', linewidth=2.5, linestyle='-', marker='o', zorder=20,
            label='Net NEP Anomaly')

    # Annotate Net Values
    for x, y in zip(x_centers, net_nep_values):
        label_y = y + 0.15 if y > 0 else y - 0.15
        ax.text(x + 0.1, label_y, f"Net: {y:+.2f}$\sigma$", fontsize=9, fontweight='bold', ha='left', va='center')

    # -----------------------------
    # D. AESTHETICS
    # -----------------------------
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.text(-0.5, 0.05, "Avg. Baseline (0$\sigma$)", fontsize=8, style='italic')

    ax.axis('off')

    # Dynamic Limits based on data
    y_max = max([sum([d[v]['pos'] for v in var_list]) for d in data])
    y_min = min([sum([d[v]['neg'] for v in var_list]) for d in data])

    ax.set_ylim(y_min * 1.1, y_max * 1.1)
    ax.set_xlim(-0.6, 2.5)

    # Annotations for "Enhancement" vs "Suppression"
    ax.text(-0.5, y_max * 0.8, "Enhancement\n(Positive Anomaly)", ha='center', fontsize=10, color='#555555')
    ax.text(-0.5, y_min * 0.8, "Suppression\n(Negative Anomaly)", ha='center', fontsize=10, color='#555555')


# ==========================================
# 4. EXECUTE
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

draw_anomaly_budget(ax, scenario_data, VARS, PALETTE, SCENARIO_LABELS)

ax.set_title("Biophysical Drivers of Forest Carbon Anomalies",
             fontsize=14, fontweight='bold', pad=20)

# Legend
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=c, label=l) for l, c in zip(VAR_LABELS.values(), PALETTE.values())]
legend_elements.append(Line2D([0], [0], color='black', lw=2, marker='o', label='Net NEP Anomaly'))
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=9)

# Save
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'55_FIG-Biophysical_Budget_ZScore_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()