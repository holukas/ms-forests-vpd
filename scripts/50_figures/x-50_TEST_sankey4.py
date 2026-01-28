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
# We stack SWIN first on top (source of energy)
# We stack VPD last on bottom (major sink limitation)
VARS_POS_ORDER = ['SWIN_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE']
VARS_NEG_ORDER = ['SWIN_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE']

VAR_LABELS = {
    'VPD_ZSCORE': 'VPD',
    'SWC_ZSCORE': 'Soil Water',
    'TA_ZSCORE': 'Air Temp',
    'SWIN_ZSCORE': 'Radiation'
}
SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'

# Colors (Okabe-Ito)
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
# 2. DATA PREPARATION
# ==========================================
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()
df_main = df_main.loc[df_main['IGBP'].isin(IGBP_CLASSES)].copy()

# Structure: data[scenario_idx][variable] = {'pos': val, 'neg': val}
scenario_data = []
net_nep_values = []  # To store the net line

print("Calculating bi-directional budgets...")
for scen_id in SCENARIO_ORDER:
    df_scen = df_main[df_main['SCENARIO'] == scen_id].copy()

    # Calculate Net NEP (Sum of all SHAP values)
    all_cols = [v + SHAP_SUFFIX for v in VARS_POS_ORDER]
    df_scen['NET_SHAP'] = df_scen[all_cols].sum(axis=1)
    net_nep = df_scen['NET_SHAP'].mean()
    net_nep_values.append(net_nep)

    scen_dict = {}

    for var in VARS_POS_ORDER:
        col = var + SHAP_SUFFIX
        # Get all values for this variable in this scenario
        vals = df_scen[col].values

        # Calculate Mean Positive Contribution (Sum of Positives / Total N)
        # This preserves the additive property: Mean(Total) = Mean(Pos) + Mean(Neg)
        pos_contribution = np.sum(vals[vals > 0]) / len(vals)

        # Calculate Mean Negative Contribution
        neg_contribution = np.sum(vals[vals < 0]) / len(vals)

        scen_dict[var] = {'pos': pos_contribution, 'neg': neg_contribution}

    scenario_data.append(scen_dict)


# ==========================================
# 3. PLOTTING FUNCTIONS
# ==========================================
def sigmoid(x, x_start, x_end, y_start, y_end):
    """Sigmoid connection."""
    x_norm = (x - x_start) / (x_end - x_start)
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


def draw_bidirectional_alluvial(ax, data, pos_vars, neg_vars, palette, x_labels):
    x_centers = [0, 1, 2]
    bar_width = 0.12
    alpha_ribbon = 0.5

    # Store positions: node_pos[scen][var] = {'pos_top': y, 'pos_bot': y, 'neg_top': y, ...}
    node_pos = [{} for _ in range(len(data))]

    # -----------------------------
    # A. DRAW VERTICAL BARS
    # -----------------------------
    for i, d in enumerate(data):
        cx = x_centers[i]

        # 1. Stack Positive (Upwards from 0)
        current_y = 0.0
        for var in pos_vars:
            val = d[var]['pos']
            if val < 0.005:
                node_pos[i][var] = {**node_pos[i].get(var, {}), 'pos_top': current_y, 'pos_bot': current_y}
                continue

            top = current_y + val
            bottom = current_y

            # Draw Bar
            ax.bar(cx, val, width=bar_width, bottom=bottom,
                   color=palette[var], edgecolor='white', linewidth=0.3, zorder=10)

            # Save coords
            node_pos[i][var] = {**node_pos[i].get(var, {}), 'pos_top': top, 'pos_bot': bottom}

            # Label (first scenario only for clarity)
            if i == 0 and val > 0.1:
                ax.text(cx - 0.08, bottom + val / 2, VAR_LABELS[var],
                        ha='right', va='center', fontsize=8, color=palette[var], fontweight='bold')

            current_y += val

        # 2. Stack Negative (Downwards from 0)
        current_y = 0.0
        for var in neg_vars:
            val = d[var]['neg']  # This is negative number
            if abs(val) < 0.005:
                node_pos[i][var] = {**node_pos[i].get(var, {}), 'neg_top': current_y, 'neg_bot': current_y}
                continue

            top = current_y
            bottom = current_y + val  # val is negative

            # Draw Bar
            ax.bar(cx, abs(val), width=bar_width, bottom=bottom,
                   color=palette[var], edgecolor='white', linewidth=0.3, zorder=10)

            # Save coords
            node_pos[i][var] = {**node_pos[i].get(var, {}), 'neg_top': top, 'neg_bot': bottom}

            # Label (first scenario)
            if i == 0 and abs(val) > 0.1:
                ax.text(cx - 0.08, bottom + abs(val) / 2, VAR_LABELS[var],
                        ha='right', va='center', fontsize=8, color=palette[var], fontweight='bold')

            current_y += val  # move down

        # Scenario Label
        ax.text(cx, -1.8, x_labels[i], ha='center', va='top', fontweight='bold', fontsize=10)

    # -----------------------------
    # B. DRAW RIBBONS
    # -----------------------------
    for i in range(len(data) - 1):
        x_start = x_centers[i] + bar_width / 2
        x_end = x_centers[i + 1] - bar_width / 2
        x_curve = np.linspace(x_start, x_end, 300)

        # Positive Flows
        for var in pos_vars:
            start = node_pos[i].get(var, {})
            end = node_pos[i + 1].get(var, {})

            if 'pos_top' in start and 'pos_top' in end:
                # Check magnitude to avoid drawing hair-thin lines
                if (start['pos_top'] - start['pos_bot'] < 0.005) and (end['pos_top'] - end['pos_bot'] < 0.005):
                    continue

                y_top = sigmoid(x_curve, x_start, x_end, start['pos_top'], end['pos_top'])
                y_bot = sigmoid(x_curve, x_start, x_end, start['pos_bot'], end['pos_bot'])
                ax.fill_between(x_curve, y_bot, y_top, color=palette[var], alpha=alpha_ribbon, edgecolor='none')

        # Negative Flows
        for var in neg_vars:
            start = node_pos[i].get(var, {})
            end = node_pos[i + 1].get(var, {})

            if 'neg_top' in start and 'neg_top' in end:
                if (start['neg_top'] - start['neg_bot'] < 0.005) and (end['neg_top'] - end['neg_bot'] < 0.005):
                    continue

                y_top = sigmoid(x_curve, x_start, x_end, start['neg_top'], end['neg_top'])
                y_bot = sigmoid(x_curve, x_start, x_end, start['neg_bot'], end['neg_bot'])
                ax.fill_between(x_curve, y_bot, y_top, color=palette[var], alpha=alpha_ribbon, edgecolor='none')

    # -----------------------------
    # C. DRAW NET FLUX LINE
    # -----------------------------
    # Connect the net NEP dots across scenarios
    ax.plot(x_centers, net_nep_values, color='black', linewidth=2, linestyle='--', marker='o', label='Net NEP Anomaly')

    # Annotate Net Values
    for x, y in zip(x_centers, net_nep_values):
        ax.text(x + 0.1, y, f"Net: {y:.2f}", fontsize=9, fontweight='bold', ha='left', va='center')

    # -----------------------------
    # D. STYLING
    # -----------------------------
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axis('off')

    # Dynamic Limits
    all_pos_heights = [sum([d[v]['pos'] for v in pos_vars]) for d in data]
    all_neg_heights = [sum([d[v]['neg'] for v in neg_vars]) for d in data]

    ax.set_ylim(min(all_neg_heights) * 1.1, max(all_pos_heights) * 1.1)
    ax.set_xlim(-0.5, 2.5)


# ==========================================
# 4. EXECUTE
# ==========================================
fig, ax = plt.subplots(figsize=(11, 7), dpi=300)

draw_bidirectional_alluvial(ax, scenario_data, VARS_POS_ORDER, VARS_NEG_ORDER, PALETTE, SCENARIO_LABELS)

ax.set_title("Biophysical Budget: Drivers of Carbon Flux Anomalies",
             fontsize=14, fontweight='bold', pad=15)

# Add minimal legend
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor=c, label=l) for l, c in zip(VAR_LABELS.values(), PALETTE.values())]
legend_elements.append(plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Net NEP Balance'))
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

# Save
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'55_FIG-Bidirectional_Alluvial_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()