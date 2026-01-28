from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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

# Constraint Variables (To plot NEGATIVE/DOWNWARDS)
# Order: Put VPD last so it appears at the bottom of the stack (deepest impact)
CONSTRAINT_VARS = ['SWIN_ZSCORE', 'TA_ZSCORE', 'SWC_ZSCORE', 'VPD_ZSCORE']
CONSTRAINT_LABELS = {
    'VPD_ZSCORE': 'VPD Stress',
    'SWC_ZSCORE': 'Soil Water Stress',
    'TA_ZSCORE': 'Temp Stress',
    'SWIN_ZSCORE': 'Radiation Limitation'
}

# Realized Variable (To plot POSITIVE/UPWARDS)
REALIZED_VAR = 'Realized'

SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'

# Colors
PALETTE = {
    'Realized': '#595959',  # Dark Grey
    'VPD_ZSCORE': '#D55E00',  # Vermillion (Red)
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
# 2. DATA LOAD & PREP
# ==========================================
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()
df_main = df_main.loc[df_main['IGBP'].isin(IGBP_CLASSES)].copy()

# Prepare Data Dictionary
scenario_data = []

print("Processing budgets...")
for scen_id in SCENARIO_ORDER:
    df_scen = df_main[df_main['SCENARIO'] == scen_id].copy()

    # 1. Realized (Mean Total)
    # We calculate the sum of components to get the net result
    all_shap_cols = [c + SHAP_SUFFIX for c in CONSTRAINT_VARS]
    df_scen['TOTAL_SHAP'] = df_scen[all_shap_cols].sum(axis=1)
    realized_val = df_scen['TOTAL_SHAP'].mean()

    # Ensure a tiny positive value for visualization even if net is negative (optional)
    # or just plot absolute magnitude if it remains a "Sink" concept
    val_realized = max(0.02, realized_val)

    # 2. Constraints (Absolute Mean Negative Impacts)
    scen_vals = {'Realized': val_realized}

    for var_base in CONSTRAINT_VARS:
        col = var_base + SHAP_SUFFIX
        neg_impacts = df_scen[df_scen[col] < 0][col]
        penalty = abs(neg_impacts.mean()) if len(neg_impacts) > 0 else 0.0
        scen_vals[var_base] = penalty

    scenario_data.append(scen_vals)


# ==========================================
# 3. DIVERGING PLOT ENGINE
# ==========================================
def sigmoid(x, x_start, x_end, y_start, y_end):
    """Sigmoid curve logic."""
    x_norm = (x - x_start) / (x_end - x_start)
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


def draw_diverging_alluvial(ax, data, constraint_order, palette, x_labels):
    x_centers = [0, 1, 2]
    bar_width = 0.12
    alpha_ribbon = 0.6

    # Track vertical positions: pos[scen][var] = (y_top, y_bottom)
    node_pos = [{} for _ in range(len(data))]

    # -----------------------------
    # DRAW BARS (NODES)
    # -----------------------------
    for i, d in enumerate(data):
        cx = x_centers[i]

        # 1. Plot Realized (UPWARDS from 0)
        r_val = d['Realized']
        ax.bar(cx, r_val, width=bar_width, bottom=0,
               color=palette['Realized'], edgecolor='white', linewidth=0.5, zorder=10)
        node_pos[i]['Realized'] = (0, r_val)

        # Label Realized (Only first/last)
        if i == 0:
            ax.text(cx, r_val + 0.05, "Realized Sink", ha='center', va='bottom',
                    fontsize=9, fontweight='bold', color=palette['Realized'])
        if i == 2:
            ax.text(cx, r_val + 0.05, f"{r_val:.2f}", ha='center', va='bottom', fontsize=8)

        # 2. Plot Constraints (DOWNWARDS from 0)
        current_y = 0.0
        for var in constraint_order:
            val = d[var]
            if val < 0.005: continue

            # We stack downwards, so bottom is (current_y - val)
            top = current_y
            bottom = current_y - val

            ax.bar(cx, val, width=bar_width, bottom=bottom,
                   color=palette[var], edgecolor='white', linewidth=0.5, zorder=10)

            node_pos[i][var] = (top, bottom)

            # Label Constraints (Only first/last)
            if i == 0 and val > 0.05:
                ax.text(cx - 0.08, bottom + val / 2, CONSTRAINT_LABELS[var],
                        ha='right', va='center', fontsize=8, color=palette[var], fontweight='bold')

            # Update stack tracker (moving down)
            current_y -= val

        # Scenario Label (at y=0)
        ax.text(cx, 0.05, x_labels[i], ha='center', va='bottom',
                fontsize=10, fontweight='bold', color='white', zorder=15,
                bbox=dict(facecolor='black', alpha=0.3, edgecolor='none', pad=2))

    # -----------------------------
    # DRAW RIBBONS (CONNECTIONS)
    # -----------------------------
    for i in range(len(data) - 1):
        x_start = x_centers[i] + bar_width / 2
        x_end = x_centers[i + 1] - bar_width / 2
        x_curve = np.linspace(x_start, x_end, 300)

        # 1. Connect Realized
        y_start_bot, y_start_top = node_pos[i]['Realized']
        y_end_bot, y_end_top = node_pos[i + 1]['Realized']

        # Bottom is always 0 for Realized
        y_top_curve = sigmoid(x_curve, x_start, x_end, y_start_top, y_end_top)
        y_bot_curve = np.zeros_like(x_curve)

        ax.fill_between(x_curve, y_bot_curve, y_top_curve,
                        color=palette['Realized'], alpha=alpha_ribbon - 0.2, edgecolor='none')

        # 2. Connect Constraints
        for var in constraint_order:
            if var not in node_pos[i] or var not in node_pos[i + 1]: continue

            # Retrieve coordinates (Top is closer to 0, Bottom is deeper negative)
            start_top, start_bot = node_pos[i][var]
            end_top, end_bot = node_pos[i + 1][var]

            y_top_c = sigmoid(x_curve, x_start, x_end, start_top, end_top)
            y_bot_c = sigmoid(x_curve, x_start, x_end, start_bot, end_bot)

            ax.fill_between(x_curve, y_bot_c, y_top_c,
                            color=palette[var], alpha=alpha_ribbon, edgecolor='none')

    # -----------------------------
    # STYLING
    # -----------------------------
    ax.axhline(0, color='black', linewidth=1, linestyle='-', zorder=20)
    ax.axis('off')

    # Auto-scale limits
    max_y = max([d['Realized'] for d in data]) * 1.3
    min_y = min([sum([-v for k, v in d.items() if k != 'Realized']) for d in data]) * 1.2
    ax.set_ylim(min_y, max_y)
    ax.set_xlim(-0.5, 2.5)


# ==========================================
# 4. EXECUTE
# ==========================================
fig, ax = plt.subplots(figsize=(10, 7), dpi=300)

draw_diverging_alluvial(ax, scenario_data, CONSTRAINT_VARS, PALETTE, SCENARIO_LABELS)

ax.set_title("Balance of Carbon Sink vs. Climatic Constraints",
             fontsize=14, fontweight='bold', pad=15)

# Save
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'55_FIG-Diverging_Alluvial_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()