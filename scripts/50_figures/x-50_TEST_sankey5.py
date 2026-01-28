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

# Stack Order (Bottom to Top)
# We put the stressors at the bottom to show them "rising up" to squeeze the sink
STACK_VARS = ['VPD_ZSCORE', 'SWC_ZSCORE', 'TA_ZSCORE', 'Realized']

VAR_LABELS = {
    'VPD_ZSCORE': 'VPD Penalty',
    'SWC_ZSCORE': 'Soil Water Penalty',
    'TA_ZSCORE': 'Temp Penalty',
    'Realized': 'Remaining Sink (NEP)'
}
SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'

# Aesthetic Palette (Okabe-Ito + Grey)
PALETTE = {
    'VPD_ZSCORE': '#D55E00',  # Vermillion (The main threat)
    'SWC_ZSCORE': '#009E73',  # Bluish Green
    'TA_ZSCORE': '#CC79A7',  # Reddish Purple
    'Realized': '#E0E0E0'  # Light Grey (The "Empty" space / Result)
}
# Text colors for contrast
TEXT_COLOR = {
    'VPD_ZSCORE': 'white',
    'SWC_ZSCORE': 'white',
    'TA_ZSCORE': 'white',
    'Realized': '#333333'
}

# Paths
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
filepath = Path(results_outdir) / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet"

# ==========================================
# 2. DATA PREP
# ==========================================
print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)
df_main = shapvals_df.copy()
df_main = df_main.loc[df_main['IGBP'].isin(IGBP_CLASSES)].copy()

scenario_data = []

print("Calculating squeeze budgets...")
for scen_id in SCENARIO_ORDER:
    df_scen = df_main[df_main['SCENARIO'] == scen_id].copy()

    # 1. Calculate Penalties (Absolute sum of negative impacts)
    scen_vals = {}
    total_penalty = 0

    # Iterate through stress drivers
    for var in ['VPD_ZSCORE', 'SWC_ZSCORE', 'TA_ZSCORE']:
        col = var + SHAP_SUFFIX
        neg_impacts = df_scen[df_scen[col] < 0][col]
        penalty = abs(neg_impacts.mean()) if len(neg_impacts) > 0 else 0.0
        scen_vals[var] = penalty
        total_penalty += penalty

    # 2. Calculate Realized (Net Outcome)
    # To visualize the "Squeeze", we treat Realized as the "Remainder" of the Potential.
    # Potential = Realized + Penalties.
    # We estimate Potential based on the beneficial drivers (SWIN + positive Temp)
    # OR simply: Potential = Net + Penalties.

    # Calculate Net NEP
    cols_all = [c + SHAP_SUFFIX for c in ['VPD_ZSCORE', 'SWC_ZSCORE', 'TA_ZSCORE', 'SWIN_ZSCORE']]
    df_scen['NET'] = df_scen[cols_all].sum(axis=1)
    net_val = df_scen['NET'].mean()

    # For visualization: If Net is negative, the "Remaining Sink" is 0 (Collapsed).
    # We preserve the magnitude of the collapse in the text, but visually the bar is gone.
    val_realized = max(0.01, net_val)

    scen_vals['Realized'] = val_realized
    scenario_data.append(scen_vals)


# ==========================================
# 3. PLOTTING ENGINE
# ==========================================
def sigmoid(x, x_start, x_end, y_start, y_end):
    x_norm = (x - x_start) / (x_end - x_start)
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


def draw_squeeze_alluvial(ax, data, stack_order, palette, x_labels):
    x_centers = [0, 1, 2]
    bar_width = 0.15
    alpha_ribbon = 0.8  # Higher opacity for rich color

    node_pos = [{} for _ in range(len(data))]

    # -----------------------------
    # DRAW STACKED BARS
    # -----------------------------
    for i, d in enumerate(data):
        cx = x_centers[i]
        current_y = 0.0

        for var in stack_order:
            val = d[var]
            color = palette[var]

            # Draw Bar
            ax.bar(cx, val, width=bar_width, bottom=current_y,
                   color=color, edgecolor='white', linewidth=0.5, zorder=10)

            # Save Coords
            node_pos[i][var] = (current_y, current_y + val)

            # Label Inside Bar (if big enough)
            if val > 0.15:
                # Different label position for first vs last col
                ax.text(cx, current_y + val / 2, f"{val:.2f}",
                        ha='center', va='center', fontsize=8,
                        color=TEXT_COLOR[var], fontweight='bold')

            # External Labels (First Col only)
            if i == 0 and val > 0.05:
                ax.text(cx - 0.1, current_y + val / 2, VAR_LABELS[var],
                        ha='right', va='center', fontsize=9, fontweight='bold', color=color)

            current_y += val

        # Total Potential Line (Top of stack)
        ax.hlines(current_y, cx - bar_width / 2, cx + bar_width / 2, colors='black', linewidth=1)
        ax.text(cx, current_y + 0.05, "Total Potential", ha='center', va='bottom', fontsize=7, style='italic')

        # Scenario Label
        ax.text(cx, -0.15, x_labels[i], ha='center', va='top', fontsize=11, fontweight='bold')

    # -----------------------------
    # DRAW RIBBONS
    # -----------------------------
    for i in range(len(data) - 1):
        x_start = x_centers[i] + bar_width / 2
        x_end = x_centers[i + 1] - bar_width / 2
        x_curve = np.linspace(x_start, x_end, 300)

        for var in stack_order:
            start_bot, start_top = node_pos[i][var]
            end_bot, end_top = node_pos[i + 1][var]

            # Sigmoid connections
            y_bot = sigmoid(x_curve, x_start, x_end, start_bot, end_bot)
            y_top = sigmoid(x_curve, x_start, x_end, start_top, end_top)

            # Draw
            color = palette[var]
            # Make Realized ribbon more transparent to emphasize the "void" being filled
            alpha = 0.3 if var == 'Realized' else 0.6

            ax.fill_between(x_curve, y_bot, y_top, color=color, alpha=alpha, edgecolor='none')

    # -----------------------------
    # STYLING
    # -----------------------------
    ax.axis('off')
    ax.set_xlim(-0.5, 2.5)

    # Add an arrow indicating the "Squeeze"
    # From top of VPD in Normal to top of VPD in Extreme
    vpd_start = node_pos[0]['VPD_ZSCORE'][1]
    vpd_end = node_pos[2]['VPD_ZSCORE'][1]

    ax.annotate("", xy=(2.15, vpd_end), xytext=(2.15, 0),
                arrowprops=dict(arrowstyle="|-|", color=PALETTE['VPD_ZSCORE'], lw=1.5))
    ax.text(2.2, vpd_end / 2, "VPD Expansion", rotation=90, va='center', ha='left',
            color=PALETTE['VPD_ZSCORE'], fontsize=9, fontweight='bold')


# ==========================================
# 4. EXECUTE
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

draw_squeeze_alluvial(ax, scenario_data, STACK_VARS, PALETTE, SCENARIO_LABELS)

ax.set_title("The Carbon Squeeze: Rising Constraints vs. Remaining Sink",
             fontsize=14, fontweight='bold', pad=20)

# Save
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'55_FIG-Squeeze_Alluvial_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()