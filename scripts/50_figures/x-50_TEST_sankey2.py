from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import diive as dv
import src.files as files

# ==========================================
# 1. SETTINGS & CONFIGURATION
# ==========================================
FLUX = 'NEP_ZSCORE'
IGBP_CLASSES = ['ENF', 'DBF', 'MF', 'EBF']
SCENARIO_ORDER = [1, 4, 5]
SCENARIO_NAMES = ['Normal', 'Dry & Hot', 'Compound\nExtremes']

# Variables & Labels (Bottom to Top in Stack)
VARIABLES = ['Realized', 'VPD_ZSCORE', 'SWC_ZSCORE', 'TA_ZSCORE', 'SWIN_ZSCORE']
VAR_LABELS = {
    'Realized': 'Realized Sink',
    'VPD_ZSCORE': 'VPD Loss',
    'SWC_ZSCORE': 'Soil Water Loss',
    'TA_ZSCORE': 'Temp Loss',
    'SWIN_ZSCORE': 'Radiation Loss'
}
SHAP_SUFFIX = '_SHAPVALS_OVR_MEDIAN'

# Colors (Grey for Realized, Okabe-Ito for Drivers)
# Order matches VARIABLES list above
PALETTE = {
    'Realized': '#404040',  # Dark Grey
    'VPD_ZSCORE': '#D55E00',  # Vermillion (Red)
    'SWC_ZSCORE': '#009E73',  # Bluish Green
    'TA_ZSCORE': '#CC79A7',  # Reddish Purple
    'SWIN_ZSCORE': '#E69F00'  # Orange/Yellow
}

# Load Data
settings = files.read_settings_file("../../config/settings.yaml")
shap_type = 'conditional'
results_outdir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
filepath = Path(results_outdir) / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.parquet"

print("Loading data...")
shapvals_df = dv.load_parquet(filepath, sanitize_timestamp=False, output_middle_timestamp=False)

# Filter Global
df_main = shapvals_df.copy()
df_main = df_main.loc[df_main['IGBP'].isin(IGBP_CLASSES)].copy()

# ==========================================
# 2. DATA PREPARATION (SCENARIO BUDGETS)
# ==========================================
# We need a dictionary: data[scenario_index][variable] = value

scenario_data = []

print("Calculating scenario budgets...")
for scen_id in SCENARIO_ORDER:
    df_scen = df_main[df_main['SCENARIO'] == scen_id].copy()

    # 1. Calculate Realized (Mean Total SHAP)
    df_scen['TOTAL_SHAP'] = df_scen[[c + SHAP_SUFFIX for c in VARIABLES[1:]]].sum(axis=1)
    realized_val = df_scen['TOTAL_SHAP'].mean()
    # Ensure positive for stacking (visual magnitude)
    val_realized = max(0.05, realized_val)  # minimal thickness if 0 to keep flow visible

    # 2. Calculate Penalties (Mean Negative Impact)
    scen_vals = {'Realized': val_realized}

    for var_base in VARIABLES[1:]:
        col = var_base + SHAP_SUFFIX
        neg_impacts = df_scen[df_scen[col] < 0][col]
        penalty = abs(neg_impacts.mean()) if len(neg_impacts) > 0 else 0.0
        scen_vals[var_base] = penalty

    scenario_data.append(scen_vals)


# ==========================================
# 3. ALLUVIAL PLOTTING FUNCTIONS
# ==========================================
def sigmoid(x, x_start, x_end, y_start, y_end):
    """Sigmoid curve for ribbons."""
    x_norm = (x - x_start) / (x_end - x_start)
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


def draw_alluvial(ax, data, var_list, palette, x_labels):
    """
    Draws connected flows between scenarios.
    """
    # X-coordinates for the vertical bars
    x_centers = [0, 1, 2]
    bar_width = 0.1

    # Store y-positions for connections: y_pos[scenario_idx][variable] = (y_bottom, y_top)
    node_positions = [{} for _ in range(len(data))]

    # -----------------------------
    # 1. DRAW VERTICAL BARS (NODES)
    # -----------------------------
    for i, scen_dict in enumerate(data):
        current_y = 0.0
        x = x_centers[i]

        # Calculate total potential for this scenario (to center or align)
        # Here we align to bottom = 0

        for var in var_list:
            val = scen_dict[var]
            color = palette[var]

            # Draw Bar
            ax.bar(x, val, width=bar_width, bottom=current_y,
                   color=color, edgecolor='white', linewidth=0.5, zorder=10)

            # Store coordinates for ribbons
            node_positions[i][var] = (current_y, current_y + val)

            # Label only on the first and last scenario to reduce clutter
            if i == 0 and val > 0.1:
                ax.text(x - 0.08, current_y + val / 2, VAR_LABELS[var],
                        ha='right', va='center', fontsize=8, fontweight='bold', color=color)
            elif i == 2 and val > 0.1:
                ax.text(x + 0.08, current_y + val / 2, f"{val:.2f}",
                        ha='left', va='center', fontsize=8, color=color)

            current_y += val

        # Scenario Label
        ax.text(x, -0.15, x_labels[i], ha='center', va='top', fontsize=10, fontweight='bold')

    # -----------------------------
    # 2. DRAW CONNECTING RIBBONS
    # -----------------------------
    # Iterate through scenarios (0 to 1, then 1 to 2)
    for i in range(len(data) - 1):
        x_start = x_centers[i] + bar_width / 2
        x_end = x_centers[i + 1] - bar_width / 2
        x_curve = np.linspace(x_start, x_end, 300)

        for var in var_list:
            # Get start and end Y-coordinates
            y_start_bot, y_start_top = node_positions[i][var]
            y_end_bot, y_end_top = node_positions[i + 1][var]

            # Calculate Curves
            y_bot_curve = sigmoid(x_curve, x_start, x_end, y_start_bot, y_end_bot)
            y_top_curve = sigmoid(x_curve, x_start, x_end, y_start_top, y_end_top)

            # Draw Ribbon
            color = palette[var]
            ax.fill_between(x_curve, y_bot_curve, y_top_curve,
                            color=color, alpha=0.5, edgecolor='none', zorder=5)

    # -----------------------------
    # 3. STYLING
    # -----------------------------
    ax.axis('off')
    # Set limits with some padding
    ax.set_ylim(-0.3, max([sum(d.values()) for d in data]) * 1.1)
    ax.set_xlim(-0.5, 2.5)


# ==========================================
# 4. EXECUTE & SAVE
# ==========================================
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

draw_alluvial(ax, scenario_data, VARIABLES, PALETTE, SCENARIO_NAMES)

ax.set_title("Evolution of Carbon Sink Constraints Across Scenarios",
             fontsize=14, fontweight='bold', pad=20)

# Add Legend manually if needed, or rely on labels
# Save
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type
outfilepath = dir_out / f'55_FIG-Alluvial_Scenarios_{FLUX}.png'
print(f"Saved to {outfilepath}")
plt.savefig(outfilepath, bbox_inches='tight', dpi=300)

plt.show()