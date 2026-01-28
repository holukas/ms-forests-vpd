import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ==========================================
# 1. HELPER FUNCTIONS (Sigmoid Ribbons)
# ==========================================
def sigmoid(x, x_start, x_end, y_start, y_end):
    """Generates a smooth sigmoid curve for the ribbons."""
    x_norm = (x - x_start) / (x_end - x_start)
    # Using tanh for a natural organic flow
    s = 0.5 * (1 + np.tanh(6 * (x_norm - 0.5)))
    return y_start + s * (y_end - y_start)


def draw_zscore_sankey(ax, components, colors, labels, title):
    """
    Draws a Sankey showing how 'Potential Anomaly' is reduced by stressors
    to result in the 'Realized Anomaly'.
    """
    # Geometry settings
    x_left, x_right = 0.0, 1.0
    bar_width = 0.05

    # -----------------------------------------------------
    # CALCULATE GEOMETRY
    # -----------------------------------------------------
    # The "Left" bar is the Potential (Sum of all components)
    total_potential = sum(components)

    # Draw Left Bar (Potential State)
    ax.bar(x_left, total_potential, width=bar_width, color='#333333',
           align='center', edgecolor='none')

    # Label Left Bar
    ax.text(x_left, total_potential + 0.05, f"Potential State\n(+{total_potential:.2f}$\sigma$)",
            ha='center', va='bottom', fontweight='bold', fontsize=11)

    # -----------------------------------------------------
    # DRAW RIBBONS AND RIGHT BARS
    # -----------------------------------------------------
    # We stack from bottom to top.
    # Order: Realized (Bottom) -> Stressors (Top)

    current_y_left = 0.0
    current_y_right = 0.0

    x_curve = np.linspace(x_left + bar_width / 2, x_right - bar_width / 2, 300)

    for val, color, label in zip(components, colors, labels):
        # 1. Draw Right Bar Component
        # We add a small white gap between bars for visual separation
        gap = 0.02 if val > 0.05 else 0

        ax.bar(x_right, val - gap, width=bar_width, bottom=current_y_right,
               color=color, align='center', edgecolor='none')

        # 2. Draw Label (Right side)
        # Position text in the middle of the bar segment
        text_y = current_y_right + (val / 2)
        ax.text(x_right + 0.04, text_y, f"{label}\n({val:.2f}$\sigma$)",
                ha='left', va='center', fontsize=10, color='#333333')

        # 3. Draw Connecting Ribbon
        y_bot = sigmoid(x_curve, x_curve[0], x_curve[-1], current_y_left, current_y_right)
        y_top = sigmoid(x_curve, x_curve[0], x_curve[-1], current_y_left + val, current_y_right + val - gap)

        ax.fill_between(x_curve, y_bot, y_top, color=color, alpha=0.6, edgecolor='none')

        # Update vertical cursors
        current_y_left += val
        current_y_right += val

    # -----------------------------------------------------
    # AESTHETICS
    # -----------------------------------------------------
    ax.set_xlim(x_left - 0.2, x_right + 0.5)
    ax.set_ylim(0, total_value * 1.15)
    ax.axis('off')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)


# ==========================================
# 2. DATA PREPARATION (How to handle Z-Scores)
# ==========================================

# NOTE: In a SHAP analysis of z-scores:
# Realized Z-Score = Sum(SHAP values) + Expected_Value(Average)
# To simplify the Sankey, we visualize the "Drag" vs the "Result".

# Example Data (Replace with your df calculations)
# 1. "Realized": The actual mean NEP z-score for your site/biome
realized_nep_z = 0.40

# 2. "Penalties": The absolute sum of NEGATIVE SHAP values for each driver
# (How much did each driver pull the z-score down?)
vpd_penalty = 0.85  # This is huge (your main finding)
sm_penalty = 0.35  # Moderate
temp_penalty = 0.20  # Minor

# 3. "Potential": What the z-score WOULD be if these constraints didn't exist
# Potential = Realized + Penalties
# (e.g. 0.4 + 0.85 + 0.35 + 0.20 = 1.80 sigma)
# This represents the "Physiological Maximum" for that time period.

components = [realized_nep_z, vpd_penalty, sm_penalty, temp_penalty]

# Labels must match the order of 'components'
labels = [
    "Realized NEP Anomaly",  # The Outcome
    "Suppression by VPD",  # The Main Drag
    "Suppression by Soil Water",  # The Secondary Drag
    "Suppression by Heat/Cold"  # The Minor Drag
]

# Colors (Nature-style)
# Green for Outcome, Red for VPD, Brown for Soil, Grey for Temp
colors = [
    "#2ca02c",  # Green (Realized)
    "#d62728",  # Red (VPD - High Alert)
    "#8c564b",  # Brown (Soil)
    "#7f7f7f"  # Grey (Temp)
]

# ==========================================
# 3. GENERATE PLOT
# ==========================================
total_value = sum(components)  # Used for scaling

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

draw_zscore_sankey(ax, components, colors, labels,
                   title="Decomposition of the Carbon Sink Anomaly")

# Add a caption explaining the Z-score logic
fig.text(0.5, 0.05,
         "Flow widths represent impact magnitude in standard deviations ($σ$).\n"
         "Left: Potential NEP anomaly in absence of stress.\n"
         "Right: Partitioning of the reduction into climatic constraints.",
         ha='center', fontsize=9, style='italic')

# plt.savefig("Figure4_ZScore_Sankey.pdf", bbox_inches='tight')
plt.show()