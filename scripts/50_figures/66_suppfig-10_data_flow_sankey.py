"""
Supplementary Fig. 10: the record losses from raw half-hours to the modeled dataset.

One ribbon narrows at each filtering step, with the removed records branching off below it.
The per-site numbers are Supplementary Data 1 (script 67).

Reads:
    20_subsets/<VARIANT>/22_data_flow.csv

Writes, into the plot folder:
    66_SUPPFIG-10_DataFlow_Sankey.png | _DATA.csv
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import load_settings
from src.plot import sigmoid

# Run variant, matching the stage that produced 22_data_flow.csv.
VARIANT = ""

STEPS = [
    ('all_records', 'All half-hourly\nrecords'),
    ('measured_nee', 'Measured NEE\n(QC flag 0)'),
    ('daytime', 'Daytime\n(SW_IN_POT > 20 W m⁻²)'),
    ('peak_months', 'Four peak-GPP\nmonths'),
    ('year_balanced', 'Equal year coverage\nper month'),
    ('complete_predictors', 'No missing\npredictor'),
    ('complete_cases', 'GPP, RECO and\nET present'),
]

KEPT = '#2196F3'   # Material blue 500
LOST = '#B0BEC5'   # Blue grey 200
INK = '#455A64'    # Blue grey 700


def ribbon(ax, x0, x1, top0, bot0, top1, bot1, color, alpha):
    """Fill between two sigmoid edges, which is what makes a Sankey read as a flow."""
    x = np.linspace(x0, x1, 120)
    top = sigmoid(x, x0, x1, top0, top1)
    bot = sigmoid(x, x0, x1, bot0, bot1)
    ax.fill_between(x, bot, top, color=color, alpha=alpha, linewidth=0, zorder=2)


def lost_flow(ax, x0, x1, top0, depth, shaft, head, shaft_w, head_w, color, alpha):
    """Draw the removed records as a ribbon that turns down into a shaft ending at x1.

    top0 is the top of the removed slice at the node, its bottom is 0. The shaft has width
    shaft_w and ends in a downward arrowhead. Returns the (x, y) of the arrow tip.
    """
    xe = x1 - shaft_w
    rx_in, ry_in = 0.06, 0.20 * depth          # inner rounding, x and y radii
    ry_out = 0.85 * depth                      # outer rounding: from -0.15 depth down to -depth
    x_top = np.linspace(x0, xe, 120)
    top = sigmoid(x_top, x0, xe, top0, -0.15 * depth)
    x_bot = np.linspace(x0, xe - rx_in, 120)
    bot = sigmoid(x_bot, x0, xe - rx_in, 0.0, -depth)
    th = np.linspace(0, np.pi / 2, 30)
    outer = list(zip(xe + shaft_w * np.sin(th), -0.15 * depth - ry_out * (1 - np.cos(th))))
    # Fillet of the re-entrant corner, centered outside the shape: from (xe, -d-ry) to (xe-rx, -d).
    inner = list(zip(xe - rx_in + rx_in * np.sin(th[::-1]), -depth - ry_in + ry_in * np.cos(th[::-1])))
    mid = (xe + x1) / 2
    y_base = -depth - shaft
    verts = list(zip(x_top, top)) + outer
    verts += [(x1, y_base), (x1 + head_w, y_base), (mid, y_base - head),
              (xe - head_w, y_base), (xe, y_base)]
    verts += inner
    verts += list(zip(x_bot[::-1], bot[::-1]))
    ax.add_patch(plt.Polygon(verts, closed=True, facecolor=color, alpha=alpha, edgecolor='none', zorder=2))
    return mid, y_base - head


def main():
    settings = load_settings()
    indir = Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
    flow = pd.read_csv(indir / "22_data_flow.csv")
    totals = flow[[k for k, _ in STEPS]].sum()
    start = totals.iloc[0]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    node_w, gap = 0.22, 1.0
    drop_depth = 0.09 * start
    # The removed flow: shaft below the ribbon and the arrowhead, in record units, and the
    # shaft width and the arrowhead overhang in x units.
    shaft, head, shaft_w, head_w = 0.11 * start, 0.055 * start, 0.16, 0.05

    for i, (key, label) in enumerate(STEPS):
        x = i * gap
        n = totals[key]
        ax.add_patch(plt.Rectangle((x, 0), node_w, n, facecolor=INK, edgecolor='none', zorder=3))
        ax.text(x + node_w / 2, n + start * 0.065, label, ha='center', va='bottom',
                fontsize=10, color=INK, linespacing=1.35)
        ax.text(x + node_w / 2, n + start * 0.012, f"{n:,.0f}  ({n / start * 100:.1f}%)",
                ha='center', va='bottom', fontsize=9.5, color=INK, fontweight='bold')

        if i == len(STEPS) - 1:
            break
        nxt = totals[STEPS[i + 1][0]]
        removed = n - nxt
        ribbon(ax, x + node_w, x + gap, n, n - nxt, nxt, 0, KEPT, 0.55)
        tip_x, tip_y = lost_flow(ax, x + node_w, x + gap - 0.14, n - nxt, drop_depth, shaft, head,
                                 shaft_w, head_w, LOST, 0.75)
        ax.text(tip_x, tip_y - start * 0.015, f"-{removed:,.0f}", ha='center', va='top',
                fontsize=9, color=INK)

    ax.set_xlim(-0.35, (len(STEPS) - 1) * gap + node_w + 0.35)
    ax.set_ylim(-drop_depth - shaft - head - start * 0.09, start * 1.16)
    ax.axis('off')
    # No title in the figure; the caption says what the ribbon is. The site count is
    # printed for the caption: one site keeps no record after the last step.
    kept_sites = int((flow['complete_cases'] > 0).sum())
    lost = flow.loc[flow['complete_cases'] == 0, 'SITE'].tolist()
    print(f"{len(flow)} sites in, {kept_sites} with records left" + (f", none left at {', '.join(lost)}" if lost else ""))
    fig.tight_layout()

    dir_out = Path(settings['DIR_PLOTS_OUT']) / 'NEP_ZSCORE' / 'conditional' / VARIANT
    dir_out.mkdir(parents=True, exist_ok=True)
    outfile = dir_out / "66_SUPPFIG-10_DataFlow_Sankey.png"
    fig.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {outfile}")

    summary = pd.DataFrame({'step': [k for k, _ in STEPS], 'records': totals.values})
    summary['removed'] = summary['records'].shift(1) - summary['records']
    summary['share_kept_pct'] = summary['records'] / start * 100
    summary.to_csv(dir_out / "66_SUPPFIG-10_DataFlow_Sankey_DATA.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
