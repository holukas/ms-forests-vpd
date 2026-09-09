"""
Sankey diagram of the record losses from raw half-hours to the modelled dataset.

Reads `22_data_flow.csv` from stage 20 and draws one ribbon that narrows at each
filtering step, with the removed share branching off below it. Answers Reviewer
2's request for a transparent data-flow figure.
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
    ('daytime', 'Daytime\n(SW_IN_POT > 20)'),
    ('peak_months', 'Four peak-GPP\nmonths'),
    ('year_balanced', 'Equal year coverage\nper month'),
    ('complete_cases', 'No missing\npredictor or NEP'),
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


def main():
    settings = load_settings()
    indir = Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
    flow = pd.read_csv(indir / "22_data_flow.csv")
    totals = flow[[k for k, _ in STEPS]].sum()
    start = totals.iloc[0]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    node_w, gap = 0.22, 1.0
    drop_depth = 0.09 * start

    for i, (key, label) in enumerate(STEPS):
        x = i * gap
        n = totals[key]
        ax.add_patch(plt.Rectangle((x, 0), node_w, n, facecolor=INK, edgecolor='none', zorder=3))
        ax.text(x + node_w / 2, n + start * 0.035, label, ha='center', va='bottom',
                fontsize=10, color=INK, linespacing=1.35)
        ax.text(x + node_w / 2, n + start * 0.012, f"{n:,.0f}  ({n / start * 100:.1f} %)",
                ha='center', va='bottom', fontsize=9.5, color=INK, fontweight='bold')

        if i == len(STEPS) - 1:
            break
        nxt = totals[STEPS[i + 1][0]]
        removed = n - nxt
        ribbon(ax, x + node_w, x + gap, n, n - nxt, nxt, 0, KEPT, 0.55)
        ribbon(ax, x + node_w, x + gap, n - nxt, 0, -drop_depth * 0.15, -drop_depth, LOST, 0.75)
        ax.text(x + node_w + (gap - node_w) / 2, -drop_depth - start * 0.018,
                f"-{removed:,.0f}", ha='center', va='top', fontsize=9, color=INK)

    ax.set_xlim(-0.35, (len(STEPS) - 1) * gap + node_w + 0.35)
    ax.set_ylim(-drop_depth - start * 0.10, start * 1.16)
    ax.axis('off')
    kept_sites = int((flow['complete_cases'] > 0).sum())
    lost = len(flow) - kept_sites
    subtitle = f"{len(flow)} forest sites in"
    if lost:
        names = ', '.join(flow.loc[flow['complete_cases'] == 0, 'SITE'])
        subtitle += f", {kept_sites} with data left ({names} keeps none)"
    title = "From site records to modelled data" + chr(10) + subtitle
    ax.set_title(title, fontsize=13, color=INK, pad=18)
    fig.tight_layout()

    dir_out = Path(settings['DIR_PLOTS_OUT']) / 'NEP_ZSCORE' / 'conditional' / VARIANT
    dir_out.mkdir(parents=True, exist_ok=True)
    outfile = dir_out / "66_SUPPFIG-X_DataFlow_Sankey.png"
    fig.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved {outfile}")

    summary = pd.DataFrame({'step': [k for k, _ in STEPS], 'records': totals.values})
    summary['removed'] = summary['records'].shift(1) - summary['records']
    summary['share_kept_pct'] = summary['records'] / start * 100
    summary.to_csv(dir_out / "66_SUPPFIG-X_DataFlow_Sankey_DATA.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
