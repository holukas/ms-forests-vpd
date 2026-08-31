"""
Per-site VPD thresholds by forest type, with the significance test drawn on.

Figure 4 prints one threshold per forest type, taken from the polynomial fitted to the
aggregated curve, so its interval describes the fit rather than the spread between sites.
The question of whether the forest types differ needs the per-site thresholds instead,
which is what `40_aggregation/49_biome_thresholds.py` computes. This script only plots what
that script wrote, it does not recompute anything.

Reads from the plot folder:
    49_SiteThresholds.csv              one threshold per site
    49_BiomeThresholds_Summary.csv     median, IQR and bootstrap CI per forest type
    49_BiomeThresholds_PairwiseTests.csv   Holm-corrected pairwise tests

Supplementary figure, not a main display item.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import load_settings

# Open the figure in a window after saving. False by default so a script can run
# unattended: matplotlib picks the interactive TkAgg backend here, and plt.show()
# then blocks until the window is closed by hand.
SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# Order by median, so the reader sees the ranking the tests refer to.
IGBP_ORDER = ['DBF', 'EBF', 'ENF', 'MF']
IGBP_NAMES = {'ENF': 'Evergreen\nneedleleaf', 'DBF': 'Deciduous\nbroadleaf',
              'EBF': 'Evergreen\nbroadleaf', 'MF': 'Mixed'}
# Okabe-Ito, the same palette as the other figures
COLORS = {'ENF': '#009E73', 'DBF': '#D55E00', 'EBF': '#56B4E9', 'MF': '#E69F00'}

AX_LABELS_FONTSIZE = 12

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
folder = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET

sites = pd.read_csv(agg / '49_SiteThresholds.csv')
summary = pd.read_csv(agg / '49_BiomeThresholds_Summary.csv').set_index('IGBP')
pairs = pd.read_csv(agg / '49_BiomeThresholds_PairwiseTests.csv')

fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)

# One column per forest type: every site as a point, the median and its bootstrap
# interval drawn on top.
rng = np.random.default_rng(1)
for x, igbp in enumerate(IGBP_ORDER):
    vals = sites.loc[sites['IGBP'] == igbp, 'threshold_z'].dropna().values
    jitter = rng.normal(0, 0.06, len(vals))
    ax.scatter(x + jitter, vals, s=14, color=COLORS[igbp], alpha=0.45, linewidths=0, zorder=2)

    row = summary.loc[igbp]
    ax.plot([x - 0.26, x + 0.26], [row['median']] * 2, color=COLORS[igbp], lw=3,
            solid_capstyle='butt', zorder=4)
    ax.plot([x, x], [row['ci_lower'], row['ci_upper']], color=COLORS[igbp], lw=1.6, zorder=3)
    ax.text(x, ax.get_ylim()[0], '', ha='center')  # keeps autoscale honest

ax.axhline(0, color='black', lw=0.8, linestyle='--', alpha=0.6, zorder=1)

# Brackets for the pairs that survive the Holm correction. Only those are drawn, so the
# figure does not imply a difference where the test found none.
sig = pairs.loc[pairs['significant']].copy()
y0 = sites['threshold_z'].max()
step = 0.13 * (y0 - sites['threshold_z'].min())
for k, (_, row) in enumerate(sig.iterrows()):
    a, b = [IGBP_ORDER.index(s.strip()) for s in row['pair'].split('vs')]
    lo, hi = min(a, b), max(a, b)
    y = y0 + step * (k + 1)
    ax.plot([lo, lo, hi, hi], [y - step * 0.18, y, y, y - step * 0.18], color='black', lw=1)
    ax.text((lo + hi) / 2, y, f"p = {row['p_holm']:.3f}", ha='center', va='bottom',
            fontsize=AX_LABELS_FONTSIZE * 0.8)

ax.set_xticks(range(len(IGBP_ORDER)))
ax.set_xticklabels([f"{IGBP_NAMES[i]}\nn={int(summary.loc[i, 'n'])}" for i in IGBP_ORDER],
                   fontsize=AX_LABELS_FONTSIZE)
ax.set_ylabel(r'VPD threshold per site ($\sigma$)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(axis='y', labelsize=AX_LABELS_FONTSIZE)
for sp in ('top', 'right'):
    ax.spines[sp].set_visible(False)

fig.tight_layout()

outfile = folder / f'65_SUPPFIG-X_BiomeThresholdDistributions_{FLUX}.png'
fig.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')
print(f"Saved to {outfile}")
if SHOW_PLOT:
    plt.show()
