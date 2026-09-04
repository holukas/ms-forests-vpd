"""
How often each site sits above the VPD threshold.

The threshold is one number for the whole network, but the air at one site is past it in
one half-hour of fifty and at another in nine of ten. That spread is what this figure shows,
and it is the part a sentence in the text cannot carry.

Panel a ranks the 208 sites by the share of peak-season daytime half-hours above 1.26 kPa.
Panel b puts the same share against the site's own mean VPD, which is where the spread comes
from: a site whose mean already sits near the threshold is above it half the time. In each
site's own standard deviations the share is close to 38 % everywhere, so the ranking in
panel a is a ranking of climates, not of responses.

Panel c changes the threshold instead of the site. Each site is counted against the crossing
of its own forest type, from script 49, in the site's own standard deviations. The four
crossings differ, so the panel shows what that difference costs in exposure. The dashed line
is the same set of sites counted against the one network crossing, which is the comparison
that makes the four groups readable.

Reads what `40_aggregation/48_threshold_exceedance.py` writes and draws it. Nothing is
computed here.

Reads:
    48_EXCEEDANCE_PerSite_{FLUX}.csv       one row per site
    48_EXCEEDANCE_PerBiome_{FLUX}.csv      one row per forest type, plus an all-sites row
    47_THRESHOLD_Robustness_{FLUX}.csv     the published threshold, for the axis labels

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

IGBP_ORDER = ['ENF', 'DBF', 'EBF', 'MF']
IGBP_NAMES = {'ENF': 'Evergreen needleleaf', 'DBF': 'Deciduous broadleaf',
              'EBF': 'Evergreen broadleaf', 'MF': 'Mixed'}
# Okabe-Ito, the same palette as the other figures
COLORS = {'ENF': '#009E73', 'DBF': '#D55E00', 'EBF': '#56B4E9', 'MF': '#E69F00'}

AX_LABELS_FONTSIZE = 12

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
folder = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
folder.mkdir(parents=True, exist_ok=True)
agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET

per_site = pd.read_csv(agg / f'48_EXCEEDANCE_PerSite_{FLUX}.csv')
per_biome = pd.read_csv(agg / f'48_EXCEEDANCE_PerBiome_{FLUX}.csv').set_index('IGBP')
robustness = pd.read_csv(agg / f'47_THRESHOLD_Robustness_{FLUX}.csv')
threshold_kpa = float(robustness.loc[robustness['test'] == 'PUBLISHED REFERENCE',
                                     'threshold_kpa'].iloc[0])

pooled = per_biome.loc['all', 'pooled_pct']
counts = per_site['IGBP'].value_counts()

fig, axs = plt.subplots(1, 3, figsize=(16.5, 5), dpi=150,
                        gridspec_kw={'width_ratios': [1.35, 1, 1]})

# Panel a: every site as one bar, ranked. Bars rather than a curve, because the reader is
# meant to see 208 sites and not a smooth distribution.
ranked = per_site.sort_values('above_published_kpa_pct').reset_index(drop=True)
axs[0].bar(ranked.index, ranked['above_published_kpa_pct'], width=1.0,
           color=[COLORS[i] for i in ranked['IGBP']], linewidth=0)
axs[0].axhline(pooled, color='black', lw=1.2, linestyle='--', zorder=3)
axs[0].text(2, pooled + 2, f'all half-hours, {pooled:.0f} %', fontsize=AX_LABELS_FONTSIZE * 0.85)
axs[0].set_xlim(-2, len(ranked) + 1)
axs[0].set_ylim(0, 100)
axs[0].set_xlabel(f'Sites, ranked (n = {len(ranked)})', fontsize=AX_LABELS_FONTSIZE)
axs[0].set_ylabel(f'Half-hours above {threshold_kpa:.2f} kPa (%)', fontsize=AX_LABELS_FONTSIZE)

# Panel b: the same share against the site's own mean VPD. The vertical line is the
# threshold, so a site to the right of it spends more than half its records past the limit.
for igbp in IGBP_ORDER:
    g = per_site.loc[per_site['IGBP'] == igbp]
    axs[1].scatter(g['vpd_mean_kpa'], g['above_published_kpa_pct'], s=22, alpha=0.8,
                   color=COLORS[igbp], linewidths=0,
                   label=f'{IGBP_NAMES[igbp]} (n = {counts[igbp]})')
axs[1].axvline(threshold_kpa, color='black', lw=1.2, linestyle='--', zorder=1)
axs[1].text(threshold_kpa + 0.02, 3, f'{threshold_kpa:.2f} kPa',
            fontsize=AX_LABELS_FONTSIZE * 0.85, rotation=90, va='bottom')
axs[1].set_ylim(0, 100)
axs[1].set_xlabel('Site mean VPD (kPa)', fontsize=AX_LABELS_FONTSIZE)
axs[1].set_ylabel(f'Half-hours above {threshold_kpa:.2f} kPa (%)', fontsize=AX_LABELS_FONTSIZE)
axs[1].legend(fontsize=AX_LABELS_FONTSIZE * 0.8, frameon=False, loc='lower right')

# Panel c: one point per site, against the crossing of its own forest type. Points and a
# median bar rather than a bar chart, so the within-type spread stays visible.
# The jitter only separates overlapping points, so it is seeded and never changes the figure.
rng = np.random.default_rng(42)
global_sigma_median = per_site['above_published_sigma_pct'].median()
axs[2].axhline(global_sigma_median, color='black', lw=1.2, linestyle='--', zorder=1)
# The label sits in a margin on the right, because the point clouds reach the line.
axs[2].text(3.55, global_sigma_median + 1.5, f'network\ncrossing\n{global_sigma_median:.0f} %',
            fontsize=AX_LABELS_FONTSIZE * 0.85, va='bottom')

tick_labels = []
for pos, igbp in enumerate(IGBP_ORDER):
    g = per_site.loc[per_site['IGBP'] == igbp, 'above_biome_threshold_pct'].dropna()
    axs[2].scatter(pos + rng.uniform(-0.22, 0.22, len(g)), g, s=22, alpha=0.8,
                   color=COLORS[igbp], linewidths=0, zorder=2)
    axs[2].plot([pos - 0.32, pos + 0.32], [g.median()] * 2, color='black', lw=1.8, zorder=3)
    tick_labels.append(f'{igbp}\n{per_biome.loc[igbp, "biome_threshold_z"]:.2f} '
                       f'$\\sigma$\nn = {len(g)}')

axs[2].set_xlim(-0.6, len(IGBP_ORDER) + 0.5)
axs[2].set_ylim(0, 100)
axs[2].set_xticks(range(len(IGBP_ORDER)))
axs[2].set_xticklabels(tick_labels)
axs[2].set_xlabel('Forest type, with its own crossing', fontsize=AX_LABELS_FONTSIZE,
                  labelpad=8)
axs[2].set_ylabel("Half-hours above the forest type's crossing (%)",
                  fontsize=AX_LABELS_FONTSIZE)

for letter, ax in zip('abc', axs):
    ax.tick_params(labelsize=AX_LABELS_FONTSIZE)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(-0.09, 1.03, letter, transform=ax.transAxes, fontsize=AX_LABELS_FONTSIZE * 1.2,
            fontweight='bold', va='bottom')

# Three lines per label, so they need less room than the shared size gives them.
axs[2].tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE * 0.85)

fig.tight_layout()

outfile = folder / f'60_SUPPFIG-X_ThresholdExceedance_{FLUX}.png'
fig.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')

# The numbers a caption would quote, so they can be checked without opening the figure.
print(f"{len(per_site)} sites, {per_site['n_records'].sum():,} half-hours")
print(f"above {threshold_kpa:.2f} kPa: {pooled:.1f} % of all half-hours, "
      f"site median {per_site['above_published_kpa_pct'].median():.1f} %, "
      f"range {ranked['above_published_kpa_pct'].iloc[0]:.1f} to "
      f"{ranked['above_published_kpa_pct'].iloc[-1]:.1f} %")
print(f"lowest site {ranked['SITE'].iloc[0]}, highest site {ranked['SITE'].iloc[-1]}")
print("above the forest type's own crossing, site median: " + ", ".join(
    f"{i} {per_biome.loc[i, 'biome_threshold_median_pct']:.1f} % "
    f"at {per_biome.loc[i, 'biome_threshold_z']:.2f} sigma" for i in IGBP_ORDER))
print(f"Saved to {outfile}")
if SHOW_PLOT:
    plt.show()
