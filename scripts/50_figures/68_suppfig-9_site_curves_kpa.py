"""
Supplementary Fig. 9: the per-site VPD responses of NEP on one kPa axis.

Panel a: each site's fitted response as a thin line, the median and interquartile range
across sites where at least half the sites have data, the all-sites threshold of script 47
as the vertical line, and each site's crossing on the zero line. Panel b: the crossings by
forest type.

The lines repeat the stage 49 per-site fit (POLY_DEGREE, at least MIN_BINS VPD bins) on the
stage 41 cells; the crossings come from stage 49. kPa = site mean + z * sd, as in script 54.
Do not collapse the TA axis with a rolling mean instead: it shifts the crossing (script 47).

Reads:
    40_aggregation/NEP_ZSCORE/conditional/41_SHAPVALUES-conditional_meanAggregatedPerSite
        _BIN-TA_ZSCORE+BIN-VPD_ZSCORE+NEP_ZSCORE.parquet
    40_aggregation/NEP_ZSCORE/conditional/49_SiteThresholds.csv
    40_aggregation/NEP_ZSCORE/conditional/47_THRESHOLD_Robustness_NEP_ZSCORE.csv   the threshold
    20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv          site mean and sd for kPa

Writes, into the plot folder:
    68_SUPPFIG-9_SiteCurvesKpa_NEP_ZSCORE.png
    68_SUPPFIG-9_SiteCurvesKpa_Crossings_NEP_ZSCORE.csv   per site: IGBP, crossing in sigma and kPa
    68_SUPPFIG-9_SiteCurvesKpa_NEP_ZSCORE_DATA.csv        median and interquartile band on the kPa grid
    68_SUPPFIG-9_SiteCurvesKpa_NEP_ZSCORE_DATA_SiteCurves.csv   every site curve, in kPa
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.paths import load_settings

SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
POLY_DEGREE = 4
MIN_BINS = 15               # as stage 49: a site needs this many VPD bins for a fit
KPA_GRID = np.arange(0.0, 4.001, 0.02)
XMAX = 4.0

SITE_LINE = '#6b7280'
MEDIAN = '#000000'
BAND = '#9ca3af'
THRESHOLD = '#d55e00'
# Okabe and Ito, as in Figures 1 and 3.
IGBP_COLORS = {'ENF': '#0072B2', 'DBF': '#009E73', 'MF': '#E69F00', 'EBF': '#CC79A7'}
IGBP_ORDER = ['ENF', 'DBF', 'MF', 'EBF']
AX_LABELS_FONTSIZE = 12

settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'
agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT
dir_out.mkdir(parents=True, exist_ok=True)
stem = dir_out / '68_SUPPFIG-9_SiteCurvesKpa'

# The all-sites threshold, as written by script 47.
robustness = pd.read_csv(agg / f'47_THRESHOLD_Robustness_{FLUX}.csv')
THRESHOLD_KPA = float(robustness.loc[robustness['test'] == 'PUBLISHED REFERENCE', 'threshold_kpa'].iloc[0])

# Site statistics for the kPa mapping, CD-Ygb converted from Pa.
sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                    / "21_SUBSETS_parquet_vars_stats_subsets.csv").set_index('SITE')
sites.loc[sites.index.isin(['CD-Ygb']), ['VPD_Z0', 'VPD_SD']] /= 100


def to_kpa(site, z):
    return (sites.at[site, 'VPD_Z0'] + z * sites.at[site, 'VPD_SD']) / 10


# The per-site crossings of stage 49, in sigma, mapped to kPa.
thr = pd.read_csv(agg / '49_SiteThresholds.csv').set_index('SITE')
thr = thr.loc[thr.index.isin(sites.index)].copy()
thr['crossing_kpa'] = [to_kpa(s, z) for s, z in zip(thr.index, thr['threshold_z'])]

# Per-site curves: the stage 41 cells averaged per VPD bin, then the stage 49 fit, in kPa.
d = pd.read_parquet(agg / f'41_SHAPVALUES-{shap_type}_meanAggregatedPerSite_BIN-TA_ZSCORE+BIN-VPD_ZSCORE+{FLUX}.parquet',
                    columns=['SITE', 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS']).dropna()
d['BIN_VPD_ZSCORE'] = d['BIN_VPD_ZSCORE'].round(1)
curves = d.groupby(['SITE', 'BIN_VPD_ZSCORE'])['VPD_ZSCORE_SHAPVALS'].mean()

fig = plt.figure(figsize=(11, 7.6), dpi=150, facecolor='white')
# The curves sit on top as panel a and the strip of crossings underneath as panel b, because
# the crossings are read off the curves; the x axis is shared and labeled once, under the strip.
gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.16], left=0.09, right=0.98, top=0.97, bottom=0.10, hspace=0.06)
ax = fig.add_subplot(gs[0, 0])
ax_strip = fig.add_subplot(gs[1, 0], sharex=ax)

resampled = []
site_curves = []   # the thin lines, for the data file
n_drawn = 0
for site in thr.index:
    if site not in curves.index.get_level_values(0):
        continue
    c = curves.loc[site]
    if len(c) < MIN_BINS:
        continue
    z = c.index.to_numpy(float)
    y = c.to_numpy(float)
    coeffs = np.polyfit(z, y, POLY_DEGREE)
    zz = np.linspace(z.min(), z.max(), 200)
    yy = np.polyval(coeffs, zz)
    kpa = to_kpa(site, zz)
    ax.plot(kpa, yy, color=SITE_LINE, lw=0.6, alpha=0.25, zorder=2)
    site_curves.append(pd.DataFrame({'SITE': site, 'IGBP': thr.at[site, 'IGBP'], 'vpd_kpa': kpa, 'vpd_effect': yy}))
    n_drawn += 1
    inside = (KPA_GRID >= kpa.min()) & (KPA_GRID <= kpa.max())
    r = np.full_like(KPA_GRID, np.nan)
    r[inside] = np.interp(KPA_GRID[inside], kpa, yy)
    resampled.append(r)

M = np.vstack(resampled)
# The median line is drawn only where at least half the drawn sites have data.
n_per_kpa = np.isfinite(M).sum(axis=0)
keep = n_per_kpa >= len(resampled) / 2
with np.errstate(all='ignore'):
    med = np.nanmedian(M, axis=0)
    q1 = np.nanpercentile(M, 25, axis=0)
    q3 = np.nanpercentile(M, 75, axis=0)

ax.fill_between(KPA_GRID[keep], q1[keep], q3[keep], color=BAND, alpha=0.35, lw=0, zorder=3)
ax.plot(KPA_GRID[keep], med[keep], color=MEDIAN, lw=2.4, zorder=5)
ax.axhline(0, color='black', lw=0.8, zorder=1)
ax.axvline(THRESHOLD_KPA, color=THRESHOLD, lw=1.5, ls=(0, (5, 4)), zorder=4)

ok = thr.dropna(subset=['crossing_kpa'])
for igbp in IGBP_ORDER:
    sub = ok[ok['IGBP'] == igbp]
    ax.scatter(sub['crossing_kpa'], np.zeros(len(sub)), s=30, marker='|', color=IGBP_COLORS[igbp],
               lw=1.3, alpha=0.9, zorder=6)

# The strip: one dot per site at its crossing, one row per forest type.
rng = np.random.default_rng(0)
ylev = {igbp: 3 - i for i, igbp in enumerate(IGBP_ORDER)}
for igbp in IGBP_ORDER:
    sub = ok[ok['IGBP'] == igbp]
    yy = ylev[igbp] + rng.uniform(-0.28, 0.28, len(sub))
    ax_strip.scatter(sub['crossing_kpa'], yy, s=16, color=IGBP_COLORS[igbp], alpha=0.85, lw=0, zorder=3)
ax_strip.axvline(THRESHOLD_KPA, color=THRESHOLD, lw=1.5, ls=(0, (5, 4)), zorder=2)
ax_strip.set_ylim(-0.7, 3.7)
ax_strip.set_yticks(list(ylev.values()))
ax_strip.set_yticklabels(list(ylev.keys()), fontsize=AX_LABELS_FONTSIZE * 0.85)
for side in ('top', 'right'):
    ax_strip.spines[side].set_visible(False)
ax_strip.tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE * 0.9, length=4)
ax_strip.tick_params(axis='y', length=0)
ax_strip.set_xlabel('VPD (kPa)', fontsize=AX_LABELS_FONTSIZE)
ax_strip.text(-0.075, 1.0, 'b', transform=ax_strip.transAxes, fontsize=AX_LABELS_FONTSIZE * 1.3,
              fontweight='bold', ha='right', va='top')
ax.text(-0.075, 1.0, 'a', transform=ax.transAxes, fontsize=AX_LABELS_FONTSIZE * 1.3,
        fontweight='bold', ha='right', va='top')

ax.set_xlim(0, XMAX)
ymax = float(np.nanpercentile(np.abs(M), 99))
ax.set_ylim(-ymax, ymax)
ax.tick_params(axis='x', labelbottom=False)
ax.set_ylabel('VPD effect on NEP ($\\sigma$)', fontsize=AX_LABELS_FONTSIZE)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
ax.tick_params(labelsize=AX_LABELS_FONTSIZE * 0.9, length=4)

handles = [
    Line2D([], [], color=SITE_LINE, lw=1.0, alpha=0.6, label=f'single site (n = {n_drawn})'),
    Line2D([], [], color=MEDIAN, lw=2.4, label='median across sites'),
    Patch(facecolor=BAND, alpha=0.35, label='interquartile range across sites'),
    Line2D([], [], color=THRESHOLD, lw=1.5, ls=(0, (5, 4)), label=f'threshold, all sites ({THRESHOLD_KPA:.2f} kPa)'),
] + [Line2D([], [], color=IGBP_COLORS[i], marker='|', lw=0, markersize=9, markeredgewidth=1.5,
            label=f'site crossing, {i} (n = {(ok["IGBP"] == i).sum()})') for i in IGBP_ORDER]
ax.legend(handles=handles, loc='upper right', frameon=False, fontsize=AX_LABELS_FONTSIZE * 0.85,
          ncol=2, handlelength=2.2, columnspacing=1.6)

out_png = f'{stem}_{FLUX}.png'
fig.savefig(out_png, dpi=300, facecolor='white', bbox_inches='tight')
thr.reset_index()[['SITE', 'IGBP', 'threshold_z', 'crossing_kpa']].to_csv(f'{stem}_Crossings_{FLUX}.csv', index=False)
# The plotted values: the median and interquartile band (panel a) and every site curve.
pd.DataFrame({'vpd_kpa': KPA_GRID, 'n_sites': n_per_kpa, 'drawn': keep,
              'median': med, 'q25': q1, 'q75': q3}).to_csv(f'{stem}_{FLUX}_DATA.csv', index=False)
pd.concat(site_curves).to_csv(f'{stem}_{FLUX}_DATA_SiteCurves.csv', index=False)
mean_c, med_c = ok['crossing_kpa'].mean(), ok['crossing_kpa'].median()
q25, q75 = ok['crossing_kpa'].quantile([0.25, 0.75])
print(f"{n_drawn} sites drawn, {len(ok)} crossings; mean {mean_c:.3f} kPa, median {med_c:.3f}, IQR {q25:.3f} to {q75:.3f}")
print(ok.groupby('IGBP')['crossing_kpa'].median().round(3).to_string())
print(f"Saved {out_png}")

if SHOW_PLOT:
    plt.show()
