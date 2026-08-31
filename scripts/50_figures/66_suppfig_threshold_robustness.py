"""
PLANNED SUPPLEMENTARY FIGURE, not yet adopted.

Every robustness test of the VPD threshold on one axis, with a confidence interval on each,
plotted as the shift from the published value.

**The main text uses the table, not this figure.** Decided 31 August 2026. The
differences between tests are at most 0.08 kPa on a threshold of 1.26, so a figure either
magnifies them by zooming the axis or hides them by not zooming. Table 1, written by
`70_table-1_threshold_robustness.py` from this script's cache, carries the numbers in the
main text. This figure stays as the supplementary view for readers who want the shape of
the result rather than the values.

**The estimator is the published one.** The threshold is the highest zero crossing of a
fourth-order polynomial fitted to the VPD SHAP values averaged across sites, which is what
Figure 4 shows and what the abstract reports. An earlier version of this figure used the
median of per-site thresholds instead. That is a different quantity, it sat 0.07 kPa lower,
and it silently drops any site whose own curve never crosses zero, so a robustness figure
built on it would be testing an estimator the reader never sees.

**The interval is a bootstrap over sites.** Each replicate resamples sites with
replacement, re-averages their binned curves, refits the polynomial and takes the crossing.
That answers how much the threshold depends on which sites happen to be in the network,
which is the question a robustness figure is asked, and it uses the same estimator as the
point value. It is not the prediction band of the fitted curve: that band describes how well
a polynomial fits one aggregated curve and says nothing about site-to-site agreement, the
point A13 makes.

**Two rows have no site bootstrap** and are drawn with a diamond. The estimator row varies
how the curve is fitted rather than which sites are included, so it shows the spread across
its 23 settings. The ALE row has no interval at all, because script 55 derives the all-sites
value from a single pooled curve rather than from per-site curves. The caption says both.

**Why the VPD quartile split is not here.** It used to be the last row. It does not belong:
the other rows ask whether the number survives a different analytical choice, while that
one asks whether the threshold varies with site climate, and the answer is yes. Its range,
1.00 to 1.41 kPa, was the widest bar in the figure, so a reader scanning for robustness saw
the largest apparent instability where there was actually signal. Most of that signal is
also mechanical: the threshold is estimated in sigma and converted per site, and drier sites
have a larger mean and spread, so a constant sigma threshold already yields a higher kPa
value in the drier quartiles. Holding sigma at 0.20 for every quartile gives a spread of
0.74 kPa, wider than the 0.57 observed. In sigma the pattern runs the other way, 0.25, 0.29,
0.16, 0.04, falling with dryness. It belongs in its own supplementary analysis with that
caveat stated, `62_threshold_vs_site_vpd_range.py`.

Reads the per-site binned curves written by stage 41, plus what scripts 60 and 62 wrote.
Nothing is refitted from the models.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.paths import data_path, load_settings

SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# Recompute the thresholds and the bootstrap, or reuse what the last run wrote. The
# bootstrap refits the curve 2000 times for each of twelve rows and takes about three
# minutes, while the layout takes seconds, so a layout change should not pay for the
# statistics again. Set to True after changing anything that affects the numbers.
RECOMPUTE = False
N_BOOT = 2000
POLY_DEGREE = 4          # the published choice
MOVER = 0.06             # kPa, beyond this a row reads as a real shift
AX_LABELS_FONTSIZE = 12
COLOR_POINT = '#0072B2'
COLOR_MOVER = '#b2182b'
COLOR_REF = '#D55E00'

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
folder = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
agg_base = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type

CURVE_FILE = (f'41_SHAPVALUES-{shap_type}_meanAggregatedPerSite'
              f'_BIN-TA_ZSCORE+BIN-VPD_ZSCORE+{FLUX}.parquet')

# Sigma to kPa. Per-site mean and standard deviation, averaged over the sites in the run,
# so the mapping follows the site set.
#
# CD-Ygb records VPD in Pa where every other site uses hPa. Other scripts drop it for that
# reason. It is converted here instead, which keeps all 208 sites: dividing by 100 gives a
# mean of 18.1 hPa and a standard deviation of 6.6 hPa, both inside the range the other
# sites span, 4.5 to 31.8 hPa. The z-scores are per site and therefore unaffected by the
# unit, so the conversion only touches the mapping back to kPa.
PA_UNIT_SITES = ['CD-Ygb']
subsets = pd.read_csv(data_path('data/outputs/20_subsets/'
                                '21_SUBSETS_parquet_vars_stats_subsets.csv'))
is_pa = subsets['SITE'].isin(PA_UNIT_SITES)
subsets.loc[is_pa, ['VPD_Z0', 'VPD_SD']] /= 100
subsets = subsets.set_index('SITE')


def site_curves(*subfolders):
    """Per-site VPD SHAP for every TA by VPD cell, as a site by cell matrix.

    Figure 4 fits the polynomial to the two-dimensional grid of TA and VPD bins, not to a
    curve collapsed over TA, and it keeps only cells held by at least half the sites
    (`src/files.py::load_data`). Both matter: collapsing over TA and keeping every sparse
    edge cell moves the crossing by more than 0.2 kPa, so the reference would not
    reproduce the published value.
    """
    path = agg_base.joinpath(*subfolders) / CURVE_FILE
    d = pd.read_parquet(path, columns=['SITE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE',
                                       'VPD_ZSCORE_SHAPVALS']).dropna()
    d = d.loc[d['SITE'].isin(subsets.index)]
    piv = (d.groupby(['SITE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE'])['VPD_ZSCORE_SHAPVALS']
           .mean().unstack(['BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE']).sort_index())
    return piv


def crossing(x, y):
    """Highest zero crossing of the fitted polynomial, or nan."""
    ok = np.isfinite(y)
    if ok.sum() <= POLY_DEGREE + 1:
        return np.nan
    coef = np.polyfit(x[ok], y[ok], POLY_DEGREE)
    fine = np.linspace(x[ok].min(), x[ok].max(), 2000)
    vals = np.polyval(coef, fine)
    sign_change = np.where(np.diff(np.sign(vals)))[0]
    if len(sign_change) == 0:
        return np.nan
    roots = [fine[i] - vals[i] * (fine[i + 1] - fine[i]) / (vals[i + 1] - vals[i])
             for i in sign_change]
    return max(roots)


def to_kpa(z, sites):
    """Sigma to kPa for one site set."""
    use = subsets.loc[subsets.index.isin(sites)]
    return (use['VPD_Z0'].mean() + z * use['VPD_SD'].mean()) / 10


def aggregate(M, x, min_sites):
    """Median across sites per cell, keeping only cells held by enough sites.

    Median, not mean. Script 42 writes several aggregations per bin and Figure 4 reads the
    median column, so a mean here reproduces 1.252 kPa instead of the published 1.260.
    """
    counts = np.isfinite(M).sum(axis=0)
    keep = counts >= min_sites
    if keep.sum() <= POLY_DEGREE + 1:
        return np.array([]), np.array([])
    with np.errstate(invalid='ignore'):
        y = np.nanmedian(M[:, keep], axis=0)
    return x[keep], y


def threshold_with_ci(piv, seed=0):
    """Crossing of the aggregated curve, with a bootstrap over sites, both in kPa."""
    x = piv.columns.get_level_values('BIN_VPD_ZSCORE').to_numpy(dtype=float)
    M = piv.to_numpy(dtype=float)
    sites = piv.index
    n = len(M)
    min_sites = np.ceil(n / 2)

    xk, yk = aggregate(M, x, min_sites)
    point = to_kpa(crossing(xk, yk), sites)

    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOT)
    for i in range(N_BOOT):
        xk, yk = aggregate(M[rng.integers(0, n, n)], x, min_sites)
        boot[i] = crossing(xk, yk) if len(xk) else np.nan
    boot = boot[np.isfinite(boot)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(point), float(to_kpa(lo, sites)), float(to_kpa(hi, sites)), n


outfile = folder / f'66_PLANNED-SUPPFIG_ThresholdRobustness_{FLUX}.png'
cache = Path(str(outfile).replace('.png', '_DATA.csv'))
cached = (not RECOMPUTE) and cache.exists()

if cached:
    print(f'reusing {cache.name}, set RECOMPUTE = True to redo the numbers')
    stored = pd.read_csv(cache)
    ref = stored.loc[stored['test'] == 'PUBLISHED REFERENCE'].iloc[0]
    published, pub_lo, pub_hi = ref['threshold_kpa'], ref['lower'], ref['upper']
    n_base = int(ref['n_sites'])
    rows = [tuple(r) for r in stored.loc[stored['test'] != 'PUBLISHED REFERENCE']
            [['group', 'test', 'threshold_kpa', 'lower', 'upper', 'n_sites', 'note']]
            .itertuples(index=False, name=None)]
    rows = [(g, t, v, lo, hi, int(n), (None if pd.isna(note) else note))
            for g, t, v, lo, hi, n, note in rows]
else:
    rows = []  # (group, label, value, lo, hi, n_sites, note)

    # Reference: the published run, same estimator.
    base = site_curves()
    published, pub_lo, pub_hi, n_base = threshold_with_ci(base)

    # --- soil water depth -------------------------------------------------------
    for label, sub in [('Deepest available layer, all sites', ('deep-sm',)),
                       ('Matched sites, layer 1', ('', 'deeper-only')),
                       ('Matched sites, deepest layer', ('deep-sm', 'deeper-only'))]:
        piv = site_curves(*[p for p in sub if p])
        rows.append(('Soil water depth', label, *threshold_with_ci(piv, seed=1), None))

    # --- model fitting ----------------------------------------------------------
    piv = site_curves('blocked-cv')
    rows.append(('Model fitting', 'Blocked cross-validation',
                 *threshold_with_ci(piv, seed=2), None))

    # Accumulated local effects instead of SHAP. A different attribution method
    # altogether, so this is the row that shares least machinery with the published
    # analysis. Script 55 writes the all-sites value from a single pooled curve, so
    # there is no interval to show and none is invented here.
    ale = pd.read_csv(folder.parent / 'ale' /
                      f'55_SUPPFIG-X_ALE_ResponseCurve_VPD_ZSCORE_{FLUX}_THRESHOLD.csv')
    ale_z = float(ale.loc[ale['group'] == 'ALL SITES', 'threshold'].iloc[0])
    ale_kpa = to_kpa(ale_z, base.index)
    rows.append(('Model fitting', 'ALE instead of SHAP', ale_kpa, ale_kpa, ale_kpa,
                 len(base), 'single pooled curve, no interval'))

    # The estimator row varies the fit, not the sites, so it carries the spread across the 23
    # settings rather than a bootstrap.
    method = pd.read_csv(folder / '60_Threshold_MethodSensitivity.csv')
    est = np.array([to_kpa(z, base.index) for z in method['threshold_z']])
    rows.append(('Model fitting', f'Threshold estimator, {len(method)} settings',
                 float(np.median(est)), float(est.min()), float(est.max()), n_base,
                 'spread across settings'))

    # --- site set ---------------------------------------------------------------
    REGIONS = {
        'Europe': ['AT', 'BE', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR',
                   'IE', 'IT', 'NL', 'PL', 'PT', 'RU', 'SE', 'SJ', 'SK', 'UK'],
        'North America': ['US', 'CA', 'MX', 'PR', 'CR', 'GL'],
    }
    LOOKUP = {code: region for region, codes in REGIONS.items() for code in codes}
    region_of = pd.Series(base.index.str.split('-').str[0].map(LOOKUP), index=base.index)
    for drop, label in [(['Europe'], 'Europe removed'),
                        (['North America'], 'North America removed'),
                        (['Europe', 'North America'], 'Europe and North America removed')]:
        keep = base.loc[~region_of.isin(drop)]
        rows.append(('Site set', label, *threshold_with_ci(keep, seed=len(drop) + 10), None))

    years = subsets['N_YEARS']
    for min_years in (3, 5, 10):
        keep = base.loc[base.index.isin(years[years >= min_years].index)]
        rows.append(('Site set', f'Records of at least {min_years} years',
                     *threshold_with_ci(keep, seed=min_years + 20), None))

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# Plotted as the shift from the published threshold. Every test lands near 1.26 kPa, so on
# an absolute axis the points crowd into a narrow band and the reader has to work out that
# this is agreement. Centring on the published value makes zero mean something and lets the
# few tests that do move stand out.

LABEL_X, VALUE_X = -0.46, 1.02

ordered = list(reversed(rows))
ypos, group_rows, y = [], {}, 0.0
prev_group = None
for group, *_ in ordered:
    if prev_group is not None and group != prev_group:
        y += 1.15
    ypos.append(y)
    group_rows.setdefault(group, []).append(y)
    prev_group = group
    y += 1.0

fig, ax = plt.subplots(figsize=(11, 7.4), dpi=150)
# The caption runs to four lines, so it gets its own band at the bottom.
fig.subplots_adjust(left=0.34, right=0.87, top=0.88, bottom=0.24)
trans = ax.get_yaxis_transform()

ax.axvspan(pub_lo - published, pub_hi - published, color=COLOR_REF, alpha=0.10,
           zorder=0, linewidth=0)
ax.axvline(0, color=COLOR_REF, lw=1.4, zorder=2)

for (group, label, mid, lo, hi, n, note), y in zip(ordered, ypos):
    shift = mid - published
    colour = COLOR_MOVER if abs(shift) >= MOVER else COLOR_POINT
    ax.plot([0, 1], [y, y], transform=trans, color='#F2F2F2', lw=0.8, zorder=0)
    if hi > lo:
        ax.plot([lo - published, hi - published], [y, y], color=colour, lw=3.2,
                alpha=0.32 if note else 0.55, zorder=3, solid_capstyle='round')
    ax.scatter([shift], [y], s=44, color=colour, zorder=4,
               marker='D' if note else 'o', linewidths=0)
    ax.text(LABEL_X, y, label, transform=trans, ha='left', va='center',
            fontsize=AX_LABELS_FONTSIZE * 0.92, color='#1a1a1a')
    ax.text(LABEL_X, y - 0.34, f'{n} sites' + (f'   {note}' if note else ''),
            transform=trans, ha='left', va='center',
            fontsize=AX_LABELS_FONTSIZE * 0.66, color='#9a9a9a')
    ax.text(VALUE_X, y, f'{shift:+.2f}', transform=trans, ha='left', va='center',
            fontsize=AX_LABELS_FONTSIZE * 0.85,
            color=colour if abs(shift) >= MOVER else '#666666',
            fontweight='bold' if abs(shift) >= MOVER else 'normal')

for group, ys in group_rows.items():
    top = max(ys) + 0.62
    ax.text(LABEL_X, top, group.upper(), transform=trans, ha='left', va='center',
            fontsize=AX_LABELS_FONTSIZE * 0.7, color='#9a9a9a', fontweight='bold')
    ax.plot([LABEL_X, 1.0], [top - 0.28, top - 0.28], transform=trans, color='#E2E2E2',
            lw=0.9, zorder=0, clip_on=False)

ax.set_yticks([])
ax.set_ylim(min(ypos) - 1.0, max(ypos) + 1.1)
ax.set_xlabel('Shift from the published threshold (kPa)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE, colors='#555555', length=4)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color('#CCCCCC')

ax.text(0, 1.01, f' published {published:.2f} kPa', transform=ax.get_xaxis_transform(),
        ha='left', va='bottom', color=COLOR_REF, fontweight='bold',
        fontsize=AX_LABELS_FONTSIZE * 0.9, clip_on=False)
ax.text(VALUE_X, max(ypos) + 0.62, 'shift, kPa', transform=trans, ha='left', va='center',
        color='#9a9a9a', fontsize=AX_LABELS_FONTSIZE * 0.7, fontweight='bold')

fig.text(0.025, 0.015,
         f'Zero crossing of the fitted curve, published value {published:.2f} kPa '
         f'[{pub_lo:.2f}, {pub_hi:.2f}]. Bars are a bootstrap over sites, {N_BOOT} '
         f'replicates.' + chr(10) + f'Red marks a shift of at least {MOVER:.2f} kPa.'
         + chr(10) + 'Diamonds mark the two rows without a site bootstrap: the '
         'estimator row shows the spread across its 23 settings, and ALE has no'
         + chr(10) + 'interval because that method gives one pooled curve.',
         fontsize=AX_LABELS_FONTSIZE * 0.72, color='#666666', ha='left', linespacing=1.5)
fig.text(0.025, 0.962, 'PLANNED supplementary figure, not yet adopted',
         fontsize=AX_LABELS_FONTSIZE * 0.8, color=COLOR_REF, fontweight='bold', ha='left')

fig.savefig(outfile, dpi=300, facecolor='white')

# The reference goes into the cache as its own row, so a reused run does not have to
# recompute it either.
out = pd.DataFrame(rows + [('', 'PUBLISHED REFERENCE', published, pub_lo, pub_hi,
                            n_base, None)],
                   columns=['group', 'test', 'threshold_kpa', 'lower', 'upper',
                            'n_sites', 'note'])
out['shift_kpa'] = out['threshold_kpa'] - published
if not cached:
    out.to_csv(cache, index=False)
print(out.round(3).to_string(index=False))
print(f"\npublished reference: {published:.3f} [{pub_lo:.3f}, {pub_hi:.3f}] kPa, "
      f"{n_base} sites")
print(f"Saved to {outfile}")
if SHOW_PLOT:
    plt.show()
