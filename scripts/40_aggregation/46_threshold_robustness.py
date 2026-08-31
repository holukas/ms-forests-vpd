"""
Every robustness test of the VPD threshold, computed once and written to file.

This is the heavy half of what used to live in `50_figures`. Scripts under `50_figures`
draw, they do not compute, because they become notebooks under S10 and a notebook must not
aggregate a campaign. The bootstrap here refits a polynomial 2000 times for each of twelve
tests and takes about two minutes, which is exactly the kind of work that belongs at this
stage. `55_table-1+suppfig_threshold_robustness.py` reads what this writes and renders it
as Table 1 and the matching supplementary figure.

It also absorbs the estimator sweep that was `50_figures/60_threshold_method_sensitivity.py`.
That script read the CSV that figure script 54 writes, so a numbers script depended on a
figure script having run first. Here the same curve comes from the stage 42 output through
`src.files.load_data`, which is what script 54 itself plots, so the numbers are identical
and the dependency points the right way.

**The estimator is the published one.** The threshold is the highest zero crossing of a
fourth-order polynomial fitted to the VPD SHAP values averaged across sites, which is what
Figure 4 shows and what the abstract reports. An earlier version used the median of per-site
thresholds instead. That is a different quantity, it sat 0.07 kPa lower, and it silently
drops any site whose own curve never crosses zero.

**The interval is a bootstrap over sites.** Each replicate resamples sites with replacement,
re-averages their binned curves, refits the polynomial and takes the crossing. That answers
how much the threshold depends on which sites happen to be in the network. It is not the
prediction band of the fitted curve, which describes how well a polynomial fits one
aggregated curve and says nothing about site-to-site agreement, the point A13 makes.

**Two rows have no site bootstrap.** The estimator row varies how the curve is fitted rather
than which sites are included, so it carries the spread across its 23 settings. The ALE row
has no interval at all, because script 70 derives the all-sites value from a single pooled
curve rather than from per-site curves. Both are flagged in the `note` column.

Reads: stage 41 per-site curves for the baseline and the deep-sm, deeper-only and blocked-cv
variants, stage 42 for the aggregated curve, the stage 21 subsets table, and the ALE
threshold written by script 70.

Writes, into the aggregation folder:
    46_THRESHOLD_EstimatorSweep_{FLUX}.csv    23 estimator settings, thresholds in sigma
    46_THRESHOLD_Robustness_{FLUX}.csv        twelve tests plus the published reference, kPa
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

import src.files as files
from src.paths import data_path, load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True

# Run variant. An empty string reads the results behind the submitted figures and writes
# next to them. Any other value reads the matching variant folder. The aggregation must
# have run with the same value.
VARIANT = ""
SITE_SUBSET = ""

N_BOOT = 2000
POLY_DEGREE = 4          # the published choice

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
agg_base = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type
dir_out = agg_base / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)

CURVE_FILE = (f'41_SHAPVALUES-{shap_type}_meanAggregatedPerSite'
              f'_BIN-TA_ZSCORE+BIN-VPD_ZSCORE+{FLUX}.parquet')

# Sigma to kPa. Per-site mean and standard deviation, averaged over the sites in the run,
# so the mapping follows the site set.
#
# CD-Ygb records VPD in Pa where every other site uses hPa. It is converted here rather
# than dropped, which keeps all 208 sites: dividing by 100 gives a mean of 18.1 hPa and a
# standard deviation of 6.6 hPa, both inside the range the other sites span, 4.5 to
# 31.8 hPa. The z-scores are per site and therefore unaffected by the unit, so the
# conversion only touches the mapping back to kPa.
PA_UNIT_SITES = ['CD-Ygb']
subsets = pd.read_csv(data_path('data/outputs/20_subsets/'
                                '21_SUBSETS_parquet_vars_stats_subsets.csv'))
is_pa = subsets['SITE'].isin(PA_UNIT_SITES)
subsets.loc[is_pa, ['VPD_Z0', 'VPD_SD']] /= 100
subsets = subsets.set_index('SITE')


# ---------------------------------------------------------------------------
# Shared pieces
# ---------------------------------------------------------------------------

def highest_crossing(x, y):
    """Highest x where the curve crosses zero, matching src.fit.calc_threshold."""
    sign_changes = np.where(np.diff(np.sign(y)))[0]
    if len(sign_changes) == 0:
        return np.nan
    return max(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]) for i in sign_changes)


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


# ---------------------------------------------------------------------------
# Part 1: the estimator sweep, formerly script 60
# ---------------------------------------------------------------------------

def poly_threshold(x, y, degree):
    xf = np.linspace(x.min(), x.max(), 1000)
    return highest_crossing(xf, np.polyval(np.polyfit(x, y, degree), xf))


def lowess_threshold(x, y, frac):
    fitted = lowess(y, x, frac=frac, return_sorted=True)
    return highest_crossing(fitted[:, 0], fitted[:, 1])


def spline_threshold(x, y, s_factor):
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    # UnivariateSpline needs strictly increasing x, so average duplicate bins first
    frame = pd.DataFrame({'x': xs, 'y': ys}).groupby('x', as_index=False)['y'].mean()
    xu, yu = frame['x'].to_numpy(), frame['y'].to_numpy()
    spline = UnivariateSpline(xu, yu, s=s_factor * len(xu) * np.var(yu), k=3)
    xf = np.linspace(xu.min(), xu.max(), 1000)
    return highest_crossing(xf, spline(xf))


def rebin(x, y, width):
    """Coarsen the 0.1 sigma bins, averaging the cells that fall in each new bin."""
    edges = np.arange(np.floor(x.min()), np.ceil(x.max()) + width, width)
    idx = np.digitize(x, edges) - 1
    frame = pd.DataFrame({'bin': idx, 'x': x, 'y': y}).groupby('bin', as_index=False).mean()
    return frame['x'].to_numpy(), frame['y'].to_numpy()


def aggregated_curve():
    """The all-sites curve Figure 4 plots, straight from the stage 42 output.

    Script 54 writes exactly this frame to `54_FIG-4_..._ALLSITES_DATA.csv`, which is what
    the estimator sweep used to read. Calling `load_data` here gives the same numbers
    without needing a figure script to have run.
    """
    xvar, yvar, zvar = 'BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'TA_ZSCORE'
    aggfunc = 'mean'
    xagg = yagg = 'median'          # both bin coordinate columns, as in script 54
    _, subsetdf, _, _ = files.load_data(
        suffix='Sites', shap_type=shap_type, dir_res=dir_out, flux=FLUX, aggfunc=aggfunc,
        subsetcols=[(xvar, xagg), (yvar, yagg), (zvar, aggfunc), (yvar, 'sem')],
        count_vals_col=(zvar, 'count'), site_filter=None,
        x_in_filename='BIN-TA_ZSCORE', y_in_filename='BIN-VPD_ZSCORE')
    return (subsetdf.iloc[:, 0].to_numpy(float), subsetdf.iloc[:, 1].to_numpy(float))


def estimator_sweep():
    """Re-estimate the threshold under 23 settings. Same data, only the estimator changes."""
    x, y = aggregated_curve()
    print(f"{len(x)} cells over {len(np.unique(x))} VPD bins, "
          f"x from {x.min():.1f} to {x.max():.1f}")

    rows = []

    def record(family, setting, value, note=''):
        rows.append({'family': family, 'setting': setting, 'threshold_z': value, 'note': note})

    for deg in (2, 3, 4, 5, 6):
        record('polynomial', f"degree {deg}", poly_threshold(x, y, deg),
               'published choice' if deg == 4 else '')
    for frac in (0.2, 0.3, 0.4, 0.5, 0.7):
        record('lowess', f"frac {frac}", lowess_threshold(x, y, frac))
    for sf in (0.001, 0.01, 0.05, 0.1):
        record('spline', f"s factor {sf}", spline_threshold(x, y, sf))
    for width in (0.1, 0.2, 0.3, 0.5):
        xb, yb = rebin(x, y, width)
        record('bin width', f"{width} sigma", poly_threshold(xb, yb, 4), f"{len(xb)} bins")
    for limit in (2.0, 2.5, 3.0, 3.5, None):
        keep = np.ones_like(x, dtype=bool) if limit is None else np.abs(x) <= limit
        label = 'full range' if limit is None else f"|x| <= {limit}"
        record('fitting range', label, poly_threshold(x[keep], y[keep], 4),
               f"{keep.sum()} cells")

    out = pd.DataFrame(rows)
    finite = out['threshold_z'].dropna()
    print(f"estimator sweep: {len(finite)} of {len(out)} settings give a crossing, "
          f"median {finite.median():.3f}, range {finite.min():.3f} to {finite.max():.3f}")
    return out


# ---------------------------------------------------------------------------
# Part 2: the robustness rows
# ---------------------------------------------------------------------------

def site_curves(*subfolders):
    """Per-site VPD SHAP for every TA by VPD cell, as a site by cell matrix.

    Figure 4 fits the polynomial to the two-dimensional grid of TA and VPD bins, not to a
    curve collapsed over TA, and it keeps only cells held by at least half the sites
    (`src/files.py::load_data`). Both matter: collapsing over TA and keeping every sparse
    edge cell moves the crossing by more than 0.2 kPa, so the reference would not reproduce
    the published value.
    """
    path = agg_base.joinpath(*subfolders) / CURVE_FILE
    d = pd.read_parquet(path, columns=['SITE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE',
                                       'VPD_ZSCORE_SHAPVALS']).dropna()
    d = d.loc[d['SITE'].isin(subsets.index)]
    piv = (d.groupby(['SITE', 'BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE'])['VPD_ZSCORE_SHAPVALS']
           .mean().unstack(['BIN_TA_ZSCORE', 'BIN_VPD_ZSCORE']).sort_index())
    return piv


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


def robustness_rows(sweep):
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
    # analysis. Script 70 writes the all-sites value from a single pooled curve, so
    # there is no interval to show and none is invented here.
    ale_dir = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'ale' / VARIANT
    ale = pd.read_csv(ale_dir / f'70_SUPPFIG-X_ALE_ResponseCurve_VPD_ZSCORE_{FLUX}_THRESHOLD.csv')
    ale_z = float(ale.loc[ale['group'] == 'ALL SITES', 'threshold'].iloc[0])
    ale_kpa = to_kpa(ale_z, base.index)
    rows.append(('Model fitting', 'ALE instead of SHAP', ale_kpa, ale_kpa, ale_kpa,
                 len(base), 'single pooled curve, no interval'))

    # The estimator row varies the fit, not the sites, so it carries the spread across the
    # settings rather than a bootstrap.
    est = np.array([to_kpa(z, base.index) for z in sweep['threshold_z'].dropna()])
    rows.append(('Model fitting', f'Threshold estimator, {len(sweep)} settings',
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

    # The reference goes in as its own row, so the renderer needs nothing else.
    out = pd.DataFrame(rows + [('', 'PUBLISHED REFERENCE', published, pub_lo, pub_hi,
                                n_base, None)],
                       columns=['group', 'test', 'threshold_kpa', 'lower', 'upper',
                                'n_sites', 'note'])
    out['shift_kpa'] = out['threshold_kpa'] - published
    return out


def main():
    sweep = estimator_sweep()
    sweep_file = dir_out / f'46_THRESHOLD_EstimatorSweep_{FLUX}.csv'
    sweep.to_csv(sweep_file, index=False)

    out = robustness_rows(sweep)
    rows_file = dir_out / f'46_THRESHOLD_Robustness_{FLUX}.csv'
    out.to_csv(rows_file, index=False)

    ref = out.loc[out['test'] == 'PUBLISHED REFERENCE'].iloc[0]
    print()
    print(out.round(3).to_string(index=False))
    print(f"\npublished reference: {ref['threshold_kpa']:.3f} "
          f"[{ref['lower']:.3f}, {ref['upper']:.3f}] kPa, {int(ref['n_sites'])} sites")
    print(f"Saved {sweep_file}")
    print(f"Saved {rows_file}")


if __name__ == '__main__':
    main()
