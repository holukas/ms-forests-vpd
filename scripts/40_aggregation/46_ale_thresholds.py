"""
Put the per-site ALE curves on a common grid and derive VPD thresholds from them.

Each site's curve is interpolated to the grid and smoothed with a fourth-order polynomial;
its threshold is the first positive to negative crossing. A group's threshold (`ALL SITES`,
then one row per forest type) is the mean of the per-site crossings, with a 95% interval
from the t distribution; sites whose curve never crosses are left out. The
`ALL SITES (pooled curve)` row is a different quantity: the crossing of the cross-site
mean curve (median if USE_MEDIAN), kept where at least half the sites have data.

Reads the stage 32 files `{site}_ale_{FLUX}.parquet` and `{site}_ale_curves_{FLUX}.csv`
and the stage 21 subsets table.

Writes:
- 46_ALE_SiteCurves_{FEATURE}_{FLUX}.parquet, one curve per site, drawn by script 57
- 46_ALE_Thresholds_{FEATURE}_{FLUX}.csv, read by script 47 for its ALE row
"""
from pathlib import Path

import diive as dv
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d

from src.paths import data_path, load_settings

FLUX = 'NEP_ZSCORE'
FEATURE = 'VPD_ZSCORE'

# Run variant. An empty string reads the results behind the submitted figures and writes
# next to them. The ALE campaign must have run with the same value.
VARIANT = ""

USE_MEDIAN = False       # False: mean across sites, as in the main analysis
IGBPS = ['ENF', 'DBF', 'MF', 'EBF']
GRID_POINTS = 50

settings = load_settings()
dir_ale = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'ale' / VARIANT
dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / 'ale' / VARIANT
dir_out.mkdir(parents=True, exist_ok=True)


def fit_polynomial_to_curves(curves_array, grid):
    """Fit 4th-order polynomial to each curve, preserving NaN regions."""
    fitted = []
    for curve in curves_array:
        valid = ~np.isnan(curve)
        if np.sum(valid) > 4:
            try:
                x_valid, y_valid = grid[valid], curve[valid]
                poly = np.poly1d(np.polyfit(x_valid, y_valid, 4))
                result = poly(grid)
                result[~valid] = np.nan
                fitted.append(result)
            except Exception:
                fitted.append(curve)
        else:
            fitted.append(curve)
    return fitted


def find_zero_crossing(curve, grid):
    """Highest positive to negative crossing, by linear interpolation, as for SHAP. NaN if there is none."""
    valid = ~np.isnan(curve)
    if np.sum(valid) <= 1:
        return np.nan
    curve_valid = curve[valid]
    grid_valid = grid[valid]
    if np.all(curve_valid >= 0) or np.all(curve_valid <= 0):
        return np.nan
    sign_changes = np.diff(np.sign(curve_valid))
    for idx in np.where(sign_changes != 0)[0][::-1]:
        if curve_valid[idx] > 0 and curve_valid[idx + 1] <= 0:
            x1, x2 = grid_valid[idx], grid_valid[idx + 1]
            y1, y2 = curve_valid[idx], curve_valid[idx + 1]
            return x1 - y1 * (x2 - x1) / (y2 - y1) if (y2 - y1) != 0 else (x1 + x2) / 2
    return np.nan


def calc_ci_95(values):
    """95 % confidence interval for the mean, t distribution."""
    mean = values.mean()
    sem = stats.sem(values)
    ci = sem * stats.t.ppf((1 + 0.95) / 2, len(values) - 1)
    return mean, mean - ci, mean + ci


def load_site_curves():
    """One raw ALE curve per site, with its forest type.

    A site counts only if its parquet file has more than 10 records with both FEATURE and
    FLUX. The curve itself comes from the csv that stage 32 writes next to it.
    """
    subsets_df = pd.read_csv(str(data_path(
        "data/outputs/20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv")))
    curves, igbp_of, n_records = [], {}, 0

    for _, siteconfig in subsets_df.iterrows():
        site = siteconfig['SITE']
        igbp_of[site] = siteconfig['IGBP']

        filepath = dir_ale / f"{site}_ale_{FLUX}.parquet"
        if not filepath.exists():
            continue
        try:
            site_data = dv.load_parquet(filepath, sanitize_timestamp=False,
                                        output_middle_timestamp=False)
        except Exception:
            continue
        if FEATURE not in site_data.columns or FLUX not in site_data.columns:
            continue
        valid = ~(site_data[FEATURE].isna() | site_data[FLUX].isna())
        if valid.sum() <= 10:
            continue
        n_records += int(valid.sum())

        curve_file = dir_ale / f"{site}_ale_curves_{FLUX}.csv"
        if not curve_file.exists():
            continue
        try:
            frame = pd.read_csv(curve_file)
        except Exception:
            continue
        frame = frame[frame['feature'] == FEATURE]
        if len(frame) == 0:
            continue
        curves.append({'site': site,
                       'feature_value': frame['feature_value'].values,
                       'effect': frame['effect'].values})

    print(f"{len(curves)} sites with an ALE curve for {FEATURE}, {n_records:,} records")
    return curves, igbp_of


def interpolate(curves):
    """Put every site on one grid. Outside a site's own range the curve stays NaN."""
    everything = np.concatenate([c['feature_value'] for c in curves])
    grid = np.linspace(everything.min(), everything.max(), GRID_POINTS)

    rows, sites = [], []
    for c in curves:
        order = np.argsort(c['feature_value'])
        x, y = c['feature_value'][order], c['effect'][order]
        try:
            f = interp1d(x, y, kind='linear', bounds_error=False, fill_value=np.nan)
        except Exception:
            continue
        rows.append(f(grid))
        sites.append(c['site'])
    return grid, np.array(rows), sites


def group_threshold(curves_on_grid, grid):
    """Mean of the per-site crossings for one group, with 95% interval, SD, SE and counts.

    The standard deviation describes how much sites differ; the standard error and the
    interval describe how well the mean is determined.
    """
    fitted = fit_polynomial_to_curves(curves_on_grid, grid)
    crossings = np.array([find_zero_crossing(c, grid) for c in fitted])
    crossings = crossings[~np.isnan(crossings)]
    if len(crossings) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, 0, len(fitted)
    mean, lo, hi = calc_ci_95(crossings)
    sd = float(crossings.std(ddof=1)) if len(crossings) > 1 else np.nan
    se = float(stats.sem(crossings)) if len(crossings) > 1 else np.nan
    return mean, lo, hi, sd, se, len(crossings), len(fitted)


def main():
    curves, igbp_of = load_site_curves()
    grid, matrix, sites = interpolate(curves)

    # The per-site curves, as the figure draws them. Columns are the grid points.
    frame = pd.DataFrame(matrix, index=pd.Index(sites, name='SITE'),
                         columns=[f'{g:.6f}' for g in grid])
    frame.insert(0, 'IGBP', [igbp_of.get(s) for s in sites])
    curves_file = dir_out / f'46_ALE_SiteCurves_{FEATURE}_{FLUX}.parquet'
    frame.to_parquet(curves_file)

    # All sites: the mean of the per-site crossings, not the crossing of one pooled curve.
    fitted = fit_polynomial_to_curves(matrix, grid)
    mean, lo, hi, sd, se, n_cross, n_total = group_threshold(matrix, grid)

    rows = [{'feature': FEATURE, 'target': FLUX,
             'method': 'Per-site zero-crossings, mean', 'group': 'ALL SITES',
             'threshold': f'{mean:.3f}', 'ci_lower': f'{lo:.3f}', 'ci_upper': f'{hi:.3f}',
             'sd': f'{sd:.3f}', 'se': f'{se:.3f}',
             'n_sites': n_cross, 'mode': 'Per-site curves'}]

    # The pooled curve, written so the two definitions stay visibly apart. Nothing in the
    # manuscript reports this one.
    agg = np.nanmedian if USE_MEDIAN else np.nanmean
    pooled = agg(np.array(fitted), axis=0)
    coverage = np.sum(~np.isnan(np.array(fitted)), axis=0)
    pooled[coverage < len(fitted) / 2] = np.nan
    rows.append({'feature': FEATURE, 'target': FLUX, 'method': 'Pooled curve zero-crossing',
                 'group': 'ALL SITES (pooled curve)',
                 'threshold': f'{find_zero_crossing(pooled, grid):.3f}',
                 'ci_lower': '', 'ci_upper': '', 'sd': '', 'se': '', 'n_sites': n_total,
                 'mode': 'Single curve'})

    for igbp in IGBPS:
        keep = [i for i, s in enumerate(sites) if igbp_of.get(s) == igbp]
        if not keep:
            continue
        mean, lo, hi, sd, se, n_cross, _ = group_threshold(matrix[keep], grid)
        rows.append({'feature': FEATURE, 'target': FLUX,
                     'method': 'Per-site zero-crossings, mean', 'group': igbp,
                     'threshold': f'{mean:.3f}', 'ci_lower': f'{lo:.3f}',
                     'ci_upper': f'{hi:.3f}', 'sd': f'{sd:.3f}', 'se': f'{se:.3f}',
                     'n_sites': n_cross, 'mode': 'Per-site curves'})

    out = pd.DataFrame(rows)
    thresholds_file = dir_out / f'46_ALE_Thresholds_{FEATURE}_{FLUX}.csv'
    out.to_csv(thresholds_file, index=False)

    print()
    print(out.to_string(index=False))
    print(f"\nSaved {curves_file}")
    print(f"Saved {thresholds_file}")


if __name__ == '__main__':
    main()
