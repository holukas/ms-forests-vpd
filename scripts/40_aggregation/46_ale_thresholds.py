"""
ALE curves on a common grid, and the thresholds that come out of them.

The heavy half of what used to be the ALE figure script. Reading 208 per-site ALE files is
aggregation, not drawing, so it belongs at this stage. Script 57 reads what this writes
and draws the five-panel figure, and script 47 reads the threshold
table for the ALE row of the robustness analysis. Before this split, a stage 40 script had
to read a file that a figure script wrote, which is backwards.

**What a threshold is here.** Each site's ALE curve is interpolated to one grid, then
smoothed with a fourth-order polynomial, and the threshold is the first positive to negative
crossing of that smoothed curve. The number reported for a group is the mean of the per-site
crossings, with a 95 % interval from the t distribution. Sites whose curve never crosses are
left out of the mean, and the count of those that do cross is written next to it.

**The all-sites row is the same kind of number as the forest type rows**, and the file it
replaces said otherwise. Script 70 labelled it "Direct zero-crossing" and "Single curve",
but the value it wrote, 0.181, is the mean of the per-site crossings over the 207 sites whose
curve crosses. The crossing of the pooled curve is 0.107, a different number that was never
reported. Both are written here, the mean as `ALL SITES` and the pooled value as
`ALL SITES (pooled curve)`, so the two can never be confused again. The mean also gets a
95 % interval, which the old file left empty for no reason other than the wrong label.

**Coverage mask.** A grid point is only kept where at least half the sites in the group have
data, the same rule the SHAP route uses. Without it the ends of the curve rest on a handful
of sites.

Reads the per-site output of stage 32, `{site}_ale_{FLUX}.parquet` for the record check and
`{site}_ale_curves_{FLUX}.csv` for the curves, plus the stage 21 subsets table for the site
list and the forest type.

Writes, into the aggregation folder:
    46_ALE_SiteCurves_{FEATURE}_{FLUX}.parquet   one interpolated curve per site, plus IGBP
    46_ALE_Thresholds_{FEATURE}_{FLUX}.csv       all sites, then one row per forest type,
                                                 each with the spread and the interval
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

USE_MEDIAN = False       # False: mean across sites, the published choice
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
    """First positive to negative crossing, by linear interpolation. NaN if there is none."""
    valid = ~np.isnan(curve)
    if np.sum(valid) <= 1:
        return np.nan
    curve_valid = curve[valid]
    grid_valid = grid[valid]
    if np.all(curve_valid >= 0) or np.all(curve_valid <= 0):
        return np.nan
    sign_changes = np.diff(np.sign(curve_valid))
    for idx in np.where(sign_changes != 0)[0]:
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

    The parquet is read first, as in the original script, because a site only counts if it
    has more than 10 records for this feature and flux. The curve itself comes from the csv
    that stage 32 writes beside it.
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
    """Mean of the per-site crossings for one group, with its spread and its interval.

    Three numbers, because they answer different questions and get confused for each other.
    The standard deviation says how much sites differ. The standard error and the interval
    say how well the mean is determined, and they are smaller by the square root of the site
    count. For all sites that is 0.45 sigma against 0.03, a factor of fourteen.
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

    # All sites, the published value: the mean of the per-site crossings. Script 70 wrote
    # this number under a label saying it came from one pooled curve, which it does not.
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
