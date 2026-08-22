"""
How much does the VPD threshold depend on how it is estimated?

Reviewer 2 questioned the fourth-order polynomial and asked for a non-polynomial
estimate plus sensitivity to bin width, fitting range and smoothing strength. The
threshold is a zero crossing of a fitted curve, so the fit is not a detail.

Reads the Figure 4 plot data written by script 54 and re-estimates the crossing
under other choices. Everything here is the same data, only the estimator changes.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.paths import load_settings

VARIANT = ""
SITE_SUBSET = ""
FLUX = 'NEP_ZSCORE'
DATA_FILE = ("54_FIG-4_ResponseCurve_ShapMeans_NEP_ZSCORE_BIN_VPD_ZSCORE+"
             "VPD_ZSCORE_SHAPVALS+TA_ZSCORE_ALLSITES_DATA.csv")


def highest_crossing(x: np.ndarray, y: np.ndarray) -> float:
    """Highest x where the curve crosses zero, matching src.fit.calc_threshold."""
    sign_changes = np.where(np.diff(np.sign(y)))[0]
    if len(sign_changes) == 0:
        return np.nan
    return max(x[i] - y[i] * (x[i + 1] - x[i]) / (y[i + 1] - y[i]) for i in sign_changes)


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


def main():
    settings = load_settings()
    d = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    df = pd.read_csv(d / DATA_FILE)
    x, y = df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)
    print(f"{len(x)} cells over {len(np.unique(x))} VPD bins, x from {x.min():.1f} to {x.max():.1f}\n")

    rows = []

    def record(family, setting, value, note=''):
        rows.append({'family': family, 'setting': setting, 'threshold_z': value, 'note': note})
        shown = f"{value:.3f}" if np.isfinite(value) else "no crossing"
        print(f"  {family:<22}{setting:<22}{shown:>12}  {note}")

    print("polynomial degree")
    for deg in (2, 3, 4, 5, 6):
        record('polynomial', f"degree {deg}", poly_threshold(x, y, deg),
               'published choice' if deg == 4 else '')

    print("\nlocal smoother, no polynomial")
    for frac in (0.2, 0.3, 0.4, 0.5, 0.7):
        record('lowess', f"frac {frac}", lowess_threshold(x, y, frac))

    print("\nsmoothing spline")
    for sf in (0.001, 0.01, 0.05, 0.1):
        record('spline', f"s factor {sf}", spline_threshold(x, y, sf))

    print("\nbin width, fourth-order polynomial")
    for width in (0.1, 0.2, 0.3, 0.5):
        xb, yb = rebin(x, y, width)
        record('bin width', f"{width} sigma", poly_threshold(xb, yb, 4), f"{len(xb)} bins")

    print("\nfitting range, fourth-order polynomial")
    for limit in (2.0, 2.5, 3.0, 3.5, None):
        keep = np.ones_like(x, dtype=bool) if limit is None else np.abs(x) <= limit
        label = 'full range' if limit is None else f"|x| <= {limit}"
        record('fitting range', label, poly_threshold(x[keep], y[keep], 4), f"{keep.sum()} cells")

    out = pd.DataFrame(rows)
    finite = out['threshold_z'].dropna()
    print(f"\nacross all {len(finite)} estimates that produced a crossing:")
    print(f"  median {finite.median():.3f}, range {finite.min():.3f} to {finite.max():.3f}")
    print(f"  published fourth-order polynomial: "
          f"{out.loc[(out.family == 'polynomial') & (out.setting == 'degree 4'), 'threshold_z'].iloc[0]:.3f}")

    outfile = d / "60_Threshold_MethodSensitivity.csv"
    out.to_csv(outfile, index=False)
    print(f"\nSaved {outfile}")


if __name__ == '__main__':
    main()
