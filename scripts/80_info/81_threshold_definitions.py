"""
Report four distinct VPD points on the fitted response curve that a single threshold could refer to.

Uses the quartic fit and data of Figure 4 (script 54), for all sites and per forest type:

- peak: where the VPD effect on NEP is most positive
- steepest decline: where the effect falls fastest after the peak
- zero crossing: where the effect changes sign, the threshold of the main analysis
- minimum: where the effect is most negative within the fitted range

Each point is given in sigma and in kPa, converted as in script 54. Points within 2 % of
the edge of the observed range are flagged as marking where the data stop.

Reads: the 54_FIG-4_ResponseCurve_..._DATA.csv files of script 54.
Writes: 81_INFO_ThresholdDefinitions.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import load_settings

VARIANT = ""
SITE_SUBSET = ""
FLUX = 'NEP_ZSCORE'
DEGREE = 4
BIOMES = ['ENF', 'DBF', 'EBF', 'MF']
# CD-Ygb records VPD in Pa where every other site uses hPa. The factor is exactly 100:
# dividing gives a site mean of 18.10 hPa and a standard deviation of 6.56 hPa, both inside
# the 4.5 to 31.8 hPa range spanned by the other sites. The per-site z-scores are unaffected
# by the unit, so only this mapping back to kPa changes and the site is converted rather
# than dropped.
PA_UNIT_SITES = ['CD-Ygb']
PA_TO_HPA = 100

BASE = ("54_FIG-4_ResponseCurve_ShapMeans_NEP_ZSCORE_BIN_VPD_ZSCORE+"
        "VPD_ZSCORE_SHAPVALS+TA_ZSCORE")


def curve_points(x: np.ndarray, y: np.ndarray) -> dict:
    """Peak, steepest decline, zero crossing and minimum of the fitted curve."""
    coeffs = np.polyfit(x, y, DEGREE)
    xf = np.linspace(x.min(), x.max(), 2000)
    yf = np.polyval(coeffs, xf)
    slope = np.polyval(np.polyder(coeffs), xf)

    crossings = np.where((yf[:-1] > 0) & (yf[1:] <= 0))[0]
    zero = (max(xf[i] - yf[i] * (xf[i + 1] - xf[i]) / (yf[i + 1] - yf[i]) for i in crossings)
            if len(crossings) else np.nan)

    peak = xf[int(np.argmax(yf))]
    # steepest decline after the peak, which is where the fall is fastest
    after = xf >= peak
    steepest = xf[after][int(np.argmin(slope[after]))] if after.any() else np.nan
    minimum = xf[int(np.argmin(yf))]

    # A point sitting on the edge of the observed range is not a feature of the
    # response, it is where the data stopped. The quartic keeps steepening at the
    # right edge for most groups, so both the steepest decline and the minimum can
    # land there and must not be read as "the effect bottoms out here".
    edge = 0.02 * (x.max() - x.min())
    at_edge = {name: bool(abs(v - x.max()) < edge or abs(v - x.min()) < edge)
               for name, v in (('peak', peak), ('steepest_decline', steepest), ('minimum', minimum))}
    return {'peak': peak, 'steepest_decline': steepest, 'zero_crossing': zero,
            'minimum': minimum, 'at_edge': at_edge, 'x_max': x.max()}


def main():
    settings = load_settings()
    d = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    info = Path(settings['DIR_INFO_OUT']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    info.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                        / "21_SUBSETS_parquet_vars_stats_subsets.csv")
    sites.loc[sites['SITE'].isin(PA_UNIT_SITES), ['VPD_Z0', 'VPD_SD']] /= PA_TO_HPA

    def to_kpa(z, group):
        sub = sites if group == 'ALL SITES' else sites[sites['IGBP'] == group]
        return np.nan if not np.isfinite(z) else ((sub['VPD_Z0'] + z * sub['VPD_SD']) / 10).mean()

    rows = []
    for group, suffix in [('ALL SITES', 'ALLSITES_DATA.csv')] + [(b, f"{b}_DATA.csv") for b in BIOMES]:
        f = d / f"{BASE}_{suffix}"
        if not f.is_file():
            print(f"missing {f.name}, skipped")
            continue
        df = pd.read_csv(f)
        pts = curve_points(df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float))
        row = {'group': group, 'x_max': pts['x_max']}
        for name in ('peak', 'steepest_decline', 'zero_crossing', 'minimum'):
            row[f"{name}_z"] = pts[name]
            row[f"{name}_kPa"] = to_kpa(pts[name], group)
            row[f"{name}_at_edge"] = pts['at_edge'].get(name, False)
        rows.append(row)

    out = pd.DataFrame(rows)
    print("VPD anomaly, sigma")
    print(out[['group'] + [f"{k}_z" for k in ('peak', 'steepest_decline', 'zero_crossing', 'minimum')]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\nabsolute VPD, kPa")
    print(out[['group'] + [f"{k}_kPa" for k in ('peak', 'steepest_decline', 'zero_crossing', 'minimum')]]
          .to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    edge_cols = [c for c in out.columns if c.endswith('_at_edge')]
    if out[edge_cols].to_numpy().any():
        print("")
        print("Points sitting on the edge of the observed range, so not interior features:")
        for _, r in out.iterrows():
            hits = [c.replace('_at_edge', '') for c in edge_cols if r[c]]
            if hits:
                print(f"  {r['group']:<10}{', '.join(hits)}  (range ends at {r['x_max']:.2f} sigma)")
        print("  The fitted curve is still falling where the data stop, so the effect does")
        print("  not bottom out within the observed range.")

    all_sites = out[out['group'] == 'ALL SITES'].iloc[0]
    print(f"\nAcross all sites the three points on the curve are "
          f"{all_sites['peak_kPa']:.2f}, {all_sites['steepest_decline_kPa']:.2f} and "
          f"{all_sites['zero_crossing_kPa']:.2f} kPa.")
    print("Only the third is the threshold of the main analysis.")

    outfile = info / "81_INFO_ThresholdDefinitions.csv"
    out.to_csv(outfile, index=False)
    print(f"\nSaved {outfile}")


if __name__ == '__main__':
    main()
