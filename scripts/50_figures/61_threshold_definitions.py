"""
Report the distinct VPD points that a single "threshold" currently conflates.

Reviewer 2 pointed out that the VPD at maximum predicted NEP, the onset of decline
and the zero crossing of the attribution are three different quantities, and that
the manuscript reports one number for them. They answer different questions:

  peak            where the VPD effect on NEP is most positive, so where added
                  atmospheric demand stops helping
  steepest        where the effect falls fastest, the strongest onset of decline
  zero crossing   where the effect changes sign, the published threshold
  minimum         where the effect is most negative, inside the fitted range

Same fitted curve as Figure 4, same data, four readings of it. Also converted to
kPa the way script 58 does, since the manuscript quotes absolute values.
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
PA_UNIT_SITES = ['CD-Ygb']          # records VPD in Pa, see script 58

BASE = ("54_FIG-4_ResponseCurve_ShapMeans_NEP_ZSCORE_BIN_VPD_ZSCORE+"
        "VPD_ZSCORE_SHAPVALS+TA_ZSCORE")


def curve_points(x: np.ndarray, y: np.ndarray) -> dict:
    """Peak, steepest decline, zero crossing and minimum of the fitted curve."""
    coeffs = np.polyfit(x, y, DEGREE)
    xf = np.linspace(x.min(), x.max(), 2000)
    yf = np.polyval(coeffs, xf)
    slope = np.polyval(np.polyder(coeffs), xf)

    crossings = np.where(np.diff(np.sign(yf)))[0]
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
    sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                        / "21_SUBSETS_parquet_vars_stats_subsets.csv")
    sites = sites[~sites['SITE'].isin(PA_UNIT_SITES)]

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
    print(f"\nAcross all sites the three points R2 distinguishes are "
          f"{all_sites['peak_kPa']:.2f}, {all_sites['steepest_decline_kPa']:.2f} and "
          f"{all_sites['zero_crossing_kPa']:.2f} kPa.")
    print("Only the third is the published threshold.")

    outfile = d / "61_Threshold_Definitions.csv"
    out.to_csv(outfile, index=False)
    print(f"\nSaved {outfile}")


if __name__ == '__main__':
    main()
