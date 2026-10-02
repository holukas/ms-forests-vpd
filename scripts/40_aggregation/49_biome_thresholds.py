"""
Test whether the VPD threshold differs between forest types.

One threshold per site, from the stage 41 TA by VPD file: VPD SHAP averaged per VPD bin,
a fourth-order polynomial fitted, and the highest crossing from positive to negative
taken, as in `src.fit.calc_threshold`. Sites with fewer than MIN_BINS filled bins are skipped. The
per-site values feed a Kruskal-Wallis test across forest types, pairwise Mann-Whitney
tests with Holm correction, and a bootstrap interval for each forest type median. Rank
tests are used because the per-site thresholds are skewed. The Figure 4 intervals cannot
answer this: they describe fit uncertainty, not the spread between sites.

Writes 49_SiteThresholds.csv, 49_BiomeThresholds_Summary.csv and
49_BiomeThresholds_PairwiseTests.csv.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.paths import load_settings

VARIANT = ""
SITE_SUBSET = ""
FLUX = 'NEP_ZSCORE'
BIOMES = ['ENF', 'DBF', 'EBF', 'MF']
MIN_BINS = 15          # a site needs enough filled bins for a fourth-order fit to mean anything
N_BOOTSTRAP = 10000


def site_threshold(binned: pd.DataFrame) -> float:
    """Highest positive to negative crossing of the fitted curve for one site, in sigma."""
    curve = binned.groupby('BIN_VPD_ZSCORE')['VPD_ZSCORE_SHAPVALS'].mean()
    if len(curve) < MIN_BINS:
        return np.nan
    x, y = curve.index.to_numpy(float), curve.to_numpy(float)
    xf = np.linspace(x.min(), x.max(), 500)
    yf = np.polyval(np.polyfit(x, y, 4), xf)
    crossings = np.where((yf[:-1] > 0) & (yf[1:] <= 0))[0]
    if len(crossings) == 0:
        return np.nan
    return max(xf[i] - yf[i] * (xf[i + 1] - xf[i]) / (yf[i + 1] - yf[i]) for i in crossings)


def holm(pvalues: list) -> np.ndarray:
    """Holm step-down correction, monotonic and capped at 1."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(1.0, p[order] * (len(p) - np.arange(len(p))))
    adjusted = np.maximum.accumulate(adjusted)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def main():
    settings = load_settings()
    agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    infile = next(agg.glob("41_*BIN-TA_ZSCORE+BIN-VPD_ZSCORE*.parquet"))
    binned = pd.read_parquet(infile, columns=['BIN_VPD_ZSCORE', 'VPD_ZSCORE_SHAPVALS', 'SITE', 'IGBP']).dropna()

    rows = [{'SITE': site, 'IGBP': igbp, 'threshold_z': site_threshold(g)}
            for (site, igbp), g in binned.groupby(['SITE', 'IGBP'])]
    t = pd.DataFrame(rows).dropna(subset=['threshold_z'])
    print(f"per-site thresholds from {infile.name}")
    print(f"{len(t)} sites with a zero crossing\n")

    data = {b: t.loc[t['IGBP'] == b, 'threshold_z'].to_numpy() for b in BIOMES}
    rng = np.random.default_rng(42)

    print(f"{'biome':<6}{'n':>5}{'median':>9}{'IQR':>20}{'bootstrap 95 % CI':>22}")
    summary = []
    for b in BIOMES:
        v = data[b]
        q1, q3 = np.percentile(v, [25, 75])
        boot = np.median(v[rng.integers(0, len(v), size=(N_BOOTSTRAP, len(v)))], axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"{b:<6}{len(v):>5}{np.median(v):>9.3f}   [{q1:>6.3f},{q3:>6.3f}]      [{lo:>6.3f},{hi:>6.3f}]")
        summary.append({'IGBP': b, 'n': len(v), 'median': np.median(v), 'iqr_lower': q1,
                        'iqr_upper': q3, 'ci_lower': lo, 'ci_upper': hi})

    h, p = stats.kruskal(*[data[b] for b in BIOMES])
    print(f"\nKruskal-Wallis across four biomes: H = {h:.2f}, p = {p:.4f}")

    pairs = list(combinations(BIOMES, 2))
    raw = [stats.mannwhitneyu(data[a], data[b], alternative='two-sided').pvalue for a, b in pairs]
    corrected = holm(raw)
    print("\npairwise Mann-Whitney, Holm-corrected:")
    pair_rows = []
    for (a, b), r, c in zip(pairs, raw, corrected):
        verdict = 'significant' if c < 0.05 else 'not significant'
        print(f"  {a} vs {b:<4} raw p = {r:.4f}  corrected p = {c:.4f}  {verdict}")
        pair_rows.append({'pair': f"{a} vs {b}", 'p_raw': r, 'p_holm': c, 'significant': c < 0.05})

    spread = max(s['median'] for s in summary) - min(s['median'] for s in summary)
    widest = max(s['iqr_upper'] - s['iqr_lower'] for s in summary)
    print(f"\nrange of biome medians: {spread:.3f} sigma")
    print(f"widest within-biome IQR: {widest:.3f} sigma")
    print("Within-biome spread exceeds the between-biome spread."
          if widest > spread else "Between-biome spread exceeds the within-biome spread.")

    dir_out = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    dir_out.mkdir(parents=True, exist_ok=True)
    t.to_csv(dir_out / "49_SiteThresholds.csv", index=False)
    pd.DataFrame(summary).to_csv(dir_out / "49_BiomeThresholds_Summary.csv", index=False)
    pd.DataFrame(pair_rows).to_csv(dir_out / "49_BiomeThresholds_PairwiseTests.csv", index=False)
    print(f"\nSaved three tables to {dir_out}")


if __name__ == '__main__':
    main()
