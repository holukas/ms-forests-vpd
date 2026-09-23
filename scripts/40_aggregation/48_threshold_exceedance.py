"""
How often peak-season daytime VPD exceeds the threshold, per site and per forest type.

Counts stage 21 subset half-hours; no model or SHAP value is involved. Runs after
scripts 47 and 49. Four measures, each the share of half-hours with VPD above:
- the script 47 reference threshold in kPa
- the same threshold in sigma, the units the models saw
- the site's own crossing from script 49, in the site's sigma
- the median crossing of the site's forest type from script 49, in the site's sigma

The forest type table gives the record-weighted share and the site median. They differ,
so the text has to say which one it quotes. CD-Ygb VPD is in Pa and is divided by 100.

Reads the stage 21 subsets, 47_THRESHOLD_Robustness_{FLUX}.csv, 49_SiteThresholds.csv
and 49_BiomeThresholds_Summary.csv. Writes 48_EXCEEDANCE_PerSite_{FLUX}.csv and
48_EXCEEDANCE_PerBiome_{FLUX}.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""
BIOMES = ['ENF', 'DBF', 'EBF', 'MF']

# CD-Ygb records VPD in Pa where every other site uses hPa. Converted rather than dropped,
# which keeps all 208 sites, see stage 47.
PA_UNIT_SITES = ['CD-Ygb']
PA_TO_HPA = 100

SUBSET_FILE = "{site}_subset_GPPhighest4_qc0_daytime.parquet"
HPA_PER_KPA = 10


def site_row(filepath: Path, site: str, igbp: str, threshold_kpa: float,
             threshold_z: float, own_z: float, biome_z: float) -> dict:
    """Exceedance counts for one site, all four measures."""
    d = pd.read_parquet(filepath, columns=['VPD', 'VPD_ZSCORE']).dropna()
    vpd = d['VPD'] / PA_TO_HPA if site in PA_UNIT_SITES else d['VPD']
    vpd_kpa = vpd / HPA_PER_KPA

    above = vpd_kpa > threshold_kpa
    excess = vpd_kpa[above] - threshold_kpa
    return {
        'SITE': site,
        'IGBP': igbp,
        'n_records': len(d),
        'n_above': int(above.sum()),
        'vpd_mean_kpa': float(vpd_kpa.mean()),
        'vpd_max_kpa': float(vpd_kpa.max()),
        'above_published_kpa_pct': float(100 * above.mean()),
        'above_published_sigma_pct': float(100 * (d['VPD_ZSCORE'] > threshold_z).mean()),
        'above_own_threshold_pct': float(100 * (d['VPD_ZSCORE'] > own_z).mean())
        if np.isfinite(own_z) else np.nan,
        'own_threshold_z': own_z,
        'above_biome_threshold_pct': float(100 * (d['VPD_ZSCORE'] > biome_z).mean())
        if np.isfinite(biome_z) else np.nan,
        'biome_threshold_z': biome_z,
        # How far past the threshold the air goes when it is past it. A high frequency with a
        # small excess is a different climate from a low frequency with a large one.
        'mean_excess_kpa': float(excess.mean()) if above.any() else 0.0,
    }


def main():
    settings = load_settings()
    shap_type = 'conditional' if CONDITIONAL else 'interventional'
    subsets_base = Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
    agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET
    agg.mkdir(parents=True, exist_ok=True)

    sites = pd.read_csv(subsets_base / "21_SUBSETS_parquet_vars_stats_subsets.csv")

    # The threshold is read, not typed in, so this script cannot drift from Table 1.
    robustness = pd.read_csv(agg / f"47_THRESHOLD_Robustness_{FLUX}.csv")
    ref = robustness.loc[robustness['test'] == 'PUBLISHED REFERENCE'].iloc[0]
    threshold_kpa, threshold_z = float(ref['threshold_kpa']), float(ref['threshold_sigma'])
    own = pd.read_csv(agg / "49_SiteThresholds.csv").set_index('SITE')['threshold_z']
    # One crossing per forest type, the median of the sites in it, from script 49.
    biome_thresholds = pd.read_csv(agg / "49_BiomeThresholds_Summary.csv") \
        .set_index('IGBP')['median']

    print(f"threshold of the main analysis: {threshold_kpa:.3f} kPa, {threshold_z:.3f} sigma, "
          f"{int(ref['n_sites'])} sites")
    print(f"per-site crossings available for {len(own)} sites\n")

    rows = []
    for ix, (site, igbp) in enumerate(zip(sites['SITE'], sites['IGBP'])):
        filepath = subsets_base / '21_subsets_parquet' / SUBSET_FILE.format(site=site)
        if not filepath.is_file():
            print(f"  no subset file for {site}, skipping")
            continue
        rows.append(site_row(filepath, site, igbp, threshold_kpa, threshold_z,
                             float(own.get(site, np.nan)),
                             float(biome_thresholds.get(igbp, np.nan))))
        if (ix + 1) % 25 == 0:
            print(f"  {ix + 1} sites")

    per_site = pd.DataFrame(rows)
    per_site_file = agg / f"48_EXCEEDANCE_PerSite_{FLUX}.csv"
    per_site.to_csv(per_site_file, index=False)

    n_records = int(per_site['n_records'].sum())
    pooled = 100 * per_site['n_above'].sum() / n_records
    q25, q50, q75 = per_site['above_published_kpa_pct'].quantile([0.25, 0.5, 0.75])
    print(f"\n{len(per_site)} sites, {n_records:,} half-hours")
    print(f"above {threshold_kpa:.2f} kPa: {pooled:.1f} % of all half-hours, "
          f"site median {q50:.1f} %, IQR {q25:.1f} to {q75:.1f} %")
    print(f"  site range {per_site['above_published_kpa_pct'].min():.1f} to "
          f"{per_site['above_published_kpa_pct'].max():.1f} %, "
          f"{int((per_site['n_above'] == 0).sum())} sites never above it")
    print(f"  when above, the air sits {per_site['mean_excess_kpa'].median():.2f} kPa past it "
          f"at the median site")

    # The same threshold in the units the models saw. A near-constant fraction here says the
    # spread above comes from the site climates rather than from the threshold.
    z25, z50, z75 = per_site['above_published_sigma_pct'].quantile([0.25, 0.5, 0.75])
    print(f"above {threshold_z:.2f} sigma of the site's own VPD: median {z50:.1f} %, "
          f"IQR {z25:.1f} to {z75:.1f} %")
    o50 = per_site['above_own_threshold_pct'].median()
    print(f"above the site's own crossing: median {o50:.1f} %, "
          f"{int(per_site['above_own_threshold_pct'].notna().sum())} sites with a crossing")
    b50 = per_site['above_biome_threshold_pct'].median()
    print(f"above the forest type's crossing: median {b50:.1f} %, "
          f"thresholds " + ", ".join(f"{b} {biome_thresholds[b]:.2f}" for b in BIOMES) + " sigma")

    biome_rows = []
    for biome_name in BIOMES + ['all']:
        g = per_site if biome_name == 'all' else per_site.loc[per_site['IGBP'] == biome_name]
        biome_rows.append({
            'IGBP': biome_name,
            'n_sites': len(g),
            'n_records': int(g['n_records'].sum()),
            'pooled_pct': 100 * g['n_above'].sum() / g['n_records'].sum(),
            'site_median_pct': g['above_published_kpa_pct'].median(),
            'site_min_pct': g['above_published_kpa_pct'].min(),
            'site_max_pct': g['above_published_kpa_pct'].max(),
            'sigma_median_pct': g['above_published_sigma_pct'].median(),
            'own_threshold_median_pct': g['above_own_threshold_pct'].median(),
            'biome_threshold_z': float(biome_thresholds[biome_name])
            if biome_name != 'all' else np.nan,
            'biome_threshold_median_pct': g['above_biome_threshold_pct'].median(),
            'mean_excess_kpa': g['mean_excess_kpa'].median(),
        })
    per_biome = pd.DataFrame(biome_rows)
    per_biome_file = agg / f"48_EXCEEDANCE_PerBiome_{FLUX}.csv"
    per_biome.to_csv(per_biome_file, index=False)

    print(f"\n{'biome':<6}{'sites':>7}{'records':>12}{'pooled':>9}{'site median':>13}"
          f"{'site range':>18}")
    for r in biome_rows:
        site_range = f"{r['site_min_pct']:.1f} to {r['site_max_pct']:.1f}%"
        print(f"{r['IGBP']:<6}{r['n_sites']:>7}{r['n_records']:>12,}{r['pooled_pct']:>8.1f}%"
              f"{r['site_median_pct']:>12.1f}%{site_range:>18}")

    print(f"\nSaved {per_site_file}")
    print(f"Saved {per_biome_file}")


if __name__ == '__main__':
    main()
