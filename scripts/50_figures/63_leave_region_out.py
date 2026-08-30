"""
Do the headline numbers survive dropping a continent?

Reviewer 2 asked for leave-Europe-out and leave-North-America-out re-aggregation.
Between them the two regions hold about three quarters of the sites, so a result
that depends on them is a result about them rather than about forests in general.

Recomputes two things per region set: the stage 8 attribution of VPD against soil
water, and the median per-site VPD threshold. Both read files that already exist, so
nothing is refitted.

Region comes from the country prefix of the FLUXNET site code.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import load_settings

VARIANT = ""
SITE_SUBSET = ""
FLUX = 'NEP_ZSCORE'
# CD-Ygb records VPD in Pa where every other site uses hPa, see script 58. The factor is
# exactly 100: dividing gives a site mean of 18.10 hPa and a standard deviation of
# 6.56 hPa, both inside the 4.5 to 31.8 hPa range spanned by the other sites. The
# per-site z-scores are unaffected by the unit, so only the mapping to kPa changes and
# the site is converted rather than dropped, which keeps all 208 sites.
PA_UNIT_SITES = ['CD-Ygb']
PA_TO_HPA = 100

REGIONS = {
    'Europe': ['AT', 'BE', 'CH', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI', 'FR', 'GB', 'GR',
               'IE', 'IT', 'NL', 'PL', 'PT', 'RU', 'SE', 'SJ', 'SK', 'UK'],
    'North America': ['US', 'CA', 'MX', 'PR', 'CR', 'GL'],
    'Asia': ['CN', 'JP', 'KR', 'MY', 'ID', 'IN', 'TH', 'VN', 'PH', 'TW', 'IL', 'SA', 'KZ', 'GF'],
    'Oceania': ['AU', 'NZ', 'VU', 'PG'],
    'South America': ['BR', 'AR', 'CL', 'CO', 'PE', 'EC', 'GY', 'SR', 'UY', 'PA'],
    'Africa': ['ZA', 'ZM', 'CD', 'SN', 'SD', 'MA', 'TN', 'CG', 'CI', 'ET', 'GA', 'BJ', 'GH'],
}
LOOKUP = {code: region for region, codes in REGIONS.items() for code in codes}


def main():
    settings = load_settings()
    plots = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET

    sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                        / "21_SUBSETS_parquet_vars_stats_subsets.csv")
    stages = pd.read_parquet(agg / f"44_SHAPVALUES-conditional_AggregatedAcrossScenarios_{FLUX}.parquet")
    thresholds = pd.read_csv(plots / "59_SiteThresholds.csv")

    sites.loc[sites['SITE'].isin(PA_UNIT_SITES), ['VPD_Z0', 'VPD_SD']] /= PA_TO_HPA

    sites['region'] = sites['SITE'].str.split('-').str[0].map(LOOKUP)
    if sites['region'].isna().any():
        missing = sorted(sites.loc[sites['region'].isna(), 'SITE'].str.split('-').str[0].unique())
        raise SystemExit(f"Unmapped country codes: {missing}. Add them to REGIONS.")

    d = thresholds.merge(sites[['SITE', 'region', 'VPD_Z0', 'VPD_SD']], on='SITE')
    d['threshold_kpa'] = (d['VPD_Z0'] + d['threshold_z'] * d['VPD_SD']) / 10

    st8 = stages[stages['SCENARIO'] == 8].merge(sites[['SITE', 'region']], on='SITE')
    V, W = 'VPD_ZSCORE_SHAPVALS_OVR_AVG', 'SWC_ZSCORE_SHAPVALS_OVR_AVG'

    print(sites.groupby('region').size().sort_values(ascending=False).to_string())

    sets = [('all sites', []), ('minus Europe', ['Europe']),
            ('minus North America', ['North America']),
            ('minus both', ['Europe', 'North America'])]

    rows = []
    print(f"\n{'set':<22}{'sites':>7}{'stage 8 sites':>15}{'VPD/SM ratio':>14}"
          f"{'threshold sigma':>17}{'threshold kPa':>15}")
    for label, drop in sets:
        keep_thr = d[~d['region'].isin(drop)]
        keep_st8 = st8[~st8['region'].isin(drop)].dropna(subset=[V, W])
        ratio = keep_st8[V].mean() / keep_st8[W].mean()
        row = {'set': label, 'sites': len(keep_thr), 'stage8_sites': len(keep_st8),
               'ratio': ratio, 'threshold_sigma': keep_thr['threshold_z'].median(),
               'threshold_kpa': keep_thr['threshold_kpa'].median()}
        rows.append(row)
        print(f"{label:<22}{row['sites']:>7}{row['stage8_sites']:>15}{ratio:>14.2f}"
              f"{row['threshold_sigma']:>17.3f}{row['threshold_kpa']:>15.2f}")

    out = pd.DataFrame(rows)
    base = out.iloc[0]
    worst_ratio = (out['ratio'] - base['ratio']).abs().max()
    worst_thr = (out['threshold_kpa'] - base['threshold_kpa']).abs().max()
    print(f"\nLargest change against the full network: ratio {worst_ratio:.2f}, "
          f"threshold {worst_thr:.2f} kPa.")

    outfile = plots / "63_LeaveRegionOut.csv"
    out.to_csv(outfile, index=False)
    print(f"Saved {outfile}")


if __name__ == '__main__':
    main()
