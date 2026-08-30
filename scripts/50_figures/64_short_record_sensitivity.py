"""
Do the headline numbers depend on the sites with short records?

Reviewer 2 noticed that some sites keep fewer than three years even though a
three-year minimum is stated. That is expected: the minimum is applied before the
QC, daytime, peak-month and balancing filters, which then cut into the record. What
the manuscript does not show is whether those sites matter.

Recomputes the threshold and the stage 8 attribution while requiring progressively
longer retained records. Reads existing files, nothing is refitted.
"""
from pathlib import Path

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
MINIMA = [1, 2, 3, 5, 10]


def main():
    settings = load_settings()
    plots = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET

    sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                        / "21_SUBSETS_parquet_vars_stats_subsets.csv")
    stages = pd.read_parquet(agg / f"44_SHAPVALUES-conditional_AggregatedAcrossScenarios_{FLUX}.parquet")
    thresholds = pd.read_csv(plots / "59_SiteThresholds.csv")

    sites.loc[sites['SITE'].isin(PA_UNIT_SITES), ['VPD_Z0', 'VPD_SD']] /= PA_TO_HPA

    print("retained years per site")
    print(f"  median {sites['N_YEARS'].median():.0f}, quartiles "
          f"{sites['N_YEARS'].quantile(0.25):.0f} and {sites['N_YEARS'].quantile(0.75):.0f}, "
          f"range {sites['N_YEARS'].min():.0f} to {sites['N_YEARS'].max():.0f}")
    for n in (2, 3):
        below = int((sites['N_YEARS'] < n).sum())
        print(f"  fewer than {n} years: {below} sites, {100 * below / len(sites):.1f} %")

    d = thresholds.merge(sites[['SITE', 'N_YEARS', 'N_RECORDS', 'VPD_Z0', 'VPD_SD']], on='SITE')
    d['threshold_kpa'] = (d['VPD_Z0'] + d['threshold_z'] * d['VPD_SD']) / 10
    st8 = stages[stages['SCENARIO'] == 8].merge(sites[['SITE', 'N_YEARS']], on='SITE')
    V, W = 'VPD_ZSCORE_SHAPVALS_OVR_AVG', 'SWC_ZSCORE_SHAPVALS_OVR_AVG'

    rows = []
    print(f"\n{'minimum years':>14}{'sites':>7}{'records kept':>14}{'threshold sigma':>17}"
          f"{'threshold kPa':>15}{'VPD/SM ratio':>14}")
    total_records = sites['N_RECORDS'].sum()
    for minimum in MINIMA:
        keep = d[d['N_YEARS'] >= minimum]
        keep8 = st8[st8['N_YEARS'] >= minimum].dropna(subset=[V, W])
        share = 100 * sites.loc[sites['N_YEARS'] >= minimum, 'N_RECORDS'].sum() / total_records
        row = {'min_years': minimum, 'sites': len(keep), 'records_kept_pct': share,
               'threshold_sigma': keep['threshold_z'].median(),
               'threshold_kpa': keep['threshold_kpa'].median(),
               'ratio': keep8[V].mean() / keep8[W].mean()}
        rows.append(row)
        print(f"{minimum:>14}{row['sites']:>7}{share:>13.1f}%{row['threshold_sigma']:>17.3f}"
              f"{row['threshold_kpa']:>15.2f}{row['ratio']:>14.2f}")

    out = pd.DataFrame(rows)
    base = out.iloc[0]
    print(f"\nAgainst keeping every site, the largest change is "
          f"{(out['threshold_kpa'] - base['threshold_kpa']).abs().max():.2f} kPa in the threshold.")
    print(f"Requiring three years drops {int(base['sites'] - out.loc[out.min_years == 3, 'sites'].iloc[0])} "
          f"sites and {base['records_kept_pct'] - out.loc[out.min_years == 3, 'records_kept_pct'].iloc[0]:.1f} "
          f"per cent of the records.")

    outfile = plots / "64_ShortRecordSensitivity.csv"
    out.to_csv(outfile, index=False)
    print(f"Saved {outfile}")


if __name__ == '__main__':
    main()
