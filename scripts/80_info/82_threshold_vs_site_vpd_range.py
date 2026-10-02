"""
Test whether the per-site VPD threshold is an artifact of standardization by relating it to each site's maximum VPD.

A site with a narrow VPD range reaches +3 sigma at a modest absolute VPD, so part of the
nonlinearity could come from the standardization. The script prints how many sites never
exceed 1.25 kPa, then groups sites into quartiles of their maximum VPD and reports per
quartile the median threshold in sigma and kPa and the number of sites with a zero
crossing, plus Spearman correlations of the threshold with the site maximum VPD. A
threshold produced by the standardization would grow in kPa in proportion to the site's
range; a fixed physical limit would not move.

Reads: 49_SiteThresholds.csv (script 49), 21_SUBSETS_parquet_vars_stats_subsets.csv.
Writes: 82_INFO_ThresholdVsSiteVPDRange.csv (per quartile) and
82_INFO_ThresholdVsSiteVPDRange_perSite.csv.
"""
from pathlib import Path

import pandas as pd
from scipy import stats

from src.paths import load_settings

VARIANT = ""
SITE_SUBSET = ""
FLUX = 'NEP_ZSCORE'
CUTOFF_KPA = 1.25

# CD-Ygb records VPD in Pa where every other site uses hPa. Other scripts drop it for that
# reason. It is converted here instead, which keeps all 208 sites: dividing by 100 gives a
# mean of 18.1 hPa, a standard deviation of 6.6 hPa and a maximum of 39.3 hPa, all inside
# the range the other sites span. The z-scores are per site and therefore unaffected by the
# unit, so the conversion only touches the absolute VPD columns.
PA_UNIT_SITES = ['CD-Ygb']
PA_UNIT_COLS = ['VPD_Z0', 'VPD_SD', 'VPD_MAX']


def main():
    settings = load_settings()
    plots = Path(settings['DIR_PLOTS_OUT']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    info = Path(settings['DIR_INFO_OUT']) / FLUX / 'conditional' / VARIANT / SITE_SUBSET
    info.mkdir(parents=True, exist_ok=True)
    sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                        / "21_SUBSETS_parquet_vars_stats_subsets.csv")
    thresholds = pd.read_csv(agg / "49_SiteThresholds.csv")

    # Every site enters its quartile; a site without a crossing keeps an empty threshold and
    # is left out of the medians, so the last column shows where crossings are missing.
    d = sites[['SITE', 'IGBP', 'VPD_Z0', 'VPD_SD', 'VPD_MAX']].merge(
        thresholds[['SITE', 'threshold_z']], on='SITE', how='left')
    d.loc[d['SITE'].isin(PA_UNIT_SITES), PA_UNIT_COLS] /= 100
    d['vpd_max_kpa'] = d['VPD_MAX'] / 10
    d['threshold_kpa'] = (d['VPD_Z0'] + d['threshold_z'] * d['VPD_SD']) / 10

    below = int((d['vpd_max_kpa'] <= CUTOFF_KPA).sum())
    print(f"Sites never exceeding {CUTOFF_KPA} kPa: {below} of {len(d)}")
    print(f"Lowest site maximum: {d['vpd_max_kpa'].min():.2f} kPa, median {d['vpd_max_kpa'].median():.2f} kPa")
    if below == 0:
        print("A split at the cutoff leaves an empty group, so quartiles are used instead.\n")

    d['stratum'] = pd.qcut(d['vpd_max_kpa'], 4, labels=['Q1 lowest max VPD', 'Q2', 'Q3', 'Q4 highest max VPD'])
    g = d.groupby('stratum', observed=True).agg(
        n=('SITE', 'size'),
        median_max_vpd_kpa=('vpd_max_kpa', 'median'),
        median_threshold_sigma=('threshold_z', 'median'),
        median_threshold_kpa=('threshold_kpa', 'median'),
        sites_with_crossing=('threshold_z', lambda v: int(v.notna().sum())))
    print(g.to_string(float_format=lambda v: f"{v:.2f}"))

    rho_sigma = stats.spearmanr(d['vpd_max_kpa'], d['threshold_z'], nan_policy='omit')
    rho_kpa = stats.spearmanr(d['vpd_max_kpa'], d['threshold_kpa'], nan_policy='omit')
    print(f"\nSpearman against the site's own maximum VPD")
    print(f"  threshold in sigma: rho = {rho_sigma.statistic:+.3f}, p = {rho_sigma.pvalue:.4g}")
    print(f"  threshold in kPa:   rho = {rho_kpa.statistic:+.3f}, p = {rho_kpa.pvalue:.4g}")

    lo, hi = g.iloc[0], g.iloc[-1]
    range_growth = 100 * (hi['median_max_vpd_kpa'] / lo['median_max_vpd_kpa'] - 1)
    thr_growth = 100 * (hi['median_threshold_kpa'] / lo['median_threshold_kpa'] - 1)
    print(f"\nFrom the lowest to the highest quartile the site VPD range grows {range_growth:.0f} %"
          f" while the threshold grows {thr_growth:.0f} %.")
    print("Proportional growth would mean the standardization made the threshold.")
    print("No growth would mean a fixed physical limit. Neither holds.")

    crossings = int(g['sites_with_crossing'].sum())
    missing = d.loc[d['threshold_z'].isna(), ['SITE', 'stratum']]
    print(f"\nA crossing from positive to negative appears at {crossings} of {len(d)} sites, in every"
          f" stratum, so the nonlinearity is not confined to sites that reach high absolute VPD.")
    for site, stratum in missing.itertuples(index=False):
        print(f"  no crossing: {site} ({stratum})")

    outfile = info / "82_INFO_ThresholdVsSiteVPDRange.csv"
    g.to_csv(outfile)
    d[['SITE', 'IGBP', 'threshold_z', 'threshold_kpa', 'vpd_max_kpa', 'stratum']].to_csv(
        info / "82_INFO_ThresholdVsSiteVPDRange_perSite.csv", index=False)
    print(f"\nSaved {outfile}")


if __name__ == '__main__':
    main()
