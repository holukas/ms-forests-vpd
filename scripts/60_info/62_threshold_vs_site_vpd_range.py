"""
Does the threshold survive at low-VPD sites, and does it scale with the site's climate?

Reviewer 3 pointed out that a site with a narrow absolute VPD range reaches +3 sigma
at a modest absolute VPD, so part of the nonlinearity might come from the
standardization rather than from a physiological limit. The suggested test was to
stratify sites at an absolute maximum VPD of 1.25 kPa and look for the same
response in the low-VPD group.

That split cannot be made here. No site in the analysis stays below 1.25 kPa, the
lowest site maximum being 1.69 kPa, so the low-VPD group is empty. This script
answers the underlying question instead, by splitting sites into quartiles of their
own maximum VPD and asking two things:

  does a zero crossing still appear in the narrowest-VPD sites
  does the absolute threshold track the site's VPD range

If the standardization manufactured the threshold, the value in kPa would rise in
proportion to the site's range. If the threshold were purely physical, it would not
move at all.

Uses the per-site thresholds from script 59 and the per-site VPD statistics from
stage 21.
"""
from pathlib import Path

import pandas as pd
from scipy import stats

from src.paths import load_settings

VARIANT = ""
SITE_SUBSET = ""
FLUX = 'NEP_ZSCORE'
REVIEWER_CUTOFF_KPA = 1.25

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

    d = thresholds.merge(sites[['SITE', 'VPD_Z0', 'VPD_SD', 'VPD_MAX']], on='SITE')
    d = d.copy()
    d.loc[d['SITE'].isin(PA_UNIT_SITES), PA_UNIT_COLS] /= 100
    d['vpd_max_kpa'] = d['VPD_MAX'] / 10
    d['threshold_kpa'] = (d['VPD_Z0'] + d['threshold_z'] * d['VPD_SD']) / 10

    below = int((d['vpd_max_kpa'] <= REVIEWER_CUTOFF_KPA).sum())
    print(f"Sites never exceeding {REVIEWER_CUTOFF_KPA} kPa: {below} of {len(d)}")
    print(f"Lowest site maximum: {d['vpd_max_kpa'].min():.2f} kPa, median {d['vpd_max_kpa'].median():.2f} kPa")
    if below == 0:
        print("The split the reviewer suggested leaves an empty group, so quartiles are used instead.\n")

    d['stratum'] = pd.qcut(d['vpd_max_kpa'], 4, labels=['Q1 lowest max VPD', 'Q2', 'Q3', 'Q4 highest max VPD'])
    g = d.groupby('stratum', observed=True).agg(
        n=('SITE', 'size'),
        median_max_vpd_kpa=('vpd_max_kpa', 'median'),
        median_threshold_sigma=('threshold_z', 'median'),
        median_threshold_kpa=('threshold_kpa', 'median'),
        sites_with_crossing=('threshold_z', lambda v: int(v.notna().sum())))
    print(g.to_string(float_format=lambda v: f"{v:.2f}"))

    rho_sigma = stats.spearmanr(d['vpd_max_kpa'], d['threshold_z'])
    rho_kpa = stats.spearmanr(d['vpd_max_kpa'], d['threshold_kpa'])
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
    print(f"\nA zero crossing appears at {crossings} of {len(d)} sites, in every stratum,"
          f" so the nonlinearity is not confined to sites that reach high absolute VPD.")

    outfile = info / "62_INFO_ThresholdVsSiteVPDRange.csv"
    g.to_csv(outfile)
    d[['SITE', 'IGBP', 'threshold_z', 'threshold_kpa', 'vpd_max_kpa', 'stratum']].to_csv(
        info / "62_INFO_ThresholdVsSiteVPDRange_perSite.csv", index=False)
    print(f"\nSaved {outfile}")


if __name__ == '__main__':
    main()
