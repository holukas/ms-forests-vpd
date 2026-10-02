"""
How much of a peak-season day is measured, and how many Stage 8 records each time scale would keep.

Backs the reply to Reviewer 2 on why the analysis is not repeated at daily or coarser
resolution (R2-01). Two parts, both counted over the 208 sites of the analysis.

1. Daily coverage. The analysis window is the four peak-GPP months of a site and its
   daytime records (SW_IN_POT > 20 W m-2), all years, before the quality filter. Per day,
   the script counts the daytime records and those with directly measured NEE
   (NEE_*_QC == 0). Per site it takes the median day and the share of days that reach
   90, 75 and 50 % measured, then the median across sites. The pooled shares count every
   site-day once.
2. Sample size per time scale. The stage 21 subsets (measured, daytime, peak months,
   year-balanced, complete cases) are averaged to days (only days with at least
   MIN_RECORDS_PER_DAY records), calendar months and years. Averaging narrows the spread
   of the z-scores, so the means are standardized again per site at their own scale
   (population standard deviation) before the Stage 8 mask of `src.stages.stage_8` is
   applied: Stage 8 then means the same tail of the site distribution at every scale.
   The half-hourly row applies the mask to the records themselves and reproduces the
   30,398 Stage 8 records of the analysis.

Reads: 17_datasets_info_parquet_vars_stats_usedsites_era5.csv, the stage 12 merged
parquet of each site, 21_SUBSETS_parquet_vars_stats_subsets.csv and the stage 21 subsets.
Writes, into 80_info/NEP_ZSCORE/:
- 80_INFO_DailyCoverage_perSite.csv: one row per site
- 80_INFO_DailyCoverage.csv: median site and pooled shares
- 80_INFO_TimeScaleSampleSize.csv: records and Stage 8 records per time scale
"""
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import diive as dv
import src.stages as stg
from src.common import get_variable_names, peak_season_months
from src.paths import data_path, load_settings, resolve_stored_path

FLUX = 'NEP_ZSCORE'
VARIANT = ""                 # empty is the main analysis
N_WORKERS = 3                # memory is the limit, as in stage 21
COVERAGE_LEVELS = [90, 75, 50]
MIN_RECORDS_PER_DAY = 8      # a day with fewer daytime records gets no daily value
MIN_TRAINING_RECORDS = 100   # reported: sites with at least this many records per scale
ZCOLS = ['TA_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']


def daily_coverage(task: tuple) -> dict:
    """Daytime records per peak-season day and the measured share, for one site."""
    _, siteconfig = task
    varnames = get_variable_names(siteconfig)
    df = dv.load_parquet(str(resolve_stored_path(siteconfig['_FILEPATH_PARQUET'])),
                         sanitize_timestamp=False, output_middle_timestamp=False)
    # Same peak-season definition as stage 21: the full record, before any filter
    months = peak_season_months(df, gpp_col=varnames['gpp_var'])
    df = df.loc[df.index.month.isin(months) & (df[varnames['swinpot_var']] > 20)]
    measured = (df[varnames['nee_qc_var']] == 0)
    per_day = pd.DataFrame({'daytime': measured.groupby(df.index.date).size(),
                            'measured': measured.groupby(df.index.date).sum()})
    share = per_day['measured'] / per_day['daytime']
    row = {'SITE': siteconfig['SITE'], 'IGBP': siteconfig['IGBP'], 'days': len(per_day),
           'daytime_per_day_median': per_day['daytime'].median(),
           'measured_per_day_median': per_day['measured'].median(),
           'measured_share_pct': 100 * per_day['measured'].sum() / per_day['daytime'].sum()}
    for level in COVERAGE_LEVELS:
        row[f'days_ge{level}pct_measured'] = int((share >= level / 100).sum())
        row[f'days_ge{level}pct_measured_pct'] = 100 * (share >= level / 100).mean()
    return row


def time_scale_counts(path: Path) -> dict:
    """Records and Stage 8 records of one subset at each time scale."""
    df = pd.read_parquet(path, columns=ZCOLS)
    t = df.index
    daily_n = df.groupby(t.date).size()
    daily = df.groupby(t.date).mean().loc[daily_n >= MIN_RECORDS_PER_DAY]
    scales = {'half-hourly': df,
              'daily': daily,
              'monthly': df.groupby([t.year, t.month]).mean(),
              'yearly': df.groupby(t.year).mean()}
    row = {'SITE': path.name.split('_')[0]}
    for scale, d in scales.items():
        if scale != 'half-hourly':
            d = (d - d.mean()) / d.std(ddof=0)
        stage8, *_ = stg.stage_8(d)
        row[f'{scale}_records'] = len(d)
        row[f'{scale}_stage8'] = len(stage8)
    return row


def main():
    settings = load_settings()
    dir_out = Path(settings['DIR_INFO_OUT']) / FLUX
    dir_out.mkdir(parents=True, exist_ok=True)
    subsets_dir = Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
    sites = pd.read_csv(subsets_dir / "21_SUBSETS_parquet_vars_stats_subsets.csv")['SITE']
    datasets = pd.read_csv(
        data_path("data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv"))
    datasets = datasets.loc[datasets['SITE'].isin(sites)]
    print(f"{len(datasets)} sites")

    # Part 1: daily coverage in the analysis window
    tasks = list(datasets.iterrows())
    with Pool(N_WORKERS) as pool:
        cov = pd.DataFrame(pool.map(daily_coverage, tasks)).sort_values('SITE')
    cov.to_csv(dir_out / "80_INFO_DailyCoverage_perSite.csv", index=False)

    summary = {'sites': len(cov),
               'daytime_per_day_median_site': cov['daytime_per_day_median'].median(),
               'measured_per_day_median_site': cov['measured_per_day_median'].median(),
               'measured_share_pct_median_site': cov['measured_share_pct'].median(),
               'sites_median_day_unmeasured': int((cov['measured_per_day_median'] == 0).sum())}
    for level in COVERAGE_LEVELS:
        summary[f'days_ge{level}pct_measured_pct_median_site'] = \
            cov[f'days_ge{level}pct_measured_pct'].median()
        summary[f'days_ge{level}pct_measured_pct_pooled'] = \
            100 * cov[f'days_ge{level}pct_measured'].sum() / cov['days'].sum()
    pd.Series(summary).to_frame('value').to_csv(dir_out / "80_INFO_DailyCoverage.csv",
                                                 index_label='quantity')
    print("\nDaily coverage in the analysis window (four peak-GPP months, daytime)")
    for k, v in summary.items():
        print(f"  {k:<46}{v:>10.1f}")

    # Part 2: sample size per time scale
    paths = sorted((subsets_dir / "21_subsets_parquet").glob("*_subset_*.parquet"))
    paths = [p for p in paths if p.name.split('_')[0] in set(sites)]
    counts = pd.DataFrame([time_scale_counts(p) for p in paths])
    rows = []
    for scale in ('half-hourly', 'daily', 'monthly', 'yearly'):
        n, s8 = counts[f'{scale}_records'], counts[f'{scale}_stage8']
        rows.append({'scale': scale, 'records': int(n.sum()), 'median_per_site': n.median(),
                     'stage8_records': int(s8.sum()), 'sites_without_stage8': int((s8 == 0).sum()),
                     f'sites_with_ge{MIN_TRAINING_RECORDS}_records': int((n >= MIN_TRAINING_RECORDS).sum())})
    sample = pd.DataFrame(rows)
    sample.to_csv(dir_out / "80_INFO_TimeScaleSampleSize.csv", index=False)
    print(f"\nSample size per time scale ({len(counts)} sites; days need "
          f"{MIN_RECORDS_PER_DAY} records)")
    print(sample.to_string(index=False))


if __name__ == '__main__':
    main()
