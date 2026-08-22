"""
Count how many records each filtering step of stage 21 removes.

Reviewer 2 asked for a transparent path from the records a site delivers to the
records a model sees, with the losses named. The steps mirror
`src.files.create_subsets_parquet_files` in the same order, so the number in the
last column must equal `N_RECORDS` in the stage 21 summary. The script checks
that per site and refuses to write a table that disagrees.

Writes `22_data_flow.csv`, one row per site, into the stage 20 folder of the
run variant.
"""
import re
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

import diive as dv
from src.common import get_variable_names, peak_season_months
from src.paths import data_path, load_settings, resolve_stored_path

# Run variant, matching 21_prepare_input_data.py.
VARIANT = ""

# Sites run in parallel. Memory is the limit, as in stage 21.
N_WORKERS = 3

STEPS = ["all_records", "measured_nee", "daytime", "peak_months", "year_balanced", "complete_cases"]


def count_site(task: tuple) -> dict:
    """Walk one site through the stage 21 filters, counting rows after each."""
    ix, siteconfig = task
    site = str(siteconfig['SITE'])
    varnames = get_variable_names(siteconfig)

    df = dv.load_parquet(str(resolve_stored_path(siteconfig['_FILEPATH_PARQUET'])),
                         sanitize_timestamp=False, output_middle_timestamp=False)
    counts = {'SITE': site, 'IGBP': siteconfig['IGBP'], 'all_records': len(df)}

    df = df.copy()
    df['MONTH'] = df.index.month
    df['YEAR'] = df.index.year
    gpp_top4 = peak_season_months(df, gpp_col=varnames['gpp_var'])

    df = df.loc[df[varnames['nee_qc_var']] == 0]
    counts['measured_nee'] = len(df)

    df = df.loc[df[varnames['swinpot_var']] > 20]
    counts['daytime'] = len(df)

    df = df.loc[df['MONTH'].isin(gpp_top4)]
    counts['peak_months'] = len(df)

    # Equal year coverage per month, oldest years trimmed
    min_years = df.groupby('MONTH')['YEAR'].nunique().min()
    keep = pd.DatetimeIndex([], name=df.index.name)
    for month in gpp_top4:
        month_data = df.loc[df['MONTH'] == month]
        years = month_data['YEAR'].unique()
        if len(years) > min_years:
            years = sorted(years, reverse=True)[:min_years]
        keep = keep.union(month_data.loc[month_data['YEAR'].isin(years)].index)
    df = df.loc[keep]
    counts['year_balanced'] = len(df)

    required = [varnames[v] for v in
                ('nee_var', 'le_var', 'gpp_var', 'reco_var', 'ta_var', 'vpd_var', 'swin_var', 'swc_var')]
    counts['complete_cases'] = len(df[required].dropna())
    counts['SWC_VAR'] = varnames['swc_var']
    return counts


def main():
    settings = load_settings()
    datasets_df = pd.read_csv(
        data_path("data/outputs/10_datasets/17_datasets_info_parquet_vars_stats_usedsites_era5.csv"))

    tasks = [(ix, siteconfig) for ix, siteconfig in datasets_df.iterrows()]
    print(f"Counting {len(tasks)} sites with {N_WORKERS} worker(s).")

    if N_WORKERS == 1:
        rows = [count_site(t) for t in tasks]
    else:
        with Pool(N_WORKERS) as pool:
            rows = pool.map(count_site, tasks)

    flow = pd.DataFrame(rows).sort_values('SITE').reset_index(drop=True)

    # A site that ends with nothing is dropped at stage 21 and has no subset
    dropped = flow.loc[flow['complete_cases'] == 0, 'SITE'].tolist()

    outdir = Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
    subsets = pd.read_csv(outdir / "21_SUBSETS_parquet_vars_stats_subsets.csv")
    check = flow.merge(subsets[['SITE', 'N_RECORDS']], on='SITE', how='inner')
    mismatch = check.loc[check['complete_cases'] != check['N_RECORDS'],
                         ['SITE', 'complete_cases', 'N_RECORDS']]
    if not mismatch.empty:
        raise SystemExit(f"Counts disagree with stage 21 for {len(mismatch)} site(s):\n"
                         f"{mismatch.to_string(index=False)}")
    print(f"Final counts match stage 21 for all {len(check)} sites.")

    flow.to_csv(outdir / "22_data_flow.csv", index=False)
    print(f"Wrote {outdir / '22_data_flow.csv'}")

    print(f"\nSites: {len(datasets_df)} into stage 21, {len(check)} with a subset")
    if dropped:
        print(f"No records left after filtering: {', '.join(dropped)}")

    totals = flow[STEPS].sum()
    print(f"\n{'step':<18}{'records':>14}{'removed':>14}{'share kept':>12}")
    prev = None
    for step in STEPS:
        n = int(totals[step])
        removed = "" if prev is None else f"{prev - n:>14,}"
        print(f"{step:<18}{n:>14,}{removed:>14}{n / totals[STEPS[0]] * 100:>11.1f}%")
        prev = n


if __name__ == '__main__':
    main()
