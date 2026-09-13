"""
Does adding extreme VPD still cost net uptake when the other conditions are matched?

The Results compare Stage 7 (hot, driest soil, VPD not extreme) with Stage 8 (the
same plus extreme VPD) as cross-site means of the site means, and as the share of the
common sites at which the net effect is more negative at Stage 8. Both comparisons use
whatever records fall into each stage, so the two stages can differ in more than VPD:
within the cut-offs Stage 8 records are hotter and drier than Stage 7 records, and they
carry more light, since shortwave radiation is not constrained by the stages.

This script matches every Stage 8 record of a site to the closest Stage 7 record of
the same site, with replacement, and reads the difference in the attributed effects
across the matched pairs. Two matchings, so the price of the time constraint is visible:

  strict     same calendar month, hour of day within one hour, and every covariate
             (TA, SWC, SWIN in site-standardized units) within CALIPER
  covariate  every covariate within CALIPER, any month and hour

Per site the script reports how many Stage 8 records found a partner, the covariate
balance after matching, and the mean difference (Stage 8 minus Stage 7) of the net
effect and of the four driver contributions over the matched pairs. Across sites it
reports the median and mean of those site differences, the share of sites at which the
matched net difference is negative, and the unmatched comparison on the same sites for
reference.

Nothing is refitted. Reads the per-site SHAP files of stage 31 and applies the stage
masks of `src/stages.py`, as stage 44 does.

Reads:
    20_subsets/21_SUBSETS_parquet_vars_stats_subsets.csv
    30_shap/<FLUX>/conditional/<SITE>_shap-conditional_<FLUX>.parquet

Writes:
    80_info/<FLUX>/conditional/89_INFO_StageMatching_<FLUX>.csv          per site and matching
    80_info/<FLUX>/conditional/89_INFO_StageMatching_<FLUX>_SUMMARY.csv  across sites
"""
from pathlib import Path

import numpy as np
import pandas as pd

import src.stages as s
from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
VARIANT = ""

CALIPER = 0.5          # largest allowed |difference| per covariate, in sigma
HOUR_WINDOW = 1.0      # strict matching: hour of day within this many hours
COVARIATES = ['TA_ZSCORE', 'SWC_ZSCORE', 'SWIN_ZSCORE']
SHAP = {'net': 'SUM', 'VPD': 'VPD_ZSCORE_SHAPVALS', 'TA': 'TA_ZSCORE_SHAPVALS',
        'SM': 'SWC_ZSCORE_SHAPVALS', 'SW': 'SWIN_ZSCORE_SHAPVALS'}

settings = load_settings()
shap_dir = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / 'conditional' / VARIANT
outdir = Path(settings['DIR_INFO_OUT']) / FLUX / 'conditional' / VARIANT
outdir.mkdir(parents=True, exist_ok=True)
sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                    / "21_SUBSETS_parquet_vars_stats_subsets.csv")


def match(s8, s7, strict):
    """Index of the closest Stage 7 record for every Stage 8 record, or -1 if none is
    within the caliper (and, when strict, within the same month and hour window)."""
    x8 = s8[COVARIATES].to_numpy()
    x7 = s7[COVARIATES].to_numpy()
    # squared distance, n8 by n7
    d = ((x8[:, None, :] - x7[None, :, :]) ** 2).sum(axis=2)
    ok = (np.abs(x8[:, None, :] - x7[None, :, :]) <= CALIPER).all(axis=2)
    if strict:
        m8 = s8.index.month.to_numpy(); m7 = s7.index.month.to_numpy()
        h8 = (s8.index.hour + s8.index.minute / 60).to_numpy()
        h7 = (s7.index.hour + s7.index.minute / 60).to_numpy()
        ok &= (m8[:, None] == m7[None, :])
        ok &= (np.abs(h8[:, None] - h7[None, :]) <= HOUR_WINDOW)
    d = np.where(ok, d, np.inf)
    j = d.argmin(axis=1)
    j[~np.isfinite(d[np.arange(len(j)), j])] = -1
    return j


rows = []
for _, r in sites.iterrows():
    site, igbp = r['SITE'], r['IGBP']
    f = shap_dir / f"{site}_shap-conditional_{FLUX}.parquet"
    if not f.is_file():
        print(f"  No results for {site}, skipping.")
        continue
    df = pd.read_parquet(f)
    s7 = s.stage_7(df)[0]
    s8 = s.stage_8(df)[0]
    if len(s7) == 0 or len(s8) == 0:
        continue
    for name, strict in [('strict', True), ('covariate', False)]:
        j = match(s8, s7, strict)
        hit = j >= 0
        row = {'SITE': site, 'IGBP': igbp, 'matching': name,
               'n_stage7': len(s7), 'n_stage8': len(s8), 'n_matched': int(hit.sum()),
               'share_matched': hit.mean(),
               # unmatched comparison on the same site, all records of each stage
               'net_stage7_all': s7['SUM'].mean(), 'net_stage8_all': s8['SUM'].mean(),
               'vpd_stage7_all': s7['VPD_ZSCORE_SHAPVALS'].mean(),
               'vpd_stage8_all': s8['VPD_ZSCORE_SHAPVALS'].mean()}
        if hit.sum() == 0:
            rows.append(row)
            continue
        a = s8.loc[hit]
        b = s7.iloc[j[hit]]
        for cov in COVARIATES + ['VPD_ZSCORE']:
            row[f'd_{cov}'] = a[cov].to_numpy().mean() - b[cov].to_numpy().mean()
        for short, col in SHAP.items():
            row[f'd_{short}'] = a[col].to_numpy().mean() - b[col].to_numpy().mean()
        rows.append(row)
    print(f"{site}: stage 7 {len(s7)}, stage 8 {len(s8)}, "
          f"matched strict {rows[-2]['n_matched']}, covariate {rows[-1]['n_matched']}")

per_site = pd.DataFrame(rows)
per_site.to_csv(outdir / f'89_INFO_StageMatching_{FLUX}.csv', index=False)

# Across sites. Sites with no matched pair carry no difference and are counted as such.
summary = []
for name in ['strict', 'covariate']:
    p = per_site.loc[per_site['matching'] == name]
    m = p.loc[p['n_matched'] > 0]
    row = {'matching': name, 'sites_both_stages': len(p), 'sites_with_pairs': len(m),
           'stage8_records': int(p['n_stage8'].sum()),
           'stage8_records_matched': int(p['n_matched'].sum()),
           'share_records_matched': p['n_matched'].sum() / p['n_stage8'].sum()}
    for cov in COVARIATES + ['VPD_ZSCORE']:
        row[f'balance_{cov}_mean'] = m[f'd_{cov}'].mean()
    for short in SHAP:
        row[f'd_{short}_median'] = m[f'd_{short}'].median()
        row[f'd_{short}_mean'] = m[f'd_{short}'].mean()
        row[f'd_{short}_share_negative'] = (m[f'd_{short}'] < 0).mean()
    # the unmatched comparison, on the sites with pairs and on all common sites
    row['unmatched_net_diff_mean_sites_with_pairs'] = (m['net_stage8_all'] - m['net_stage7_all']).mean()
    row['unmatched_net_share_negative_sites_with_pairs'] = ((m['net_stage8_all'] - m['net_stage7_all']) < 0).mean()
    row['unmatched_net_diff_mean_all_common'] = (p['net_stage8_all'] - p['net_stage7_all']).mean()
    row['unmatched_net_share_negative_all_common'] = ((p['net_stage8_all'] - p['net_stage7_all']) < 0).mean()
    summary.append(row)
summary = pd.DataFrame(summary)
summary.to_csv(outdir / f'89_INFO_StageMatching_{FLUX}_SUMMARY.csv', index=False)

pd.set_option('display.width', 200)
print()
print(summary.T.to_string())
print(f"\nSaved to {outdir}")
