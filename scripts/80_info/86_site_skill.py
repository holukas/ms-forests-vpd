"""
Per-site model skill (n, R2, RMSE, MAE, bias) for all records, Stage 8 and the high-VPD tail.

Uses the out-of-sample predictions that stage 31 stores in the per-site SHAP files
(`NEP_ZSCORE_PRED`, the five folds concatenated), so nothing is refitted. Runs for each
entry of VARIANTS: the main analysis with random folds and the leave-one-year-out
cross-validation (`blocked-cv`).

- Bias is prediction minus observation.
- The VPD tail is VPD_ZSCORE above the Stage 8 VPD cutoff.
- Subsets with fewer than MIN_RECORDS records get no skill numbers.
- R2 in Stage 8 and the tail is computed against the subset's own mean, where the NEP
  variance is small, so R2 can be low or negative while RMSE is modest. Read both together.

Reads, per variant: 30_shap/NEP_ZSCORE/conditional/{variant}/{site}_shap-conditional_NEP_ZSCORE.parquet
Writes:
- 80_info/NEP_ZSCORE/86_INFO_SiteSkill_NEP_ZSCORE.csv: one row per site and variant
- 80_info/NEP_ZSCORE/86_INFO_SiteSkill_summary.csv: median and quartiles across sites
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import src.stages as stg
from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANTS = ["", "blocked-cv"]     # empty is the submitted run
VPD_TAIL_SIGMA = 1.2815515655446   # the Stage 8 VPD cut-off, top decile of a normal
MIN_RECORDS = 10                   # a subset with fewer records gets no skill numbers

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
dir_out = Path(settings['DIR_INFO_OUT']) / FLUX
dir_out.mkdir(parents=True, exist_ok=True)


def skill(y, p):
    if len(y) < MIN_RECORDS:
        return dict(n=len(y), r2=np.nan, rmse=np.nan, mae=np.nan, bias=np.nan)
    e = p - y
    ss_tot = np.sum((y - y.mean()) ** 2)
    return dict(n=len(y), r2=1 - np.sum(e ** 2) / ss_tot if ss_tot > 0 else np.nan,
                rmse=np.sqrt(np.mean(e ** 2)), mae=np.mean(np.abs(e)), bias=np.mean(e))


rows = []
for variant in VARIANTS:
    d = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / variant
    files = sorted(glob.glob(str(d / f'*_shap-{shap_type}_{FLUX}.parquet')))
    for f in files:
        site = Path(f).name.split('_')[0]
        t = pq.read_table(f, columns=[FLUX, f'{FLUX}_PRED', 'TA_ZSCORE', 'VPD_ZSCORE', 'SWC_ZSCORE']).to_pandas()
        t = t.dropna(subset=[FLUX, f'{FLUX}_PRED'])
        y, p = t[FLUX].to_numpy(float), t[f'{FLUX}_PRED'].to_numpy(float)
        s8, *_ = stg.stage_8(t)
        tail = t[t['VPD_ZSCORE'] > VPD_TAIL_SIGMA]
        row = dict(site=site, variant=variant or 'random-folds')
        for tag, sub in (('all', t), ('stage8', s8), ('vpdtail', tail)):
            sk = skill(sub[FLUX].to_numpy(float), sub[f'{FLUX}_PRED'].to_numpy(float))
            row.update({f'{tag}_{k}': v for k, v in sk.items()})
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(dir_out / f'86_INFO_SiteSkill_{FLUX}.csv', index=False)

metrics = [c for c in out.columns if c.split('_')[-1] in ('r2', 'rmse', 'mae', 'bias', 'n')]
summary = (out.groupby('variant')[metrics]
           .describe(percentiles=[0.25, 0.5, 0.75]).stack(level=0, future_stack=True)
           .reset_index().rename(columns={'level_1': 'metric'}))
summary = summary[['variant', 'metric', 'count', '25%', '50%', '75%']]
summary.to_csv(dir_out / '86_INFO_SiteSkill_summary.csv', index=False)

pd.set_option('display.width', 200)
for variant, g in out.groupby('variant'):
    print(f"\n{variant}: {len(g)} sites")
    for tag in ('all', 'stage8', 'vpdtail'):
        n_ok = g[f'{tag}_r2'].notna().sum()
        med = g[[f'{tag}_n', f'{tag}_r2', f'{tag}_rmse', f'{tag}_mae', f'{tag}_bias']].median()
        print(f"  {tag:8s} sites with >= {MIN_RECORDS} records: {n_ok:3d} | median n {med[f'{tag}_n']:.0f}, "
              f"R2 {med[f'{tag}_r2']:.3f}, RMSE {med[f'{tag}_rmse']:.3f}, MAE {med[f'{tag}_mae']:.3f}, bias {med[f'{tag}_bias']:+.3f}")
print(f"\nSaved {dir_out / f'86_INFO_SiteSkill_{FLUX}.csv'} and 86_INFO_SiteSkill_summary.csv")
