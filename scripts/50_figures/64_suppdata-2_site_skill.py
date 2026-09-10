"""
Supplementary Data 2: the skill of every site model under both cross-validation splits.

Lays out what `scripts/80_info/86_site_skill.py` writes as the file that goes to the
journal as a Supplementary Data file (two rows per site, so it exceeds an A4 page and
cannot be a Supplementary Table). One row per site and split, the forest type, and for
all records, the compound-extreme stage (Stage 8) and the high-VPD tail the number of
records, R2, RMSE, MAE and bias of the out-of-sample prediction. Nothing is recomputed
here.

Reads:
    80_info/<FLUX>/86_INFO_SiteSkill_<FLUX>.csv
    20_subsets/<VARIANT>/21_SUBSETS_parquet_vars_stats_subsets.csv   for the forest type

Writes, into the plot folder next to the other supplementary items:
    64_SUPPDATA-2_SiteSkill_<FLUX>.xlsx | .csv
"""
from pathlib import Path

import pandas as pd

from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""

# The split labels of script 86 and the names the reader sees, in the order of the Methods.
SPLITS = [('random-folds', 'Shuffled 5-fold'), ('blocked-cv', 'Leave-one-year-out')]
# The record subsets of script 86 and their headers.
SUBSETS = [('all', 'All records'), ('stage8', 'Stage 8'), ('vpdtail', 'VPD tail (> 1.28σ)')]
METRICS = [('n', 'n', 0), ('r2', 'R2', 3), ('rmse', 'RMSE (σ)', 3), ('mae', 'MAE (σ)', 3),
           ('bias', 'Bias (σ)', 3)]

settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT
dir_out.mkdir(parents=True, exist_ok=True)

skill = pd.read_csv(Path(settings['DIR_INFO_OUT']) / FLUX / f'86_INFO_SiteSkill_{FLUX}.csv')
sites = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT
                    / '21_SUBSETS_parquet_vars_stats_subsets.csv')[['SITE', 'IGBP']]

split_order = {key: i for i, (key, _) in enumerate(SPLITS)}
skill = skill.merge(sites, left_on='site', right_on='SITE', how='left')
skill['split_order'] = skill['variant'].map(split_order)
skill = skill.sort_values(['site', 'split_order']).reset_index(drop=True)

table = pd.DataFrame({
    'Site': skill['site'],
    'Forest type (IGBP)': skill['IGBP'],
    'Cross-validation split': skill['variant'].map(dict(SPLITS)),
})
for subset, subset_name in SUBSETS:
    for metric, metric_name, decimals in METRICS:
        values = skill[f'{subset}_{metric}']
        table[f'{subset_name}, {metric_name}'] = (values.round().astype('Int64') if decimals == 0
                                                  else values.round(decimals))

stem = dir_out / f'64_SUPPDATA-2_SiteSkill_{FLUX}'
table.to_excel(f'{stem}.xlsx', index=False)
table.to_csv(f'{stem}.csv', index=False, encoding='utf-8-sig')

for key, name in SPLITS:
    g = table[table['Cross-validation split'] == name]
    print(f"{name}: {len(g)} sites, median R2 over all records {g['All records, R2'].median():.3f}, "
          f"Stage 8 skill at {g['Stage 8, R2'].notna().sum()} sites")
print(f"{len(table)} rows, {table['Forest type (IGBP)'].isna().sum()} without a forest type")
print(f"Saved {stem}.xlsx and .csv")
