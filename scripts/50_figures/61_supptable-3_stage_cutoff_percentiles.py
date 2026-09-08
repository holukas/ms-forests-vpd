"""
Supplementary Table 3, lower part: where the fixed stage cut-offs fall in the site distributions.

The stages are defined by fixed z-values, the same at every site, and not by the
empirical percentiles of each site. Standardisation does not make the drivers Gaussian,
so the share of records below a cut-off differs from the share a standard normal
distribution would give. This table shows by how much: one row per cut-off, with the
percentile it corresponds to under a standard normal distribution and, for each driver,
the median across sites of the share of peak-season daytime records at or below it.

It lays out the summary written by `scripts/80_info/87_stage_cutoff_percentiles.py`;
nothing is recomputed here. Meant as the lower part of the stage definition table.

Reads:
    80_info/<FLUX>/87_INFO_StageCutoffPercentiles_summary.csv

Writes, into the plot folder next to the other supplementary tables:
    61_SUPPTABLE-3_StageCutoffPercentiles_<FLUX>.csv | .xlsx
"""
from pathlib import Path

import pandas as pd

from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""

# Driver order and the names the manuscript uses. The files carry the FLUXNET names.
DRIVERS = [('TA_ZSCORE', 'TA'), ('VPD_ZSCORE', 'VPD'), ('SWC_ZSCORE', 'SM'),
           ('SWIN_ZSCORE', 'SW')]

# Cut-offs from low to high, as script 87 labels them.
CUTOFF_ORDER = ['-c', '-b', '-a', 'a', 'b', 'c']

settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'
dir_info = Path(settings['DIR_INFO_OUT']) / FLUX
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT
dir_out.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(dir_info / '87_INFO_StageCutoffPercentiles_summary.csv')

rows = []
for cutoff in CUTOFF_ORDER:
    block = summary[summary['cutoff'] == cutoff].set_index('driver')
    row = {
        'Cut-off (sigma)': round(float(block['sigma'].iloc[0]), 2),
        'Standard normal (%)': round(100 * float(block['gaussian'].iloc[0]), 1),
    }
    for variable, name in DRIVERS:
        row[f'{name} (%)'] = round(100 * float(block.loc[variable, 'median']), 1)
    rows.append(row)

table = pd.DataFrame(rows)

n_sites = int(summary['sites'].iloc[0])
print(f"Median share of records at or below each cut-off, across {n_sites} sites\n")
print(table.to_string(index=False))

table.to_csv(dir_out / f"61_SUPPTABLE-3_StageCutoffPercentiles_{FLUX}.csv",
             index=False, encoding='utf-8-sig')
table.to_excel(dir_out / f"61_SUPPTABLE-3_StageCutoffPercentiles_{FLUX}.xlsx", index=False)
print(f"\nSaved two files to {dir_out}")
