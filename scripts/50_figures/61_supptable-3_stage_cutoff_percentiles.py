"""
Supplementary Table 3: the stage definitions, and where the fixed cut-offs fall in the site
distributions.

Upper part: the eight stages as conditions on the site-standardized drivers. The text rules
are checked against the stage functions in `src/stages.py` on random z-values, and the script
stops before writing if they disagree.

Lower part: the cut-offs are fixed z-values, and standardizing does not make the drivers
Gaussian. One row per cut-off, with its standard normal percentile and, per driver, the
median across sites of the share of peak-season daytime records at or below it.

Reads:
    80_info/<FLUX>/87_INFO_StageCutoffPercentiles_summary.csv   from script 87 of 80_info

Writes, into the plot folder:
    61_SUPPTABLE-3_StageDefinitions_<FLUX>.xlsx          both parts on one sheet, for Word
    61_SUPPTABLE-3_StageDefinitions_<FLUX>.csv           the upper part
    61_SUPPTABLE-3_StageCutoffPercentiles_<FLUX>.csv     the lower part
"""
from pathlib import Path

import numpy as np
import pandas as pd

import src.stages as stg
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

cutoffs = pd.DataFrame(rows)

# The upper part. Bounds are (low, high, low inclusive, high inclusive) on the cut-offs a, b, c
# of src/stages.py; None means unbounded. SW is unrestricted in every stage.
A, B, C = 0.32, 0.71, 1.28
STAGE_RULES = {
    1: {'TA': (-A, A, True, True), 'SM': (-A, A, True, True), 'VPD': (-A, A, True, True)},
    2: {'TA': (A, B, False, True), 'SM': (-A, A, True, True), 'VPD': (None, C, False, True)},
    3: {'TA': (A, B, False, True), 'SM': (-B, -A, True, False), 'VPD': (None, C, False, True)},
    4: {'TA': (B, C, False, True), 'SM': (-B, -A, True, False), 'VPD': (None, C, False, True)},
    5: {'TA': (B, C, False, True), 'SM': (-C, -B, True, False), 'VPD': (None, C, False, True)},
    6: {'TA': (C, None, False, False), 'SM': (-C, -B, True, False), 'VPD': (None, C, False, True)},
    7: {'TA': (C, None, False, False), 'SM': (None, -C, False, False), 'VPD': (None, C, False, True)},
    8: {'TA': (C, None, False, False), 'SM': (None, -C, False, False), 'VPD': (C, None, False, False)},
}
STAGE_FUNCTIONS = {k: getattr(stg, f'stage_{k}') for k in STAGE_RULES}
RULE_COLUMNS = {'TA': 'TA_ZSCORE', 'SM': 'SWC_ZSCORE', 'VPD': 'VPD_ZSCORE'}


def rule_text(name, bounds):
    """One cell of the grid, in the notation of the printed table."""
    low, high, low_inc, high_inc = bounds
    if low is not None and high is not None and low == -high:
        return f'|{name}| ≤ {high:.2f}'
    if low is None:
        return f'{name} {"≤" if high_inc else "<"} {high:.2f}'
    if high is None:
        return f'{name} {"≥" if low_inc else ">"} {low:.2f}'
    return f'{low:.2f} {"≤" if low_inc else "<"} {name} {"≤" if high_inc else "<"} {high:.2f}'


def rule_mask(z, bounds):
    """The same rule applied to values, for the check against the stage functions."""
    low, high, low_inc, high_inc = bounds
    mask = np.ones(len(z), dtype=bool)
    if low is not None:
        mask &= (z >= low) if low_inc else (z > low)
    if high is not None:
        mask &= (z <= high) if high_inc else (z < high)
    return mask


# The check: the text rules and the functions must select the same records. Values are
# kept away from the rounded cut-offs, where 0.32 and 0.31863936 would disagree.
rng = np.random.default_rng(0)
sample = pd.DataFrame({col: rng.uniform(-3, 3, 200_000) for col in RULE_COLUMNS.values()})
sample['SWIN_ZSCORE'] = rng.uniform(-3, 3, len(sample))
for col in RULE_COLUMNS.values():
    near = np.zeros(len(sample), dtype=bool)
    for edge in (A, B, C):
        near |= (np.abs(np.abs(sample[col]) - edge) < 0.005).to_numpy()
    sample = sample[~near]
sample = sample.reset_index(drop=True)
for stage, rules in STAGE_RULES.items():
    from_rules = np.ones(len(sample), dtype=bool)
    for name, bounds in rules.items():
        from_rules &= rule_mask(sample[RULE_COLUMNS[name]].to_numpy(), bounds)
    from_code = STAGE_FUNCTIONS[stage](sample)[0].index
    assert set(sample.index[from_rules]) == set(from_code), f'Stage {stage}: table and code disagree'

grid = pd.DataFrame([{'Stage': f'Stage {stage}',
                      **{name: rule_text(name, rules[name]) for name in ('TA', 'SM', 'VPD')},
                      'SW': 'all'} for stage, rules in STAGE_RULES.items()])

n_sites = int(summary['sites'].iloc[0])
print("Stage definitions, checked against src/stages.py\n")
print(grid.to_string(index=False))
print(f"\nMedian share of records at or below each cut-off, across {n_sites} sites\n")
print(cutoffs.to_string(index=False))

stem = dir_out / f"61_SUPPTABLE-3_StageDefinitions_{FLUX}"
grid.to_csv(f"{stem}.csv", index=False, encoding='utf-8-sig')
cutoffs.to_csv(dir_out / f"61_SUPPTABLE-3_StageCutoffPercentiles_{FLUX}.csv",
               index=False, encoding='utf-8-sig')
# One sheet for Word: the grid, one empty row, then the cut-off block with its own header.
with pd.ExcelWriter(f"{stem}.xlsx") as writer:
    grid.to_excel(writer, sheet_name='Supplementary Table 3', index=False)
    cutoffs.to_excel(writer, sheet_name='Supplementary Table 3', index=False,
                     startrow=len(grid) + 2)
print(f"\nSaved {stem}.xlsx and two csv files to {dir_out}")
