"""
Supplementary Table 9: three readings of the fitted VPD response, for all sites and per
forest type.

The fitted curve of Fig. 4 has a peak (the most positive VPD effect on NEP), a steepest
decline and a zero crossing, which the manuscript reports as the threshold. The table gives
all three in site-standardized units and in kPa. A value on the edge of the fitted range is
the last point fitted, not a feature of the curve, and is marked "(range edge)". The minimum
of the effect is left out because it sits on the range edge for every group.

Reads:
    80_info/<FLUX>/conditional/81_INFO_ThresholdDefinitions.csv   from script 81 of 80_info

Writes, into the plot folder:
    62_SUPPTABLE-9_ThresholdDefinitions_<FLUX>.csv | .xlsx
"""
from pathlib import Path

import pandas as pd

from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# Group labels of script 81 and the names the reader sees, in the order of Fig. 4.
GROUPS = [('ALL SITES', 'Global forests'), ('ENF', 'ENF'), ('DBF', 'DBF'),
          ('MF', 'MF'), ('EBF', 'EBF')]
# The three readings, in the order they occur along the VPD axis.
DEFINITIONS = [('peak', 'Peak of the VPD effect'),
               ('steepest_decline', 'Steepest decline'),
               ('zero_crossing', 'Zero crossing, the threshold')]
EDGE_NOTE = ' (range edge)'

settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'
dir_info = Path(settings['DIR_INFO_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)

definitions = pd.read_csv(dir_info / '81_INFO_ThresholdDefinitions.csv').set_index('group')


def cell(row, key, unit, decimals):
    """One value, with the edge flag when the point is the last one fitted."""
    text = f"{row[f'{key}_{unit}']:.{decimals}f}"
    return text + EDGE_NOTE if bool(row[f'{key}_at_edge']) else text


rows = []
for key, label in GROUPS:
    r = definitions.loc[key]
    row = {'Group': label, 'Fitted range, upper limit (σ)': f"{r['x_max']:.1f}"}
    for definition, name in DEFINITIONS:
        row[f'{name} (σ)'] = cell(r, definition, 'z', 2)
        row[f'{name} (kPa)'] = cell(r, definition, 'kPa', 2)
    rows.append(row)
table = pd.DataFrame(rows)

# The console on Windows may not take the sigma; the files keep it.
print(table.to_string(index=False).replace('σ', 'sigma'))
flagged = int(table.apply(lambda c: c.astype(str).str.contains(EDGE_NOTE.strip(), regex=False)).to_numpy().sum())
print(f"\n{flagged} values sit on the edge of the fitted range")

stem = dir_out / f"62_SUPPTABLE-9_ThresholdDefinitions_{FLUX}"
table.to_csv(f"{stem}.csv", index=False, encoding='utf-8-sig')
table.to_excel(f"{stem}.xlsx", index=False)
print(f"\nSaved {stem}.csv and .xlsx")
