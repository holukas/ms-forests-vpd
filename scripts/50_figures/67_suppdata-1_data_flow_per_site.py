"""
Supplementary Data 1: the record losses per site, from delivered half-hours to the
modelled records.

Lays out `22_data_flow.csv` from stage 20 as the table that goes to the journal as a
Supplementary Data file (it has one row per site and exceeds an A4 page, so it cannot be a
Supplementary Table). One row per site, the six steps of Supplementary Fig. 10 as columns
in reading order, the soil water layer the site used, and a last row with the totals.
Nothing is recomputed here.

Reads:
    20_subsets/<VARIANT>/22_data_flow.csv

Writes, into the plot folder next to the other supplementary items:
    67_SUPPDATA-1_DataFlowPerSite.xlsx | .csv
"""
from pathlib import Path

import pandas as pd

from src.paths import load_settings

# Run variant, matching the stage that produced 22_data_flow.csv.
VARIANT = ""

# Column order and the headers the reader sees. The step names match the labels of
# Supplementary Fig. 10 and the Methods sentence on the data flow.
COLUMNS = [
    ('SITE', 'Site'),
    ('IGBP', 'Forest type (IGBP)'),
    ('all_records', 'All half-hourly records'),
    ('measured_nee', 'Measured NEE (QC flag 0)'),
    ('daytime', 'Daytime (SW_IN_POT > 20 W m-2)'),
    ('peak_months', 'Four peak-GPP months'),
    ('year_balanced', 'Equal year coverage per month'),
    ('complete_cases', 'Complete cases (records used)'),
    ('SWC_VAR', 'Soil water layer used'),
]
COUNT_COLUMNS = [c for c, _ in COLUMNS if c not in ('SITE', 'IGBP', 'SWC_VAR')]

settings = load_settings()
flow = pd.read_csv(Path(settings['DIR_DATA_PROC_SUBSETS_BASE']) / VARIANT / "22_data_flow.csv")
flow = flow.sort_values('SITE').reset_index(drop=True)

total = {c: flow[c].sum() for c in COUNT_COLUMNS}
total.update({'SITE': 'All sites', 'IGBP': '', 'SWC_VAR': ''})
table = pd.concat([flow[[c for c, _ in COLUMNS]], pd.DataFrame([total])], ignore_index=True)
table.columns = [name for _, name in COLUMNS]

dir_out = Path(settings['DIR_PLOTS_OUT']) / 'NEP_ZSCORE' / 'conditional' / VARIANT
dir_out.mkdir(parents=True, exist_ok=True)
stem = dir_out / "67_SUPPDATA-1_DataFlowPerSite"
table.to_excel(f"{stem}.xlsx", index=False)
table.to_csv(f"{stem}.csv", index=False, encoding='utf-8-sig')

kept = int((flow['complete_cases'] > 0).sum())
none_left = flow.loc[flow['complete_cases'] == 0, 'SITE'].tolist()
print(f"{len(flow)} sites, {kept} with records left"
      + (f", none left at {', '.join(none_left)}" if none_left else ""))
print(f"Totals: " + ", ".join(f"{c} {total[c]:,}" for c in COUNT_COLUMNS))
print(f"Saved {stem}.xlsx and .csv")
