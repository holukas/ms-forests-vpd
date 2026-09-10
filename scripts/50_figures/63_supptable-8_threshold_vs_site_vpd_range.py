"""
Supplementary Table 8: the per-site threshold against the VPD range of the site.

Sites are split into quartiles of their own maximum peak-season daytime VPD. For each
quartile the table gives the number of sites, the median site maximum, the median per-site
threshold in the site's own standard deviations and in kPa, and how many sites have a zero
crossing at all. If the standardisation had made the threshold, the value in kPa would
rise in proportion to the site's range; if the threshold were a fixed physical limit, it
would not move. The table shows neither.

It lays out the summary written by `scripts/80_info/82_threshold_vs_site_vpd_range.py`;
nothing is recomputed here.

Reads:
    80_info/<FLUX>/conditional/82_INFO_ThresholdVsSiteVPDRange.csv

Writes, into the plot folder next to the other supplementary tables:
    63_SUPPTABLE-8_ThresholdVsSiteVPDRange_<FLUX>.csv | .xlsx
"""
from pathlib import Path

import pandas as pd

from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# The stratum labels of script 82 and the names the reader sees.
STRATA = [
    ('Q1 lowest max VPD', 'Q1, lowest maximum VPD'),
    ('Q2', 'Q2'),
    ('Q3', 'Q3'),
    ('Q4 highest max VPD', 'Q4, highest maximum VPD'),
]
COLUMNS = [
    ('n', 'Sites', 0),
    ('median_max_vpd_kpa', 'Median site maximum VPD (kPa)', 2),
    ('median_threshold_sigma', 'Median threshold (σ)', 2),
    ('median_threshold_kpa', 'Median threshold (kPa)', 2),
    ('sites_with_crossing', 'Sites with a zero crossing', 0),
]

settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'
dir_info = Path(settings['DIR_INFO_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)

summary = pd.read_csv(dir_info / '82_INFO_ThresholdVsSiteVPDRange.csv').set_index('stratum')

rows = []
for key, label in STRATA:
    row = {'Quartile of site maximum VPD': label}
    for column, name, decimals in COLUMNS:
        value = summary.loc[key, column]
        row[name] = int(value) if decimals == 0 else round(float(value), decimals)
    rows.append(row)
table = pd.DataFrame(rows)

print(f"Per-site threshold by quartile of the site's maximum VPD, {int(summary['n'].sum())} sites\n")
# The console on Windows may not take the sigma; the files keep it.
print(table.to_string(index=False).replace('σ', 'sigma'))
lo, hi = summary.iloc[0], summary.iloc[-1]
print(f"\nSite maximum VPD grows {100 * (hi['median_max_vpd_kpa'] / lo['median_max_vpd_kpa'] - 1):.0f} % "
      f"from Q1 to Q4, the threshold in kPa {100 * (hi['median_threshold_kpa'] / lo['median_threshold_kpa'] - 1):.0f} %")

stem = dir_out / f"63_SUPPTABLE-8_ThresholdVsSiteVPDRange_{FLUX}"
table.to_csv(f"{stem}.csv", index=False, encoding='utf-8-sig')
table.to_excel(f"{stem}.xlsx", index=False)
print(f"\nSaved {stem}.csv and .xlsx")
