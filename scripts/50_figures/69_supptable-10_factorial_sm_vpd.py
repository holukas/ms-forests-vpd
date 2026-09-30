"""
Supplementary Table 10: SM and VPD effects in the sixteen cells of four soil water classes by
four VPD classes.

The stage sequences escalate one driver while the other is only kept off its extreme, so
they are no factorial design. This table crosses the soil water classes with the VPD classes
at the stage cut-offs, with temperature free. The cells at central VPD show the SM effect
while VPD stays near its baseline, the cells at central SM show the VPD effect while soil
water stays near its baseline, and no driver comes first.

Per cell: the number of sites and records, and the SM, VPD and net effects as the mean of the
site means, as in Fig. 3 and Supplementary Table 6, with the interquartile range across sites.

Reads:
    40_aggregation/<FLUX>/conditional/factorial-cells/44_SHAPVALUES-conditional_AggregatedAcrossScenarios_<FLUX>.csv
        from script 44 with STAGE_SEQUENCE = "factorial"

Writes, into the plot folder:
    69_SUPPTABLE-10_FactorialSmVpd_<FLUX>.csv | .xlsx
    69_SUPPTABLE-10_FactorialSmVpd_<FLUX>_DATA.csv   mean, median and quartiles per cell
"""
from pathlib import Path

import pandas as pd

import src.stages as s
from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

# Class ranges as the reader sees them, in the order of src/stages.py
SWC_LABELS = ['central (|z| ≤ 0.32σ)', 'dry (−0.71σ to −0.32σ)',
              'very dry (−1.28σ to −0.71σ)', 'extreme (< −1.28σ)']
VPD_LABELS = ['central (|z| ≤ 0.32σ)', 'higher (0.32σ to 0.71σ)',
              'high (0.71σ to 1.28σ)', 'extreme (> 1.28σ)']
EFFECTS = [
    ('SWC_ZSCORE_SHAPVALS_OVR_AVG', 'SM effect (σ)'),
    ('VPD_ZSCORE_SHAPVALS_OVR_AVG', 'VPD effect (σ)'),
    ('NET_SHAPVALS_OVR_AVG', 'Net effect (σ)'),
]

settings = load_settings()
shap_type = 'conditional' if CONDITIONAL else 'interventional'
dir_agg = (Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT
           / SITE_SUBSET / 'factorial-cells')
dir_out = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
dir_out.mkdir(parents=True, exist_ok=True)

sites = pd.read_csv(dir_agg / f"44_SHAPVALUES-{shap_type}_AggregatedAcrossScenarios_{FLUX}.csv")
# A site enters a cell when it has at least one record there, as in the stage tables
sites = sites[sites['N_VALUES'] > 0]

rows, data_rows = [], []
for (swc_k, vpd_k), cell in sites.groupby(['SWC_CLASS', 'VPD_CLASS']):
    row = {'SM class': SWC_LABELS[swc_k], 'VPD class': VPD_LABELS[vpd_k],
           'Sites': cell['SITE'].nunique(), 'Records': int(cell['N_VALUES'].sum())}
    data_row = {'SWC_CLASS': swc_k, 'VPD_CLASS': vpd_k,
                'SWC_CLASS_NAME': s.SWC_CLASS_NAMES[swc_k], 'VPD_CLASS_NAME': s.VPD_CLASS_NAMES[vpd_k],
                'sites': row['Sites'], 'records': row['Records']}
    for column, name in EFFECTS:
        values = cell[column]
        q25, q75 = values.quantile(0.25), values.quantile(0.75)
        row[name] = f"{values.mean():.2f} [{q25:.2f}, {q75:.2f}]"
        stem = column.replace('_SHAPVALS_OVR_AVG', '')
        data_row.update({f'{stem}_mean': values.mean(), f'{stem}_median': values.median(),
                         f'{stem}_q25': q25, f'{stem}_q75': q75})
    rows.append(row)
    data_rows.append(data_row)
table = pd.DataFrame(rows)
data = pd.DataFrame(data_rows)

print(f"SM and VPD effects in {len(table)} cells, {sites['SITE'].nunique()} sites\n")
# The console on Windows may not take the sigma; the files keep it.
print(table.to_string(index=False).replace('σ', 'sigma').replace('≤', '<=').replace('−', '-'))
for column, name in EFFECTS[:2]:
    stem = column.replace('_SHAPVALS_OVR_AVG', '')
    grid = data.pivot(index='SWC_CLASS_NAME', columns='VPD_CLASS_NAME', values=f'{stem}_mean')
    grid = grid.reindex(index=list(s.SWC_CLASS_NAMES), columns=list(s.VPD_CLASS_NAMES)).round(2)
    print(f"\n{name.replace('σ', 'sigma')}, mean of site means, rows SM class, columns VPD class\n{grid.to_string()}")

stem = dir_out / f"69_SUPPTABLE-10_FactorialSmVpd_{FLUX}"
table.to_csv(f"{stem}.csv", index=False, encoding='utf-8-sig')
table.to_excel(f"{stem}.xlsx", index=False)
data.to_csv(f"{stem}_DATA.csv", index=False)
print(f"\nSaved {stem}.csv, .xlsx and _DATA.csv")
