"""
Threshold robustness as a table.

The same content as the planned Figure 5, written as a table instead. The differences
between tests are small, at most 0.083 kPa on a threshold of 1.26, and any figure of them
either magnifies the differences by zooming the axis or hides them by not zooming. A table
states the numbers exactly and leaves nothing to be read off a scale.

Reads what `66_fig-5_threshold_robustness.py` cached, so the numbers cannot drift from the
figure. Run that first, or with RECOMPUTE = True if the inputs changed.

No shift column. Rounded to two decimals four of the shifts read as +0.00 or -0.00, and
three decimals would imply a precision the bootstrap interval, about plus or minus 0.02,
does not support. The threshold column against the reference row says the same thing.

Writes both formats:
  xlsx   for pasting into Word, which keeps the table structure
  csv    for the repository and the data deposit
"""
from pathlib import Path

import pandas as pd

from src.paths import load_settings

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
folder = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET

source = folder / f'66_PLANNED-FIG-5_ThresholdRobustness_{FLUX}_DATA.csv'
if not source.is_file():
    raise FileNotFoundError(f"No cached numbers at {source}. Run "
                            f"66_fig-5_threshold_robustness.py first.")

d = pd.read_csv(source)
ref = d.loc[d['test'] == 'PUBLISHED REFERENCE'].iloc[0]
tests = d.loc[d['test'] != 'PUBLISHED REFERENCE'].copy()

# What each test varies, so the table reads without the figure beside it.
WHAT_IT_VARIES = {
    'Deepest available layer, all sites':
        'Soil water taken from the deepest layer each site has',
    'Matched sites, layer 1':
        'Only the sites that have a deeper layer, using their shallow layer',
    'Matched sites, deepest layer':
        'The same sites, using their deepest layer',
    'Blocked cross-validation':
        'One calendar year left out at a time instead of a shuffled split',
    'Threshold estimator, 23 settings':
        'Polynomial degree, smoothing, bin width and fitting range',
    'Europe removed': 'European sites excluded',
    'North America removed': 'North American sites excluded',
    'Europe and North America removed': 'Both regions excluded',
    'Records of at least 3 years': 'Sites with shorter records excluded',
    'Records of at least 5 years': 'Sites with shorter records excluded',
    'Records of at least 10 years': 'Sites with shorter records excluded',
}


def interval(row):
    """The interval, with a note where it is not a site bootstrap."""
    text = f"{row['lower']:.2f} to {row['upper']:.2f}"
    return text + (' *' if isinstance(row['note'], str) else '')


out = pd.DataFrame({
    'Group': tests['group'],
    'Test': tests['test'],
    'What it varies': tests['test'].map(WHAT_IT_VARIES),
    'Sites': tests['n_sites'].astype(int),
    'Threshold (kPa)': tests['threshold_kpa'].round(2),
    '95% interval (kPa)': tests.apply(interval, axis=1),
})

# The reference on top, since every shift is measured against it.
header = pd.DataFrame([{
    'Group': 'Reference',
    'Test': 'Published analysis',
    'What it varies': 'None, this is the value reported in the manuscript',
    'Sites': int(ref['n_sites']),
    'Threshold (kPa)': round(ref['threshold_kpa'], 2),
    '95% interval (kPa)': f"{ref['lower']:.2f} to {ref['upper']:.2f}",
}])
out = pd.concat([header, out], ignore_index=True)

biggest = tests.loc[tests['shift_kpa'].abs().idxmax()]
footnote = (
    f"Threshold is the highest zero crossing of a fourth-order polynomial fitted to the "
    f"VPD SHAP values, the same estimator as in Figure 4. Intervals are a bootstrap over "
    f"sites, 2000 replicates. Rows marked * show the spread across the settings varied "
    f"rather than a bootstrap. The published value is {ref['threshold_kpa']:.2f} kPa. The "
    f"largest departure from it is {biggest['test'].lower()} at "
    f"{biggest['threshold_kpa']:.2f} kPa; every other test lands within 0.06 kPa of the "
    f"published value."
)

stem = folder / f'70_TABLE-1_ThresholdRobustness_{FLUX}'
out.to_csv(f'{stem}.csv', index=False)
with pd.ExcelWriter(f'{stem}.xlsx', engine='openpyxl') as writer:
    out.to_excel(writer, sheet_name='Table 1', index=False, startrow=0)
    sheet = writer.sheets['Table 1']
    # Column widths, so the pasted table does not need manual fixing in Word.
    for column, width in zip('ABCDEF', (16, 34, 52, 8, 16, 20)):
        sheet.column_dimensions[column].width = width
    sheet.cell(row=len(out) + 3, column=1, value=footnote)

print(out.to_string(index=False))
print()
print(footnote)
print(f"\nSaved {stem}.xlsx and {stem}.csv")
