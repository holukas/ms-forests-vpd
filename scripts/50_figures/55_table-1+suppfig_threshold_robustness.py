"""
Threshold robustness: main-text Table 1 and the matching supplementary figure.

One script, because both display items show the same fourteen tests. Rendering them from two
scripts is how a table and a figure drift apart.

**It computes nothing.** `40_aggregation/47_threshold_robustness.py` builds the rows and
writes them to file, which is where the two-minute bootstrap belongs. This script reads that
file and draws. It runs in about a second, which is what every script under `50_figures`
should do, because they become notebooks under S10.

**The table is the main-text item, the figure is supplementary.** Decided 31 August 2026.
The differences between tests are at most 0.08 kPa on a threshold of 1.26, so a figure
either magnifies them by zooming the axis or hides them by not zooming. The table states
the numbers exactly. The figure keeps the shape of the result for readers who want it.

**The interval is the prediction band of the fitted curve**, the same one Figure 4 and the
coefficient table show, so the reference row reads 1.26 [1.16, 1.36] in both places. Script
47 also writes a bootstrap over sites, about five times narrower, which answers how much the
threshold depends on which sites are in the network. That one is not shown here, because two
different intervals on the same number would confuse a reader. Use it in the response letter.

**Three rows have no interval.** The figure draws them with a diamond and the table marks
them with an asterisk. The estimator row shows the spread across its 23 settings instead, and
the leave-one-site-out row the spread across its 208 removals, which are not independent of
each other. The ALE row has no curve fitted here at all, since its value is a mean of per-site
crossings. Script 47 flags all three in the `note` column.

**No shift column in the table.** Rounded to two decimals several of the shifts read as +0.00
or -0.00, and three decimals would imply a precision the bootstrap interval, about plus or
minus 0.02, does not support. The threshold column read against the reference row says the
same thing. The figure plots the shift, because there an axis centred on the published value
is what makes agreement visible.

**Why the VPD quartile split is not here.** It used to be the last row. It does not belong:
the other rows ask whether the number survives a different analytical choice, while that one
asks whether the threshold varies with site climate, and the answer is yes. Its range, 1.00
to 1.41 kPa, was the widest bar in the figure, so a reader scanning for robustness saw the
largest apparent instability where there was actually signal. Most of that signal is also
mechanical: the threshold is estimated in sigma and converted per site, and drier sites have
a larger mean and spread, so a constant sigma threshold already yields a higher kPa value in
the drier quartiles. It belongs in its own supplementary analysis with that caveat stated,
`80_info/82_threshold_vs_site_vpd_range.py`.

Writes:
  png    the supplementary figure
  xlsx   Table 1, for pasting into Word, which keeps the table structure
  csv    Table 1 for the repository and the data deposit
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import load_settings

SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

MOVER = 0.06             # kPa, beyond this a row reads as a real shift
AX_LABELS_FONTSIZE = 12
COLOR_POINT = '#0072B2'
COLOR_MOVER = '#b2182b'
COLOR_REF = '#D55E00'

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
folder = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET
folder.mkdir(parents=True, exist_ok=True)
agg = Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS_AGG']) / FLUX / shap_type / VARIANT / SITE_SUBSET

source = agg / f'47_THRESHOLD_Robustness_{FLUX}.csv'
if not source.is_file():
    raise FileNotFoundError(f"No rows at {source}. Run "
                            f"40_aggregation/47_threshold_robustness.py first.")

out = pd.read_csv(source)
ref_row = out.loc[out['test'] == 'PUBLISHED REFERENCE'].iloc[0]
published, pub_lo, pub_hi = ref_row['threshold_kpa'], ref_row['lower'], ref_row['upper']
rows = [(g, t, v, lo, hi, int(n), (None if pd.isna(note) else note))
        for g, t, v, lo, hi, n, note in
        out.loc[out['test'] != 'PUBLISHED REFERENCE']
        [['group', 'test', 'threshold_kpa', 'lower', 'upper', 'n_sites', 'note']]
        .itertuples(index=False, name=None)]

# The published analysis is the first row, the way the table lists it, so the reader sees
# the value being tested rather than inferring it from a line.
rows.insert(0, ('Reference', 'Published analysis', published, pub_lo, pub_hi,
                int(ref_row['n_sites']), None))

outfile = folder / f'55_SUPPFIG-X_ThresholdRobustness_{FLUX}.png'


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# Plotted as the shift from the published threshold. Every test lands near 1.26 kPa, so on
# an absolute axis the points crowd into a narrow band and the reader has to work out that
# this is agreement. Centring on the published value makes zero mean something and lets the
# few tests that do move stand out.

LABEL_X, VALUE_X = -0.46, 1.02

ordered = list(reversed(rows))
ypos, group_rows, y = [], {}, 0.0
prev_group = None
for group, *_ in ordered:
    if prev_group is not None and group != prev_group:
        y += 1.15
    ypos.append(y)
    group_rows.setdefault(group, []).append(y)
    prev_group = group
    y += 1.0

fig, ax = plt.subplots(figsize=(11, 8.2), dpi=150)
# The caption runs to four lines, so it gets its own band at the bottom.
fig.subplots_adjust(left=0.34, right=0.87, top=0.88, bottom=0.24)
trans = ax.get_yaxis_transform()

ax.axvline(0, color=COLOR_REF, lw=1.4, zorder=2)

for (group, label, mid, lo, hi, n, note), y in zip(ordered, ypos):
    shift = mid - published
    colour = COLOR_MOVER if abs(shift) >= MOVER else COLOR_POINT
    ax.plot([0, 1], [y, y], transform=trans, color='#F2F2F2', lw=0.8, zorder=0)
    if hi > lo:
        ax.plot([lo - published, hi - published], [y, y], color=colour, lw=3.2,
                alpha=0.32 if note else 0.55, zorder=3, solid_capstyle='round')
    ax.scatter([shift], [y], s=44, color=colour, zorder=4,
               marker='D' if note else 'o', linewidths=0)
    ax.text(LABEL_X, y, label, transform=trans, ha='left', va='center',
            fontsize=AX_LABELS_FONTSIZE * 0.92, color='#1a1a1a')
    ax.text(LABEL_X, y - 0.34, f'{n} sites' + (f'   {note}' if note else ''),
            transform=trans, ha='left', va='center',
            fontsize=AX_LABELS_FONTSIZE * 0.66, color='#9a9a9a')
    ax.text(VALUE_X, y, f'{shift:+.2f}', transform=trans, ha='left', va='center',
            fontsize=AX_LABELS_FONTSIZE * 0.85,
            color=colour if abs(shift) >= MOVER else '#666666',
            fontweight='bold' if abs(shift) >= MOVER else 'normal')

for group, ys in group_rows.items():
    top = max(ys) + 0.62
    ax.text(LABEL_X, top, group.upper(), transform=trans, ha='left', va='center',
            fontsize=AX_LABELS_FONTSIZE * 0.7, color='#9a9a9a', fontweight='bold')
    ax.plot([LABEL_X, 1.0], [top - 0.28, top - 0.28], transform=trans, color='#E2E2E2',
            lw=0.9, zorder=0, clip_on=False)

ax.set_yticks([])
ax.set_ylim(min(ypos) - 1.0, max(ypos) + 1.1)
ax.set_xlabel('Shift from the published threshold (kPa)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE, colors='#555555', length=4)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color('#CCCCCC')

ax.text(0, 1.01, f' published {published:.2f} kPa [{pub_lo:.2f}, {pub_hi:.2f}]',
        transform=ax.get_xaxis_transform(),
        ha='left', va='bottom', color=COLOR_REF, fontweight='bold',
        fontsize=AX_LABELS_FONTSIZE * 0.9, clip_on=False)
ax.text(VALUE_X, max(ypos) + 0.62, 'shift, kPa', transform=trans, ha='left', va='center',
        color='#9a9a9a', fontsize=AX_LABELS_FONTSIZE * 0.7, fontweight='bold')

fig.text(0.025, 0.015,
         f'Zero crossing of the fitted curve, published value {published:.2f} kPa '
         f'[{pub_lo:.2f}, {pub_hi:.2f}]. Bars are the 95 % prediction band of that fit.'
         + chr(10) + f'Red marks a shift of at least {MOVER:.2f} kPa.'
         + chr(10) + 'Diamonds mark the three rows with no band: the estimator row and '
         'the leave-one-site-out row show the spread across' + chr(10) + 'their settings and '
         'their removals, and ALE fits no curve here.',
         fontsize=AX_LABELS_FONTSIZE * 0.72, color='#666666', ha='left', linespacing=1.5)
fig.savefig(outfile, dpi=300, facecolor='white')
print(f"Saved {outfile}")

# ---------------------------------------------------------------------------
# Table 1
# ---------------------------------------------------------------------------
# The same rows the figure used, so the two cannot disagree.
ref = out.loc[out['test'] == 'PUBLISHED REFERENCE'].iloc[0]
tests = out.loc[out['test'] != 'PUBLISHED REFERENCE'].copy()

# Seven tests for the main table, one per kind of challenge. The figure keeps all of them.
# The full set is too many for a main display item, and the reason is redundancy: three
# record-length rows make one point, three region rows make one point, and each matched pair
# is a two-row argument about one comparison. Both largest movers stay, North America and the
# ten-year records, so the selection cannot be read as flattering. The estimator row is left
# out on purpose: dropping it removes one of the three things the interval column means, so
# the main table carries two instead of three.
MAIN_ROWS = [
    'Deepest available layer, all sites',
    'Air temperature dropped from the model',
    'Blocked cross-validation',
    'ALE instead of SHAP',
    'Any one site removed',
    'North America removed',
    'Records of at least 10 years',
]
missing = [t for t in MAIN_ROWS if t not in set(tests['test'])]
if missing:
    raise SystemExit(f'Main table rows not found in the input: {missing}')
main = tests.loc[tests['test'].isin(MAIN_ROWS)].copy()

# What each test varies, so the table reads without the figure beside it.
WHAT_IT_VARIES = {
    'Deepest available layer, all sites':
        'Soil water taken from the deepest layer each site has',
    'Matched sites, layer 1':
        'Only the sites that have a deeper layer, using their shallow layer',
    'Matched sites, deepest layer':
        'The same sites, using their deepest layer',
    'Air temperature dropped from the model':
        'The model never sees temperature, so nothing of it can reach the VPD values',
    'Blocked cross-validation':
        'One calendar year left out at a time instead of a shuffled split',
    'ALE instead of SHAP':
        'Accumulated local effects, a different attribution method entirely',
    'Threshold estimator, 23 settings':
        'Polynomial degree, smoothing, bin width and fitting range',
    'Any one site removed':
        'Each site dropped from the network in turn, one at a time',
    'Europe removed': 'European sites excluded',
    'North America removed': 'North American sites excluded',
    'Europe and North America removed': 'Both regions excluded',
    'Records of at least 3 years': 'Sites with shorter records excluded',
    'Records of at least 5 years': 'Sites with shorter records excluded',
    'Records of at least 10 years': 'Sites with shorter records excluded',
}


def interval(row):
    """The interval, with a note where it is not a site bootstrap.

    Script 47 writes equal bounds for a row with no fitted curve, which is the case for ALE.
    Printing "1.25 to 1.25" would read as a precision claim, so the cell says none instead
    and the note column carries the reason.
    """
    if row['upper'] <= row['lower']:
        return 'none *'
    text = f"{row['lower']:.2f} to {row['upper']:.2f}"
    return text + (' *' if isinstance(row['note'], str) else '')


table1 = pd.DataFrame({
    'Group': main['group'],
    'Test': main['test'],
    'What it varies': main['test'].map(WHAT_IT_VARIES),
    'Sites': main['n_sites'].astype(int),
    'Threshold (kPa)': main['threshold_kpa'].round(2),
    '95% interval (kPa)': main.apply(interval, axis=1),
})

# The reference on top, since every shift is measured against it.
reference_row = pd.DataFrame([{
    'Group': 'Reference',
    'Test': 'Published analysis',
    'What it varies': 'None, this is the value reported in the manuscript',
    'Sites': int(ref['n_sites']),
    'Threshold (kPa)': round(ref['threshold_kpa'], 2),
    '95% interval (kPa)': f"{ref['lower']:.2f} to {ref['upper']:.2f}",
}])
table1 = pd.concat([reference_row, table1], ignore_index=True)

biggest = tests.loc[tests['shift_kpa'].abs().idxmax()]
footnote = (
    f"Threshold is the highest zero crossing of a fourth-order polynomial fitted to the "
    f"VPD SHAP values, the same estimator as in Figure 4. Intervals are the 95 % prediction "
    f"band of that fit, as in Figure 4. Rows marked * have no band: the leave-one-site-out "
    f"row shows the spread across the 208 removals, and the ALE row fits no curve. "
    f"The published value is {ref['threshold_kpa']:.2f} kPa. Of all {len(tests)} tests, the "
    f"largest departure from it is {biggest['test'].lower()} at "
    f"{biggest['threshold_kpa']:.2f} kPa, and every other test lands within {MOVER:.2f} kPa "
    f"of the published value. The full set is in Supplementary Fig. X."
)

stem = folder / f'55_TABLE-1_ThresholdRobustness_{FLUX}'
table1.to_csv(f'{stem}.csv', index=False)
with pd.ExcelWriter(f'{stem}.xlsx', engine='openpyxl') as writer:
    table1.to_excel(writer, sheet_name='Table 1', index=False, startrow=0)
    sheet = writer.sheets['Table 1']
    # Column widths, so the pasted table does not need manual fixing in Word.
    for column, width in zip('ABCDEF', (16, 34, 52, 8, 16, 20)):
        sheet.column_dimensions[column].width = width
    sheet.cell(row=len(table1) + 3, column=1, value=footnote)

print(table1.to_string(index=False))
print()
print(footnote)
print(f"\nSaved {stem}.xlsx and {stem}.csv")

if SHOW_PLOT:
    plt.show()
