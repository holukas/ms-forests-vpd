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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.paths import load_settings

# The row labels carry a greater-or-equal sign, and the Windows console is cp1252, so a
# plain print of the table raises. Writing the files never did, only the echo.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

MOVER = 0.06             # kPa, beyond this a row reads as a real shift
AX_LABELS_FONTSIZE = 12
COLOR_POINT = '#0072B2'
COLOR_GREY = '#8A8A8A'
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
# A table on the left and the bars on the right, sharing one row grid. The earlier version
# plotted the shift from the published value on a centred axis, which made agreement easy to
# see but hid the numbers the reader wants. Here the axis carries absolute kPa, the value and
# its interval are printed, and the table half mirrors main-text Table 1 so the two read as
# one argument.
#
# One blue throughout. Colouring the movers red made the eye go to two rows out of seventeen
# and read them as failures, which is the opposite of what the figure says.

SHORT = {
    'Deepest available layer, all sites': 'Deepest layer, all sites',
    'Matched sites, layer 1': 'Matched sites, shallow',
    'Matched sites, deepest layer': 'Matched sites, deepest',
    'Layer 5+ sites, shallow soil water': 'Layer 5+ sites, shallow',
    'Layer 5+ sites, deep soil water': 'Layer 5+ sites, deep',
    'Air temperature dropped from the model': 'No air temperature',
    'Threshold estimator, 23 settings': 'Estimator, 23 settings',
    'Europe and North America removed': 'Europe and N. America removed',
    'Records of at least 3 years': 'Records ≥ 3 years',
    'Records of at least 5 years': 'Records ≥ 5 years',
    'Records of at least 10 years': 'Records ≥ 10 years',
}

# The reference row was inserted under a friendlier name than the file uses, so the
# lookup falls back to the reference sigma rather than failing on it.
sigma_of = dict(zip(out['test'], out['threshold_sigma']))
plot_rows = [(g, SHORT.get(t, t), v, lo, hi, n, note,
              sigma_of.get(t, ref_row['threshold_sigma']))
             for g, t, v, lo, hi, n, note in rows]

ordered = list(reversed(plot_rows))
ypos, group_rows, y, prev_group = [], {}, 0.0, None
for group, *_ in ordered:
    if prev_group is not None and group != prev_group:
        y += 1.15
    ypos.append(y)
    group_rows.setdefault(group, []).append(y)
    prev_group = group
    y += 1.0

fig, (axt, ax) = plt.subplots(1, 2, figsize=(12.6, 8.6), dpi=150,
                              gridspec_kw={'width_ratios': [1.18, 1.0], 'wspace': 0.04})
fig.subplots_adjust(left=0.02, right=0.97, top=0.90, bottom=0.10)

axt.axis('off')
for a in (axt, ax):
    a.set_ylim(min(ypos) - 1.0, max(ypos) + 1.3)

TAB = axt.get_yaxis_transform()
COL_LABEL, COL_N, COL_SIGMA, COL_VALUE = 0.00, 0.545, 0.655, 0.70


def value_text(mid, lo, hi, note):
    """What the value column prints. It has to say which kind of interval each row carries,
    so the word travels with the number rather than sitting in a column of its own where the
    neighbouring axes would cover it."""
    if hi <= lo:
        return f'{mid:.2f}   (no band)'
    if note:
        return f'{mid:.2f}   ({lo:.2f} to {hi:.2f} range)'
    return f'{mid:.2f}   [{lo:.2f}, {hi:.2f}]'


for (group, label, mid, lo, hi, n, note, sigma), y in zip(ordered, ypos):
    axt.text(COL_LABEL, y, label, transform=TAB, ha='left', va='center',
             fontsize=AX_LABELS_FONTSIZE * 0.92, color='#1a1a1a')
    axt.text(COL_N, y, f'{n}', transform=TAB, ha='right', va='center',
             fontsize=AX_LABELS_FONTSIZE * 0.8, color=COLOR_GREY)
    axt.text(COL_SIGMA, y, f'{sigma:.2f}', transform=TAB, ha='right', va='center',
             fontsize=AX_LABELS_FONTSIZE * 0.8, color=COLOR_GREY)
    axt.text(COL_VALUE, y, value_text(mid, lo, hi, note), transform=TAB, ha='left',
             va='center', fontsize=AX_LABELS_FONTSIZE * 0.8, color='#444444')

for group, ys in group_rows.items():
    top = max(ys) + 0.66
    axt.text(COL_LABEL, top, group.upper(), transform=TAB, ha='left', va='center',
             fontsize=AX_LABELS_FONTSIZE * 0.68, color=COLOR_GREY, fontweight='bold')
axt.text(COL_N, max(ypos) + 0.66, 'sites', transform=TAB, ha='right', va='center',
         fontsize=AX_LABELS_FONTSIZE * 0.68, color=COLOR_GREY, fontweight='bold')
axt.text(COL_SIGMA, max(ypos) + 0.66, 'sigma', transform=TAB, ha='right', va='center',
         fontsize=AX_LABELS_FONTSIZE * 0.68, color=COLOR_GREY, fontweight='bold')
axt.text(COL_VALUE, max(ypos) + 0.66, 'kPa [95 % band]', transform=TAB, ha='left',
         va='center', fontsize=AX_LABELS_FONTSIZE * 0.68, color=COLOR_GREY,
         fontweight='bold')

# A rule under the reference row, so the eye separates the value being tested from the tests.
rule_y = ypos[-1] - 0.62
for a, x0, x1 in ((axt, COL_LABEL, 1.10), (ax, 0.0, 1.0)):
    a.plot([x0, x1], [rule_y, rule_y], transform=a.get_yaxis_transform(), color='#B8B8B8',
           lw=1.0, zorder=1, clip_on=False)

ax.axvspan(pub_lo, pub_hi, color=COLOR_REF, alpha=0.08, lw=0, zorder=0)
ax.axvline(published, color=COLOR_REF, lw=1.4, zorder=2)

for (group, label, mid, lo, hi, n, note, sigma), y in zip(ordered, ypos):
    if hi <= lo:
        pass                                   # ALE: a marker on its own, no interval
    elif note:
        # Whiskers, not a band. These rows carry a spread across settings or removals, which
        # is a different quantity from a prediction band, so it must not look the same.
        ax.plot([lo, hi], [y, y], color=COLOR_POINT, lw=1.2, zorder=3)
        for edge in (lo, hi):
            ax.plot([edge, edge], [y - 0.26, y + 0.26], color=COLOR_POINT, lw=1.2, zorder=3)
    else:
        ax.plot([lo, hi], [y, y], color=COLOR_POINT, lw=3.2, alpha=0.45, zorder=3,
                solid_capstyle='round')
    ax.scatter([mid], [y], s=44, color=COLOR_POINT, zorder=4, linewidths=0)

ax.set_yticks([])
ax.set_xlabel('VPD threshold (kPa)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE * 0.9, colors='#555555', length=4)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color('#CCCCCC')
ax.text(published, 1.01, f'  published {published:.2f} kPa',
        transform=ax.get_xaxis_transform(), ha='left', va='bottom', color=COLOR_REF,
        fontweight='bold', fontsize=AX_LABELS_FONTSIZE * 0.85, clip_on=False)

# No caption text inside the image. The journal wants the legend as text beside the
# display item, not baked into the png, so both the summary line and the explanation
# live in the caption drafted in the revision notes. The numbers behind the summary
# line are printed below instead, so a rerun still reports them.
_points = [r[2] for r in rows if r[1] != 'Published analysis']
_inside = sum(pub_lo <= v <= pub_hi for v in _points)
print(f'{_inside} of {len(_points)} tests fall inside the band of the published '
      f'threshold, {pub_lo:.2f} to {pub_hi:.2f} kPa')

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
#
# Depth is represented by the layer 5+ pair rather than by the all-sites row. That row mixes
# the 80 sites that never moved with 128 that did, most of them by one layer, so its small
# shift is diluted. The pair holds 59 sites fixed and swaps layer 1 for layer 5 or deeper,
# which is the largest departure from the layer 1 selection the published number rests on.
MAIN_ROWS = [
    'Layer 5+ sites, shallow soil water',
    'Layer 5+ sites, deep soil water',
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
    'Layer 5+ sites, shallow soil water':
        'The 59 sites with a layer 5 or deeper sensor, using their surface layer',
    'Layer 5+ sites, deep soil water':
        'The same 59 sites, using soil water from layer 5 or deeper',
    'GAM instead of a polynomial':
        'A spline whose shape the data set, instead of a fourth-order polynomial',
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


# Group names become sub-header rows rather than a column, which saves the width a Nature
# Communications page does not have. The prose column is gone for the same reason: the test
# names carry it, and the caption holds anything they do not.
#
# Sigma and kPa both, because every row converts with its own site set. A row can move in kPa
# while standing still in sigma, and the reader cannot see that from kPa alone. It is also
# the column that ties this table to Figure 4, which reports sigma.
COLUMNS = ['Test', f'Sites (of {int(ref["n_sites"])})', 'Threshold (sigma)',
           'Threshold (kPa)', '95% interval (kPa)']


def _row(test, n, sigma, kpa, interval_text):
    return dict(zip(COLUMNS, [test, n, sigma, kpa, interval_text]))


records = [_row('Published analysis', int(ref['n_sites']), f"{ref['threshold_sigma']:.2f}",
                f"{ref['threshold_kpa']:.2f}",
                f"{ref['lower']:.2f} to {ref['upper']:.2f}")]
for group in ['Soil water depth', 'Model fitting', 'Site set']:
    block = main.loc[main['group'] == group]
    if block.empty:
        continue
    records.append(_row(group, '', '', '', ''))          # sub-header row
    for _, r in block.iterrows():
        records.append(_row(SHORT.get(r['test'], r['test']), int(r['n_sites']),
                            f"{r['threshold_sigma']:.2f}", f"{r['threshold_kpa']:.2f}",
                            interval(r)))
table1 = pd.DataFrame(records, columns=COLUMNS)




# Which tests actually pass the cutoff, rather than a sentence that assumes one does. Adding
# the layer 5+ pair made the old wording false: its shallow row sits 0.069 kPa below the
# reference because it uses 59 sites, so two rows now clear 0.06 kPa rather than one.
biggest = tests.loc[tests['shift_kpa'].abs().idxmax()]
movers = tests.loc[tests['shift_kpa'].abs() >= MOVER].sort_values('shift_kpa')
mover_text = '; '.join(f"{SHORT.get(r['test'], r['test']).lower()} at "
                       f"{r['threshold_kpa']:.2f} kPa" for _, r in movers.iterrows())
footnote = (
    f"Threshold is the highest zero crossing of a fourth-order polynomial fitted to the "
    f"VPD SHAP values, the same estimator as in Figure 4. Intervals are the 95 % prediction "
    f"band of that fit, as in Figure 4. Rows marked * have no band: the leave-one-site-out "
    f"row shows the spread across the 208 removals, and the ALE row fits no curve. "
    f"The published value is {ref['threshold_kpa']:.2f} kPa. Of all {len(tests)} tests, "
    f"{len(movers)} depart from it by more than {MOVER:.2f} kPa: {mover_text}. Every other "
    f"test lands closer. The two layer 5+ rows are a pair and are read against each other, "
    f"not against the reference: they use 59 sites rather than 208, so their offset from it "
    f"is a difference in sample, while the difference between them, "
    f"{abs(main.loc[main['test'].str.startswith('Layer'), 'threshold_kpa'].diff().iloc[-1]):.2f} "
    f"kPa, is the effect of soil water depth. The full set is in Supplementary Fig. X."
)

stem = folder / f'55_TABLE-1_ThresholdRobustness_{FLUX}'
table1.to_csv(f'{stem}.csv', index=False)
with pd.ExcelWriter(f'{stem}.xlsx', engine='openpyxl') as writer:
    table1.to_excel(writer, sheet_name='Table 1', index=False, startrow=0)
    sheet = writer.sheets['Table 1']
    # Column widths, so the pasted table does not need manual fixing in Word.
    for column, width in zip('ABCDE', (38, 13, 16, 16, 20)):
        sheet.column_dimensions[column].width = width
    sheet.cell(row=len(table1) + 3, column=1, value=footnote)

print(table1.to_string(index=False))
print()
print(footnote)
print(f"\nSaved {stem}.xlsx and {stem}.csv")

if SHOW_PLOT:
    plt.show()
