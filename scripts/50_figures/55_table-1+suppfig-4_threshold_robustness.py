"""
Table 1 and Supplementary Fig. 4: robustness of the VPD threshold.

Both display items show the same robustness tests, so one script draws both from the
same rows. It computes nothing: 40_aggregation/47_threshold_robustness.py builds the
rows, and this script reads and draws them.

The interval is the 95% prediction band of the fitted curve, as in Figure 4. Rows
flagged in the `note` column carry a different interval (a spread across settings or
removals, or for ALE a confidence interval) and get an asterisk and whiskers. The figure
shows all tests, Table 1 only those in MAIN_ROWS. The split by site VPD quartile is not a
robustness test; see 80_info/82_threshold_vs_site_vpd_range.py.

Reads:
- 47_THRESHOLD_Robustness_<FLUX>.csv from the aggregation folder

Writes to the plot folder:
- 55_SUPPFIG-4_ThresholdRobustness_<FLUX>.png
- 55_TABLE-1_ThresholdRobustness_<FLUX>.xlsx (with the footnote) and .csv
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

# The main analysis is the first row, the way the table lists it, so the reader sees
# the value being tested rather than inferring it from a line.
rows.insert(0, ('Reference', 'Main analysis', published, pub_lo, pub_hi,
                int(ref_row['n_sites']), None))

outfile = folder / f'55_SUPPFIG-4_ThresholdRobustness_{FLUX}.png'


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
# A table on the left and the bars on the right, sharing one row grid. The axis carries
# absolute kPa, the value and its interval are printed, and the table half mirrors
# main-text Table 1 so the two read as one argument.
#
# One blue throughout. Coloring the movers red made the eye go to two rows out of eighteen
# and read them as failures, which is the opposite of what the figure says.

SHORT = {
    'Deepest available layer, all sites': 'Deepest layer, all sites',
    'Matched sites, shallowest depth': 'Matched sites, shallowest',
    'Matched sites, deepest depth': 'Matched sites, deepest',
    'Sites with 5+ SM depths, shallowest': 'Sites with 5+ SM depths, shallowest',
    'Sites with 5+ SM depths, deepest': 'Sites with 5+ SM depths, deepest',
    'Air temperature dropped from the model': 'No air temperature',
    'Mean instead of median per bin': 'Mean per bin',
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
    """Return the value column text, which names the kind of interval each row carries."""
    if hi <= lo:
        return f'{mid:.2f}   (no band)'
    if note and 'confidence' in note:
        return f'{mid:.2f}   ({lo:.2f} to {hi:.2f} CI)'
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
axt.text(COL_SIGMA, max(ypos) + 0.66, 'σ', transform=TAB, ha='right', va='center',
         fontsize=AX_LABELS_FONTSIZE * 0.68, color=COLOR_GREY, fontweight='bold')
axt.text(COL_VALUE, max(ypos) + 0.66, 'kPa [95% band]', transform=TAB, ha='left',
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
ax.text(published, 1.01, f'  main analysis {published:.2f} kPa',
        transform=ax.get_xaxis_transform(), ha='left', va='bottom', color=COLOR_REF,
        fontweight='bold', fontsize=AX_LABELS_FONTSIZE * 0.85, clip_on=False)

# No caption text inside the image. The legend goes in the caption beside the display
# item, not into the png. The numbers behind the summary line are printed below instead,
# so a rerun still reports them.
_points = [r[2] for r in rows if r[1] != 'Main analysis']
_inside = sum(pub_lo <= v <= pub_hi for v in _points)
print(f'{_inside} of {len(_points)} tests fall inside the band of the main '
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
# record-length rows make one point, and each matched pair is a two-row argument about one
# comparison. Both largest movers stay, North America and the ten-year records, so the
# selection cannot be read as flattering. All three region rows stay: Europe removed gives the
# highest threshold of all tests, so without it the table would show the low end of the
# region effect and not the high end. The estimator row is left
# out on purpose: dropping it removes one of the three things the interval column means, so
# the main table carries two instead of three.
#
# Depth takes five rows. The all-sites row is the main analysis rerun on the deepest
# layer each site has, which is the test the text names first; it mixes the 80 sites that
# never moved with 128 that did, most of them by one layer, so its shift is small. The
# matched pair holds the 128 sites that have a deeper layer fixed and changes only the
# layer, which is the clean read on depth. The five-or-more-depths pair holds 59 sites
# fixed and swaps their shallowest depth for their deepest, which is the largest departure
# from the shallow selection the main analysis rests on. The pairs answer whether depth
# moves the threshold, the all-sites row whether the main analysis survives the deepest
# layer the network offers.
MAIN_ROWS = [
    'Deepest available layer, all sites',
    'Matched sites, shallowest depth',
    'Matched sites, deepest depth',
    'Sites with 5+ SM depths, shallowest',
    'Sites with 5+ SM depths, deepest',
    'Air temperature dropped from the model',
    'Blocked cross-validation',
    'Mean instead of median per bin',
    'GAM instead of a polynomial',
    'ALE instead of SHAP',
    'Any one site removed',
    'Europe removed',
    'North America removed',
    'Europe and North America removed',
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
    'Matched sites, shallowest depth':
        'Only the sites that have a deeper layer, using their shallow layer',
    'Matched sites, deepest depth':
        'The same sites, using their deepest layer',
    'Air temperature dropped from the model':
        'The model never sees temperature, so nothing of it can reach the VPD values',
    'Sites with 5+ SM depths, shallowest':
        'The 59 sites with five or more soil water depths, using their shallowest',
    'Sites with 5+ SM depths, deepest':
        'The same 59 sites, using their deepest depth',
    'GAM instead of a polynomial':
        'A spline whose shape the data set, instead of a fourth-order polynomial',
    'Blocked cross-validation':
        'One calendar year left out at a time instead of a shuffled split',
    'Mean instead of median per bin':
        'The cross-site mean of each bin fitted instead of the median',
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


def cell(value, lo, hi, note):
    """Format a value with its interval in brackets.

    A row with equal bounds (ALE, which has no fitted curve) prints the value and an
    asterisk only. A row with a note keeps the bracket and takes an asterisk.
    """
    # Rounding first and adding zero keeps a bound like -0.004 from printing as -0.00.
    value, lo, hi = (round(v, 2) + 0.0 for v in (value, lo, hi))
    if hi <= lo:
        return f"{value:.2f} *"
    text = f"{value:.2f} [{lo:.2f}, {hi:.2f}]"
    return text + (' *' if isinstance(note, str) else '')


# Group names become sub-header rows rather than a column, which saves the width a Nature
# Communications page does not have. The prose column is gone for the same reason: the test
# names carry it, and the caption holds anything they do not.
#
# Sigma and kPa both, because every row converts with its own site set. A row can move in kPa
# while standing still in sigma, and the reader cannot see that from kPa alone. It is also
# the column that ties this table to Figure 4, which reports sigma. Each value carries its
# 95 % prediction band in brackets, as Figure 4 and the text do, which saves the interval
# column the table used to have.
COLUMNS = ['Test', f'Sites (of {int(ref["n_sites"])})', 'Threshold (σ)', 'Threshold (kPa)']


def _row(test, n, sigma, kpa):
    return dict(zip(COLUMNS, [test, n, sigma, kpa]))


records = [_row('Main analysis', int(ref['n_sites']),
                cell(ref['threshold_sigma'], ref['lower_sigma'], ref['upper_sigma'], None),
                cell(ref['threshold_kpa'], ref['lower'], ref['upper'], None))]
for group in ['Soil water depth', 'Model fitting', 'Site set']:
    block = main.loc[main['group'] == group]
    if block.empty:
        continue
    records.append(_row(group, '', '', ''))          # sub-header row
    for _, r in block.iterrows():
        records.append(_row(SHORT.get(r['test'], r['test']), int(r['n_sites']),
                            cell(r['threshold_sigma'], r['lower_sigma'], r['upper_sigma'], r['note']),
                            cell(r['threshold_kpa'], r['lower'], r['upper'], r['note'])))
table1 = pd.DataFrame(records, columns=COLUMNS)




# Which tests actually pass the cutoff, rather than a sentence that assumes one does. Adding
# the five-or-more-depths pair made the old wording false: its shallow row sits 0.069 kPa below the
# reference because it uses 59 sites, so two rows now clear 0.06 kPa rather than one.
biggest = tests.loc[tests['shift_kpa'].abs().idxmax()]
movers = tests.loc[tests['shift_kpa'].abs() >= MOVER].sort_values('shift_kpa')
mover_text = '; '.join(f"{SHORT.get(r['test'], r['test']).lower()} at "
                       f"{r['threshold_kpa']:.2f} kPa" for _, r in movers.iterrows())
footnote = (
    "The threshold is the highest zero crossing of a fourth-order polynomial fitted to the "
    "cross-site median VPD effect per bin; brackets give its 95% prediction band, as in "
    "Fig. 4. Rows marked with an asterisk have a different interval: for the "
    f"leave-one-site-out row it is the range across the {int(ref['n_sites'])} removals, and "
    "for the ALE row the 95% confidence interval of the mean per-site crossing. The two "
    "matched rows use the 128 sites that have a deeper soil water layer, and the two rows for "
    "sites with five or more soil water depths use the same 59 sites. Within each pair the "
    f"rows differ only in the soil water layer. All {len(tests)} tests, described in Methods, "
    "are shown in Supplementary Fig. 4."
)

stem = folder / f'55_TABLE-1_ThresholdRobustness_{FLUX}'
table1.to_csv(f'{stem}.csv', index=False)
with pd.ExcelWriter(f'{stem}.xlsx', engine='openpyxl') as writer:
    table1.to_excel(writer, sheet_name='Table 1', index=False, startrow=0)
    sheet = writer.sheets['Table 1']
    # Column widths, so the pasted table does not need manual fixing in Word.
    for column, width in zip('ABCD', (38, 13, 22, 22)):
        sheet.column_dimensions[column].width = width
    sheet.cell(row=len(table1) + 3, column=1, value=footnote)

print(table1.to_string(index=False))
print()
print(footnote)
print(f"\nSaved {stem}.xlsx and {stem}.csv")

if SHOW_PLOT:
    plt.show()
