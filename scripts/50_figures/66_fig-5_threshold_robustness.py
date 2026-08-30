"""
PLANNED MAIN FIGURE 5, not yet adopted.

Every robustness test of the VPD threshold on one axis. The published estimate is the
reference line; each row shows what the threshold becomes when one methodological choice
is changed. Nothing is recomputed here, the script only reads what scripts 58 and 60 to
64 already wrote.

Rows are grouped by what the test varies, so eight rows read as four questions: does the
soil water layer matter, does the model fitting matter, does the site set matter, does the
site climate matter. The answer is the same in every group.

The figure is labelled as planned in its file name, so it cannot be mistaken for an adopted
display item. Decide later whether it goes in the manuscript.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import common
from src.paths import data_path, load_settings

SHOW_PLOT = False

FLUX = 'NEP_ZSCORE'
CONDITIONAL = True
VARIANT = ""
SITE_SUBSET = ""

AX_LABELS_FONTSIZE = 12
COLOR_POINT = '#0072B2'
COLOR_REF = '#D55E00'

shap_type = 'conditional' if CONDITIONAL else 'interventional'
settings = load_settings()
folder = Path(settings['DIR_PLOTS_OUT']) / FLUX / shap_type / VARIANT / SITE_SUBSET

# Sigma to kPa. The conversion is a per-site mean plus z times a per-site standard
# deviation, averaged over sites, so it is linear in z. It also depends on which sites are
# in the run, so a 128-site variant must not be converted with the 208-site mapping.
# CD-Ygb is excluded because it records VPD in Pa where every other site uses hPa.
subsets = pd.read_csv(data_path('data/outputs/20_subsets/'
                                '21_SUBSETS_parquet_vars_stats_subsets.csv'))
subsets = subsets.loc[subsets['SITE'] != 'CD-Ygb']


def converter(sites=None):
    """Return a function turning sigma into kPa for one site set."""
    use = subsets if sites is None else subsets.loc[subsets['SITE'].isin(sites)]
    a_, b_ = use['VPD_Z0'].mean() / 10, use['VPD_SD'].mean() / 10  # hPa to kPa
    return lambda z: a_ + b_ * np.asarray(z, dtype=float)


to_kpa = converter()

# Published estimate, taken as written rather than reconverted.
conv = pd.read_csv(folder / '58_Threshold_zscore_to_kPa.csv').set_index('group').loc['ALL SITES']

published, pub_lo, pub_hi = conv['kPa'], conv['kPa_lower'], conv['kPa_upper']

rows = []  # (group, label, values in kPa)

COEFF_NAME = (f'54_FIG-4_ResponseCurve_ShapMeans_{FLUX}_BIN_VPD_ZSCORE'
              f'+VPD_ZSCORE_SHAPVALS+TA_ZSCORE_DATA_COEFFICIENTS.csv')


def threshold_z(*subfolders):
    """All-sites threshold in sigma from a run's Figure 4 coefficient file."""
    path = folder.joinpath(*subfolders) / COEFF_NAME
    coeff = pd.read_csv(path)
    return float(coeff.loc[coeff['IGBP'] == 'ALL SITES', 'Threshold'].iloc[0].split('[')[0])


# Deepest available soil water layer. This run is a mixture: all 208 sites stay in, but
# only 128 have a usable layer below the first, so 80 are still on layer 1.
rows.append(('Soil water depth', 'Deepest available layer, all sites',
             to_kpa([threshold_z('deep-sm')])))

# The matched pair isolates depth from site composition: the same 128 sites, once on
# layer 1 and once on their deepest layer. These are 128-site runs, so they belong next to
# each other and not next to the published 208-site value.
to_kpa_matched = converter(common.deepest_swc_per_site().keys())
rows.append(('Soil water depth', 'Matched sites, layer 1',
             to_kpa_matched([threshold_z('deeper-only')])))
rows.append(('Soil water depth', 'Matched sites, deepest layer',
             to_kpa_matched([threshold_z('deep-sm', 'deeper-only')])))

# Temporally blocked cross-validation, leave one calendar year out. Sites with a single
# year cannot be split that way and are skipped, so this run has its own site set.
bcv_dir = (Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / 'blocked-cv')
bcv_sites = [f.name.split('_shap-')[0]
             for f in bcv_dir.glob(f'*_shap-{shap_type}_{FLUX}.parquet')]
rows.append(('Model fitting', 'Blocked cross-validation',
             converter(bcv_sites)([threshold_z('blocked-cv')])))

# Estimator choice: polynomial degree, lowess span, spline smoothing, bin width, fit range
method = pd.read_csv(folder / '60_Threshold_MethodSensitivity.csv')
rows.append(('Model fitting', f'Threshold estimator, {len(method)} variants',
             to_kpa(method['threshold_z'])))

# Whole regions removed
region = pd.read_csv(folder / '63_LeaveRegionOut.csv')
region = region.loc[region['set'] != 'all sites']
rows.append(('Site set', 'Europe or North America removed',
             region['threshold_kpa'].values))

# Progressively longer minimum record
record = pd.read_csv(folder / '64_ShortRecordSensitivity.csv')
rows.append(('Site set', f"Minimum record length, {record['min_years'].min()} to "
                         f"{record['min_years'].max()} years",
             record['threshold_kpa'].values))

# Sites split by their own VPD range
strata = pd.read_csv(folder / '62_Threshold_vs_SiteVPDRange.csv')
rows.append(('Site climate', 'Sites split by their own VPD range, quartiles',
             strata['median_threshold_kpa'].values))

# Rows bottom to top, with a gap between groups so the four questions separate.
ordered = list(reversed(rows))
ypos, group_rows, y = [], {}, 0.0
prev_group = None
for group, label, vals in ordered:
    if prev_group is not None and group != prev_group:
        y += 0.9  # gap between groups
    ypos.append(y)
    group_rows.setdefault(group, []).append(y)
    prev_group = group
    y += 1.0

fig, ax = plt.subplots(figsize=(8.2, 5.4), dpi=150)

ax.axvspan(pub_lo, pub_hi, color=COLOR_REF, alpha=0.12, zorder=0)
ax.axvline(published, color=COLOR_REF, lw=1.6, zorder=1)

rng = np.random.default_rng(0)
for (group, label, vals), y in zip(ordered, ypos):
    vals = np.asarray(vals, dtype=float)
    jitter = rng.normal(0, 0.06, len(vals)) if len(vals) > 3 else np.zeros(len(vals))
    ax.plot([vals.min(), vals.max()], [y, y], color=COLOR_POINT, lw=1.3, alpha=0.45,
            zorder=2, solid_capstyle='round')
    ax.scatter(vals, y + jitter, s=26, color=COLOR_POINT, alpha=0.7, linewidths=0,
               zorder=3)

ax.set_yticks(ypos)
ax.set_yticklabels([label for _, label, _ in ordered], fontsize=AX_LABELS_FONTSIZE * 0.92)
ax.set_ylim(min(ypos) - 0.8, max(ypos) + 1.0)
ax.set_xlim(0.96, 1.45)
ax.set_xlabel('VPD threshold (kPa)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE)
ax.tick_params(axis='y', length=0)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.grid(axis='x', color='#EEEEEE', zorder=0)

# The reference label sits above the plot area, clear of the topmost row.
ax.annotate(f'published, {published:.2f} kPa', xy=(published, 1.0),
            xycoords=('data', 'axes fraction'), xytext=(4, 4),
            textcoords='offset points', color=COLOR_REF, ha='left', va='bottom',
            fontsize=AX_LABELS_FONTSIZE * 0.85, fontweight='bold')

# Group headers, above the first row of each group, at the left edge of the label column.
for group, ys in group_rows.items():
    ax.annotate(group, xy=(0, max(ys) + 0.55), xycoords=('axes fraction', 'data'),
                xytext=(-8, 0), textcoords='offset points', ha='right', va='center',
                fontsize=AX_LABELS_FONTSIZE * 0.78, color='#777777', fontweight='bold')

# The measured range in its own column to the right, aligned rather than trailing the
# points, so the numbers can be read down the column.
for (group, label, vals), y in zip(ordered, ypos):
    vals = np.asarray(vals, dtype=float)
    txt = (f'{vals.min():.2f}' if len(vals) == 1
           else f'{vals.min():.2f} to {vals.max():.2f}')
    ax.annotate(txt, xy=(1.0, y), xycoords=('axes fraction', 'data'), xytext=(8, 0),
                textcoords='offset points', ha='left', va='center',
                fontsize=AX_LABELS_FONTSIZE * 0.82, color='#444444')

fig.tight_layout()
outfile = folder / f'66_PLANNED-FIG-5_ThresholdRobustness_{FLUX}.png'
fig.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')

out_data = pd.DataFrame([(group, label, v) for group, label, vals in rows for v in vals],
                        columns=['group', 'test', 'threshold_kpa'])
out_data.to_csv(str(outfile).replace('.png', '_DATA.csv'), index=False)
print(f"Saved to {outfile}")
if SHOW_PLOT:
    plt.show()
