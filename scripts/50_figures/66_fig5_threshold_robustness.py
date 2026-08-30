"""
PLANNED MAIN FIGURE 5, not yet adopted.

Every robustness test of the VPD threshold on one axis. The published estimate is the
reference line; each row shows what the threshold becomes when one methodological choice
is changed. Nothing is recomputed here, the script only reads what scripts 58 and 60 to
64 already wrote.

The figure is labelled as planned in its title and its file name, so it cannot be mistaken
for an adopted display item. Decide later whether it goes in the manuscript.

Not covered here: the blocked cross-validation run, which only has stage 44 output and so
has no response curve to take a threshold from, and the matched deep against shallow site
pair, which reports a ratio rather than a threshold.
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

rows = []  # (label, values in kPa, n shown)

COEFF_NAME = (f'54_FIG-4_ResponseCurve_ShapMeans_{FLUX}_BIN_VPD_ZSCORE'
              f'+VPD_ZSCORE_SHAPVALS+TA_ZSCORE_DATA_COEFFICIENTS.csv')


def threshold_z(*subfolders):
    """All-sites threshold in sigma from a run's Figure 4 coefficient file."""
    path = folder.joinpath(*subfolders) / COEFF_NAME
    coeff = pd.read_csv(path)
    return float(coeff.loc[coeff['IGBP'] == 'ALL SITES', 'Threshold'].iloc[0].split('[')[0])


# Deepest available soil water layer. This run is a mixture: all 208 sites stay in, but
# only 128 have a usable layer below the first, so 80 are still on layer 1.
rows.append(('Deepest available soil water layer (128 of 208 sites move)',
             to_kpa([threshold_z('deep-sm')])))

# The matched pair isolates depth from site composition: the same 128 sites, once on
# layer 1 and once on their deepest layer. These are 128-site runs, so they belong next to
# each other and not next to the published 208-site value.
to_kpa_matched = converter(common.deepest_swc_per_site().keys())
rows.append(('Matched 128 sites, layer 1',
             to_kpa_matched([threshold_z('deeper-only')])))
rows.append(('Matched 128 sites, deepest layer',
             to_kpa_matched([threshold_z('deep-sm', 'deeper-only')])))

# Temporally blocked cross-validation, leave one calendar year out. Sites with a single
# year cannot be split that way and are skipped, so this run has its own site set.
bcv_dir = (Path(settings['DIR_DATA_OUT_SHAP_ANALYSIS']) / FLUX / shap_type / 'blocked-cv')
bcv_sites = [f.name.split('_shap-')[0]
             for f in bcv_dir.glob(f'*_shap-{shap_type}_{FLUX}.parquet')]
rows.append((f'Blocked cross-validation ({len(bcv_sites)} sites)',
             converter(bcv_sites)([threshold_z('blocked-cv')])))

# Estimator choice: polynomial degree, lowess span, spline smoothing, bin width, fit range
method = pd.read_csv(folder / '60_Threshold_MethodSensitivity.csv')
rows.append((f'Threshold estimator ({len(method)} variants)', to_kpa(method['threshold_z'])))

# Whole regions removed
region = pd.read_csv(folder / '63_LeaveRegionOut.csv')
region = region.loc[region['set'] != 'all sites']
rows.append(('Region removed (Europe, North America)', region['threshold_kpa'].values))

# Progressively longer minimum record
record = pd.read_csv(folder / '64_ShortRecordSensitivity.csv')
rows.append((f"Minimum record length ({record['min_years'].min()} to "
             f"{record['min_years'].max()} years)", record['threshold_kpa'].values))

# Sites split by their own VPD range
strata = pd.read_csv(folder / '62_Threshold_vs_SiteVPDRange.csv')
rows.append(('Sites split by their own VPD range (quartiles)',
             strata['median_threshold_kpa'].values))

fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)

ax.axvspan(pub_lo, pub_hi, color=COLOR_REF, alpha=0.12, zorder=0)
ax.axvline(published, color=COLOR_REF, lw=1.6, zorder=1)
ax.text(published, len(rows) - 0.35, f'  published, {published:.2f} kPa',
        color=COLOR_REF, fontsize=AX_LABELS_FONTSIZE * 0.85, va='bottom', ha='left')

rng = np.random.default_rng(0)
for y, (label, vals) in enumerate(reversed(rows)):
    vals = np.asarray(vals, dtype=float)
    jitter = rng.normal(0, 0.055, len(vals)) if len(vals) > 3 else np.zeros(len(vals))
    ax.scatter(vals, y + jitter, s=26, color=COLOR_POINT, alpha=0.65, linewidths=0, zorder=3)
    ax.plot([vals.min(), vals.max()], [y, y], color=COLOR_POINT, lw=1.2, alpha=0.5, zorder=2)
    ax.text(vals.max() + 0.02, y, f'  {vals.min():.2f} to {vals.max():.2f}',
            fontsize=AX_LABELS_FONTSIZE * 0.8, va='center', color='#444444')

ax.set_yticks(range(len(rows)))
ax.set_yticklabels([label for label, _ in reversed(rows)], fontsize=AX_LABELS_FONTSIZE)
ax.set_xlabel('VPD threshold (kPa)', fontsize=AX_LABELS_FONTSIZE)
ax.tick_params(axis='x', labelsize=AX_LABELS_FONTSIZE)
ax.set_ylim(-0.6, len(rows) - 0.25)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis='y', length=0)
ax.grid(axis='x', color='#EEEEEE', zorder=0)

ax.set_title('PLANNED main Fig. 5, not yet adopted', fontsize=AX_LABELS_FONTSIZE,
             loc='left', color=COLOR_REF, fontweight='bold')

fig.tight_layout()
outfile = folder / f'66_PLANNED-FIG-5_ThresholdRobustness_{FLUX}.png'
fig.savefig(outfile, dpi=300, facecolor='white', bbox_inches='tight')

out_data = pd.DataFrame([(label, v) for label, vals in rows for v in vals],
                        columns=['test', 'threshold_kpa'])
out_data.to_csv(str(outfile).replace('.png', '_DATA.csv'), index=False)
print(f"Saved to {outfile}")
if SHOW_PLOT:
    plt.show()
